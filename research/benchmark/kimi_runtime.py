"""External Kimi CLI runtime adapter for auditable benchmark agents."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

from core.agent import load_runtime_policy


KIMI_RUNTIME_VERSION = "1.49.0"
KIMI_RUNTIME_SOURCE = "https://pypi.org/project/kimi-cli/"
RETRYABLE_EXIT_CODE = 75


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_runtime_root() -> Path:
    return Path(__file__).resolve().parents[1] / ".benchmark_runtime" / "kimi-cli"


def runtime_dir(version: str = KIMI_RUNTIME_VERSION, root: Path | None = None) -> Path:
    return (root or default_runtime_root()) / version


def runtime_executable(version: str = KIMI_RUNTIME_VERSION, root: Path | None = None) -> Path:
    base = runtime_dir(version, root)
    return base / ("Scripts/kimi.exe" if os.name == "nt" else "bin/kimi")


def _python_executable(base: Path) -> Path:
    return base / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def install_kimi_runtime(
    *, version: str = KIMI_RUNTIME_VERSION, root: Path | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Install an exact Kimi release into an isolated virtual environment."""
    target = runtime_dir(version, root)
    target.mkdir(parents=True, exist_ok=True)
    base_python = python_executable
    if base_python is None:
        base_python = sys.executable if sys.version_info[:2] == (3, 12) else shutil.which("python3.12")
    if not base_python:
        raise RuntimeError("Kimi benchmark runtime requires a Python 3.12 executable")
    spec = {
        "base": "python3.12",
        "pip_phases": [[f"kimi-cli=={version}"]],
        "runtime": "local-venv",
        "smoke": {"cli_checks": ["kimi --version"]},
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    (target / "env-spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    python = _python_executable(target)
    if not python.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [base_python, "-m", "venv", str(target)],
            check=True,
        )
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check",
         f"kimi-cli=={version}"],
        check=True,
    )
    info = check_kimi_runtime(version=version, root=root)
    info["env_spec_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    lock = target / "deep-dream-runtime-lock.json"
    lock.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def check_kimi_runtime(
    *, version: str = KIMI_RUNTIME_VERSION, root: Path | None = None,
    executable: Path | None = None,
) -> dict[str, Any]:
    """Verify the executable, version and immutable entrypoint fingerprint."""
    binary = executable or runtime_executable(version, root)
    if not binary.exists():
        raise FileNotFoundError(
            f"Kimi runtime is not installed: {binary}. Run `deep-dream benchmark runtime install`."
        )
    completed = subprocess.run(
        [str(binary), "--version"], text=True, capture_output=True, timeout=30,
    )
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    output = (completed.stdout or completed.stderr).strip()
    if version not in output:
        raise RuntimeError(f"Expected Kimi {version}, got: {output}")
    return {
        "runtime": "kimi",
        "version": version,
        "source": KIMI_RUNTIME_SOURCE,
        "executable": str(binary.resolve()),
        "executable_sha256": _sha256(binary),
        "version_output": output,
        "python": str(_python_executable(binary.parent.parent).resolve()),
    }


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _write_runtime_config(
    share_dir: Path, *, base_url: str, model: str, max_steps: int,
    context_window: int, thinking: bool,
) -> None:
    share_dir.mkdir(parents=True, exist_ok=True)
    config = f'''default_model = "deep-dream-qwen"
default_thinking = {str(thinking).lower()}
default_yolo = true
default_plan_mode = false
show_thinking_stream = false
merge_all_available_skills = false
telemetry = false

[providers.deep-dream-dashscope]
type = "openai_legacy"
base_url = {_toml_string(base_url)}
api_key = ""
reasoning_key = "reasoning_content"

[models.deep-dream-qwen]
provider = "deep-dream-dashscope"
model = {_toml_string(model)}
max_context_size = {int(context_window)}
capabilities = ["thinking"]

[loop_control]
max_steps_per_turn = {int(max_steps)}
max_retries_per_step = 3
max_ralph_iterations = 0
reserved_context_size = 4096
compaction_trigger_ratio = 0.85

[mcp.client]
tool_call_timeout_ms = 120000
'''
    (share_dir / "config.toml").write_text(config, encoding="utf-8")


def _write_agent_files(
    work_dir: Path, *, agent_directed_steps: bool = False,
) -> tuple[Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = work_dir / "system.md"
    policy = load_runtime_policy().strip()
    prompt_path.write_text(
        f"""# Deep-Dream LoCoMo Memory Agent

{policy}

## Benchmark contract

- You may use only the Deep-Dream MCP tools exposed in this run.
- Never ask the user, use a shell, access files, browse the web, or call subagents.
- The question category, reference answer, and gold evidence are unavailable by design.
- Before finishing, call `submit_evidence` with only IDs actually returned by tools.
{("- Decide autonomously how many retrieval steps are needed. Continue while another query can materially improve the evidence." if agent_directed_steps else "- By step 10, stop exploring and call `submit_evidence` with the best evidence already found\n  (or three empty ID arrays). Never spend the final two steps on another search.")}
- After the tool accepts the submission, return exactly one JSON object and no prose:
  {{"answer":"concise final answer","session_ids":[],"episode_ids":[],"turn_ids":[],"confidence":0.0,"stop_reason":"submit_evidence"}}
- Copy the accepted evidence IDs into the final object. The answer may make ordinary inferences,
  but memory claims must be supported by the submitted original session/turn evidence.
- If the evidence is insufficient or the question has a false premise, say so concisely.
""",
        encoding="utf-8",
    )
    agent_path = work_dir / "agent.yaml"
    agent_path.write_text(
        "version: 1\nagent:\n  name: deep-dream-benchmark\n"
        "  system_prompt_path: ./system.md\n  tools: []\n",
        encoding="utf-8",
    )
    return agent_path, prompt_path


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S | re.I)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(stripped)
    first, last = stripped.find("{"), stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first:last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Kimi final message is not a JSON object")


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [clean for item in value if (clean := _sanitize_value(item)) is not None]
    if isinstance(value, dict):
        if str(value.get("type") or "").lower() in {"think", "thinking", "reasoning"}:
            return None
        return {
            key: clean for key, item in value.items()
            if key not in {"reasoning_content", "thinking", "thinking_content", "analysis"}
            and (clean := _sanitize_value(item)) is not None
        }
    return value


def _sanitize_event(value: dict[str, Any], *, final: bool = False) -> dict[str, Any]:
    """Keep auditable actions while dropping hidden reasoning text."""
    clean = _sanitize_value(value) or {}
    if clean.get("role") == "assistant" and not final:
        clean.pop("content", None)
    return clean


def _event_usage(event: dict[str, Any]) -> tuple[int, int]:
    usage = event.get("usage") or event.get("token_usage") or {}
    return (
        int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
    )


def _accepted_tool_content(value: Any) -> bool:
    if isinstance(value, str):
        try:
            return _accepted_tool_content(json.loads(value))
        except json.JSONDecodeError:
            return False
    if isinstance(value, dict):
        if value.get("accepted") is True:
            return True
        return any(_accepted_tool_content(item) for item in value.values())
    if isinstance(value, list):
        return any(_accepted_tool_content(item) for item in value)
    return False


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_content_text(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") == "text" and value.get("text") is not None:
            return str(value["text"])
        return _content_text(value.get("content") or value.get("text") or "")
    return ""


def _id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if not isinstance(value, list):
        raise ValueError("Evidence IDs must be an array")
    return list(dict.fromkeys(str(item) for item in value if str(item)))


@dataclass(slots=True)
class KimiAgentResult:
    final: dict[str, Any]
    events: list[dict[str, Any]]
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    tool_counts: dict[str, int]
    steps: int
    exit_code: int


class KimiAgentRuntime:
    """Launch one isolated Kimi print-mode process for one benchmark question."""

    def __init__(
        self, *, executable: Path, run_dir: Path, config_path: Path,
        model: str, thinking: bool, max_steps: int = 12, timeout_seconds: int = 600,
        context_window: int = 32000, api_key: str | None = None,
        lifecycle: str = "per-question", agent_step_policy: str = "autonomous",
    ):
        if lifecycle not in {"per-question", "persistent"}:
            raise ValueError("lifecycle must be per-question or persistent")
        if agent_step_policy not in {"legacy", "autonomous"}:
            raise ValueError("agent_step_policy must be legacy or autonomous")
        self.executable = executable
        self.run_dir = run_dir.resolve()
        self.config_path = config_path.resolve()
        self.model = model
        self.thinking = thinking
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.context_window = context_window
        self.api_key = api_key
        self.lifecycle = lifecycle
        self.agent_step_policy = agent_step_policy
        self._bridges: dict[int, subprocess.Popen[str]] = {}
        self._bridge_lock = threading.Lock()

    def run(self, *, scope_id: str, question_id: str, question: str,
            question_date: str = "") -> KimiAgentResult:
        if self.lifecycle == "persistent":
            return self._run_persistent(
                scope_id=scope_id, question_id=question_id, question=question,
                question_date=question_date,
            )
        safe = hashlib.sha256(question_id.encode()).hexdigest()[:16]
        work_dir = self.run_dir / "kimi_runtime" / safe
        share_dir = work_dir / "share"
        agent_path, _ = _write_agent_files(work_dir)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        llm = config.get("llm") or {}
        base_url = str(llm.get("base_url") or "").rstrip("/")
        if not base_url:
            raise ValueError("Kimi Agent requires llm.base_url")
        key_env = str(llm.get("api_key_env") or "OPENAI_API_KEY")
        api_key = str(self.api_key or os.getenv(key_env) or llm.get("api_key") or "")
        if not api_key:
            raise RuntimeError(f"Required API key environment variable is not set: {key_env}")
        _write_runtime_config(
            share_dir, base_url=base_url, model=self.model, max_steps=self.max_steps,
            context_window=int(llm.get("context_window_tokens") or self.context_window),
            thinking=self.thinking,
        )
        mcp_path = work_dir / "mcp.json"
        mcp_path.write_text(json.dumps({
            "mcpServers": {
                "deep-dream": {
                    "command": sys.executable,
                    "args": [
                        "-m", "research.benchmark.mcp_server",
                        "--run-dir", str(self.run_dir),
                        "--scope-id", scope_id,
                        "--config", str(self.config_path),
                    ],
                    "env": {
                        "PYTHONPATH": str(_project_root()),
                        "DEEPDREAM_JSON_OUTPUT": "1",
                        "DEEPDREAM_LOG_LEVEL": "ERROR",
                    },
                }
            }
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        prompt = f"Question: {question}"
        if question_date:
            prompt += f"\nQuestion date: {question_date}"
        prompt += "\nUse Deep-Dream autonomously, submit source evidence, then return the required JSON."
        command = [
            str(self.executable), "--print", "-p", prompt,
            "--output-format=stream-json",
            "--agent-file", str(agent_path), "--mcp-config-file", str(mcp_path),
            "--max-steps-per-turn", str(self.max_steps),
            "--thinking" if self.thinking else "--no-thinking",
        ]
        env = os.environ.copy()
        env.update({
            "KIMI_SHARE_DIR": str(share_dir),
            "KIMI_CLI_NO_AUTO_UPDATE": "1",
            "OPENAI_BASE_URL": base_url,
            "OPENAI_API_KEY": api_key,
        })
        started = time.monotonic()
        completed = subprocess.run(
            command, cwd=str(_project_root()), env=env, text=True,
            capture_output=True, timeout=self.timeout_seconds,
        )
        latency = round(time.monotonic() - started, 3)
        if completed.returncode:
            message = "\n".join(
                part for part in (completed.stdout, completed.stderr) if part
            )[-4000:].strip().replace(api_key, "[REDACTED]")
            error = RuntimeError(f"Kimi exited {completed.returncode}: {message}")
            lowered = message.lower()
            setattr(error, "retryable", bool(
                completed.returncode == RETRYABLE_EXIT_CODE
                or "resume this session" in lowered
                or "429" in lowered
                or re.search(r"\b5\d\d\b", lowered)
            ))
            raise error
        raw_events = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                raw_events.append(value)
        assistant = [row for row in raw_events if row.get("role") == "assistant"]
        if not assistant:
            raise ValueError("Kimi stream contained no assistant message")
        final_text = _content_text(assistant[-1].get("content"))
        final = _parse_json_object(final_text)
        final["answer"] = str(final.get("answer") or "").strip()
        if not final["answer"]:
            raise ValueError("Kimi final JSON has an empty answer")
        for key in ("session_ids", "episode_ids", "turn_ids"):
            try:
                final[key] = _id_list(final.get(key))
            except ValueError as exc:
                raise ValueError(f"Kimi final JSON field {key} must be an array") from exc
        prompt_tokens = completion_tokens = 0
        tool_counts: dict[str, int] = {}
        steps = 0
        submitted: dict[str, list[str]] | None = None
        submit_call_ids: set[str] = set()
        for event in raw_events:
            prompt_count, completion_count = _event_usage(event)
            prompt_tokens += prompt_count
            completion_tokens += completion_count
            for call in event.get("tool_calls") or []:
                function = call.get("function") or {}
                name = str(function.get("name") or call.get("name") or "")
                if name:
                    tool_counts[name] = tool_counts.get(name, 0) + 1
                    steps += 1
                if name.endswith("submit_evidence"):
                    if call.get("id"):
                        submit_call_ids.add(str(call["id"]))
                    arguments = function.get("arguments") or call.get("arguments") or {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    if isinstance(arguments, dict):
                        submitted = {
                            key: _id_list(arguments.get(key))
                            for key in ("session_ids", "episode_ids", "turn_ids")
                        }
        if submitted is None:
            raise ValueError("Kimi did not call submit_evidence")
        accepted = False
        for event in raw_events:
            if event.get("role") != "tool" or str(event.get("tool_call_id") or "") not in submit_call_ids:
                continue
            if _accepted_tool_content(event.get("content")):
                accepted = True
        if not accepted:
            raise ValueError("Kimi submit_evidence call was not accepted")
        for key, accepted in submitted.items():
            if final[key] != accepted:
                raise ValueError(f"Kimi final {key} does not match submitted evidence")
        events = [
            _sanitize_event(row, final=index == len(raw_events) - 1)
            for index, row in enumerate(raw_events)
        ]
        return KimiAgentResult(
            final=final, events=events, latency_seconds=latency,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            tool_counts=tool_counts, steps=steps, exit_code=completed.returncode,
        )

    def _bridge(self) -> subprocess.Popen[str]:
        worker_id = threading.get_ident()
        with self._bridge_lock:
            process = self._bridges.get(worker_id)
            if process is not None and process.poll() is None:
                return process
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            llm = config.get("llm") or {}
            base_url = str(llm.get("base_url") or "").rstrip("/")
            key_env = str(llm.get("api_key_env") or "OPENAI_API_KEY")
            api_key = str(self.api_key or os.getenv(key_env) or llm.get("api_key") or "")
            if not base_url or not api_key:
                raise RuntimeError(f"Kimi persistent runtime is missing {key_env} or llm.base_url")
            root = self.run_dir / "kimi_runtime_persistent" / f"worker-{len(self._bridges) + 1}"
            share_dir = root / "share"
            agent_path, _ = _write_agent_files(
                root, agent_directed_steps=self.agent_step_policy == "autonomous",
            )
            _write_runtime_config(
                share_dir, base_url=base_url, model=self.model, max_steps=self.max_steps,
                context_window=int(llm.get("context_window_tokens") or self.context_window),
                thinking=self.thinking,
            )
            runtime_python = _python_executable(self.executable.parent.parent)
            bridge_path = Path(__file__).with_name("kimi_persistent_bridge.py")
            command = [
                str(runtime_python), str(bridge_path),
                "--project-root", str(_project_root()), "--config", str(share_dir / "config.toml"),
                "--agent-file", str(agent_path), "--max-steps", str(self.max_steps),
            ]
            if self.thinking:
                command.append("--thinking")
            env = os.environ.copy()
            env.update({
                "KIMI_SHARE_DIR": str(share_dir),
                "KIMI_CLI_NO_AUTO_UPDATE": "1",
                "KIMI_DISABLE_TELEMETRY": "1",
                "OPENAI_BASE_URL": base_url,
                "OPENAI_API_KEY": api_key,
                "PYTHONPATH": str(_project_root()),
            })
            process = subprocess.Popen(
                command, cwd=str(_project_root()), env=env, text=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=1,
            )
            self._bridges[worker_id] = process
            return process

    def _run_persistent(
        self, *, scope_id: str, question_id: str, question: str, question_date: str,
    ) -> KimiAgentResult:
        process = self._bridge()
        prompt = f"Question: {question}"
        if question_date:
            prompt += f"\nQuestion date: {question_date}"
        prompt += "\nUse Deep-Dream autonomously, submit source evidence, then return the required JSON."
        request = {
            "request_id": question_id,
            "run_dir": str(self.run_dir),
            "scope_id": scope_id,
            "service_config": str(self.config_path),
            "mcp_python": sys.executable,
            "prompt": prompt,
            "timeout_seconds": self.timeout_seconds,
        }
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"Kimi persistent bridge exited {process.poll()}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(
                f"Kimi persistent bridge error {response.get('error_type')}: {response.get('error')}"
            )
        if response.get("request_id") != question_id or response.get("fresh_context") is not True:
            raise RuntimeError("Kimi persistent bridge violated fresh-context contract")
        updates = list(response.get("updates") or [])
        text_chunks = []
        calls: dict[str, dict[str, Any]] = {}
        tool_counts: dict[str, int] = {}
        submitted: dict[str, list[str]] | None = None
        accepted = False
        for update in updates:
            kind = update.get("sessionUpdate")
            if kind == "agent_message_chunk":
                text_chunks.append(_content_text(update.get("content")))
            elif kind in {"tool_call", "tool_call_update"}:
                call_id = str(update.get("toolCallId") or "")
                state = calls.setdefault(call_id, {})
                title = str(update.get("title") or state.get("title") or "")
                if title:
                    state["title"] = title
                content = _content_text(update.get("content"))
                if content:
                    if update.get("status") == "completed":
                        state["result"] = content
                    else:
                        state["arguments"] = content
                if kind == "tool_call":
                    name = title.split(":", 1)[0]
                    state["name"] = name
                    tool_counts[name] = tool_counts.get(name, 0) + 1
                if update.get("status") == "completed":
                    name = str(state.get("name") or "")
                    if name.endswith("submit_evidence"):
                        try:
                            args = json.loads(str(state.get("arguments") or "{}"))
                        except json.JSONDecodeError:
                            args = {}
                        try:
                            accepted_result = json.loads(str(state.get("result") or "{}"))
                        except json.JSONDecodeError:
                            accepted_result = {}
                        accepted = bool(accepted_result.get("accepted") is True)
                        source = accepted_result if accepted else args
                        submitted = {
                            key: _id_list(source.get(key))
                            for key in ("session_ids", "episode_ids", "turn_ids")
                        }
        final_text = "".join(text_chunks).strip()
        try:
            final = _parse_json_object(final_text)
        except ValueError:
            # Some OpenAI-compatible models occasionally obey the evidence
            # contract but return a plain-text answer instead of the requested
            # JSON wrapper. Evidence acceptance remains the trust boundary.
            evidence_accepted = submitted is not None and accepted
            submitted = submitted or {
                "session_ids": [], "episode_ids": [], "turn_ids": [],
            }
            final = {
                "answer": final_text or "Insufficient evidence.",
                **submitted,
                "confidence": 0.0,
                "stop_reason": (
                    "submit_evidence" if evidence_accepted else "agent_exhausted_without_evidence"
                ),
                "format_fallback": (
                    "plain-text-after-accepted-evidence"
                    if evidence_accepted else "plain-text-without-evidence"
                ) if final_text else "empty-output-abstention",
            }
        final["answer"] = str(final.get("answer") or "").strip()
        if not final["answer"]:
            raise ValueError("Kimi final JSON has an empty answer")
        for key in ("session_ids", "episode_ids", "turn_ids"):
            final[key] = _id_list(final.get(key))
        no_evidence_fallback = final.get("format_fallback") in {
            "plain-text-without-evidence", "empty-output-abstention",
        }
        if (submitted is None or not accepted) and not no_evidence_fallback:
            raise ValueError("Kimi submit_evidence call was not accepted")
        if submitted is not None:
            for key, values in submitted.items():
                if final[key] != values:
                    raise ValueError(f"Kimi final {key} does not match submitted evidence")
        events = [_sanitize_event(update) for update in updates]
        events.append({
            "type": "runtime_session",
            "session_id": response["session_id"],
            "bridge_pid": response["bridge_pid"],
            "fresh_context": True,
        })
        return KimiAgentResult(
            final=final, events=events,
            latency_seconds=float(response.get("latency_seconds") or 0),
            prompt_tokens=0, completion_tokens=0,
            tool_counts=tool_counts, steps=sum(tool_counts.values()), exit_code=0,
        )

    def close(self) -> None:
        with self._bridge_lock:
            processes = list(self._bridges.values())
            self._bridges.clear()
        for process in processes:
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
