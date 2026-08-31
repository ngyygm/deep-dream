"""BigCodeBench 直答（direct-completion）轨道：kimi-k3 单轮生成 → 官方 sanitize/evaluate。

区别于 agent 轨道：每题一次 chat 调用，官方 instruct 协议（chat 模型用
``instruct_prompt``，响应为 markdown 代码块），产物 ``raw.<tag>.jsonl`` 的
``{task_id, solution}`` 即官方 sanitize/evaluate 的输入格式。

链路（bigcode-venv 侧执行后两步）::

    # 1) 生成（repo .venv）
    .venv/bin/python -m research.benchmark.bigcode_track <run_dir> \
        --config research/service_config.kimi-par.json --workers 4
    # 2) 净化 + 3) 评测（bigcode-venv，见下方 macOS 坑位说明）
    export BIGCODEBENCH_OVERRIDE_PATH=$PWD/research/.benchmark_data/bigcodebench/BigCodeBench-v0.1.4.jsonl
    $BCB_VENV/bin/bigcodebench.sanitize <run_dir>/raw.completion.jsonl --calibrate
    $BCB_VENV/bin/python -m research.benchmark.bigcode_eval_local \
        <run_dir>/raw.completion-sanitized-calibrated.jsonl - 4 "$(paste -sd, <run_dir>/ids.completion.txt)"

macOS arm64 本地评测的四个坑（2026-08-24 实测）：
1. ``reliability_guard`` 的 setrlimit(RLIMIT_AS/RLIMIT_DATA) 在 Darwin 任何值
   都抛 "current limit exceeds maximum limit" → 必须 --max_as_limit 0
   --max_data_limit 0 跳过 guard（直接调 evaluate() 传 0）。
2. fire 会把 ``--pass_k 1`` 解析成 int，内部 ``for k in passk`` 直接
   TypeError —— 必须传字符串（直接调 evaluate(pass_k="1")）。
3. GT 结果缓存在 ~/Library/Caches/bigcodebench/<md5>.pkl，首次全量跑一次后
   复用；换数据文件（hash 变化）自动失效。
4. 端点 kimi-k3 的模板不认 enable_thinking=false，必须额外传
   chat_template_kwargs.thinking=false，否则隐性 thinking 会吃光 completion
   预算（max_tokens=4096 时 content 为空、finish=length）。

id 选择：``--ids-file``（每行一个 task_id）或 ``--n``（数据集前 N 题）。
``--prepare-ids`` 只做 AST import 分析并落盘 ``smoke50.ids``（纯 stdlib 前 50
题，本机可执行）与 ``full.ids``（剔除 BLOCKED_LIBS，macOS 装不上的重库）。
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import textwrap
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_FILE = REPO_ROOT / ".benchmark_data" / "bigcodebench" / "BigCodeBench-v0.1.4.jsonl"
if not DEFAULT_TASKS_FILE.exists():  # 直接从源码树跑（未安装为包）时的相对布局兜底
    DEFAULT_TASKS_FILE = REPO_ROOT / "research" / ".benchmark_data" / "bigcodebench" / "BigCodeBench-v0.1.4.jsonl"

# 官方 chat 协议前缀：bigcodebench/generate.py run_codegen 的默认值，保持一致
# 以便与官方 leaderboard 口径可比。
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)

# macOS（arm64）装不上或跑不动的任务依赖 → full.ids 直接剔除（约 8 题）
BLOCKED_LIBS = {"tensorflow", "keras", "geopandas", "shapely", "pytesseract"}

# 重试：指数退避，覆盖限流/网络抖动；端点超时由 openai client 控制
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 4.0


# ---------------------------------------------------------------------------
# 数据与 id 选择
# ---------------------------------------------------------------------------

def load_tasks(tasks_file: Path) -> list[dict[str, Any]]:
    """按行读取任务 jsonl，保留数据集顺序（--n first-N 依赖该顺序）。"""
    tasks: list[dict[str, Any]] = []
    with tasks_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    if not tasks:
        raise ValueError(f"No tasks loaded from {tasks_file}")
    return tasks


def _root_module(name: str) -> str:
    return name.split(".", 1)[0].strip()


def _parse_lenient(code: str) -> ast.Module | None:
    """宽松解析 BigCodeBench 的非完整片段。

    complete_prompt 以悬空 ``def`` 头结尾、canonical_solution 是缩进的函数体
    片段，二者单独 ``ast.parse`` 都会 IndentationError/SyntaxError。逐级尝试：
    原文 → dedent（体片段）→ dedent+补 ``pass``（悬空 def 头）。全部失败返回
    None，由调用方兜底为空集。
    """
    for candidate in (code, textwrap.dedent(code), textwrap.dedent(code) + "\n    pass\n"):
        try:
            # docstring 里的非法转义（如 '\d'）会触发 SyntaxWarning，静默之
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return ast.parse(candidate)
        except (SyntaxError, ValueError):
            continue
    return None


def imported_modules(code: str) -> set[str]:
    """AST 解析 import 面；解析失败返回空集由调用方兜底。"""
    modules: set[str] = set()
    tree = _parse_lenient(code)
    if tree is None:
        return modules
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(_root_module(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                modules.add(_root_module(node.module))
            elif node.level and node.level > 0:
                # ``from . import sibling``：module 为 None，取导入名本身
                modules |= {_root_module(alias.name) for alias in node.names}
    return modules


def task_modules(task: dict[str, Any]) -> set[str]:
    """任务的完整 import 面：代码脚手架 + 参考解 + 测试三者并集。

    只看 complete_prompt 不够——canonical_solution/test 可能引入额外模块；
    取并集后 stdlib 判定才不会漏判（宁可错杀进 full，不可漏放进 smoke）。
    """
    modules: set[str] = set()
    for field in ("complete_prompt", "canonical_solution", "test"):
        modules |= imported_modules(task.get(field) or "")
    # 数据集自带的 libs 元数据（形如 "['random', 'itertools']"）一并并入
    libs = task.get("libs")
    if isinstance(libs, str):
        try:
            libs = json.loads(libs)
        except json.JSONDecodeError:
            libs = [part.strip().strip("'\"") for part in libs.strip("[]").split(",")]
    if isinstance(libs, list):
        modules |= {_root_module(str(lib)) for lib in libs if lib}
    return modules


def is_stdlib_only(task: dict[str, Any], stdlib: set[str]) -> bool:
    return bool(task_modules(task)) and task_modules(task) <= stdlib


def is_blocked(task: dict[str, Any]) -> bool:
    return bool(task_modules(task) & BLOCKED_LIBS)


def prepare_ids(tasks_file: Path, out_dir: Path, smoke_n: int = 50) -> dict[str, int]:
    """AST import 分析：smoke50.ids（纯 stdlib 前 N）+ full.ids（去 BLOCKED）。

    输出沿用 pi_track 风格：一行一个 task_id，顺序即数据集顺序。
    """
    tasks = load_tasks(tasks_file)
    stdlib = set(sys.stdlib_module_names)
    smoke_ids = [t["task_id"] for t in tasks if is_stdlib_only(t, stdlib)][:smoke_n]
    full_ids = [t["task_id"] for t in tasks if not is_blocked(t)]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke50.ids").write_text("\n".join(smoke_ids) + "\n", encoding="utf-8")
    (out_dir / "full.ids").write_text("\n".join(full_ids) + "\n", encoding="utf-8")
    stats = {
        "total": len(tasks),
        "stdlib_smoke": len(smoke_ids),
        "full": len(full_ids),
        "blocked": len(tasks) - len(full_ids),
    }
    print(
        f"prepare-ids: {stats['total']} tasks -> smoke50.ids {stats['stdlib_smoke']}"
        f" / full.ids {stats['full']} (blocked {stats['blocked']}) in {out_dir}"
    )
    return stats


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------

def _load_endpoint(config_path: Path) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    llm = cfg.get("llm") or {}
    base_url = str(llm.get("base_url") or "").rstrip("/")
    if not base_url:
        raise ValueError(f"llm.base_url missing in {config_path}")
    # 实测（2026-08-24 probe）：该端点 kimi-k3 的模板不认 enable_thinking，
    # 隐式 thinking 仍会烧掉全部 completion 预算（max_tokens=4096 时 content
    # 为空、finish=length）；thinking=False 才真正关闭。两个旗标都传，模板
    # 认哪个都能关掉。
    extra_body = llm.get("extra_body") or {}
    chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_kwargs.setdefault("enable_thinking", False)
    chat_kwargs["thinking"] = False
    return {
        "base_url": base_url,
        "api_key": str(llm.get("api_key") or "EMPTY"),
        "model": str(llm.get("model") or "kimi-k3"),
        "max_tokens": int(llm.get("max_tokens") or 4096),
        "timeout": float(llm.get("timeout_seconds") or 300),
        "extra_body": {"chat_template_kwargs": chat_kwargs},
    }


def _build_user_prompt(task: dict[str, Any]) -> str:
    """官方 chat 协议：instruction + instruct_prompt + response 引导，不预填。"""
    return (
        f"{INSTRUCTION_PREFIX}\n"
        f"{task['instruct_prompt'].strip()}\n\n"
        f"{RESPONSE_PREFIX}"
    )


def _generate_one(client: Any, endpoint: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """单题生成，带指数退避重试；solution 为模型原始输出（含 markdown 围栏）。"""
    user_prompt = _build_user_prompt(task)
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=endpoint["model"],
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0,
                max_tokens=endpoint["max_tokens"],
                extra_body=endpoint["extra_body"],
            )
            latency = time.monotonic() - started
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            usage = getattr(response, "usage", None)
            return {
                "task_id": task["task_id"],
                "solution": text,
                "prompt_chars": len(user_prompt),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "latency": round(latency, 2),
            }
        except Exception as exc:  # noqa: BLE001 - 限流/断连需重试而非终止轨道
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                wait = BACKOFF_BASE_S * (2 ** (attempt - 1))
                print(f"retry {task['task_id']} attempt={attempt} in {wait:.0f}s: {exc}", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"generation failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _select_ids(tasks: list[dict[str, Any]], ids_file: Path | None, n: int | None) -> list[str]:
    if ids_file is not None:
        wanted = [
            line.strip()
            for line in ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        known = {t["task_id"] for t in tasks}
        missing = [tid for tid in wanted if tid not in known]
        if missing:
            raise ValueError(f"{len(missing)} ids not in tasks file, e.g. {missing[:3]}")
        return wanted
    return [t["task_id"] for t in tasks[: (n or len(tasks))]]


def run_generation(args: argparse.Namespace) -> int:
    import openai  # repo .venv 已装；延迟导入便于 --prepare-ids 无依赖跑通

    run_dir: Path = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(Path(args.tasks_file))
    by_id = {t["task_id"]: t for t in tasks}
    selected = _select_ids(tasks, args.ids_file, args.n)
    # ids.<tag>.txt：评测 --selective_evaluate 的唯一事实来源，须与 raw 一一对应
    ids_path = run_dir / f"ids.{args.track_tag}.txt"
    ids_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

    results_path = run_dir / f"raw.{args.track_tag}.jsonl"
    done: set[str] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").split("\n"):  # 只按\n切，记录内可能含\u2028 等
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("task_id") and rec.get("solution"):
                done.add(rec["task_id"])

    endpoint = _load_endpoint(args.config)
    client = openai.OpenAI(
        base_url=endpoint["base_url"],
        api_key=endpoint["api_key"],
        timeout=endpoint["timeout"],
    )
    pending = [by_id[tid] for tid in selected if tid not in done]
    print(
        f"bigcode track[{args.track_tag}]: {len(selected)} tasks "
        f"({len(done)} already done) -> {results_path}"
    )

    write_lock = threading.Lock()
    counters = {"ok": 0, "errors": 0, "finished": 0}

    def _worker(task: dict[str, Any]) -> dict[str, Any]:
        try:
            return _generate_one(client, endpoint, task)
        except Exception as exc:  # noqa: BLE001 - 单题失败不终止轨道
            return {"task_id": task["task_id"], "solution": "", "error": str(exc)[:500]}

    def _record(record: dict[str, Any]) -> None:
        with write_lock:
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record.get("solution"):
                counters["ok"] += 1
            else:
                counters["errors"] += 1
            counters["finished"] += 1
            print(
                f"[{counters['finished']}/{len(pending)}] {record['task_id']} "
                f"{'ok' if record.get('solution') else 'ERROR'} "
                f"({record.get('completion_tokens', 0)} tok, {record.get('latency', 0)}s)",
                flush=True,
            )

    if args.workers <= 1:
        for task in pending:
            _record(_worker(task))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, task) for task in pending]
            for future in as_completed(futures):
                _record(future.result())

    print(f"Done: {counters['ok']} ok, {counters['errors']} errors -> {results_path}")
    return 0 if counters["errors"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BigCodeBench direct-completion track (kimi-k3)")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True, help="service config JSON (llm.base_url/api_key/model)")
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    parser.add_argument("--ids-file", type=Path, default=None, help="One task_id per line (overrides --n)")
    parser.add_argument("--n", type=int, default=None, help="First-N tasks in dataset order")
    parser.add_argument("--track-tag", default="completion", help="Output tag (raw.<tag>.jsonl / ids.<tag>.txt)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent chat calls (同端点还有 LME ingest，smoke 保持 <=4)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-ids", action="store_true", help="Only write smoke50.ids/full.ids into run_dir and exit")
    parser.add_argument("--smoke-n", type=int, default=50, help="stdlib-only smoke id count for --prepare-ids")
    args = parser.parse_args(argv)

    if args.prepare_ids:
        prepare_ids(Path(args.tasks_file), args.run_dir.resolve(), smoke_n=args.smoke_n)
        return 0
    if args.ids_file is None and args.n is None:
        parser.error("specify --ids-file or --n (default: all tasks in file)")
    return run_generation(args)


if __name__ == "__main__":
    sys.exit(main())
