"""Bounded, evidence-first Deep-Dream benchmark agent."""
from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
import json
import os
import re
import time
from typing import Any

from core.agent import load_runtime_policy
from core.cli._helpers import concept_source_evidence, relation_evidence

from .datasets import BenchmarkItem, MemorySession


AGENT_TOOLS = {
    "search_documents",
    "explore_memory",
    "search_concepts",
    "trace_concept",
    "expand_neighbors",
    "relation_evidence",
    "read_episode",
    "read_session",
    "submit_evidence",
}


def _agent_extra_body(config: dict[str, Any], thinking: bool) -> dict[str, Any] | None:
    """Apply an Agent-only thinking override without changing the answer model."""
    raw = config.get("agent_extra_body")
    body = copy.deepcopy(raw if raw is not None else config.get("extra_body") or {})
    if "agent_think" in config:
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = thinking
    return body or None


def _compact(value: Any, max_chars: int = 4000) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _compact(item, max_chars=max_chars) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(item, max_chars=max_chars) for item in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "…"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
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
    raise ValueError("Agent response is not a JSON object")


class AgentDecisionModel:
    """OpenAI-compatible JSON action policy; benchmark labels are never shown."""

    TOOL_PROMPT = """Available tools:
- search_documents {"query": string, "terms": [string], "limit": 1..20}
- explore_memory {"query": string, "terms": [string], "limit": 1..20}
- search_concepts {"query": string, "limit": 1..20}
- trace_concept {"family_id": string, "limit": 1..20}
- expand_neighbors {"family_id": string, "depth": 1..2, "limit": 1..20}
- relation_evidence {"concept_a": string, "concept_b": string, "limit": 1..20}
- read_episode {"episode_id": string}
- read_session {"session_id": string}
- submit_evidence {"episode_ids": [string], "session_ids": [string], "turn_ids": [string]}

Return exactly one JSON object and no prose:
{"tool":"tool_name","arguments":{...}}
Do not output an answer. `submit_evidence` ends retrieval and a separate model answers."""

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("llm") or {}
        self.system_prompt = load_runtime_policy() + "\n\n" + self.TOOL_PROMPT

    def _history(self, trajectory: list[dict[str, Any]]) -> str:
        rows = [{
            "step": row.get("step"),
            "tool": row.get("tool"),
            "arguments": row.get("arguments"),
            "observation": row.get("observation"),
            "error": row.get("error"),
        } for row in trajectory]
        serialized = json.dumps(rows, ensure_ascii=False)
        return serialized[-int(self.config.get("agent_history_max_chars") or 24000):]

    def decide(self, item: BenchmarkItem, trajectory: list[dict[str, Any]], step: int) -> dict[str, Any]:
        from core.llm.chat_api import ollama_chat, openai_compatible_chat

        date_line = f"\nQuestion date: {item.question_date}" if item.question_date else ""
        prompt = (
            f"Question: {item.question}{date_line}\nStep: {step}\n"
            f"Previous tool observations:\n{self._history(trajectory)}\n"
            "Choose the single best next retrieval action. Submit only source-backed evidence."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        base_url = str(self.config.get("base_url") or "https://api.openai.com/v1")
        thinking = bool(self.config.get("agent_think", self.config.get("think", False)))
        agent_max_tokens = int(self.config.get("agent_max_tokens") or 600)
        if thinking:
            agent_max_tokens = int(
                self.config.get("agent_thinking_max_tokens") or max(agent_max_tokens, 2048)
            )
        started = time.monotonic()
        if "11434" in base_url and not base_url.rstrip("/").endswith("/v1"):
            response = ollama_chat(
                messages,
                model=str(self.config.get("model") or "qwen3.5:4b"),
                base_url=base_url,
                think=thinking,
                timeout=int(self.config.get("timeout_seconds") or 300),
                num_predict=agent_max_tokens,
                json_format=True,
            )
        else:
            response = openai_compatible_chat(
                messages,
                model=str(self.config.get("model") or "gpt-4o-mini"),
                base_url=base_url,
                api_key=str(self.config.get("api_key") or os.getenv("OPENAI_API_KEY") or ""),
                timeout=int(self.config.get("timeout_seconds") or 300),
                max_tokens=agent_max_tokens,
                temperature=float(self.config.get("temperature", 0) or 0),
                extra_body=_agent_extra_body(self.config, thinking),
            )
        return {
            "action": _parse_json_object(response.content),
            "raw_response": response.content,
            "prompt": prompt,
            "latency_seconds": round(time.monotonic() - started, 3),
            "prompt_tokens": response.prompt_eval_count,
            "completion_tokens": response.eval_count,
            "model": response.model or self.config.get("model"),
        }


class AgenticMemoryTools:
    """Conversation-isolated, source-backed tool surface."""

    def __init__(self, storage: Any, retriever: Any, document_to_session: dict[str, str],
                 sessions: list[MemorySession]):
        self.storage = storage
        self.retriever = retriever
        self.document_to_session = dict(document_to_session)
        self.allowed_document_ids = set(document_to_session)
        self.sessions = {session.session_id: session for session in sessions}
        self.turn_to_session = {
            turn_id: session.session_id for session in sessions for turn_id in session.turn_ids
        }

    @staticmethod
    def _limit(arguments: dict[str, Any], default: int = 10) -> int:
        return max(1, min(int(arguments.get("limit") or default), 20))

    @staticmethod
    def _required(arguments: dict[str, Any], key: str, max_chars: int = 500) -> str:
        value = str(arguments.get(key) or "").strip()
        if not value:
            raise ValueError(f"{key} is required")
        return value[:max_chars]

    def _episode_payload(self, episode_id: str) -> dict[str, Any]:
        row = self.storage._conn().execute(
            """SELECT episode_id, document_id, source_text, memory_text, event_time
               FROM episodes WHERE episode_id = ? AND status = 'active'""",
            (episode_id,),
        ).fetchone()
        if not row or str(row[1]) not in self.allowed_document_ids:
            raise KeyError(f"Unknown or unavailable episode_id: {episode_id}")
        session_id = self.document_to_session[str(row[1])]
        session = self.sessions[session_id]
        turn_ids = [
            turn_id for turn_id in re.findall(r"(?m)^\[([^\]]+)\]", str(row[2] or ""))
            if turn_id in set(session.turn_ids)
        ]
        return {
            "episode_id": str(row[0]),
            "session_id": session_id,
            "turn_ids": turn_ids,
            "source_text": str(row[2] or ""),
            "memory_text": str(row[3] or ""),
            "event_time": str(row[4] or ""),
        }

    def search_documents(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        query = self._required(arguments, "query")
        result = self.retriever.explore(
            query, terms=arguments.get("terms") or None, limit=self._limit(arguments),
        )
        rows = result.get("channel_evidence", {}).get("raw-document", [])
        rows = sorted(
            rows,
            key=lambda row: (
                -float(row.get("match_score") or 0),
                int(row.get("channel_rank") or 1),
                str(row.get("session_id") or ""),
                str(row.get("turn_id") or ""),
            ),
        )
        return [_compact(row) for row in rows[:self._limit(arguments)]]

    def explore_memory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = self._required(arguments, "query")
        result = self.retriever.explore(
            query, terms=arguments.get("terms") or None, limit=self._limit(arguments),
        )
        return _compact({
            "evidence": result["evidence"][:self._limit(arguments)],
            "ranked_session_ids": result["ranked_session_ids"][:self._limit(arguments)],
            "ranked_turn_ids": result["ranked_turn_ids"][:self._limit(arguments)],
            "concepts": result["explore"].get("semantic_hits", [])[:self._limit(arguments)],
            "neighbors": result["explore"].get("neighbors", [])[:self._limit(arguments)],
        })

    def search_concepts(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        query = self._required(arguments, "query")
        limit = self._limit(arguments)
        result = self.retriever.explore(query, limit=limit)
        allowed_families = {
            str(row.get("target_family_id") or "")
            for row in result["explore"].get("source_evidence", [])
        }
        return [_compact(row) for row in result["explore"].get("semantic_hits", [])
                if str(row.get("family_id") or "") in allowed_families][:limit]

    def trace_concept(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        family_id = self._required(arguments, "family_id")
        limit = self._limit(arguments)
        rows = concept_source_evidence(self.storage, [family_id], limit=max(limit * 3, 20))
        results = []
        for row in rows:
            episode_id = str(row.get("episode_version_id") or "")
            try:
                payload = self._episode_payload(episode_id)
            except KeyError:
                continue
            results.append(_compact({**row, **payload}))
            if len(results) >= limit:
                break
        return results

    def expand_neighbors(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        family_id = self._required(arguments, "family_id")
        depth = max(1, min(int(arguments.get("depth") or 1), 2))
        limit = self._limit(arguments)
        results = []
        for raw in self.storage.get_concept_neighbors(family_id, max_depth=depth, max_results=limit * 3):
            row = dict(raw) if isinstance(raw, dict) else vars(raw)
            candidate = str(row.get("family_id") or row.get("target_family_id") or "")
            if candidate and self.trace_concept({"family_id": candidate, "limit": 1}):
                results.append(_compact(row))
            if len(results) >= limit:
                break
        return results

    def relation_evidence(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        left = self._required(arguments, "concept_a")
        right = self._required(arguments, "concept_b")
        results = []
        for row in relation_evidence(self.storage, left, right, limit=self._limit(arguments) * 3):
            episode_id = str(row.get("episode_version_id") or "")
            try:
                payload = self._episode_payload(episode_id)
            except KeyError:
                continue
            results.append(_compact({**row, **payload}))
            if len(results) >= self._limit(arguments):
                break
        return results

    def read_episode(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return _compact(self._episode_payload(self._required(arguments, "episode_id")))

    def read_session(self, arguments: dict[str, Any]) -> dict[str, Any]:
        session_id = self._required(arguments, "session_id")
        session = self.sessions.get(session_id)
        if not session or self.session_to_document_id(session_id) not in self.allowed_document_ids:
            raise KeyError(f"Unknown or unavailable session_id: {session_id}")
        return _compact({
            "session_id": session.session_id,
            "timestamp": session.timestamp,
            "turn_ids": session.turn_ids,
            "text": session.text,
        })

    def session_to_document_id(self, session_id: str) -> str:
        for document_id, candidate in self.document_to_session.items():
            if candidate == session_id:
                return document_id
        return ""

    def execute(self, tool: str, arguments: dict[str, Any]) -> Any:
        if tool not in AGENT_TOOLS - {"submit_evidence"}:
            raise ValueError(f"Unknown or non-executable tool: {tool}")
        return getattr(self, tool)(arguments)

    def contexts_for_submission(
        self,
        session_ids: list[str],
        episode_ids: list[str],
        turn_ids: list[str],
        *,
        limit: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        ordered_sessions = list(dict.fromkeys(session_ids))
        episode_payloads = []
        for episode_id in dict.fromkeys(episode_ids):
            payload = self._episode_payload(episode_id)
            episode_payloads.append(payload)
            if payload["session_id"] not in ordered_sessions:
                ordered_sessions.append(payload["session_id"])
        selected_turns = list(dict.fromkeys(turn_ids))
        for turn_id in selected_turns:
            session_id = self.turn_to_session.get(turn_id)
            if session_id and session_id not in ordered_sessions:
                ordered_sessions.append(session_id)
        if not selected_turns:
            for payload in episode_payloads:
                for turn_id in payload["turn_ids"]:
                    if turn_id not in selected_turns:
                        selected_turns.append(turn_id)
        contexts = []
        for rank, session_id in enumerate(ordered_sessions[:limit], 1):
            session = self.sessions.get(session_id)
            if not session:
                continue
            relevant_episodes = [row for row in episode_payloads if row["session_id"] == session_id]
            contexts.append({
                "session_id": session_id,
                "timestamp": session.timestamp,
                "text": session.text,
                "turn_ids": session.turn_ids,
                "matched_turn_ids": [turn for turn in selected_turns if turn in set(session.turn_ids)],
                "score": 1.0 / rank,
                "evidence": [{
                    "episode_id": row["episode_id"], "session_id": session_id,
                    "turn_ids": row["turn_ids"], "source_text": row["source_text"],
                    "retrieval_channel": "agent-submitted",
                } for row in relevant_episodes] or [{"retrieval_channel": "agent-submitted-session"}],
            })
        return contexts, selected_turns

    def validate_submission(self, session_ids: list[str], episode_ids: list[str], turn_ids: list[str]) -> None:
        """Reject evidence that stops at an unresolved indirect reference."""
        contexts, _ = self.contexts_for_submission(
            session_ids, episode_ids, turn_ids, limit=max(1, len(self.sessions)),
        )
        selected_turns = set(turn_ids)
        if selected_turns:
            lines = []
            for row in contexts:
                for line in str(row.get("text") or "").splitlines():
                    match = re.match(r"^\[([^\]]+)\]", line.strip())
                    if match and match.group(1) in selected_turns:
                        lines.append(line)
            text = "\n".join(lines)
        else:
            text = "\n".join(str(row.get("text") or "") for row in contexts)
        if re.search(r"\bhome country\b", text, re.I) and not re.search(
            r"\bhome country\s*(?:is|was|,|:)\s*(?:the\s+)?[A-Z][A-Za-z-]+", text
        ):
            raise ValueError(
                "Indirect reference 'home country' is unresolved; submit an additional source that names the country."
            )


class AgenticMemoryRunner:
    """Use tools to submit evidence, then always call the shared answerer."""

    def __init__(self, tools: AgenticMemoryTools, policy: Any, answerer: Any,
                 *, max_steps: int = 8, answer_top_k: int = 5,
                 allow_unsurfaced_evidence: bool = False):
        self.tools = tools
        self.policy = policy
        self.answerer = answerer
        self.max_steps = max(1, max_steps)
        self.answer_top_k = max(1, answer_top_k)
        # X7 arm #6: when True, the submit_evidence surfaced-set gate is
        # relaxed so an answer may consume derived memory the agent never
        # surfaced via tools. Defaults to False to preserve the auditable
        # memory contract for every existing caller.
        self.allow_unsurfaced_evidence = allow_unsurfaced_evidence

    @staticmethod
    def _decision_payload(decision: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if "action" in decision:
            return decision["action"], {key: value for key, value in decision.items() if key != "action"}
        return decision, {}

    @staticmethod
    def _collect_ids(value: Any, sessions: list[str], episodes: list[str], turns: list[str]) -> None:
        if isinstance(value, dict):
            sid = value.get("session_id")
            eid = value.get("episode_id") or value.get("episode_version_id")
            if sid and str(sid) not in sessions:
                sessions.append(str(sid))
            if eid and str(eid) not in episodes:
                episodes.append(str(eid))
            singular_turn = value.get("turn_id")
            if singular_turn and str(singular_turn) not in turns:
                turns.append(str(singular_turn))
            for turn_id in value.get("turn_ids") or value.get("matched_turn_ids") or []:
                if turn_id and str(turn_id) not in turns:
                    turns.append(str(turn_id))
            for nested in value.values():
                AgenticMemoryRunner._collect_ids(nested, sessions, episodes, turns)
        elif isinstance(value, list):
            for nested in value:
                AgenticMemoryRunner._collect_ids(nested, sessions, episodes, turns)

    @staticmethod
    def _submitted(arguments: dict[str, Any], key: str) -> list[str]:
        value = arguments.get(key) or []
        if not isinstance(value, list):
            raise ValueError(f"{key} must be an array")
        return list(dict.fromkeys(str(item) for item in value if str(item)))

    def retrieve(self, item: BenchmarkItem) -> dict[str, Any]:
        """Run only the auditable tool trajectory and return submitted evidence."""
        trajectory: list[dict[str, Any]] = []
        surfaced_sessions: list[str] = []
        surfaced_episodes: list[str] = []
        surfaced_turns: list[str] = []
        selected_sessions: list[str] = []
        selected_episodes: list[str] = []
        selected_turns: list[str] = []
        prompt_tokens = completion_tokens = 0
        model = None
        stop_reason = "max_steps"
        started = time.monotonic()
        repeated: dict[str, int] = {}

        for step in range(1, self.max_steps + 1):
            row: dict[str, Any] = {"step": step}
            try:
                decision = self.policy.decide(item, trajectory, step)
                action, metadata = self._decision_payload(decision)
                tool = str(action.get("tool") or "")
                arguments = action.get("arguments") or {
                    key: value for key, value in action.items() if key != "tool"
                }
                if tool not in AGENT_TOOLS:
                    raise ValueError(f"Unknown tool: {tool}")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                row.update({"tool": tool, "arguments": _compact(arguments)})
                row.update({key: _compact(value) for key, value in metadata.items()
                            if key not in {"prompt", "raw_response"}})
                prompt_tokens += int(metadata.get("prompt_tokens") or 0)
                completion_tokens += int(metadata.get("completion_tokens") or 0)
                model = metadata.get("model") or model
                signature = json.dumps([tool, arguments], ensure_ascii=False, sort_keys=True)
                repeated[signature] = repeated.get(signature, 0) + 1
                if repeated[signature] > 2:
                    raise ValueError("Identical tool action repeated more than twice")
                if tool == "submit_evidence":
                    selected_sessions = self._submitted(arguments, "session_ids")
                    selected_episodes = self._submitted(arguments, "episode_ids")
                    selected_turns = self._submitted(arguments, "turn_ids")
                    if not self.allow_unsurfaced_evidence:
                        unknown = (
                            set(selected_sessions) - set(surfaced_sessions)
                            or set(selected_episodes) - set(surfaced_episodes)
                            or set(selected_turns) - set(surfaced_turns)
                        )
                        if unknown:
                            raise ValueError(f"Submitted evidence was not surfaced by tools: {sorted(unknown)}")
                    validate = getattr(self.tools, "validate_submission", None)
                    if validate:
                        validate(selected_sessions, selected_episodes, selected_turns)
                    row["observation"] = {"accepted": True, "evidence_count": (
                        len(selected_sessions) + len(selected_episodes) + len(selected_turns)
                    )}
                    trajectory.append(row)
                    stop_reason = "submit_evidence"
                    break
                tool_started = time.monotonic()
                observation = self.tools.execute(tool, arguments)
                row["tool_latency_seconds"] = round(time.monotonic() - tool_started, 3)
                row["observation"] = _compact(observation)
                self._collect_ids(observation, surfaced_sessions, surfaced_episodes, surfaced_turns)
            except Exception as exc:
                row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            trajectory.append(row)

        if stop_reason == "max_steps":
            selected_sessions = surfaced_sessions
            selected_episodes = surfaced_episodes
            selected_turns = surfaced_turns
        contexts, ranked_turns = self.tools.contexts_for_submission(
            selected_sessions, selected_episodes, selected_turns, limit=self.answer_top_k,
        )
        return {
            "retrieval_latency_seconds": round(time.monotonic() - started, 3),
            "retrieval_prompt_tokens": prompt_tokens,
            "retrieval_completion_tokens": completion_tokens,
            "retrieval_model": model,
            "retrieved": contexts,
            "retrieved_turn_ids": ranked_turns,
            "submitted_evidence": {
                "session_ids": selected_sessions,
                "episode_ids": selected_episodes,
                "turn_ids": selected_turns,
            },
            "trajectory": trajectory,
            "agent_steps": len(trajectory),
            "agent_stop_reason": stop_reason,
            "agent_tool_counts": {
                tool: sum(row.get("tool") == tool for row in trajectory)
                for tool in sorted(AGENT_TOOLS) if any(row.get("tool") == tool for row in trajectory)
            },
        }

    def run(self, item: BenchmarkItem) -> dict[str, Any]:
        """Backward-compatible retrieve-then-answer convenience method."""
        retrieval = self.retrieve(item)
        answer = self.answerer.answer(item, retrieval["retrieved"])
        retrieval_prompt_tokens = int(retrieval.get("retrieval_prompt_tokens") or 0)
        retrieval_completion_tokens = int(retrieval.get("retrieval_completion_tokens") or 0)
        answer_latency = float(answer.get("answer_latency_seconds") or 0)
        return {
            **retrieval,
            **answer,
            "latency_seconds": round(
                float(retrieval.get("retrieval_latency_seconds") or 0) + answer_latency, 3
            ),
            "prompt_tokens": retrieval_prompt_tokens + int(answer.get("prompt_tokens") or 0),
            "completion_tokens": retrieval_completion_tokens + int(answer.get("completion_tokens") or 0),
            "model": answer.get("model") or retrieval.get("retrieval_model"),
        }
