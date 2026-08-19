"""Official dataset download and normalization for LongMemEval and LoCoMo."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen


DATASETS = {
    "longmemeval-s": {
        "filename": "longmemeval_s_cleaned.json",
        "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
        "source": "https://github.com/xiaowu0162/LongMemEval",
    },
    "locomo": {
        "filename": "locomo10.json",
        "url": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        "source": "https://github.com/snap-research/locomo",
    },
}


@dataclass(slots=True)
class MemorySession:
    session_id: str
    timestamp: str
    text: str
    turn_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkItem:
    dataset: str
    scope_id: str
    question_id: str
    question: str
    answer: str
    question_type: str
    question_date: str
    sessions: list[MemorySession]
    evidence_session_ids: list[str]
    evidence_turn_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(name: str, data_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """Download one official dataset and write a source/hash manifest."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    spec = DATASETS[name]
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / spec["filename"]
    if force or not target.exists():
        request = Request(spec["url"], headers={"User-Agent": "deep-dream-benchmark/1.0"})
        temp = target.with_suffix(target.suffix + ".part")
        with urlopen(request, timeout=300) as response, temp.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        # Validate before replacing any existing cache.
        json.loads(temp.read_text(encoding="utf-8"))
        temp.replace(target)
    record = {
        "dataset": name,
        "path": str(target.resolve()),
        "url": spec["url"],
        "source": spec["source"],
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }
    manifest_path = data_dir / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    existing[name] = record
    manifest_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return "" if value is None else str(value)


def _turn_text(turn: dict[str, Any], turn_id: str) -> str:
    role = turn.get("role") or turn.get("speaker") or "unknown"
    text = _as_text(turn.get("content", turn.get("text", "")))
    caption = turn.get("blip_caption")
    if caption:
        text += f" [Image caption: {caption}]"
    return f"[{turn_id}] {role}: {text}".strip()


def load_longmemeval(path: Path) -> list[BenchmarkItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items: list[BenchmarkItem] = []
    for row in data:
        sessions = []
        evidence_turns = []
        for sid, timestamp, turns in zip(
            row["haystack_session_ids"], row["haystack_dates"], row["haystack_sessions"]
        ):
            turn_ids, lines = [], []
            for index, turn in enumerate(turns, start=1):
                turn_id = f"{sid}:{index}"
                turn_ids.append(turn_id)
                lines.append(_turn_text(turn, turn_id))
                if turn.get("has_answer"):
                    evidence_turns.append(turn_id)
            sessions.append(MemorySession(str(sid), str(timestamp), "\n".join(lines), turn_ids))
        qid = str(row["question_id"])
        items.append(BenchmarkItem(
            dataset="longmemeval-s", scope_id=qid, question_id=qid,
            question=_as_text(row["question"]), answer=_as_text(row["answer"]),
            question_type=_as_text(row["question_type"]),
            question_date=_as_text(row.get("question_date")), sessions=sessions,
            evidence_session_ids=[str(v) for v in row.get("answer_session_ids", [])],
            evidence_turn_ids=evidence_turns,
            metadata={"abstention": qid.endswith("_abs")},
        ))
    return items


def _locomo_sessions(conversation: dict[str, Any]) -> list[MemorySession]:
    numbered = []
    for key, turns in conversation.items():
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                number = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue
            numbered.append((number, key, turns))
    sessions = []
    for _, key, turns in sorted(numbered):
        timestamp = _as_text(conversation.get(f"{key}_date_time"))
        lines, turn_ids = [], []
        for index, turn in enumerate(turns, start=1):
            turn_id = _as_text(turn.get("dia_id")) or f"{key}:{index}"
            turn_ids.append(turn_id)
            lines.append(_turn_text(turn, turn_id))
        sessions.append(MemorySession(key, timestamp, "\n".join(lines), turn_ids))
    return sessions


def load_locomo(path: Path) -> list[BenchmarkItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for sample in data:
        scope = _as_text(sample["sample_id"])
        sessions = _locomo_sessions(sample["conversation"])
        turn_to_session = {
            turn_id: session.session_id for session in sessions for turn_id in session.turn_ids
        }
        for index, qa in enumerate(sample.get("qa", [])):
            qid = _as_text(qa.get("question_id")) or f"{scope}:{index}"
            evidence = []
            for value in qa.get("evidence", []):
                turn_id = _as_text(value)
                if turn_id not in turn_to_session and turn_id.startswith("D:"):
                    repaired = "D" + turn_id[2:]
                    if repaired in turn_to_session:
                        turn_id = repaired
                if turn_id in turn_to_session:
                    evidence.append(turn_id)
            evidence_sessions = sorted({turn_to_session[v] for v in evidence})
            items.append(BenchmarkItem(
                dataset="locomo", scope_id=scope, question_id=qid,
                question=_as_text(qa.get("question")), answer=_as_text(qa.get("answer")),
                question_type=_as_text(qa.get("category")), question_date="",
                sessions=sessions, evidence_session_ids=evidence_sessions,
                evidence_turn_ids=evidence,
                metadata={"category": qa.get("category"), "sample_id": scope},
            ))
    return items


def load_benchmark(name: str, data_dir: Path) -> tuple[list[BenchmarkItem], Path]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    path = data_dir / DATASETS[name]["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset is not prepared: {path}")
    loader = load_longmemeval if name == "longmemeval-s" else load_locomo
    return loader(path), path


def group_by_scope(items: Iterable[BenchmarkItem]) -> dict[str, list[BenchmarkItem]]:
    grouped: dict[str, list[BenchmarkItem]] = {}
    for item in items:
        grouped.setdefault(item.scope_id, []).append(item)
    return grouped


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    for candidate in (normalized, normalized.replace(" UTC", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    for fmt in (
        "%Y/%m/%d (%a) %H:%M",
        "%I:%M %p on %d %B, %Y",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            pass
    return None
