"""Official dataset download and normalization for LongMemEval, LoCoMo and MemoryAgentBench."""
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
    # MemoryAgentBench 无单文件下载 URL：manifest 记录 4 个 parquet 分片，由
    # 数据集自带脚本/HF 下载后经 load_memoryagentbench 校验使用。
    "memoryagentbench": {
        "filename": "memoryagentbench/manifest.json",
        "source": "https://github.com/HUST-AI-HYZ/MemoryAgentBench",
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
    # 官方 LLM-judge 评分点（MemoryAgentBench keypoints / InfBench 参考要点）。
    judge_rubric: list[str] = field(default_factory=list)
    # 题目级可见会话白名单（可见性边界，独立于 scorer 专用的 gold evidence）。
    visible_session_ids: list[str] = field(default_factory=list)


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
    if "url" not in spec:
        # 无 URL 的数据集（memoryagentbench）：只能登记既有分片清单，不做下载。
        if not target.exists():
            raise FileNotFoundError(
                f"Dataset has no download URL and is not prepared: {target}"
            )
    else:
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
        "url": spec.get("url", ""),
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


_MEMORYAGENTBENCH_COMPETENCY = {
    "Accurate_Retrieval": "AR",
    "Conflict_Resolution": "CR",
    "Long_Range_Understanding": "LRU",
    "Test_Time_Learning": "TTL",
}


def _ordered_session_ids(sessions: Iterable[MemorySession]) -> list[str]:
    """Return a stable question-local visibility allowlist."""
    result = list(dict.fromkeys(session.session_id for session in sessions))
    if not result:
        raise ValueError("Benchmark question visibility cannot be empty")
    return result


def _partition_exact_text(text: str, *, max_chars: int = 120_000) -> list[str]:
    """Partition a long stream without dropping or duplicating source text."""
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            floor = start + int(max_chars * 0.75)
            boundary = text.rfind("\n\n", floor, hard_end)
            if boundary < floor:
                boundary = text.rfind("\n", floor, hard_end)
            if boundary >= floor:
                end = boundary + (2 if text.startswith("\n\n", boundary) else 1)
        chunks.append(text[start:end])
        start = end
    if "".join(chunks) != text:  # defensive evidence-preservation invariant
        raise ValueError("MemoryAgentBench context partition changed source text")
    return chunks


def _resolve_memoryagentbench_shard(path: Path, file_record: dict[str, Any]) -> Path:
    """Resolve one manifest shard path, tolerating stale recorded absolute paths.

    官方 manifest 里的 ``files[].path`` 是下载数据的机器上的绝对路径，跨机器
    迁移后必然失效；此时回退到 manifest 同目录（含 ``data/`` 子目录）下按
    ``filename`` 查找。哈希校验仍按 manifest 记录执行，回退不降低完整性。
    """
    filename = str(file_record.get("filename") or "")
    candidates = []
    recorded = Path(str(file_record.get("path") or ""))
    if recorded.is_absolute():
        candidates.append(recorded)
    else:
        candidates.append(path.parent / filename)
    candidates.append(path.parent / filename)
    candidates.append(path.parent / "data" / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "MemoryAgentBench shard is missing: "
        f"tried {', '.join(str(c) for c in dict.fromkeys(candidates))}"
    )


def load_memoryagentbench(path: Path) -> list[BenchmarkItem]:
    """Load the fixed ICLR 2026 MemoryAgentBench release.

    Each official Parquet row becomes one isolated memory scope.  Its long
    context is partitioned into stable, ordered source documents so a failed
    document can resume without rebuilding a million-token haystack.  The
    partition is exact: concatenating the documents reproduces the official
    context byte-for-byte.
    """
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "memoryagentbench":
        raise ValueError(f"Invalid MemoryAgentBench manifest: {path}")
    validated_shards: list[tuple[Path, str]] = []
    for file_record in manifest.get("files") or []:
        parquet_path = _resolve_memoryagentbench_shard(path, file_record)
        expected_hash = str(file_record.get("sha256") or "")
        if expected_hash and sha256_file(parquet_path) != expected_hash:
            raise ValueError(
                f"MemoryAgentBench shard hash changed: {parquet_path}"
            )
        stem = parquet_path.name.split("-00000-", 1)[0]
        competency = _MEMORYAGENTBENCH_COMPETENCY.get(stem)
        if competency is None:
            raise ValueError(
                f"Unknown MemoryAgentBench split: {parquet_path.name}"
            )
        validated_shards.append((parquet_path, competency))

    # Verify immutable shard identity before importing the optional parser.
    # This keeps integrity failures observable even in a minimal install and
    # avoids reporting a missing extra in place of a certified-byte mismatch.
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - minimal installs
        raise RuntimeError(
            "Loading MemoryAgentBench requires pyarrow: pip install pyarrow"
        ) from exc

    items: list[BenchmarkItem] = []
    seen_question_ids: set[str] = set()
    for parquet_path, competency in validated_shards:
        rows = parquet.read_table(parquet_path).to_pylist()
        for row_index, row in enumerate(rows):
            scope_id = f"mab-{competency.lower()}-{row_index:03d}"
            context = _as_text(row.get("context"))
            partitions = _partition_exact_text(context)
            if not partitions:
                raise ValueError(f"Empty MemoryAgentBench context: {scope_id}")
            sessions = [
                MemorySession(
                    session_id=f"context_{index:04d}",
                    timestamp="",
                    text=chunk,
                    turn_ids=[f"{scope_id}:context:{index:04d}"],
                )
                for index, chunk in enumerate(partitions)
            ]
            metadata = row.get("metadata") or {}
            questions = list(row.get("questions") or [])
            answers = list(row.get("answers") or [])
            official_ids = list(metadata.get("qa_pair_ids") or [])
            question_types = list(metadata.get("question_types") or [])
            question_dates = list(metadata.get("question_dates") or [])
            keypoints = list(metadata.get("keypoints") or [])
            previous_events = list(metadata.get("previous_events") or [])
            for question_index, question in enumerate(questions):
                official_id = (
                    _as_text(official_ids[question_index])
                    if question_index < len(official_ids) else ""
                )
                question_id = (
                    f"{scope_id}:{official_id}"
                    if official_id else f"{scope_id}:q{question_index:04d}"
                )
                if question_id in seen_question_ids:
                    raise ValueError(
                        f"Duplicate MemoryAgentBench question ID: {question_id}"
                    )
                seen_question_ids.add(question_id)
                aliases = answers[question_index] if question_index < len(answers) else []
                if not isinstance(aliases, list):
                    aliases = [aliases]
                aliases = [_as_text(value) for value in aliases]
                qtype = (
                    _as_text(question_types[question_index])
                    if question_index < len(question_types) else competency
                )
                qdate = (
                    _as_text(question_dates[question_index])
                    if question_index < len(question_dates) else ""
                )
                # The fixed release represents InfBench's single summary
                # question with a flat list of all reference key points.  A
                # few other producers use one nested list per question, so
                # preserve both encodings without silently dropping all but
                # the first InfBench key point.
                if len(questions) == 1 and keypoints and all(
                    not isinstance(value, (list, tuple)) for value in keypoints
                ):
                    per_question_keypoints = list(keypoints)
                elif question_index < len(keypoints) and keypoints[question_index]:
                    value = keypoints[question_index]
                    per_question_keypoints = (
                        list(value) if isinstance(value, (list, tuple)) else [value]
                    )
                else:
                    per_question_keypoints = []
                items.append(BenchmarkItem(
                    dataset="memoryagentbench",
                    scope_id=scope_id,
                    question_id=question_id,
                    question=_as_text(question),
                    answer=aliases[0] if aliases else "",
                    question_type=qtype,
                    question_date=qdate,
                    sessions=sessions,
                    evidence_session_ids=[],
                    judge_rubric=[_as_text(v) for v in per_question_keypoints],
                    metadata={
                        "competency": competency,
                        "official_split": stem,
                        "official_source": metadata.get("source"),
                        "official_question_id": official_id,
                        "answer_aliases": aliases,
                        "demo": metadata.get("demo"),
                        "previous_events": previous_events,
                        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
                        "context_documents": len(sessions),
                        "context_partition_max_chars": 120_000,
                    },
                    visible_session_ids=_ordered_session_ids(sessions),
                ))
    return items


def load_benchmark(name: str, data_dir: Path) -> tuple[list[BenchmarkItem], Path]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    path = data_dir / DATASETS[name]["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset is not prepared: {path}")
    loaders = {
        "longmemeval-s": load_longmemeval,
        "locomo": load_locomo,
        "memoryagentbench": load_memoryagentbench,
    }
    return loaders[name](path), path


def data_dir_for_dataset_path(name: str, dataset_path: Path) -> Path:
    """从落盘的 dataset_path 反推 load_benchmark 的 data_dir。

    load_benchmark 拼的是 data_dir / DATASETS[name]["filename"]，filename 可含
    子目录（memoryagentbench/manifest.json），反推要按目录深度取 parents——
    无脑 .parent 会把子目录型数据集拼成双层路径（LME 平文件不炸，MAB 炸）。"""
    depth = len(DATASETS[name]["filename"].strip("/").split("/")) - 1
    return dataset_path.parents[depth]


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
