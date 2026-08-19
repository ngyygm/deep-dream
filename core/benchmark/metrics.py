"""Deterministic retrieval and LoCoMo QA metrics."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import string
from typing import Any, Iterable


def normalize_answer(text: str) -> str:
    text = str(text).lower().replace(",", "")
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def _stem(word: str) -> str:
    # The official LoCoMo scorer uses PorterStemmer. Import lazily so data and
    # retrieval-only workflows do not require the benchmark extra.
    try:
        from nltk.stem import PorterStemmer
        return PorterStemmer().stem(word)
    except ImportError:
        return word


def token_f1(prediction: str, answer: str) -> float:
    predicted = [_stem(w) for w in normalize_answer(prediction).split()]
    expected = [_stem(w) for w in normalize_answer(answer).split()]
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    same = sum(common.values())
    if not same:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(expected)
    return 2 * precision * recall / (precision + recall)


def locomo_f1(prediction: str, answer: str, category: int | str) -> float:
    category = int(category)
    if category == 5:
        lowered = prediction.lower()
        return float("no information available" in lowered or "not mentioned" in lowered)
    if category == 3:
        answer = answer.split(";", 1)[0].strip()
    if category == 1:
        predictions = [v.strip() for v in prediction.split(",")]
        answers = [v.strip() for v in answer.split(",")]
        return sum(max(token_f1(pred, ans) for pred in predictions) for ans in answers) / max(len(answers), 1)
    if category in (2, 3, 4):
        return token_f1(prediction, answer)
    raise ValueError(f"Unsupported LoCoMo category: {category}")


def retrieval_at_k(ranked_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> dict[str, float]:
    ranked = list(dict.fromkeys(ranked_ids))[:k]
    relevant = set(relevant_ids)
    if not relevant:
        return {
            f"recall_any@{k}": 0.0,
            f"recall_all@{k}": 0.0,
            f"evidence_recall@{k}": 0.0,
            f"ndcg_any@{k}": 0.0,
        }
    hits = [1 if item in relevant else 0 for item in ranked]
    found = sum(hits)
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(relevant), k)))
    return {
        f"recall_any@{k}": float(found > 0),
        f"recall_all@{k}": float(found == len(relevant)),
        f"evidence_recall@{k}": found / len(relevant),
        f"ndcg_any@{k}": dcg / ideal if ideal else 0.0,
    }


def aggregate_records(records: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    scored = [r for r in records if r.get("score") is not None]
    groups: dict[str, list[float]] = defaultdict(list)
    for record in scored:
        groups[str(record.get("question_type", "unknown"))].append(float(record["score"]))
    result: dict[str, Any] = {
        "dataset": dataset,
        "total": len(records),
        "scored": len(scored),
        "overall": sum(r["score"] for r in scored) / len(scored) if scored else None,
        "by_type": {key: {"score": sum(values) / len(values), "count": len(values)} for key, values in sorted(groups.items())},
    }
    metric_values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, value in (record.get("retrieval_metrics") or {}).items():
            metric_values[key].append(float(value))
    result["retrieval"] = {key: sum(values) / len(values) for key, values in sorted(metric_values.items()) if values}
    latencies = [float(row["total_latency_seconds"]) for row in records if row.get("total_latency_seconds") is not None]
    result["runtime"] = {
        "completed": sum(row.get("status", "completed") != "error" for row in records),
        "failed": sum(row.get("status") == "error" for row in records),
        "average_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "median_latency_seconds": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "p95_latency_seconds": sorted(latencies)[min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)]
        if latencies else None,
    }
    agent_rows = [row for row in records if row.get("retrieval_mode") == "agentic"
                  or row.get("source_track") == "skill-agent"
                  or str(row.get("track") or "").startswith("skill-agent")]
    if agent_rows:
        step_counts = [int(row.get("agent_steps") or 0) for row in agent_rows]
        tool_counts: Counter[str] = Counter()
        stop_reasons: Counter[str] = Counter()
        for row in agent_rows:
            tool_counts.update(row.get("agent_tool_counts") or {})
            stop_reasons[str(row.get("agent_stop_reason") or "unknown")] += 1
        result["agent"] = {
            "questions": len(agent_rows),
            "average_steps": sum(step_counts) / len(step_counts),
            "tool_counts": dict(sorted(tool_counts.items())),
            "stop_reasons": dict(sorted(stop_reasons.items())),
        }
    return result
