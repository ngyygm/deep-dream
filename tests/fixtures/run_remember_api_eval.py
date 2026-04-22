#!/usr/bin/env python3
"""
Remember API 单向评估脚本。

目标：
- 逐条提交样本到 /api/v1/remember
- 轮询任务结果
- 查询该 graph_id 下的实体与关系
- 产出面向 remember 重构的 JSON 报告
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List
from urllib.parse import quote

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_dataset_extended import EXTENDED_DATASETS


SESSION = requests.Session()
SESSION.trust_env = False
BASE = os.environ.get("DEEP_DREAM_API", "http://localhost:16200/api/v1")


def api_get(path: str) -> requests.Response:
    return SESSION.get(f"{BASE}{path}", timeout=30)


def api_post(path: str, data: Dict[str, Any]) -> requests.Response:
    return SESSION.post(f"{BASE}{path}", json=data, timeout=180)


def wait_for_task(task_id: str, graph_id: str, timeout: int = 300) -> Dict[str, Any]:
    started = time.time()
    while time.time() - started < timeout:
        resp = api_get(f"/remember/tasks/{task_id}?graph_id={quote(graph_id)}")
        if resp.status_code == 200:
            payload = resp.json().get("data", {})
            status = payload.get("status", "unknown")
            if status in {"completed", "failed"}:
                return payload
        time.sleep(2)
    return {"status": "timeout"}


def fetch_graph_snapshot(graph_id: str) -> Dict[str, Any]:
    entities_resp = api_get(f"/find/entities?graph_id={quote(graph_id)}&limit=200")
    relations_resp = api_get(f"/find/relations?graph_id={quote(graph_id)}&limit=200")
    entities = entities_resp.json().get("data", {}).get("entities", []) if entities_resp.status_code == 200 else []
    relations = relations_resp.json().get("data", {}).get("relations", []) if relations_resp.status_code == 200 else []
    return {
        "entity_count": len(entities),
        "relation_count": len(relations),
        "entity_names": [entity.get("name", "") for entity in entities],
    }


def evaluate_dataset(name: str, dataset: Dict[str, Any]) -> Dict[str, Any]:
    graph_id = f"remember_eval_{name}"
    text = dataset["text"]
    use_wait = len(text) < 400
    payload = {
        "graph_id": graph_id,
        "text": text,
        "source_name": f"eval:{name}",
        "wait": use_wait,
    }
    started = time.time()
    resp = api_post("/remember", payload)
    elapsed_submit = time.time() - started
    if resp.status_code not in (200, 202):
        return {
            "status": "error",
            "http_code": resp.status_code,
            "elapsed": elapsed_submit,
            "graph_id": graph_id,
        }

    data = resp.json().get("data", {})
    if data.get("status") == "completed":
        task_payload = data
    else:
        task_id = data.get("task_id")
        if not task_id:
            return {
                "status": "error",
                "error": "missing_task_id",
                "elapsed": elapsed_submit,
                "graph_id": graph_id,
            }
        task_payload = wait_for_task(task_id, graph_id)

    snapshot = fetch_graph_snapshot(graph_id)
    expected_entities = dataset.get("expected_entities", [])
    matched_expected = [
        expected for expected in expected_entities
        if any(expected in name or name in expected for name in snapshot["entity_names"])
    ]
    result_inner = task_payload.get("result", {}) or {}
    elapsed_total = time.time() - started
    return {
        "status": task_payload.get("status", "unknown"),
        "graph_id": graph_id,
        "elapsed": elapsed_total,
        "submit_elapsed": elapsed_submit,
        "task_entities": result_inner.get("entities", task_payload.get("entities_count", 0)),
        "task_relations": result_inner.get("relations", task_payload.get("relations_count", 0)),
        "graph_entities": snapshot["entity_count"],
        "graph_relations": snapshot["relation_count"],
        "expected_entities": expected_entities,
        "matched_expected_entities": matched_expected,
        "matched_expected_ratio": (
            len(matched_expected) / len(expected_entities) if expected_entities else 1.0
        ),
    }


def main() -> int:
    try:
        health = api_get("/health")
        health.raise_for_status()
    except Exception as exc:
        print(f"Server unavailable: {exc}")
        return 1

    results: Dict[str, Any] = {}
    for name, dataset in EXTENDED_DATASETS.items():
        print(f"[remember-eval] {name}")
        results[name] = evaluate_dataset(name, dataset)
        current = results[name]
        print(
            f"  status={current.get('status')} "
            f"task_ents={current.get('task_entities', 0)} "
            f"task_rels={current.get('task_relations', 0)} "
            f"matched={len(current.get('matched_expected_entities', []))}/{len(current.get('expected_entities', []))}"
        )

    report = {
        "base": BASE,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    output_path = os.path.join(os.path.dirname(__file__), "remember_api_eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
