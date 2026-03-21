#!/usr/bin/env python3
"""
TMG 技能文档（SKILL.md + reference.md）与 service_api 对齐的冒烟测试。

用法（需已启动服务）：
  python scripts/tmg_skill_api_smoke.py --base-url http://127.0.0.1:16200

可选：
  --skip-remember       不跑写入与队列相关步骤
  --strict-llm          要求 GET /health/llm 必须 200（默认允许 503）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple


def _request(
    method: str,
    url: str,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = 120.0,
) -> Tuple[int, Dict[str, Any]]:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return resp.status, {}
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return e.code, {"_raw": raw}


def _get(base: str, path: str, timeout: float = 120.0) -> Tuple[int, Dict[str, Any]]:
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return e.code, {"_raw": raw}


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description="TMG API smoke test (SKILL + reference)")
    ap.add_argument("--base-url", default="http://127.0.0.1:16200", help="TMG base URL")
    ap.add_argument("--skip-remember", action="store_true", help="Skip remember + queue tests")
    ap.add_argument("--strict-llm", action="store_true", help="Require /health/llm 200")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    # --- Tier 1: SKILL minimal set ---
    st, j = _get(base, "/health")
    if st >= 500 or not j.get("success"):
        _fail(f"/health -> {st} {j}")
        return 1
    d = j.get("data") or {}
    if "storage_path" not in d:
        _fail("/health missing storage_path")
        return 1
    _ok("/health")

    st_llm, j_llm = _get(base, "/health/llm", timeout=90.0)
    if args.strict_llm:
        if st_llm != 200 or not j_llm.get("success"):
            _fail(f"/health/llm -> {st_llm} {j_llm}")
            return 1
        _ok("/health/llm (strict)")
    else:
        if st_llm == 200 and j_llm.get("success"):
            _ok("/health/llm")
        else:
            print(f"[WARN] /health/llm -> {st_llm} (continuing)")

    body_find = {
        "query": "我最近在做什么",
        "expand": True,
        "max_entities": 10,
        "max_relations": 20,
    }
    st, j = _request("POST", base + "/api/find", body_find)
    if st >= 500 or not j.get("success"):
        _fail(f"/api/find -> {st} {j}")
        return 1
    fd = j.get("data") or {}
    entities = fd.get("entities") or []
    relations = fd.get("relations") or []
    if "entity_count" not in fd or "relation_count" not in fd:
        _fail("/api/find response missing counts")
        return 1
    _ok(f"/api/find (entities={fd.get('entity_count')} relations={fd.get('relation_count')})")

    # Pick ids for atomic tests
    ent0 = entities[0] if entities else None
    logical_eid = ent0.get("entity_id") if ent0 else None
    abs_eid = ent0.get("id") if ent0 else None
    rel0 = relations[0] if relations else None
    abs_rid = rel0.get("id") if rel0 else None
    logical_rid = rel0.get("relation_id") if rel0 else None
    logical_eid_2 = entities[1].get("entity_id") if len(entities) > 1 else logical_eid

    # --- Tier 2: reference ---
    st, j = _get(base, "/api/find/stats")
    if st >= 500 or not j.get("success"):
        _fail(f"/api/find/stats -> {st} {j}")
        return 1
    _ok("/api/find/stats")

    st, j = _get(base, "/api/remember/queue?limit=10")
    if st >= 500 or not j.get("success"):
        _fail(f"/api/remember/queue -> {st} {j}")
        return 1
    _ok("/api/remember/queue")

    qn = urllib.parse.quote("测试", safe="")
    st, j = _get(base, f"/api/find/entities/search?query_name={qn}&max_results=3&threshold=0.35")
    if st >= 500 or not j.get("success"):
        _fail(f"/api/find/entities/search -> {st} {j}")
        return 1
    _ok("/api/find/entities/search")

    st, j = _get(base, "/api/find/relations/search?query_text=%E6%B5%8B%E8%AF%95&max_results=3&threshold=0.35")
    if st >= 500 or not j.get("success"):
        _fail(f"/api/find/relations/search -> {st} {j}")
        return 1
    _ok("/api/find/relations/search")

    if logical_eid:
        eid_enc = urllib.parse.quote(logical_eid, safe="")
        st, j = _get(base, f"/api/find/entities/{eid_enc}")
        if st >= 500 or not j.get("success"):
            _fail(f"/api/find/entities/<id> -> {st} {j}")
            return 1
        _ok("/api/find/entities/<entity_id>")

        st, j = _get(base, f"/api/find/entities/{eid_enc}/versions")
        if st >= 500 or not j.get("success"):
            _fail(f"/api/find/entities/versions -> {st} {j}")
            return 1
        _ok("/api/find/entities/<id>/versions")

        st, j = _get(
            base,
            f"/api/find/entities/{eid_enc}/at-time?time_point=2099-12-31T23:59:59",
        )
        if st not in (200, 404) or (st == 200 and not j.get("success")):
            _fail(f"/api/find/entities/at-time -> {st} {j}")
            return 1
        _ok("/api/find/entities/<id>/at-time")

        st, j = _get(base, f"/api/find/entities/{eid_enc}/version-count")
        if st not in (200, 404) or (st == 200 and not j.get("success")):
            _fail(f"/api/find/entities/version-count -> {st} {j}")
            return 1
        _ok("/api/find/entities/<id>/version-count")

        st, j = _get(base, f"/api/find/relations/by-entity/{eid_enc}?limit=5")
        if st >= 500 or not j.get("success"):
            _fail(f"/api/find/relations/by-entity -> {st} {j}")
            return 1
        _ok("/api/find/relations/by-entity/<entity_id>")

    st, j = _request(
        "POST",
        base + "/api/find/query-one",
        {
            "query_text": "测试",
            "similarity_threshold": 0.35,
            "max_entities": 5,
            "max_relations": 5,
            "include_entities": True,
            "include_relations": True,
        },
    )
    if st >= 500 or not j.get("success"):
        _fail(f"/api/find/query-one -> {st} {j}")
        return 1
    _ok("/api/find/query-one")

    # --- Tier 3: extended ---
    st, j = _get(base, "/api/find/entities/all?limit=3")
    if st >= 500 or not j.get("success"):
        _fail(f"/api/find/entities/all -> {st} {j}")
        return 1
    _ok("/api/find/entities/all?limit=3")

    st, j = _get(base, "/api/find/entities/all-before-time?time_point=2099-12-31T23:59:59&limit=3")
    if st >= 500 or not j.get("success"):
        _fail(f"/api/find/entities/all-before-time -> {st} {j}")
        return 1
    _ok("/api/find/entities/all-before-time")

    if logical_eid:
        st, j = _request(
            "POST",
            base + "/api/find/entities/version-counts",
            {"entity_ids": [logical_eid]},
        )
        if st >= 500 or not j.get("success"):
            _fail(f"/api/find/entities/version-counts -> {st} {j}")
            return 1
        _ok("POST /api/find/entities/version-counts")

    if abs_eid:
        ae = urllib.parse.quote(abs_eid, safe="")
        st, j = _get(base, f"/api/find/entities/by-absolute-id/{ae}")
        if st not in (200, 404) or (st == 200 and not j.get("success")):
            _fail(f"/api/find/entities/by-absolute-id -> {st} {j}")
            return 1
        _ok("/api/find/entities/by-absolute-id/<absolute_id>")

        st, j = _get(base, f"/api/find/entities/by-absolute-id/{ae}/embedding-preview?num_values=4")
        if st not in (200, 404) or (st == 200 and not j.get("success")):
            _fail(f"/api/find/entities/embedding-preview -> {st} {j}")
            return 1
        _ok("/api/find/entities/by-absolute-id/.../embedding-preview")

    st, j = _get(base, "/api/find/relations/all")
    if st >= 500 or not j.get("success"):
        _fail(f"/api/find/relations/all -> {st} {j}")
        return 1
    _ok("/api/find/relations/all")

    if logical_eid and logical_eid_2:
        st, j = _get(
            base,
            "/api/find/relations/between?"
            + urllib.parse.urlencode(
                {"from_entity_id": logical_eid, "to_entity_id": logical_eid_2}
            ),
        )
        if st >= 500 or not j.get("success"):
            _fail(f"/api/find/relations/between -> {st} {j}")
            return 1
        _ok("/api/find/relations/between")

    if abs_rid and logical_rid:
        r_enc = urllib.parse.quote(logical_rid, safe="")
        st, j = _get(base, f"/api/find/relations/{r_enc}/versions")
        if st not in (200, 404) or (st == 200 and not j.get("success")):
            _fail(f"/api/find/relations/versions -> {st} {j}")
            return 1
        _ok("/api/find/relations/<relation_id>/versions")

        st, j = _get(base, f"/api/find/relations/by-absolute-id/{urllib.parse.quote(abs_rid, safe='')}")
        if st not in (200, 404) or (st == 200 and not j.get("success")):
            _fail(f"/api/find/relations/by-absolute-id -> {st} {j}")
            return 1
        _ok("/api/find/relations/by-absolute-id/<entity_absolute_id>")

        st, j = _get(
            base,
            f"/api/find/relations/by-absolute-id/{urllib.parse.quote(abs_rid, safe='')}/embedding-preview",
        )
        if st not in (200, 404) or (st == 200 and not j.get("success")):
            _fail(f"/api/find/relations/embedding-preview -> {st} {j}")
            return 1
        _ok("/api/find/relations/by-absolute-id/.../embedding-preview")

    st, j = _get(base, "/api/find/memory-cache/latest/metadata")
    if st >= 500 or (st == 200 and not j.get("success")):
        _fail(f"/api/find/memory-cache/latest/metadata -> {st} {j}")
        return 1
    if st not in (200, 404):
        _fail(f"/api/find/memory-cache/latest/metadata -> {st} {j}")
        return 1
    _ok("/api/find/memory-cache/latest/metadata")

    st, j = _get(base, "/api/find/memory-cache/latest")
    if st not in (200, 404) or (st == 200 and not j.get("success")):
        _fail(f"/api/find/memory-cache/latest -> {st} {j}")
        return 1
    _ok("/api/find/memory-cache/latest")

    # memory-cache by id: use latest id if present
    mcd = j.get("data") if st == 200 else {}
    if isinstance(mcd, dict) and mcd.get("id"):
        cid = mcd["id"]
        st, j = _get(base, f"/api/find/memory-cache/{urllib.parse.quote(cid, safe='')}")
        if st not in (200, 404) or (st == 200 and not j.get("success")):
            _fail(f"/api/find/memory-cache/<id> -> {st} {j}")
            return 1
        _ok("/api/find/memory-cache/<cache_id>")

        st, j = _get(base, f"/api/find/memory-cache/{urllib.parse.quote(cid, safe='')}/text")
        if st not in (200, 404) or (st == 200 and not j.get("success")):
            _fail(f"/api/find/memory-cache/<id>/text -> {st} {j}")
            return 1
        _ok("/api/find/memory-cache/<cache_id>/text")

    if not args.skip_remember:
        q = urllib.parse.urlencode(
            {
                "text": "冒烟测试写入：这是一条用于验证 GET /api/remember 与状态轮询的短叙事。时间：测试会话。",
                "source_name": "tmg_skill_api_smoke",
                "event_time": "2026-03-22T12:00:00",
                "load_cache_memory": "false",
            },
            encoding="utf-8",
        )
        st, j = _get(base, "/api/remember?" + q, timeout=120.0)
        if st not in (200, 202) or not j.get("success"):
            _fail(f"/api/remember GET -> {st} {j}")
            return 1
        task_id = (j.get("data") or {}).get("task_id")
        if not task_id:
            _fail("remember response missing task_id")
            return 1
        _ok(f"/api/remember GET task_id={task_id[:8]}…")

        deadline = time.time() + 600.0
        status = ""
        while time.time() < deadline:
            st2, j2 = _get(base, f"/api/remember/status/{urllib.parse.quote(task_id, safe='')}")
            if st2 >= 500:
                _fail(f"/api/remember/status -> {st2} {j2}")
                return 1
            if j2.get("success"):
                status = (j2.get("data") or {}).get("status", "")
                if status in ("completed", "failed"):
                    _ok(f"/api/remember/status -> {status}")
                    break
            time.sleep(1.0)
        else:
            _fail("/api/remember/status poll timeout")
            return 1

        q2 = urllib.parse.urlencode(
            {
                "text": "冒烟第二段：更短的文本。",
                "source_name": "tmg_smoke_second",
                "event_time": "2026-03-22T12:01:00",
            },
            encoding="utf-8",
        )
        st, j = _get(base, "/api/remember?" + q2, timeout=120.0)
        if st not in (200, 202) or not j.get("success"):
            _fail(f"/api/remember GET (2nd) -> {st} {j}")
            return 1
        _ok("/api/remember GET (second enqueue)")

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
