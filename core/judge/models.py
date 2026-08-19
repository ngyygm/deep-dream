"""裁决服务的 key 构建与规范化。

key 设计目标：
- 同一对判断（名称+内容相同）在任意窗口/文档/processor 上得到同一 key
- A/B 顺序无关（对齐判断是对称的）
- 内容截断与真实 LLM prompt 的截断一致（guard 500 / 候选 200），
  避免"prompt 相同但 key 不同"的漏命中
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

# 与 entity_search._alignment_guard 的 LRU key 截断保持一致
_GUARD_CONTENT_TRUNC = 200
# 与 LLMClient.judge_entity_alignment 的 prompt 截断一致
_PROMPT_CONTENT_TRUNC = 500
_CANDIDATE_CONTENT_TRUNC = 200

_WS_RE = re.compile(r"\s+")


def norm_name(name: str) -> str:
    """名称规范化：casefold + 压缩空白。用于 key 与 family 名归一。"""
    return _WS_RE.sub(" ", str(name or "").strip()).casefold()


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _content_hash(content: Optional[str], trunc: int) -> str:
    c = str(content or "")
    if len(c) > trunc:
        c = c[:trunc]
    return _sha1(c)


def _digest(payload: Any) -> str:
    return _sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def guard_key(name_a: str, content_a: Optional[str],
              name_b: str, content_b: Optional[str],
              name_match_type: str = "none") -> str:
    """实体对齐 guard 判断的 memo key。A/B 对称：按规范名排序后拼装。"""
    a = (norm_name(name_a), _content_hash(content_a, _PROMPT_CONTENT_TRUNC))
    b = (norm_name(name_b), _content_hash(content_b, _PROMPT_CONTENT_TRUNC))
    first, second = (a, b) if a <= b else (b, a)
    return _digest(["guard", first, second, str(name_match_type or "none")])


def resolve_entity_key(entity: Dict[str, Any],
                       candidates: Sequence[Dict[str, Any]],
                       context_text: Optional[str] = None) -> str:
    """resolve_entity_candidates_batch 的 memo key。

    候选以 (family_id, name, content前200) 参与——family_id 保证候选身份，
    content hash 保证候选内容变化后 key 随之变化（配合 invalidate_for_family）。
    """
    cand_parts = sorted(
        [
            [
                str(c.get("family_id", "")),
                norm_name(str(c.get("name", ""))),
                _content_hash(c.get("content"), _CANDIDATE_CONTENT_TRUNC),
                str(c.get("name_match_type", "none") or "none"),
            ]
            for c in (candidates or [])
        ]
    )
    return _digest([
        "resolve_ent",
        norm_name(str(entity.get("name", ""))),
        _content_hash(entity.get("content"), _CANDIDATE_CONTENT_TRUNC),
        cand_parts,
        _content_hash(context_text, 500),
    ])


def resolve_relation_key(entity1_name: str, entity2_name: str,
                         new_relation_contents: Sequence[str],
                         existing_relations: Sequence[Dict[str, Any]],
                         new_source_document: str = "") -> str:
    """resolve_relation_pair_batch 的 memo key。端点对对称。"""
    e1, e2 = sorted([norm_name(entity1_name), norm_name(entity2_name)])
    existing_parts = sorted(
        [
            [
                str(r.get("family_id", "")),
                _content_hash(r.get("content"), _CANDIDATE_CONTENT_TRUNC),
            ]
            for r in (existing_relations or [])
        ]
    )
    new_parts = sorted(str(c) for c in (new_relation_contents or []))
    return _digest([
        "resolve_rel", e1, e2, new_parts, existing_parts,
        str(new_source_document or ""),
    ])


def families_touched(entity: Dict[str, Any],
                     candidates: Sequence[Dict[str, Any]]) -> List[str]:
    """从 resolve_ent 调用参数中提取涉及的 family_id 列表（用于合并失效）。"""
    fids = set()
    for item in [entity] + list(candidates or []):
        fid = str((item or {}).get("family_id", "") or "")
        if fid:
            fids.add(fid)
    return sorted(fids)


def families_touched_for_relation(existing_relations: Iterable[Dict[str, Any]]) -> List[str]:
    fids = set()
    for r in existing_relations or []:
        fid = str((r or {}).get("family_id", "") or "")
        if fid:
            fids.add(fid)
    return sorted(fids)
