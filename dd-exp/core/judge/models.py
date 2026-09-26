"""裁决服务的名称规范化与存储侧解析。

P1 删除 memo 缓存模块后，key builder（guard_key/resolve_*_key/
families_touched*）不再有生产消费方，随之移除——judge 判断只剩单一
路径（LLMClient 直调），无双份缓存截断一致性问题。
"""
from __future__ import annotations

import re
from typing import Optional

from core.utils import entity_match_key, entity_name_variants

# LIKE 通配符转义（前缀召回腿用）
_LIKE_ESCAPE = re.compile(r"[~%_]")


def norm_name(name: str) -> str:
    """名称规范化：委托 core.utils.entity_match_key（全库唯一语义）。

    P2 起含括号注记与中文称号后缀剥离——gate 与候选匹配对
    '张三教授' vs '张三' 得出同一结论（此前 gate 视为不同名，正是
    重复 family 竞态漏网路径）。
    """
    return entity_match_key(name)


def resolve_family_id_from_conn(conn, name: str) -> Optional[str]:
    """在给定连接上按统一名称语义解析 family_id（FamilyWriteGate 存储腿）。

    召回：原文/核心名变体精确匹配（COLLATE NOCASE）+ 核心名前缀 LIKE
    （覆盖 DB 侧以全名存储的形式，如查询 "张三" 召回 "张三教授"）。
    精度：命中行必须 norm_name 相等——LIKE 只是召回手段，误报被过滤。

    registry（短只读连接）与 LibraryManager.find_family_id_by_name 共用。
    gate 是竞态兜底而非主匹配器，前缀腿 LIMIT 有界。
    """
    norm = norm_name(name)
    if not norm:
        return None
    variants = [v for v in dict.fromkeys(entity_name_variants(name)) if v]
    placeholders = ",".join("?" for _ in variants)
    rows = conn.execute(
        f"SELECT entity_family_id, canonical_name FROM entity_families "
        f"WHERE canonical_name COLLATE NOCASE IN ({placeholders}) "
        f"ORDER BY updated_at DESC LIMIT 8",
        variants,
    ).fetchall()
    like_pat = _LIKE_ESCAPE.sub(lambda m: "~" + m.group(0), variants[-1]) + "%"
    rows += conn.execute(
        "SELECT entity_family_id, canonical_name FROM entity_families "
        "WHERE canonical_name LIKE ? ESCAPE '~' "
        "ORDER BY updated_at DESC LIMIT 20",
        (like_pat,),
    ).fetchall()
    for fid, db_name in rows:
        if norm_name(db_name or "") == norm:
            return fid
    return None
