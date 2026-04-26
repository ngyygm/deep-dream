"""Shared Neo4j helper functions and constants.

Used by Neo4jStorageManager and all mixins.

借鉴 Graphiti (Zep) 的分层节点架构：
    Neo4j       → 图结构存储（Entity / Relation / Episode 节点及边）
    sqlite-vec  → embedding 向量存储与 KNN 搜索

与 StorageManager 保持完全相同的公共接口，可作为 drop-in replacement。
"""


import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from ...models import ContentPatch, Episode, Entity, Relation

# Try to get neo4j.time.DateTime for isinstance check (faster than hasattr)
try:
    import neo4j.time as _neo4j_time
    _Neo4jDateTime = _neo4j_time.DateTime
except ImportError:
    _Neo4jDateTime = type(None)  # isinstance will never match
from ...utils import clean_markdown_code_blocks
from ...perf import _perf_timer
from ..cache import QueryCache
from ..vector_store import VectorStore
from functools import lru_cache as _lru_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cypher RETURN 子句片段 — 所有 Entity 查询共用
# ---------------------------------------------------------------------------
_ENTITY_RETURN_FIELDS = """\
e.uuid AS uuid, e.family_id AS family_id, e.name AS name,
e.content AS content, e.summary AS summary,
e.attributes AS attributes, e.confidence AS confidence,
e.content_format AS content_format, e.community_id AS community_id,
e.valid_at AS valid_at, e.invalid_at AS invalid_at,
e.event_time AS event_time, e.processed_time AS processed_time,
e.episode_id AS episode_id, e.source_document AS source_document"""

# ---------------------------------------------------------------------------
# Cypher RETURN 子句片段 — 所有 Relation 查询共用
# ---------------------------------------------------------------------------
_RELATION_RETURN_FIELDS = """\
r.uuid AS uuid, r.family_id AS family_id,
r.entity1_absolute_id AS entity1_absolute_id,
r.entity2_absolute_id AS entity2_absolute_id,
r.content AS content, r.event_time AS event_time,
r.processed_time AS processed_time, r.episode_id AS episode_id,
r.source_document AS source_document, r.valid_at AS valid_at,
r.invalid_at AS invalid_at, r.summary AS summary,
r.attributes AS attributes, r.confidence AS confidence,
r.provenance AS provenance"""

# 占位符：在普通字符串中写 RETURN __ENT_FIELDS__ / __REL_FIELDS__，
# _expand_cypher() 展开为实际字段列表
_ENT_FIELDS_RETURN = f"RETURN {_ENTITY_RETURN_FIELDS}"
_REL_FIELDS_RETURN = f"RETURN {_RELATION_RETURN_FIELDS}"


@_lru_cache(maxsize=256)
def _expand_cypher(cypher: str) -> str:
    """展开 Cypher 查询中的 __ENT_FIELDS__ / __REL_FIELDS__ 占位符为实际 RETURN 字段。"""
    cypher = cypher.replace("RETURN DISTINCT __ENT_FIELDS__", f"RETURN DISTINCT {_ENTITY_RETURN_FIELDS}")
    cypher = cypher.replace("RETURN __ENT_FIELDS__", _ENT_FIELDS_RETURN)
    cypher = cypher.replace("RETURN DISTINCT __REL_FIELDS__", f"RETURN DISTINCT {_RELATION_RETURN_FIELDS}")
    cypher = cypher.replace("RETURN __REL_FIELDS__", _REL_FIELDS_RETURN)
    return cypher


# 向后兼容旧名
_q = _expand_cypher


# ---------------------------------------------------------------------------
# graph_id 自动注入：Community Edition 属性级图谱隔离
# ---------------------------------------------------------------------------

# 需要注入 graph_id 过滤的节点标签
_GRAPH_SCOPED_LABELS = frozenset({"Entity", "Relation", "Episode"})

# 匹配 MATCH/OPTIONAL MATCH 中的 (alias:Label) 模式
# 捕获: 可选的 OPTIONAL 前缀 + MATCH 关键字 + 节点模式 (alias:Label ...)
_MATCH_NODE_RE = re.compile(
    r'(OPTIONAL\s+)?MATCH\s+\((\w+):(Entity|Relation|Episode)\b',
    re.IGNORECASE,
)

# 匹配 BM25 fulltext 的 YIELD node 模式
_BM25_YIELD_RE = re.compile(
    r'(CALL\s+db\.index\.fulltext\.queryNodes\([^)]+\)\s*\n?\s*YIELD\s+node\b)',
    re.IGNORECASE,
)

# Pre-compiled regex patterns used in _inject_graph_id_filter
_WHERE_RE = re.compile(r'\bWHERE\b', re.IGNORECASE)
_RETURN_ORDER_LIMIT_WITH_RE = re.compile(r'\b(RETURN|ORDER|LIMIT|WITH)\b', re.IGNORECASE)
_DETACH_DELETE_RE = re.compile(r'(DETACH\s+)?DELETE\b', re.IGNORECASE)
_NEXT_CLAUSE_RE = re.compile(
    r'\b(MATCH|MERGE|OPTIONAL\s+MATCH|RETURN|WITH|UNWIND|CREATE|SET|DELETE|DETACH)\b',
    re.IGNORECASE,
)


def _scan_path_end(cypher: str, start: int) -> int:
    """从节点关闭括号后扫描，跳过所有 边-节点 延续，返回路径模式的终点位置。

    处理: (a:Entity)-[r:TYPE]->(b)-[s]->(c) 等链式路径模式。
    """
    pos = start
    while pos < len(cypher):
        # 跳过空白
        while pos < len(cypher) and cypher[pos] in ' \t\n\r':
            pos += 1
        if pos >= len(cypher):
            break

        # 检测边的开始: '-' 或 '<-'
        if cypher[pos] == '<' and pos + 1 < len(cypher) and cypher[pos + 1] == '-':
            pos += 2
        elif cypher[pos] == '-':
            pos += 1
        else:
            break  # 不是边延续，路径结束

        # 跳过边内容: [...]
        while pos < len(cypher) and cypher[pos] in ' \t':
            pos += 1
        if pos < len(cypher) and cypher[pos] == '[':
            depth = 1
            pos += 1
            while pos < len(cypher) and depth > 0:
                if cypher[pos] == '[':
                    depth += 1
                elif cypher[pos] == ']':
                    depth -= 1
                pos += 1

        # 跳过边结尾: 可选的 '-' 或 '->'
        while pos < len(cypher) and cypher[pos] in ' \t':
            pos += 1
        if pos < len(cypher) and cypher[pos] == '-':
            pos += 1
            while pos < len(cypher) and cypher[pos] == '>':
                pos += 1

        # 节点模式: (...)
        while pos < len(cypher) and cypher[pos] in ' \t\n\r':
            pos += 1
        if pos < len(cypher) and cypher[pos] == '(':
            depth = 1
            pos += 1
            while pos < len(cypher) and depth > 0:
                if cypher[pos] == '(':
                    depth += 1
                elif cypher[pos] == ')':
                    depth -= 1
                pos += 1
        else:
            # 边后无节点，路径结束
            break

    return pos


@_lru_cache(maxsize=256)
def _inject_graph_id_filter(cypher: str) -> str:
    """向 Cypher 查询自动注入 graph_id WHERE 过滤。

    处理的节点标签: Entity, Relation, Episode
    跳过的模式: MERGE（graph_id 通过 SET 子句设置）、DETACH DELETE（通过具体 ID 定位）

    对每个 MATCH (alias:Label) 模式：
    - 路径模式 (a:Entity)-[r]->(b:Entity): 扫描到路径终点后注入
    - DETACH DELETE/DELETE: 跳过（使用具体标识符，无需额外过滤）
    - 有 WHERE → 在 WHERE 条件前插入 alias.graph_id = $graph_id AND
    - 无 WHERE → 在路径/节点模式后添加 WHERE alias.graph_id = $graph_id
    """
    # 跟踪所有已处理的 alias:position，避免重复注入
    processed: set[tuple[int, str]] = set()

    # 1. 处理 BM25 fulltext YIELD node 模式
    for m in _BM25_YIELD_RE.finditer(cypher):
        yield_end = m.end()
        rest = cypher[yield_end:]
        where_match = _WHERE_RE.search(rest)
        if where_match:
            insert_pos = yield_end + where_match.end()
            cypher = cypher[:insert_pos] + " node.graph_id = $graph_id AND" + cypher[insert_pos:]
        else:
            insert_match = _RETURN_ORDER_LIMIT_WITH_RE.search(rest)
            if insert_match:
                insert_pos = yield_end + insert_match.start()
                cypher = cypher[:insert_pos] + "WHERE node.graph_id = $graph_id\n                       " + cypher[insert_pos:]

    # 2. 处理 MATCH (alias:Label) 模式
    # 需要从后向前处理，避免偏移量错乱
    matches = list(_MATCH_NODE_RE.finditer(cypher))

    for m in reversed(matches):
        alias = m.group(2)
        label = m.group(3)
        match_start = m.start()

        # 跳过已处理的（同一位置的同一 alias）
        if (match_start, alias) in processed:
            continue
        processed.add((match_start, alias))

        # 找到节点模式的关闭括号 ')'
        # 从 m.end() 开始（此时在 Label 之后），查找第一个未嵌套的 ')'
        paren_depth = 1  # 已经有一个 '(' 被匹配
        pos = m.end()
        while pos < len(cypher) and paren_depth > 0:
            if cypher[pos] == '(':
                paren_depth += 1
            elif cypher[pos] == ')':
                paren_depth -= 1
            pos += 1
        # pos 现在指向关闭 ')' 之后的位置
        node_end = pos

        # 检测 DETACH DELETE / DELETE — 跳过注入（使用具体标识符定位）
        rest_raw = cypher[node_end:].lstrip()
        if _DETACH_DELETE_RE.match(rest_raw):
            continue

        # 扫描路径模式 — 跳过边-节点延续到路径终点
        path_end = _scan_path_end(cypher, node_end)

        # 路径扫描后再次检测 DETACH DELETE / DELETE
        # 处理 MATCH (e:Entity)-[r:RELATES_TO]-() DETACH DELETE r 等模式
        rest_after_path = cypher[path_end:].lstrip()
        if _DETACH_DELETE_RE.match(rest_after_path):
            continue

        # 处理逗号分隔的多节点 MATCH 模式: MATCH (a:X), (b:Y), ...
        # 需要跳过所有逗号后续的节点，在最后一个节点后才注入 WHERE
        scan_pos = path_end
        while True:
            # 跳过空白
            while scan_pos < len(cypher) and cypher[scan_pos] in ' \t\n\r':
                scan_pos += 1
            if scan_pos < len(cypher) and cypher[scan_pos] == ',':
                scan_pos += 1  # 跳过逗号
                # 跳过空白
                while scan_pos < len(cypher) and cypher[scan_pos] in ' \t\n\r':
                    scan_pos += 1
                # 跳过节点模式 (...)
                if scan_pos < len(cypher) and cypher[scan_pos] == '(':
                    depth = 1
                    scan_pos += 1
                    while scan_pos < len(cypher) and depth > 0:
                        if cypher[scan_pos] == '(':
                            depth += 1
                        elif cypher[scan_pos] == ')':
                            depth -= 1
                        scan_pos += 1
                    # 跳过路径延续（边-节点模式）
                    scan_pos = _scan_path_end(cypher, scan_pos)
                    path_end = scan_pos
                    continue
            break  # 不是逗号延续，结束扫描

        # path_end 指向路径模式后第一个非空白字符（如 RETURN/WHERE 等关键字）
        # cypher[:path_end] 保留 node_end..path_end 之间的空白
        rest = cypher[path_end:]
        where_search = _WHERE_RE.search(rest)

        # 确保 WHERE 属于当前 MATCH（不是后续子句的 WHERE）
        next_clause = _NEXT_CLAUSE_RE.search(rest)

        if where_search and (not next_clause or where_search.start() < next_clause.start()):
            # 有 WHERE 子句，在其后插入 graph_id 过滤
            insert_pos = path_end + where_search.end()
            cypher = cypher[:insert_pos] + f" {alias}.graph_id = $graph_id AND" + cypher[insert_pos:]
        else:
            # 无 WHERE 子句，在路径终点后添加 WHERE（保留原始空白分隔）
            cypher = cypher[:path_end] + f"\n                WHERE {alias}.graph_id = $graph_id\n                " + cypher[path_end:]

    return cypher


# ---------------------------------------------------------------------------
# Neo4j 节点 / 边 属性 → Entity / Relation 转换
# ---------------------------------------------------------------------------

# Lightweight now-cache: avoids a datetime.now() syscall per record in batch queries.
# Within a single batch of 500 records (~10ms), the same "now" is perfectly fine.
_cached_now_time: float = 0.0
_cached_now_val: Optional[datetime] = None
_cached_now_lock = threading.Lock()


def _get_cached_now() -> datetime:
    """Return a cached datetime.now() value, refreshed every ~1 second."""
    global _cached_now_time, _cached_now_val
    _t = time.time()
    if _cached_now_val is None or (_t - _cached_now_time) > 1.0:
        _cached_now_val = datetime.now()
        _cached_now_time = _t
    return _cached_now_val


def _neo4j_record_to_entity(record, _now: Optional[datetime] = None) -> Entity:
    """将 Neo4j 查询返回的单条记录转为 Entity dataclass。"""
    _pd = _parse_dt
    if _now is None:
        _now = _get_cached_now()
    return Entity(
        absolute_id=record["uuid"],
        family_id=record["family_id"],
        name=record.get("name", ""),
        content=record.get("content", ""),
        event_time=_pd(record.get("event_time")) or _now,
        processed_time=_pd(record.get("processed_time")) or _now,
        episode_id=record.get("episode_id", ""),
        source_document=record.get("source_document") or "",
        embedding=record.get("embedding"),
        valid_at=_pd(record.get("valid_at")),
        invalid_at=_pd(record.get("invalid_at")),
        summary=record.get("summary"),
        attributes=record.get("attributes"),
        confidence=float(_c) if (_c := record.get("confidence")) is not None else None,
        content_format=record.get("content_format", "plain"),
        community_id=record.get("community_id"),
    )


def _neo4j_record_to_relation(record, _now: Optional[datetime] = None) -> Relation:
    """将 Neo4j 查询返回的单条记录转为 Relation dataclass。"""
    _pd = _parse_dt
    if _now is None:
        _now = _get_cached_now()
    return Relation(
        absolute_id=record["uuid"],
        family_id=record["family_id"],
        entity1_absolute_id=record.get("entity1_absolute_id", ""),
        entity2_absolute_id=record.get("entity2_absolute_id", ""),
        content=record.get("content", ""),
        event_time=_pd(record.get("event_time")) or _now,
        processed_time=_pd(record.get("processed_time")) or _now,
        episode_id=record.get("episode_id", ""),
        source_document=record.get("source_document") or "",
        embedding=record.get("embedding"),
        valid_at=_pd(record.get("valid_at")),
        invalid_at=_pd(record.get("invalid_at")),
        summary=record.get("summary"),
        attributes=record.get("attributes"),
        confidence=float(_c) if (_c := record.get("confidence")) is not None else None,
        provenance=record.get("provenance"),
        content_format=record.get("content_format", "plain"),
    )


def _parse_dt(value: Any) -> Optional[datetime]:
    """安全解析日期时间。返回 None 表示缺失，而非 fallback 到当前时间。"""
    if value is None:
        return None
    # Fast path: standard datetime (covers 99%+ of records)
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    # Neo4j driver returns neo4j.time.DateTime — use isinstance instead of hasattr
    if isinstance(value, _Neo4jDateTime):
        try:
            native = value.to_native()
            if isinstance(native, datetime):
                return native.replace(tzinfo=None) if native.tzinfo else native
        except Exception:
            pass
    # Fallback: try string conversion
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            pass
    # Last resort: objects with isoformat method
    if hasattr(value, 'isoformat'):
        try:
            return datetime.fromisoformat(value.isoformat()).replace(tzinfo=None)
        except Exception:
            pass
    logger.warning("_parse_dt received unexpected type %s, returning None", type(value).__name__)
    return None


def _fmt_dt(value: Any) -> Optional[str]:
    """安全格式化日期时间为 ISO 字符串。兼容 datetime、neo4j.time.DateTime 和字符串。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.replace(tzinfo=None).isoformat() if value.tzinfo else value.isoformat()
    # neo4j.time.DateTime 等非 datetime 子类
    if hasattr(value, 'to_native'):
        try:
            native = value.to_native()
            if isinstance(native, datetime):
                return native.replace(tzinfo=None).isoformat() if native.tzinfo else native.isoformat()
        except Exception:
            pass
    if hasattr(value, 'isoformat'):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


_NATIVE_TYPES = (str, int, float, bool, type(None))

def _neo4j_types_to_native(obj):
    """将 Neo4j 返回的 dict/list 中不可 JSON 序列化的类型转为 Python 原生类型。"""
    if isinstance(obj, _NATIVE_TYPES):
        return obj
    if isinstance(obj, dict):
        return {k: _neo4j_types_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_neo4j_types_to_native(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    # neo4j.time.DateTime / Date / Duration 等
    if hasattr(obj, 'iso_format') or hasattr(obj, 'isoformat'):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    return obj


