"""Concept DAL — 统一概念原语的角色分发门面（Option B）。

这是 ACL-2026 论文所宣称的"统一 NL Concept 原语"的规范入口。
entity / relation / episode / document 在概念层都是同一个 Concept，
只是 ``role`` 不同。本模块对外暴露 **一个** Concept 接口（按 role 参数分派），
底层物理存储仍保持 **双轨制**：

  - entity_families / entity_observations   （实体轨道）
  - relation_families / relation_assertions （关系轨道）

不做物理表合并，因为合并会带来 NULL 扩散、丢失索引、以及实体/关系
身份语义不兼容（实体用 canonical_name 唯一，关系用 (subject, object) 唯一）。
本门面只是把两条轨道的 repository 函数（``entities.py`` / ``relations.py``）
按 role 映射到统一签名，使调用方只看到 Concept 抽象。

分派是 **数据驱动** 的（ROLE_REGISTRY 字典），不是 if/elif 链，
新增角色只需在 registry 中登记。

注意：episode / document 是"容器"角色，它们的概念族存储在各自的
repository（episodes.py / documents.py）中，本门面只通过
``role_to_owner_type`` 为 embedding 层提供 owner_type 映射，
不直接承担 episode/document 的家族读写。
"""

from __future__ import annotations

import logging
from typing import Optional

from core.models import Concept
from . import entities, relations

logger = logging.getLogger(__name__)

# ── 角色常量 ───────────────────────────────────────────────
ROLE_ENTITY = "entity"
ROLE_RELATION = "relation"
# 容器角色（仅用于 owner_type 映射，不参与 family 读写分发）
ROLE_EPISODE = "episode"
ROLE_DOCUMENT = "document"

#: 所有合法的 concept 角色（含容器角色）
ALL_ROLES = (ROLE_ENTITY, ROLE_RELATION, ROLE_EPISODE, ROLE_DOCUMENT)

#: 参与 family 读写分发的角色（实体/关系两条物理轨道）
DISPATCHABLE_ROLES = (ROLE_ENTITY, ROLE_RELATION)

#: embedding owner_type 映射 —— Concept 角色 -> embeddings.owner_type
ROLE_TO_OWNER_TYPE = {
    ROLE_ENTITY: "entity_obs",
    ROLE_RELATION: "relation_assert",
    ROLE_EPISODE: "episode",
    ROLE_DOCUMENT: "document_version",
}


def role_to_owner_type(role: str) -> str:
    """Concept 角色 -> embeddings.owner_type 列的取值。

    用于 embedding 层按概念角色检索时确定 owner_type。
    """
    try:
        return ROLE_TO_OWNER_TYPE[role]
    except KeyError:
        raise ValueError(
            f"未知 concept 角色: {role!r}（合法值: {sorted(ROLE_TO_OWNER_TYPE)}）"
        )


# ── 行 -> Concept DTO 转换 ────────────────────────────────

def _row_to_concept(role: str, row: dict) -> Concept:
    """把物理轨道的 dict 行转换为统一的 Concept DTO。

    委托给 ``Concept.from_row``（与 dto-agent 协调的统一行->DTO 映射），
    避免在本门面里重复维护列名映射。在委托前先做一次列名规范化，
    把 family 表特有的列提升到 ``from_row`` 认识的键上：

      - canonical_content -> content   （entity/relation family 的 NL 内容）
      - canonical_name     已由 from_row 处理（-> name）
      - subject/object_entity_family_id 已由 from_row 处理
        （-> subject/object_family_id）

    未消费的列（attributes、source_text 等）原样保留在 ``Concept.extra``。
    """
    row = dict(row or {})
    # canonical_content 是 family 表的内容列；from_row 只认 content/memory_text
    if "content" not in row and row.get("canonical_content"):
        row["content"] = row["canonical_content"]
    return Concept.from_row(role, row)


# ── 关系显示名合成 ──────────────────────────────────────────

def format_relation_display_name(subject_name: str, object_name: str,
                                 content: str) -> str:
    """合成关系的可读显示名（PURE，无 conn 访问）。

    与 ``library_manager._relation_display_name`` 的命名逻辑 **逐字一致**，
    使 DAL 与 API 层产生完全相同的 label：

        if content and len(content) <= 80: label = f"{e1} {content} {e2}".strip()
        else: label = f"{e1} → {e2}".strip(" →") if (e1 or e2) else (content or "")

    分支语义：
      - 短内容（<=80 字符）作为谓词嵌入 "e1 predicate e2"。
      - 否则退化成 "e1 → e2"；若两端都为空则回退到 content 本身。
    """
    e1 = subject_name or ""
    e2 = object_name or ""
    content = content or ""
    if content and len(content) <= 80:
        return f"{e1} {content} {e2}".strip()
    if e1 or e2:
        return f"{e1} → {e2}".strip(" →")
    return content or ""


def _enrich_relation_name(conn, concept: Concept) -> Concept:
    """对 role=='relation' 的 Concept 合成 .name 并 stash 端点名到 extra。

    非关系角色原样返回。解析 subject/object family id -> canonical_name 经
    ``entities.get_entity_family_names``（批量、纯 repo），按
    ``format_relation_display_name`` 产出 label 写入 ``concept.name``，并把
    endpoint 名存进 ``concept.extra["entity1_name"]`` /
    ``["entity2_name"]`` 以与 manager 的 display dict 保持对等。

    防御：family id 为 None/空时，对应端点名视为 ""。
    """
    if getattr(concept, "role", None) != ROLE_RELATION:
        return concept

    sub_fid = concept.subject_family_id or ""
    obj_fid = concept.object_family_id or ""
    lookup_ids = [fid for fid in (sub_fid, obj_fid) if fid]
    names = entities.get_entity_family_names(conn, lookup_ids) if lookup_ids else {}
    e1 = names.get(sub_fid, "") if sub_fid else ""
    e2 = names.get(obj_fid, "") if obj_fid else ""

    concept.name = format_relation_display_name(e1, e2, concept.content)
    if concept.extra is None:
        concept.extra = {}
    concept.extra["entity1_name"] = e1
    concept.extra["entity2_name"] = e2
    return concept


# ── 角色注册表（数据驱动分派） ────────────────────────────

# 每个 dispatchable role 登记其物理表与对应的 repository 函数。
# 这是本门面的核心：把"按 role 选函数"从 if/elif 链变成一次 dict 查找。
ROLE_REGISTRY = {
    ROLE_ENTITY: {
        "family_table": "entity_families",
        "version_table": "entity_observations",
        "owner_type": "entity_obs",
        "family_id_col": "entity_family_id",
        "upsert": entities.upsert_entity_family,
        "get": entities.get_entity_family,
        "find": entities.find_entity_family_by_name,
        "insert_version": entities.insert_entity_observation,
        "active_version": entities.get_active_observation,
        "supersede": entities.supersede_observations_by_episodes,
        "reactivate": entities.reactivate_observations_by_episodes,
        "list": entities.list_entity_families,
    },
    ROLE_RELATION: {
        "family_table": "relation_families",
        "version_table": "relation_assertions",
        "owner_type": "relation_assert",
        "family_id_col": "relation_family_id",
        "upsert": relations.upsert_relation_family,
        "get": relations.get_relation_family,
        "find": relations.find_relation_family,
        "insert_version": relations.insert_relation_assertion,
        # 关系没有"按 family 取单个 active 版本"的原语，只有按 episode 列表；
        # get_active_version 对 relation 退化为按 episode 取第一条 active assertion。
        "active_version": relations.get_active_assertions_by_episode,
        "supersede": relations.supersede_assertions_by_episodes,
        "reactivate": relations.reactivate_assertions_by_episodes,
        "list": relations.list_relation_families,
    },
}


def _resolve(role: str) -> dict:
    """解析角色对应的 registry 项；非法角色抛 ValueError。"""
    try:
        return ROLE_REGISTRY[role]
    except KeyError:
        raise ValueError(
            f"概念角色 {role!r} 不支持家族读写（合法值: {DISPATCHABLE_ROLES}）"
        )


# ── 统一 Concept 接口 ──────────────────────────────────────

def upsert_concept_family(conn, role: str, family_id: str, **kwargs) -> None:
    """按角色 upsert 一个概念家族。

    role='entity' kwargs: canonical_name, canonical_content, created_at, updated_at
    role='relation' kwargs: subject_entity_family_id, object_entity_family_id,
                            canonical_content, created_at, updated_at
    """
    reg = _resolve(role)
    reg["upsert"](conn, family_id, **kwargs)


def get_concept_family(conn, role: str, family_id: str) -> Optional[Concept]:
    """按家族 ID 取一个 Concept，返回统一 DTO；不存在返回 None。"""
    reg = _resolve(role)
    row = reg["get"](conn, family_id)
    if row is None:
        return None
    return _enrich_relation_name(conn, _row_to_concept(role, row))


def find_concept_family(conn, role: str, **kwargs) -> Optional[Concept]:
    """按业务键查找一个 Concept。

    role='entity': canonical_name=<...>
    role='relation': subject_family_id=<...>, object_family_id=<...>
    """
    reg = _resolve(role)
    if role == ROLE_ENTITY:
        row = reg["find"](conn, kwargs["canonical_name"])
    elif role == ROLE_RELATION:
        row = reg["find"](conn, kwargs["subject_family_id"], kwargs["object_family_id"])
    else:  # pragma: no cover — _resolve 已保证不达此分支
        raise ValueError(f"find 不支持角色 {role!r}")
    if row is None:
        return None
    return _enrich_relation_name(conn, _row_to_concept(role, row))


def insert_concept_version(conn, role: str, version_id: str,
                           family_id: str, episode_id: str, **kwargs) -> None:
    """插入一个概念版本（observation / assertion）。

    role='entity': name, content, extra_json, processed_at, run_id
    role='relation': subject_entity_id, object_entity_id,
                     subject_entity_family_id, object_entity_family_id,
                     content, evidence_text, evidence_*, extra_json,
                     processed_at, run_id
    """
    reg = _resolve(role)
    if role == ROLE_ENTITY:
        reg["insert_version"](
            conn, version_id, family_id, episode_id, **kwargs,
        )
    elif role == ROLE_RELATION:
        reg["insert_version"](
            conn, version_id, family_id, episode_id, **kwargs,
        )
    else:  # pragma: no cover
        raise ValueError(f"insert_concept_version 不支持角色 {role!r}")


def get_active_version(conn, role: str, episode_id: str, family_id: str):
    """取某 episode 下某家族的当前 active 版本。

    entity -> 单条 observation dict（或 None）
    relation -> list[assertion dict]（关系没有"单版本"原语，返回该 episode
                下该... 注：关系原语按 episode 取全部 active assertion，
                不带 family_id 过滤；调用方需自行从结果中筛 family_id。）
    """
    reg = _resolve(role)
    if role == ROLE_ENTITY:
        return reg["active_version"](conn, episode_id, family_id)
    elif role == ROLE_RELATION:
        # 关系原语签名是 get_active_assertions_by_episode(conn, episode_id)
        # 不接受 family_id；返回该 episode 全部 active assertions。
        return reg["active_version"](conn, episode_id)
    else:  # pragma: no cover
        raise ValueError(f"get_active_version 不支持角色 {role!r}")


def supersede_by_episodes(conn, role: str, episode_ids: list) -> int:
    """把指定 episodes 下该角色的 active 版本置为 superseded。返回受影响行数。"""
    reg = _resolve(role)
    return reg["supersede"](conn, list(episode_ids or []))


def reactivate_by_episodes(conn, role: str, episode_ids: list) -> int:
    """把指定 episodes 下该角色的 superseded 版本恢复为 active。返回受影响行数。"""
    reg = _resolve(role)
    return reg["reactivate"](conn, list(episode_ids or []))


def list_concept_families(conn, role: Optional[str] = None,
                          limit: int = 100, offset: int = 0) -> list:
    """列出概念家族。

    role=None 时跨角色列出全部（entity + relation），合并后按 updated_at 排序。
    指定 role 时只列该角色。
    返回 list[Concept]。
    """
    if limit is None or limit < 0:
        limit = 100
    if offset is None or offset < 0:
        offset = 0

    if role is not None:
        reg = _resolve(role)
        rows = reg["list"](conn, limit=limit, offset=offset)
        concepts = [_row_to_concept(role, r) for r in rows]
        if role == ROLE_RELATION:
            return [_enrich_relation_name(conn, c) for c in concepts]
        return concepts

    # role=None: 跨角色合并。为保证 limit/offset 语义在合并后仍合理，
    # 分别按角色取（每角色最多 limit+offset），合并后排序再裁剪。
    # 时间字段在当前 Concept DTO 中走 extra（updated_at/last_seen_at），
    # 缺失时降级为空串排序。
    def _sort_key(c: Concept) -> str:
        ex = getattr(c, "extra", {}) or {}
        return (ex.get("updated_at") or ex.get("last_seen_at")
                or ex.get("created_at") or "")

    gathered = []
    for r in DISPATCHABLE_ROLES:
        reg = ROLE_REGISTRY[r]
        rows = reg["list"](conn, limit=limit + offset, offset=0)
        for row in rows:
            gathered.append(_row_to_concept(r, row))
    gathered = [_enrich_relation_name(conn, c) if c.role == ROLE_RELATION else c
                for c in gathered]
    gathered.sort(key=_sort_key, reverse=True)
    return gathered[offset: offset + limit]


__all__ = [
    "ROLE_ENTITY",
    "ROLE_RELATION",
    "ROLE_EPISODE",
    "ROLE_DOCUMENT",
    "ALL_ROLES",
    "DISPATCHABLE_ROLES",
    "ROLE_TO_OWNER_TYPE",
    "role_to_owner_type",
    "format_relation_display_name",
    "upsert_concept_family",
    "get_concept_family",
    "find_concept_family",
    "insert_concept_version",
    "get_active_version",
    "supersede_by_episodes",
    "reactivate_by_episodes",
    "list_concept_families",
]
