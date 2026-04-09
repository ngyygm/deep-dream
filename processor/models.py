"""
核心数据结构定义

旧模型（Entity, Relation, Episode）保留用于向后兼容。
新模型（Concept）是统一的概念原语，参见 docs/vision.md。
"""
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Episode:
    """Episode — 知识图谱的一等节点

    每次写入（remember / dream）产生一个 Episode，包含当时的记忆上下文和原始文本。
    抽取出的实体/关系通过 MENTIONS 边连接回 Episode，实现事实溯源。
    """
    absolute_id: str
    content: str  # Markdown格式的完整描述
    event_time: datetime  # 事件发生时间
    source_document: str  # 来源文档名称
    activity_type: Optional[str] = None  # 可选的活动类型，如"阅读小说"、"处理文档"等
    episode_type: Optional[str] = None  # Episode 类型: "narrative" | "fact" | "conversation" | "dream"


@dataclass
class Entity:
    """实体 - 带版本链"""
    absolute_id: str  # 主键，版本唯一标识符（DB 列名 id）
    family_id: str  # 实体的家族ID，同一实体的不同版本具有相同的family_id
    name: str  # 实体名称
    content: str  # 实体的自然语言描述
    event_time: datetime  # 事件发生时间
    processed_time: datetime  # 系统实际处理时间
    episode_id: str  # 记录当前更新是基于什么记忆环境下的判断
    source_document: str  # 来源文档名称
    embedding: Optional[bytes] = None  # Embedding向量（BLOB格式，可选）
    valid_at: Optional[datetime] = None  # 事实生效时间
    invalid_at: Optional[datetime] = None  # 事实失效时间（被新版本替代）
    summary: Optional[str] = None  # 实体摘要（由 LLM 进化维护）
    attributes: Optional[str] = None  # JSON 字符串，结构化属性字典
    confidence: Optional[float] = None  # 置信度评分 (0.0-1.0)
    content_format: str = "plain"  # "plain" (旧) | "markdown" (新)
    community_id: Optional[str] = None  # 社区检测分配的社区ID


@dataclass
class Relation:
    """关系 - 带版本链的概念边（无向关系）

    关系是无向的，不区分方向，只表示两个实体之间的关联。
    entity1_absolute_id 和 entity2_absolute_id 只是用来标识关系涉及的两个实体，没有方向性。
    存储时，实体对按字母顺序排序（entity1 < entity2），确保 (A,B) 和 (B,A) 被视为同一个关系。
    """
    absolute_id: str  # 主键，版本唯一标识符（DB 列名 id）
    family_id: str  # 关系的家族ID，同一关系的不同版本具有相同的family_id
    entity1_absolute_id: str  # 第一个实体的绝对ID（版本唯一ID，可以通过此ID找到family_id），按字母顺序排序
    entity2_absolute_id: str  # 第二个实体的绝对ID（版本唯一ID，可以通过此ID找到family_id），按字母顺序排序
    content: str  # 关系的自然语言描述
    event_time: datetime  # 事件发生时间
    processed_time: datetime  # 系统实际处理时间
    episode_id: str  # 记录当前更新是基于什么记忆环境下的判断
    source_document: str  # 来源文档名称
    embedding: Optional[bytes] = None  # Embedding向量（BLOB格式，可选）
    valid_at: Optional[datetime] = None  # 事实生效时间
    invalid_at: Optional[datetime] = None  # 事实失效时间（被新版本替代）
    summary: Optional[str] = None  # 关系摘要（由 LLM 进化维护）
    attributes: Optional[str] = None  # JSON 字符串，结构化属性字典
    confidence: Optional[float] = None  # 置信度评分 (0.0-1.0)
    provenance: Optional[str] = None  # JSON: [{"episode_id": "...", "confidence": 0.9}, ...]
    content_format: str = "plain"  # "plain" (旧) | "markdown" (新)


@dataclass
class ContentPatch:
    """Section 级变更记录"""
    uuid: str
    target_type: str  # "Entity" | "Relation"
    target_absolute_id: str  # 哪个版本节点
    target_family_id: str  # 逻辑 ID
    section_key: str  # 哪个 section
    change_type: str  # "added" | "modified" | "unchanged" | "removed" | "restructured"
    old_hash: str  # 旧 section 内容 hash
    new_hash: str  # 新 section 内容 hash
    diff_summary: str  # 变更摘要
    source_document: str  # 触发来源
    event_time: datetime


# ---------------------------------------------------------------------------
# 新统一模型 — Concept（概念）
# ---------------------------------------------------------------------------

# 概念角色常量
ROLE_ENTITY = "entity"
ROLE_RELATION = "relation"
ROLE_OBSERVATION = "observation"


@dataclass
class ConceptVersion:
    """概念的单个版本快照"""
    absolute_id: str  # 版本唯一标识
    content: str  # 该版本的内容快照
    source_concept_id: str  # 产生该版本的源 observation 概念 ID
    processed_time: datetime  # 该版本的产生时间
    valid_at: Optional[datetime] = None  # 有效期起始
    invalid_at: Optional[datetime] = None  # 有效期结束（被新版本替代时设置）


@dataclass
class Concept:
    """统一概念模型 — 系统的唯一原语

    万物皆概念。Entity、Relation、Observation 只是角色的不同，
    不是类型系统的区分。所有概念遵循相同的存储、版本、检索、遍历规则。

    详见 docs/vision.md 第二节"核心概念模型"。
    """
    family_id: str  # 逻辑身份，跨版本不变
    role: str  # 角色：entity | relation | observation
    name: Optional[str] = None  # 显示名称（entity 用）
    content: Optional[str] = None  # 当前版本的内容
    embedding: Optional[bytes] = None  # 语义向量
    summary: Optional[str] = None  # 概念摘要（派生视图，可由 LLM 重新生成）
    confidence: Optional[float] = None  # 置信度 (0.0-1.0)
    attributes: Optional[str] = None  # JSON 字符串，结构化属性

    # 关系型概念专用：指向两端的概念
    connects: Optional[List[str]] = None  # [concept_family_id, ...]

    # 版本信息
    versions: Optional[List[ConceptVersion]] = None

    # 时间信息（当前版本）
    created_at: Optional[datetime] = None  # 概念首次创建时间
    updated_at: Optional[datetime] = None  # 最近更新时间

    # 溯源：提及过此概念的源文本列表（observation concept family_ids）
    sources: Optional[List[str]] = None

    # 社区信息
    community_id: Optional[str] = None


# ---------------------------------------------------------------------------
# 转换工具：旧模型 ↔ Concept
# ---------------------------------------------------------------------------

def entity_to_concept(e: Entity) -> Concept:
    """将旧 Entity 转换为统一 Concept"""
    return Concept(
        family_id=e.family_id,
        role=ROLE_ENTITY,
        name=e.name,
        content=e.content,
        embedding=e.embedding,
        summary=e.summary,
        confidence=e.confidence,
        attributes=e.attributes,
        versions=[ConceptVersion(
            absolute_id=e.absolute_id,
            content=e.content,
            source_concept_id=e.episode_id or "",
            processed_time=e.processed_time,
            valid_at=e.valid_at,
            invalid_at=e.invalid_at,
        )],
        created_at=e.event_time,
        updated_at=e.processed_time,
        sources=[e.episode_id] if e.episode_id else [],
        community_id=e.community_id,
    )


def relation_to_concept(r: Relation) -> Concept:
    """将旧 Relation 转换为统一 Concept

    注意：connects 使用 absolute_id（版本级ID），迁移时需通过
    存储层查询或缓存将 absolute_id 映射为 family_id，
    以确保 Concept.connects 指向逻辑身份而非特定版本。
    """
    return Concept(
        family_id=r.family_id,
        role=ROLE_RELATION,
        content=r.content,
        embedding=r.embedding,
        summary=r.summary,
        confidence=r.confidence,
        attributes=r.attributes,
        # TODO(Phase 0): 迁移时需将 absolute_id 解析为 family_id
        connects=[r.entity1_absolute_id, r.entity2_absolute_id],
        versions=[ConceptVersion(
            absolute_id=r.absolute_id,
            content=r.content,
            source_concept_id=r.episode_id or "",
            processed_time=r.processed_time,
            valid_at=r.valid_at,
            invalid_at=r.invalid_at,
        )],
        created_at=r.event_time,
        updated_at=r.processed_time,
        sources=[r.episode_id] if r.episode_id else [],
    )


def episode_to_concept(ep: Episode) -> Concept:
    """将旧 Episode 转换为统一 Concept（observation 角色）"""
    return Concept(
        family_id=ep.absolute_id,  # Episode 没有独立 family_id，用 absolute_id
        role=ROLE_OBSERVATION,
        content=ep.content,
        created_at=ep.event_time,
        updated_at=ep.event_time,
        attributes=None,
    )
