"""
核心数据结构定义

旧模型（Entity, Relation, Episode）保留用于向后兼容。
新模型（Concept）是统一的概念原语；Entity/Relation/Episode DTO 仅作为流水线适配对象。
"""
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass(slots=True)
class Episode:
    """Episode — 知识图谱的一等节点

    每次写入产生一个 Episode，包含当时的记忆上下文和原始文本。
    抽取出的实体/关系通过 MENTIONS 边连接回 Episode，实现事实溯源。
    """
    absolute_id: str
    content: str  # Markdown格式的完整描述
    event_time: datetime  # 事件发生时间
    source_document: str  # 来源文档名称
    processed_time: Optional[datetime] = None  # 系统处理时间
    activity_type: Optional[str] = None  # 可选的活动类型，如"阅读小说"、"处理文档"等
    episode_type: Optional[str] = None  # Episode 类型: "narrative" | "fact" | "conversation"
    heading_path: Optional[str] = None  # full heading breadcrumb (e.g. "Chapter 1 > Section 2")


@dataclass(slots=True)
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
    entity1_family_id: str = ""  # 第一个实体的家族ID（冗余存储，便于查询）
    entity2_family_id: str = ""  # 第二个实体的家族ID（冗余存储，便于查询）
    version_seq: int = 1  # 版本序号，每次跨 Episode 提及递增
    embedding: Optional[bytes] = None  # Embedding向量（BLOB格式，可选）
    valid_at: Optional[datetime] = None  # 事实生效时间
    attributes: Optional[str] = None  # JSON 字符串，结构化属性字典
    confidence: Optional[float] = None  # 置信度评分 (0.0-1.0)
    content_format: str = "plain"  # "plain" (旧) | "markdown" (新)
    community_id: Optional[str] = None  # 社区检测分配的社区ID
    _pending_patches: list = None  # 内部用：ContentPatch 缓冲，flush 后清空
    _score: float = 0.0  # search relevance score (BM25/embedding)


@dataclass(slots=True)
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
    entity1_family_id: str = ""  # 第一个实体的家族ID（冗余存储，便于查询）
    entity2_family_id: str = ""  # 第二个实体的家族ID（冗余存储，便于查询）
    version_seq: int = 1  # 版本序号，每次跨 Episode 提及递增
    embedding: Optional[bytes] = None  # Embedding向量（BLOB格式，可选）
    valid_at: Optional[datetime] = None  # 事实生效时间
    attributes: Optional[str] = None  # JSON 字符串，结构化属性字典
    confidence: Optional[float] = None  # 置信度评分 (0.0-1.0)
    provenance: Optional[str] = None  # JSON: [{"episode_id": "...", "confidence": 0.9}, ...]
    content_format: str = "plain"  # "plain" (旧) | "markdown" (新)
    evidence_text: Optional[str] = None  # the source text evidence for this relation
    evidence_start_offset: Optional[int] = None
    evidence_end_offset: Optional[int] = None
    evidence_line_start: Optional[int] = None
    evidence_line_end: Optional[int] = None
    _pending_patches: list = None  # 内部用：ContentPatch 缓冲，flush 后清空
    _score: float = 0.0  # search relevance score (BM25/embedding)


@dataclass(slots=True)
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

# 概念角色常量。四个真实角色：entity / relation / episode / document
# （observation 已作为未实现项移除，不要重新加入）。
ROLE_ENTITY = "entity"
ROLE_RELATION = "relation"
ROLE_EPISODE = "episode"
ROLE_DOCUMENT = "document"
ALL_ROLES = (ROLE_ENTITY, ROLE_RELATION, ROLE_EPISODE, ROLE_DOCUMENT)


@dataclass(slots=True)
class ConceptVersion:
    """概念的单个版本快照"""
    absolute_id: str  # 版本唯一标识
    content: str  # 该版本的内容快照
    source_concept_id: str  # 产生该版本的源 observation 概念 ID
    processed_time: datetime  # 该版本的产生时间
    version_seq: int = 1  # 版本序号
    valid_at: Optional[datetime] = None  # 有效期起始


@dataclass
class Concept:
    """Unified Concept primitive — entity/relation/episode/document are one Concept with different role (ACL thesis, Option B).

    统一概念原语：物理上仍保留 entity_*/relation_* 分表（不迁移数据），
    但代码面统一为单个 role 参数化的 Concept，配合 v_latest_concept UNION
    视图，使 thesis 端到端诚实。

    DAL（概念数据访问层）对所有角色返回本对象；role 字段决定如何解读
    name/content 与（仅 relation 角色有意义的）端点字段。
    """
    family_id: str  # 逻辑身份，跨版本不变
    role: str  # 角色：entity | relation | episode | document
    name: str = ''  # 显示名称（entity/episode/document 用；relation 通常为空）
    content: str = ''  # 当前版本的 NL 内容
    version_id: str = ''  # 当前版本绝对 ID（DB 行主键）
    status: str = 'active'  # 状态：active | superseded | deleted ...
    confidence: float = 0.0  # 置信度 (0.0-1.0)
    episode_id: str = ''  # 该版本产生时的记忆环境 episode

    # 关系型概念专用（None for non-relation roles）
    subject_family_id: Optional[str] = None
    object_family_id: Optional[str] = None

    # 透传字段：未在强类型字段中建模的列（attributes、source_text、document_version_id 等）
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转成纯 dict（JSON 友好，不做任何转义）。

        extra 中的键会被提升到顶层（DAL/视图行直接消费的扁平形状），
        顶层强类型字段与 extra 同名时以强类型字段为准。
        """
        out = dict(self.extra) if self.extra else {}
        out.update({
            'family_id': self.family_id,
            'role': self.role,
            'name': self.name,
            'content': self.content,
            'version_id': self.version_id,
            'status': self.status,
            'confidence': self.confidence,
            'episode_id': self.episode_id,
            'subject_family_id': self.subject_family_id,
            'object_family_id': self.object_family_id,
        })
        return out

    @classmethod
    def from_row(cls, role: str, row: dict) -> "Concept":
        """从一个 repo dict 行构造 Concept。

        兼容 v_latest_concept 视图行（扁平统一形状）以及各角色的原始表行：

        - entity: name <- row['name'] 或 row['canonical_name']；
                  family_id <- row['family_id'] 或 row['entity_family_id']
        - relation: 端点 <- row['subject_family_id']/['object_family_id']
                    或 row['subject_entity_family_id']/['object_entity_family_id']
        - episode: content <- row['content'] 或 row['memory_text']
        - document: name <- row['name'] 或 row['title']

        已知的强类型列消费后会从 extra 副本中剔除，避免重复。
        未知列原样保留在 extra 里供调用方使用。
        """
        if role not in ALL_ROLES:
            raise ValueError(f"unknown Concept role: {role!r} (expected one of {ALL_ROLES})")

        row = dict(row or {})  # 防御性拷贝，避免污染调用方的 dict

        def pick(*keys, default=''):
            for k in keys:
                if k in row and row[k] is not None:
                    return row[k]
            return default

        family_id = pick('family_id', 'entity_family_id',
                         'relation_family_id', 'episode_family_id', 'document_id')
        version_id = pick('version_id', 'id', 'entity_id', 'relation_id',
                          'episode_id', 'document_version_id')

        # 角色：允许行里自带 role，但显式参数优先（DAL 按 role 分支查询）
        row_role = pick('role', default=role) or role

        name = ''
        content = pick('content', 'memory_text')
        subject_family_id: Optional[str] = None
        object_family_id: Optional[str] = None

        if role == ROLE_ENTITY:
            name = pick('name', 'canonical_name')
        elif role == ROLE_RELATION:
            subject_family_id = pick('subject_family_id', 'subject_entity_family_id',
                                     'entity1_family_id', default=None) or None
            object_family_id = pick('object_family_id', 'object_entity_family_id',
                                    'entity2_family_id', default=None) or None
        elif role == ROLE_EPISODE:
            name = pick('name', default='')
        elif role == ROLE_DOCUMENT:
            name = pick('name', 'title')

        # 置信度：行里可能给 None / 缺失，统一回退到 0.0
        try:
            confidence = float(pick('confidence', default=0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        episode_id = pick('episode_id', 'episode_version_id')

        status = pick('status', default='active') or 'active'

        # extra：剔除已消费的已知列，保留其余（attributes/source_text/...）
        consumed = {
            'family_id', 'entity_family_id', 'relation_family_id',
            'episode_family_id', 'document_id',
            'version_id', 'id', 'entity_id', 'relation_id',
            'episode_id', 'episode_version_id', 'document_version_id',
            'role', 'name', 'canonical_name', 'content', 'memory_text',
            'confidence', 'status',
            'subject_family_id', 'subject_entity_family_id', 'entity1_family_id',
            'object_family_id', 'object_entity_family_id', 'entity2_family_id',
        }
        extra = {k: v for k, v in row.items() if k not in consumed}

        return cls(
            family_id=family_id,
            role=row_role,
            name=name,
            content=content,
            version_id=version_id,
            status=status,
            confidence=confidence,
            episode_id=episode_id,
            subject_family_id=subject_family_id,
            object_family_id=object_family_id,
            extra=extra,
        )

