"""
核心数据结构定义
"""
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class MemoryCache:
    """记忆缓存 - 文档化设计"""
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
    entity_id: str  # 实体的逻辑ID，同一实体的不同版本具有相同的entity_id
    name: str  # 实体名称
    content: str  # 实体的自然语言描述
    event_time: datetime  # 事件发生时间
    processed_time: datetime  # 系统实际处理时间
    memory_cache_id: str  # 记录当前更新是基于什么记忆环境下的判断
    source_document: str  # 来源文档名称
    embedding: Optional[bytes] = None  # Embedding向量（BLOB格式，可选）
    valid_at: Optional[datetime] = None  # 事实生效时间
    invalid_at: Optional[datetime] = None  # 事实失效时间（被新版本替代）
    summary: Optional[str] = None  # 实体摘要（由 LLM 进化维护）
    attributes: Optional[str] = None  # JSON 字符串，结构化属性字典
    confidence: Optional[float] = None  # 置信度评分 (0.0-1.0)


@dataclass
class Relation:
    """关系 - 带版本链的概念边（无向关系）

    关系是无向的，不区分方向，只表示两个实体之间的关联。
    entity1_absolute_id 和 entity2_absolute_id 只是用来标识关系涉及的两个实体，没有方向性。
    存储时，实体对按字母顺序排序（entity1 < entity2），确保 (A,B) 和 (B,A) 被视为同一个关系。
    """
    absolute_id: str  # 主键，版本唯一标识符（DB 列名 id）
    relation_id: str  # 关系的逻辑ID，同一关系的不同版本具有相同的relation_id
    entity1_absolute_id: str  # 第一个实体的绝对ID（版本唯一ID，可以通过此ID找到entity_id），按字母顺序排序
    entity2_absolute_id: str  # 第二个实体的绝对ID（版本唯一ID，可以通过此ID找到entity_id），按字母顺序排序
    content: str  # 关系的自然语言描述
    event_time: datetime  # 事件发生时间
    processed_time: datetime  # 系统实际处理时间
    memory_cache_id: str  # 记录当前更新是基于什么记忆环境下的判断
    source_document: str  # 来源文档名称
    embedding: Optional[bytes] = None  # Embedding向量（BLOB格式，可选）
    valid_at: Optional[datetime] = None  # 事实生效时间
    invalid_at: Optional[datetime] = None  # 事实失效时间（被新版本替代）
    summary: Optional[str] = None  # 关系摘要（由 LLM 进化维护）
    attributes: Optional[str] = None  # JSON 字符串，结构化属性字典
    confidence: Optional[float] = None  # 置信度评分 (0.0-1.0)
    provenance: Optional[str] = None  # JSON: [{"episode_id": "...", "confidence": 0.9}, ...]
