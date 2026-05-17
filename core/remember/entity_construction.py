"""
Entity construction helpers: factory functions for building/creating entities.
Extracted from EntityProcessor for modularity.
"""
from typing import Optional
from datetime import datetime, timezone
import uuid
import logging

from core.models import Entity
from core.storage.sqlite.manager import SQLiteGraphStorageManager as Neo4jStorageManager
from core.content_schema import (
    ENTITY_SECTIONS,
    compute_content_patches,
)
from core.remember._shared import _doc_basename

logger = logging.getLogger(__name__)


def _extract_summary(name: str, content: str) -> str:
    """从实体名称和内容中提取简短摘要（无需额外LLM调用）。"""
    # 跳过 markdown 标题行，取第一行非空正文
    if not content:
        return name[:100]
    for line in content.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        return stripped[:200] if len(stripped) > 200 else stripped
    # 回退到名称
    return name[:100]


def _construct_entity(name: str, content: str, episode_id: str,
                      family_id: str, source_document: str = "",
                      base_time: Optional[datetime] = None,
                      confidence: Optional[float] = None) -> Entity:
    """Shared helper: construct an Entity object with standard fields.

    Args:
        confidence: Initial confidence from LLM extraction (0.0-1.0).
                    Falls back to 0.7 if not provided.
    """
    # Guard: never create entities with empty names
    name = (name or "").strip()
    if not name:
        logger.warning("_construct_entity called with empty name — using fallback")
        name = "未命名概念"
    _now = datetime.now(timezone.utc)
    event_time = base_time if base_time is not None else _now
    processed_time = _now
    entity_record_id = f"entity_{processed_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    source_document_only = _doc_basename(source_document)
    # Use LLM-provided confidence if available, otherwise default
    initial_confidence = confidence if confidence is not None else 0.7
    initial_confidence = max(0.0, min(1.0, initial_confidence))
    return Entity(
        absolute_id=entity_record_id,
        family_id=family_id,
        name=name,
        content=content,
        event_time=event_time,
        processed_time=processed_time,
        episode_id=episode_id,
        source_document=source_document_only,
        content_format="markdown",
        summary=_extract_summary(name, content),
        confidence=initial_confidence,
    )


def _build_new_entity(name: str, content: str, episode_id: str,
                      source_document: str = "", base_time: Optional[datetime] = None,
                      confidence: Optional[float] = None) -> Entity:
    """构建新实体对象，但不立即写库。"""
    return _construct_entity(
        name, content, episode_id,
        family_id=f"ent_{uuid.uuid4().hex[:12]}",
        source_document=source_document, base_time=base_time,
        confidence=confidence,
    )


def _create_new_entity(storage: Neo4jStorageManager,
                       name: str, content: str, episode_id: str,
                       source_document: str = "", base_time: Optional[datetime] = None,
                       confidence: Optional[float] = None) -> Entity:
    """创建新实体"""
    entity = _build_new_entity(name, content, episode_id, source_document, base_time=base_time,
                               confidence=confidence)
    storage.save_entity(entity)
    return entity


def _compute_entity_patches(
    family_id: str,
    old_content: str,
    old_content_format: str,
    new_content: str,
    new_absolute_id: str,
    source_document: str = "",
    event_time: Optional[datetime] = None,
) -> list:
    return compute_content_patches(
        family_id=family_id,
        old_content=old_content,
        old_content_format=old_content_format,
        new_content=new_content,
        new_absolute_id=new_absolute_id,
        target_type="Entity",
        schema=ENTITY_SECTIONS,
        source_document=source_document,
        event_time=event_time,
    )


def _build_entity_version(family_id: str, name: str, content: str,
                          episode_id: str, source_document: str = "",
                          base_time: Optional[datetime] = None,
                          old_content: str = "",
                          old_content_format: str = "plain") -> Entity:
    """构建实体新版本对象，但不立即写库。附带 section patch 计算。"""
    entity = _construct_entity(
        name, content, episode_id,
        family_id=family_id,
        source_document=source_document, base_time=base_time,
    )
    if old_content:
        patches = _compute_entity_patches(
            family_id=family_id,
            old_content=old_content,
            old_content_format=old_content_format,
            new_content=content,
            new_absolute_id=entity.absolute_id,
            source_document=_doc_basename(source_document),
            event_time=entity.event_time,
        )
        if patches:
            entity._pending_patches = patches
    return entity


def _create_entity_version(storage: Neo4jStorageManager,
                           family_id: str, name: str, content: str,
                           episode_id: str, source_document: str = "",
                           base_time: Optional[datetime] = None,
                           old_content: str = "",
                           old_content_format: str = "plain") -> Entity:
    """创建实体的新版本，并记录 section 级 patches。"""
    # 始终创建新版本（每个 episode 提及的概念都版本化）

    entity = _build_entity_version(family_id, name, content, episode_id, source_document, base_time=base_time)
    storage.save_entity(entity)

    # 注意：置信度 corroboration 在 extraction.py Phase C-1b 统一处理，不在此处重复调用

    # 计算 section patches
    _source_document_only = _doc_basename(source_document)
    if old_content:
        patches = _compute_entity_patches(
            family_id=family_id,
            old_content=old_content,
            old_content_format=old_content_format,
            new_content=content,
            new_absolute_id=entity.absolute_id,
            source_document=_source_document_only,
            event_time=entity.event_time,
        )
        if patches:
            storage.save_content_patches(patches)

    return entity
