"""
Section 级合并协调器。

在 Entity / Relation 合并时，将 content 解析为 Markdown sections，
只对变更 section 调用 LLM 合并，未变更 section 直接引用。

对外提供两个高层函数：
- merge_entity_section_level(old_entity, new_content, llm, storage, source_document, ...)
- merge_relation_section_level(old_relation, new_content, llm, storage, source_document, ...)
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..models import ContentPatch, Entity, Relation
from ..content_schema import (
    ENTITY_SECTIONS,
    RELATION_SECTIONS,
    collect_changed_sections,
    compute_section_diff,
    content_to_sections,
    has_any_change,
    render_markdown_sections,
    section_hash,
)


def merge_entity_section_level(
    old_entity: Entity,
    new_content: str,
    llm_client,
    storage,
    source_document: str = "",
    old_source_document: str = "",
    entity_name: str = "",
) -> Tuple[str, List[ContentPatch], bool]:
    """对实体执行 section 级合并。

    Returns:
        (merged_content, patches, format_upgraded)
        - merged_content: 合并后的完整 markdown 内容
        - patches: section 变更记录列表
        - format_upgraded: 是否从 plain 升级为 markdown
    """
    old_format = getattr(old_entity, 'content_format', 'plain')
    new_format = 'markdown'

    old_sections = content_to_sections(old_entity.content, old_format, ENTITY_SECTIONS)
    new_sections = content_to_sections(new_content, 'plain', ENTITY_SECTIONS)

    diff = compute_section_diff(old_sections, new_sections)

    if not has_any_change(diff):
        # 无变更，直接返回旧内容
        return old_entity.content, [], False

    # 只对变更的 section 调用 LLM 合并
    merged_sections = dict(old_sections)  # start from old
    patches: List[ContentPatch] = []
    changed = collect_changed_sections(diff)

    for key, old_body, new_body in changed:
        if key not in old_sections:
            # 新增 section，直接使用新内容
            merged_sections[key] = new_body
        elif key not in new_sections:
            # section 被删除，保留旧版本（不删除）
            merged_sections[key] = old_body
        else:
            # section 修改 → 调用 LLM section 级合并
            merged_body = llm_client.merge_entity_section(
                section_key=key,
                old_section=old_body,
                new_section=new_body,
                old_source_document=old_source_document,
                new_source_document=source_document,
                entity_name=entity_name or old_entity.name,
            )
            merged_sections[key] = merged_body

        # 生成 patch
        info = diff[key]
        patches.append(ContentPatch(
            uuid=str(uuid.uuid4()),
            target_type="Entity",
            target_absolute_id=old_entity.absolute_id,
            target_entity_id=old_entity.entity_id,
            section_key=key,
            change_type=info.get("change_type", "modified"),
            old_hash=section_hash(old_body) if old_body else "",
            new_hash=section_hash(merged_sections.get(key, "")),
            diff_summary=f"Section '{key}' {info.get('change_type', 'modified')}",
            source_document=source_document,
            event_time=datetime.now(),
        ))

    merged_content = render_markdown_sections(merged_sections, ENTITY_SECTIONS)
    format_upgraded = old_format == 'plain'
    return merged_content, patches, format_upgraded


def merge_relation_section_level(
    old_relation: Relation,
    new_content: str,
    llm_client,
    storage,
    source_document: str = "",
    old_source_document: str = "",
    entity1_name: str = "",
    entity2_name: str = "",
) -> Tuple[str, List[ContentPatch], bool]:
    """对关系执行 section 级合并。

    Returns:
        (merged_content, patches, format_upgraded)
    """
    old_format = getattr(old_relation, 'content_format', 'plain')
    new_format = 'markdown'

    old_sections = content_to_sections(old_relation.content, old_format, RELATION_SECTIONS)
    new_sections = content_to_sections(new_content, 'plain', RELATION_SECTIONS)

    diff = compute_section_diff(old_sections, new_sections)

    if not has_any_change(diff):
        return old_relation.content, [], False

    merged_sections = dict(old_sections)
    patches: List[ContentPatch] = []
    changed = collect_changed_sections(diff)

    for key, old_body, new_body in changed:
        if key not in old_sections:
            merged_sections[key] = new_body
        elif key not in new_sections:
            merged_sections[key] = old_body
        else:
            merged_body = llm_client.merge_relation_section(
                section_key=key,
                old_section=old_body,
                new_section=new_body,
                old_source_document=old_source_document,
                new_source_document=source_document,
                entity1_name=entity1_name,
                entity2_name=entity2_name,
            )
            merged_sections[key] = merged_body

        info = diff[key]
        patches.append(ContentPatch(
            uuid=str(uuid.uuid4()),
            target_type="Relation",
            target_absolute_id=old_relation.absolute_id,
            target_entity_id=old_relation.relation_id,
            section_key=key,
            change_type=info.get("change_type", "modified"),
            old_hash=section_hash(old_body) if old_body else "",
            new_hash=section_hash(merged_sections.get(key, "")),
            diff_summary=f"Section '{key}' {info.get('change_type', 'modified')}",
            source_document=source_document,
            event_time=datetime.now(),
        ))

    merged_content = render_markdown_sections(merged_sections, RELATION_SECTIONS)
    format_upgraded = old_format == 'plain'
    return merged_content, patches, format_upgraded
