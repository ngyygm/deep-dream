#!/usr/bin/env python3
"""
迁移脚本： 将旧 plain 格式的 Entity/Relation content 转为结构化 Markdown。

用法:
    python scripts/migrate_plain_to_markdown.py [--config CONFIG_PATH] [--dry-run] [--batch-size N]
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processor.content_schema import (
    ENTITY_SECTIONS,
    RELATION_SECTIONS,
    render_markdown_sections,
    content_to_sections,
    section_hash,
)
from processor.models import ContentPatch
from server.config import load_config
from processor.storage import create_storage_manager
from processor.llm.client import LLMClient
from processor.utils import wprint
from processor.perf import _perf_timer
from processor.embedding import EmbeddingClient
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Migrate plain content to structured Markdown')
    parser.add_argument('--config', default='service_config.json', help='Config file path')
    parser.add_argument('--dry-run', action='store_true', help='Only show what would be migrated. Don\' write')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for LLM calls')
    return parser.parse_args()


def migrate_entities(storage, llm_client, dry_run=False, batch_size=10):
    """迁移所有 plain 格式的实体内容为 Markdown 格式。"""
    entities = storage.get_all_entities()
    plain_entities = [e for e in entities if getattr(e, 'content_format', 'plain') == 'plain']
    if not plain_entities:
        print("所有实体已经是 markdown 格式， 无需迁移。")
        return 0, 0

    print(f"发现 {len(plain_entities)}/{len(entities)} 个 plain 格式实体需要迁移.")
    total_migrated = 0
    total_patched = 0

    for i in range(0, len(plain_entities), batch_size):
        batch = plain_entities[i:i + batch_size]
        print(f"  处理批次 {i // batch_size + 1} ({len(batch)} 个实体)...")
        for entity in batch:
            if dry_run:
                print(f"    [DRY-RUN] {entity.name} ({entity.entity_id})")
                continue

            prompt = f"""将以下实体的纯文本描述重写为结构化的 Markdown 格式。
每个 section 必须用 ## 标题开头，section 列表: {ENTITY_SECTIONS}

实体名称: {entity.name}
当前描述:
{entity.content}

请直接输出重构后的 Markdown 内容，不要包含任何解释。"""

            try:
                response = llm_client._call_llm(prompt, "")
                response = response.strip()
                if response.startswith("```"):
                    lines = response.split("\n")
                    if lines[-1] == "```":
                        response = "\n".join(lines[:-1])
                        if response.startswith("```markdown\n"):
                            response = response[len("```markdown\n"):]
                        elif response.startswith("```\n"):
                            response = response[len("```\n"):]

                markdown_content = response
                event_time = entity.event_time if entity.event_time else datetime.now()
                source_doc = entity.source_document or ""
                now = datetime.now()

                from processor.models import Entity
                new_entity = Entity(
                    absolute_id=f"entity_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                    entity_id=entity.entity_id,
                    name=entity.name,
                    content=markdown_content,
                    event_time=event_time,
                    processed_time=now(),
                    episode_id=entity.episode_id,
                    source_document=source_doc,
                    content_format="markdown",
                )

                if not dry_run:
                    storage.save_entity(new_entity)
                    old_sections = content_to_sections(entity.content, 'plain', ENTITY_SECTIONS)
                    new_sections = content_to_sections(markdown_content, 'markdown', ENTITY_SECTIONS)
                    patches = []
                    for key in ENTITY_SECTIONS:
                        old_body = old_sections.get(key, "")
                        new_body = new_sections.get(key, "")
                        if old_body != new_body:
                            patches.append(ContentPatch(
                                uuid=str(uuid.uuid4()),
                                target_type="Entity",
                                target_absolute_id=new_entity.absolute_id,
                                target_entity_id=entity.entity_id,
                                section_key=key,
                                change_type="restructured",
                                old_hash=section_hash(old_body) if old_body else "",
                                new_hash=section_hash(new_body) if new_body else "",
                                diff_summary="Migrated from plain to markdown",
                                source_document=source_doc,
                                event_time=now,
                            ))
                    if patches:
                        storage.save_content_patches(patches)
                    total_migrated += 1
                    total_patched += len(patches)
                print(f"  已迁移: {entity.name}")
            except Exception as e:
                print(f"  迁移失败: {entity.name}: {e}")
                continue

    print(f"实体迁移完成: {total_migrated}/{len(plain_entities)} 迁移成功, {total_patched} patches 保存")
    return total_migrated, total_patched


def migrate_relations(storage, llm_client, dry_run=False, batch_size=10):
    """迁移所有 plain 格式的关系内容为 Markdown 格式。"""
    relations = storage.get_all_relations()
    plain_relations = [r for r in relations if getattr(r, 'content_format', 'plain') == 'plain']

    if not plain_relations:
        print("所有关系已经是 markdown 格式, 无需迁移。")
        return 0, 0

    print(f"发现 {len(plain_relations)}/{len(relations)} 个 plain 格式关系需要迁移.")
    total_migrated = 0
    total_patched = 0

    for i in range(0, len(plain_relations), batch_size):
        batch = plain_relations[i:i + batch_size]
        print(f"  处理批次 {i // batch_size + 1} ({len(batch)} 个关系)...")
        for relation in batch:
            if dry_run:
                print(f"    [DRY-RUN] {relation.relation_id}")
                continue

            prompt = f"""将以下关系的纯文本描述重写为结构化的 Markdown 格式。
每个 section 必须用 ## 标题开头, section 列表: {RELATION_SECTIONS}

关系 ID: {relation.relation_id}
当前描述:
{relation.content}

请直接输出重构后的 Markdown 内容，不要包含任何解释。"""

            try:
                response = llm_client._call_llm(prompt, "")
                response = response.strip()
                if response.startswith("```"):
                    lines = response.split("\n")
                    if lines[-1] == "```":
                        response = "\n".join(lines[:-1])
                        if response.startswith("```markdown\n"):
                            response = response[len("```markdown\n"):]
                        elif response.startswith("```\n"):
                            response = response[len("```\n"):]

                markdown_content = response
                event_time = relation.event_time if relation.event_time else datetime.now()
                source_doc = relation.source_document or ""
                now = datetime.now()

                from processor.models import Relation
                new_relation = Relation(
                    absolute_id=f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                    relation_id=relation.relation_id,
                    entity1_absolute_id=relation.entity1_absolute_id,
                    entity2_absolute_id=relation.entity2_absolute_id,
                    content=markdown_content,
                    event_time=event_time,
                    processed_time=now(),
                    episode_id=relation.episode_id,
                    source_document=source_doc,
                    content_format="markdown",
                )

                if not dry_run:
                    storage.save_relation(new_relation)
                    old_sections = content_to_sections(relation.content, 'plain', RELATION_SECTIONS)
                    new_sections = content_to_sections(markdown_content, 'markdown', RELATION_SECTIONS)
                    patches = []
                    for key in RELATION_SECTIONS:
                        old_body = old_sections.get(key, "")
                        new_body = new_sections.get(key, "")
                        if old_body != new_body:
                            patches.append(ContentPatch(
                                uuid=str(uuid.uuid4()),
                                target_type="Relation",
                                target_absolute_id=new_relation.absolute_id,
                                target_entity_id=relation.relation_id,
                                section_key=key,
                                change_type="restructured",
                                old_hash=section_hash(old_body) if old_body else "",
                                new_hash=section_hash(new_body) if new_body else "",
                                diff_summary="Migrated from plain to markdown",
                                source_document=source_doc,
                                event_time=now,
                            ))
                    if patches:
                        storage.save_content_patches(patches)
                    total_migrated += 1
                    total_patched += len(patches)
                print(f"  已迁移: {relation.relation_id}")
            except Exception as e:
                print(f"  迁移失败: {relation.relation_id}: {e}")
                continue

    print(f"关系迁移完成: {total_migrated}/{len(plain_relations)} 迁移成功, {total_patched} patches 保存")
    return total_migrated, total_patched


def main():
    args = parse_args()
    config = load_config(args.config)
    print(f"配置已加载: {args.config}")

    storage_config = config.get("storage") or {}
    backend = storage_config.get("backend", "sqlite")
    embedding_client = None
    if backend == "neo4j":
        neo4j_config = storage_config.get("neo4j") or {}
        embedding_client = EmbeddingClient(
            model_name=config.get("embedding", {}).get("model", "text-embedding-3-small"),
        )
        storage = create_storage_manager(config, embedding_client=embedding_client)
    else:
        storage = create_storage_manager(config)

    llm_client = LLMClient(config)
    print(f"存储后端: {type(storage).__name__}")

    print("=" * 60)
    print("开始迁移实体...")
    e_migrated, e_patched = migrate_entities(storage, llm_client, dry_run=args.dry_run, batch_size=args.batch_size)
    print(f"实体迁移完成: {e_migrated} 迁移, {e_patched} patches")

    print("=" * 60)
    print("开始迁移关系...")
    r_migrated, r_patched = migrate_relations(storage, llm_client, dry_run=args.dry_run, batch_size=args.batch_size)
    print(f"关系迁移完成: {r_migrated} 迁移, {r_patched} patches")
    print("=" * 60)
    print("全部迁移完成!")

    # Neo4j 迁移
    if backend == "neo4j":
        try:
            with storage._driver.session() as session:
                session.run(
                    "MATCH (e:Entity) WHERE e.content_format IS NULL SET e.content_format = 'plain'"
                )
                session.run(
                    "MATCH (r:Relation) WHERE r.content_format IS NULL SET r.content_format = 'plain'"
                )
            print("Neo4j content_format 已设置")
        except Exception as e:
            print(f"Neo4j 迁移警告: {e}")

    storage.close()
    print("存储已关闭.")


if __name__ == "__main__":
    main()
