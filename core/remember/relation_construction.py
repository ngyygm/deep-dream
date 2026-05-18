"""Relation construction mixin: deduplication, merging, single-relation processing, and factory methods.

Split from relation.py — contains relation dedupe/merge, single-relation alignment,
and all relation construction/versioning factory methods.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

from core.models import Relation
from core.debug_log import log as dbg, log_section as dbg_section, _ENABLED as _dbg_enabled
from core.content_schema import RELATION_SECTIONS, compute_content_patches
from core.utils import wprint_info, normalize_entity_pair

from .helpers import MIN_RELATION_CONTENT_LENGTH


class _RelationConstructionMixin:
    """Relation dedupe, merge, construction, and versioning methods."""

    def _dedupe_and_merge_relations(self, extracted_relations: List[Dict[str, str]],
                                    entity_name_to_id: Dict[str, str]) -> List[Dict[str, str]]:
        """对相同实体对的关系进行去重和合并"""
        from .relation import _get_entity_names
        relations_by_pair = {}
        filtered_count = 0
        filtered_relations = []
        dbg_section("RelationProcessor._dedupe_and_merge_relations")
        dbg(f"输入关系数: {len(extracted_relations)}")
        if _dbg_enabled:
            dbg(f"entity_name_to_id 映射 ({len(entity_name_to_id)} 个): {list(entity_name_to_id)[:20]}")

        for relation in extracted_relations:
            entity1_name, entity2_name = _get_entity_names(relation)

            if not entity1_name or not entity2_name:
                filtered_count += 1
                filtered_relations.append({
                    'entity1': entity1_name or '(空)',
                    'entity2': entity2_name or '(空)',
                    'reason': '实体名称为空'
                })
                dbg(f"  过滤(空名): e1='{entity1_name}' e2='{entity2_name}'")
                continue

            missing_entities = []
            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)
            if not entity1_id:
                missing_entities.append(f'entity1: {entity1_name}')
            if not entity2_id:
                missing_entities.append(f'entity2: {entity2_name}')

            if missing_entities:
                filtered_count += 1
                filtered_relations.append({
                    'entity1': entity1_name,
                    'entity2': entity2_name,
                    'reason': f'实体不在当前窗口的实体列表中: {", ".join(missing_entities)}'
                })
                dbg(f"  过滤(不在映射): e1='{entity1_name}' e2='{entity2_name}' 缺少: {missing_entities}")
                continue

            if entity1_id and entity2_id and entity1_id == entity2_id:
                filtered_count += 1
                filtered_relations.append({
                    'entity1': entity1_name,
                    'entity2': entity2_name,
                    'reason': f'两个实体是同一个实体（family_id: {entity1_id}）'
                })
                dbg(f"  过滤(自关系): e1='{entity1_name}' e2='{entity2_name}' family_id={entity1_id}")
                continue

            normalized_pair = normalize_entity_pair(entity1_name, entity2_name)

            if normalized_pair not in relations_by_pair:
                relations_by_pair[normalized_pair] = []
            _needs_copy = (entity1_name != normalized_pair[0] or entity2_name != normalized_pair[1])
            if _needs_copy:
                relation_copy = relation.copy()
                relation_copy['entity1_name'] = normalized_pair[0]
                relation_copy['entity2_name'] = normalized_pair[1]
            else:
                relation_copy = relation
            relations_by_pair[normalized_pair].append(relation_copy)

        merged_relations = []
        for pair, relations in relations_by_pair.items():
            if self.preserve_distinct_relations_per_pair:
                seen_contents = set()
                for relation in relations:
                    content_key = (relation.get('content') or '').strip().lower()
                    if not content_key or content_key in seen_contents:
                        continue
                    seen_contents.add(content_key)
                    merged_relations.append(relation)
                continue
            if len(relations) == 1:
                merged_relations.append(relations[0])
            else:
                merged_relation = self._merge_relations_for_pair(pair, relations)
                if merged_relation:
                    merged_relations.append(merged_relation)

        dbg(f"去重合并结果: 过滤 {filtered_count}, 合并后通过 {len(merged_relations)}")
        for _mr in merged_relations:
            dbg(f"  通过: '{_mr.get('entity1_name', '')}' <-> '{_mr.get('entity2_name', '')}'  content='{_mr.get('content', '')[:100]}'")

        return merged_relations

    def _merge_relations_for_pair(self, pair: tuple,
                                  relations: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """合并同一实体对的多个关系"""
        if not relations:
            return None

        if len(relations) == 1:
            return relations[0]

        relation_contents = [c for rel in relations if (c := rel.get('content', ''))]

        if not relation_contents:
            return relations[0]

        if len(relation_contents) == 1:
            return relations[0]

        merged_content = "；".join(relation_contents)

        merged_relation = {
            'entity1_name': pair[0],
            'entity2_name': pair[1],
            'content': merged_content
        }

        return merged_relation

    def _process_single_relation(self, extracted_relation: Dict[str, str],
                                 entity1_id: str,
                                 entity2_id: str,
                                 episode_id: str,
                                 entity1_name: str = "",
                                 entity2_name: str = "",
                                 verbose_relation: bool = True,
                                 source_document: str = "",
                                 base_time: Optional[datetime] = None,
                                 pre_fetched_relations: Optional[List[Relation]] = None,
                                 _pre_built_relations_info: Optional[List[Dict]] = None) -> Optional[Relation]:
        """处理单个关系

        注意：参数 entity1_id 和 entity2_id 是实体的 family_id（不是绝对ID）
        在创建关系时，会通过 family_id 获取实体的最新版本，然后使用绝对ID存储
        """
        from ._shared import _doc_basename
        from .relation import _get_entity_names

        relation_content = extracted_relation['content']
        if not entity1_name or not entity2_name:
            _e1, _e2 = _get_entity_names(extracted_relation)
            entity1_name = entity1_name or _e1
            entity2_name = entity2_name or _e2
        if pre_fetched_relations is not None:
            existing_relations = pre_fetched_relations
        else:
            existing_relations = self.storage.get_relations_by_entities(
                entity1_id,
                entity2_id
            )

        if not existing_relations:
            return self._create_new_relation(
                entity1_id,
                entity2_id,
                relation_content,
                episode_id,
                entity1_name,
                entity2_name,
                verbose_relation,
                source_document,
                base_time=base_time,
            )

        existing_relations_info = _pre_built_relations_info or [
            {
                'family_id': r.family_id,
                'content': r.content,
                'source_document': r.source_document,
            }
            for r in existing_relations
        ]

        match_result = self.llm_client.judge_relation_match(
            extracted_relation,
            existing_relations_info,
            new_source_document=_doc_basename(source_document),
        )
        if isinstance(match_result, list) and match_result:
            match_result = match_result[0] if isinstance(match_result[0], dict) else None
        elif not isinstance(match_result, dict):
            match_result = None

        if match_result and match_result.get('family_id'):
            family_id = match_result['family_id']

            latest_relation = next(
                (r for r in existing_relations if r.family_id == family_id), None
            )
            if not latest_relation:
                return self._create_new_relation(
                    entity1_id,
                    entity2_id,
                    relation_content,
                    episode_id,
                    entity1_name,
                    entity2_name,
                    verbose_relation,
                    source_document,
                    base_time=base_time,
                )

            _old_content = (latest_relation.content or "").strip()
            _new_content = relation_content.strip()
            if _old_content == _new_content:
                new_relation = self._create_relation_version(
                    family_id,
                    entity1_id,
                    entity2_id,
                    latest_relation.content,
                    episode_id,
                    verbose_relation,
                    source_document,
                    entity1_name,
                    entity2_name,
                    base_time=base_time,
                )
                return new_relation
            else:
                record_count = 0
                if verbose_relation:
                    try:
                        vc_map = self.storage.get_relation_version_counts([family_id])
                        record_count = vc_map.get(family_id, 0)
                    except Exception:
                        pass

                merged_content = self.llm_client.merge_relation_content(
                    latest_relation.content,
                    relation_content,
                    old_source_document=latest_relation.source_document,
                    new_source_document=source_document,
                    entity1_name=entity1_name,
                    entity2_name=entity2_name,
                )

                if verbose_relation:
                    wprint_info(f"[关系操作] 🔄 更新关系: {entity1_name} <-> {entity2_name} (family_id: {family_id}, 版本数: {record_count})")

                new_relation = self._create_relation_version(
                    family_id,
                    entity1_id,
                    entity2_id,
                    merged_content,
                    episode_id,
                    verbose_relation,
                    source_document,
                    entity1_name,
                    entity2_name,
                    base_time=base_time,
                )

                return new_relation
        else:
            return self._create_new_relation(
                entity1_id,
                entity2_id,
                relation_content,
                episode_id,
                entity1_name,
                entity2_name,
                verbose_relation,
                source_document,
                base_time=base_time,
            )

    def _construct_relation(self, entity1_id: str, entity2_id: str,
                            content: str, episode_id: str,
                            family_id: str,
                            entity1_name: str = "", entity2_name: str = "",
                            verbose_relation: bool = True, source_document: str = "",
                            base_time: Optional[datetime] = None,
                            entity_lookup: Optional[Dict[str, Any]] = None,
                            skip_label: str = "关系创建",
                            confidence: Optional[float] = None) -> Optional[Relation]:
        """Shared helper: resolve entities, validate, and construct a Relation object."""
        from ._shared import _doc_basename

        entity1 = (entity_lookup or {}).get(entity1_id) or self.storage.get_entity_by_family_id(entity1_id)
        entity2 = (entity_lookup or {}).get(entity2_id) or self.storage.get_entity_by_family_id(entity2_id)

        if not entity1 or not entity2:
            missing_info = []
            if not entity1:
                missing_info.append(f"entity1: {entity1_name or '(未提供名称)'} (family_id: {entity1_id})")
            if not entity2:
                missing_info.append(f"entity2: {entity2_name or '(未提供名称)'} (family_id: {entity2_id})")
            if verbose_relation:
                wprint_info(f"[关系操作] ⚠️  警告: 无法找到实体: {', '.join(missing_info)}，跳过{skip_label}")
            return None

        _now = datetime.now()
        ts = base_time if base_time is not None else _now
        processed_time = _now
        relation_record_id = f"relation_{processed_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        if entity1.name <= entity2.name:
            entity1_absolute_id, entity2_absolute_id = entity1.absolute_id, entity2.absolute_id
        else:
            entity1_absolute_id, entity2_absolute_id = entity2.absolute_id, entity1.absolute_id

        source_document_only = _doc_basename(source_document)
        initial_confidence = confidence if confidence is not None else 0.7
        initial_confidence = max(0.0, min(1.0, initial_confidence))
        return Relation(
            absolute_id=relation_record_id,
            family_id=family_id,
            entity1_absolute_id=entity1_absolute_id,
            entity2_absolute_id=entity2_absolute_id,
            content=content,
            event_time=ts,
            processed_time=processed_time,
            episode_id=episode_id,
            source_document=source_document_only,
            content_format="markdown",
            summary=content[:200].strip(),
            confidence=initial_confidence,
        )

    def _build_new_relation(self, entity1_id: str, entity2_id: str,
                            content: str, episode_id: str,
                            entity1_name: str = "", entity2_name: str = "",
                            verbose_relation: bool = True, source_document: str = "",
                            base_time: Optional[datetime] = None,
                            entity_lookup: Optional[Dict[str, Any]] = None,
                            confidence: Optional[float] = None) -> Optional[Relation]:
        """构建新关系对象，但不立即写库。"""
        _cs = content.strip() if content else ""
        if len(_cs) < MIN_RELATION_CONTENT_LENGTH:
            if verbose_relation:
                wprint_info(f"[关系操作] ⚠️  跳过: 关系内容过短 ({len(_cs)}字符): {entity1_name} <-> {entity2_name}")
            return None

        return self._construct_relation(
            entity1_id, entity2_id, content, episode_id,
            family_id=f"rel_{uuid.uuid4().hex[:12]}",
            entity1_name=entity1_name, entity2_name=entity2_name,
            verbose_relation=verbose_relation, source_document=source_document,
            base_time=base_time, entity_lookup=entity_lookup,
            skip_label="关系创建",
            confidence=confidence,
        )

    def _create_new_relation(self, entity1_id: str, entity2_id: str,
                            content: str, episode_id: str,
                            entity1_name: str = "", entity2_name: str = "",
                            verbose_relation: bool = True, source_document: str = "",
                            base_time: Optional[datetime] = None,
                            confidence: Optional[float] = None) -> Optional[Relation]:
        """创建新关系"""
        relation = self._build_new_relation(
            entity1_id, entity2_id, content, episode_id,
            entity1_name=entity1_name, entity2_name=entity2_name,
            verbose_relation=verbose_relation, source_document=source_document, base_time=base_time,
            confidence=confidence,
        )
        if relation:
            self.storage.save_relation(relation)
            if verbose_relation:
                wprint_info(f"[关系操作] ✅ 创建新关系: {entity1_name} <-> {entity2_name} (family_id: {relation.family_id})")
        return relation

    def _build_relation_version(self, family_id: str, entity1_id: str,
                                 entity2_id: str, content: str,
                                 episode_id: str,
                                 verbose_relation: bool = True,
                                 source_document: str = "",
                                 entity1_name: str = "",
                                 entity2_name: str = "",
                                 base_time: Optional[datetime] = None,
                                 entity_lookup: Optional[Dict[str, Any]] = None,
                                 _existing_relation: Optional[Relation] = None,
                                 old_content: str = "",
                                 old_content_format: str = "plain") -> Optional[Relation]:
        """构建关系新版本对象，但不立即写库。附带 section patch 计算。"""
        _cs = content.strip() if content else ""
        if len(_cs) < MIN_RELATION_CONTENT_LENGTH:
            if _existing_relation and _existing_relation.content and len(_existing_relation.content.strip()) >= MIN_RELATION_CONTENT_LENGTH:
                content = _existing_relation.content
            else:
                try:
                    versions = self.storage.get_relation_versions(family_id)
                    for v in versions:
                        if v.content and len(v.content.strip()) >= MIN_RELATION_CONTENT_LENGTH:
                            content = v.content
                            break
                except Exception:
                    pass
            _cs2 = content.strip() if content else ""
            if len(_cs2) < MIN_RELATION_CONTENT_LENGTH:
                if verbose_relation:
                    wprint_info(f"[关系操作] ⚠️  跳过版本: 内容过短且无可用历史内容 ({len(_cs2)}字符): {family_id}")
                return None

        relation = self._construct_relation(
            entity1_id, entity2_id, content, episode_id,
            family_id=family_id,
            entity1_name=entity1_name, entity2_name=entity2_name,
            verbose_relation=verbose_relation, source_document=source_document,
            base_time=base_time, entity_lookup=entity_lookup,
            skip_label="关系版本创建",
        )
        if relation and old_content:
            patches = compute_content_patches(
                family_id=family_id,
                old_content=old_content,
                old_content_format=old_content_format,
                new_content=content,
                new_absolute_id=relation.absolute_id,
                target_type="Relation",
                schema=RELATION_SECTIONS,
                source_document=source_document,
                event_time=relation.event_time,
            )
            if patches:
                relation._pending_patches = patches
        return relation

    def _create_relation_version(self, family_id: str, entity1_id: str,
                                 entity2_id: str, content: str,
                                 episode_id: str,
                                 verbose_relation: bool = True,
                                 source_document: str = "",
                                 entity1_name: str = "",
                                 entity2_name: str = "",
                                 base_time: Optional[datetime] = None,
                                 entity_lookup: Optional[Dict[str, Any]] = None) -> Optional[Relation]:
        """创建关系的新版本（始终创建，不跳过）。"""
        relation = self._build_relation_version(
            family_id, entity1_id, entity2_id, content, episode_id,
            verbose_relation=verbose_relation, source_document=source_document,
            entity1_name=entity1_name, entity2_name=entity2_name, base_time=base_time,
            entity_lookup=entity_lookup,
            _existing_relation=None,
        )
        if relation:
            self.storage.save_relation(relation)
            self._corroboration_queue.append(family_id)
        return relation

    def flush_corroboration_batch(self):
        """Flush queued corroboration updates as a single batch SQL operation."""
        if not self._corroboration_queue:
            return
        unique_fids = list(set(self._corroboration_queue))
        self._corroboration_queue.clear()
        try:
            self.storage.adjust_confidence_on_corroboration_batch(
                unique_fids, source_type="relation",
            )
        except Exception:
            for fid in unique_fids:
                try:
                    self.storage.adjust_confidence_on_corroboration(fid, source_type="relation")
                except Exception:
                    pass
