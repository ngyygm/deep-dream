"""Orphan entity cleanup, fallback cooccurrence, and relation recovery sub-mixin."""
from __future__ import annotations

from typing import Dict, List, Optional

from core.utils import wprint_info, wprint_debug, wprint_warn
from core.llm.prompts import RELATION_DISCOVER_SYSTEM, ORPHAN_RECOVERY_USER


class _OrphanMixin:
    """Orphan entity handling: cleanup, fallback cooccurrence relations, and LLM-based recovery."""

    def _cleanup_orphaned_entities(
        self,
        saved_entities: list,
        verbose: bool = False,
        window_text: str = "",
        all_entity_names: Optional[List[str]] = None,
        episode_id: str = "",
        source_document: str = "",
    ) -> int:
        """处理孤立实体：先尝试补救（找关系），再为无法补救的创建兜底共现关系。

        在 step10（关系存储）完成后调用。此时关系已经全部写入，
        可以准确判断哪些实体是孤立的。

        补救流程：对孤立实体调用 LLM 寻找与其他实体的关系，写入后重新检查度数，
        仍然为 0 的创建兜底共现关系（不再删除）。

        Args:
            saved_entities: step9 存入的实体列表（_AlignResult.unique_entities）
            verbose: 是否打印日志
            window_text: 当前窗口文本（补救用）
            all_entity_names: 当前窗口所有实体名称（补救用）
            episode_id: 当前 episode ID（补救写关系时使用）
            source_document: 来源文档名（补救写关系时使用）

        Returns:
            删除的孤立实体数量（始终为 0，不再删除）
        """
        if not saved_entities:
            return 0

        new_family_ids = [e.family_id for e in saved_entities if hasattr(e, 'family_id') and e.family_id]
        if not new_family_ids:
            return 0

        # 批量查询度数（关系数）
        batch_fn = getattr(self.storage, 'batch_get_entity_degrees', None)
        if batch_fn is None:
            return 0

        try:
            degree_map = batch_fn(new_family_ids)
        except Exception:
            return 0

        # 收集度数为 0 的实体（无任何关系）
        orphan_fids = [fid for fid, deg in degree_map.items() if deg == 0]
        if not orphan_fids:
            return 0

        # 区分「全新实体」和「对齐到已有实体的更新」
        # 批量查询版本数：版本数 > 1 说明实体在本次处理前就已存在
        version_counts = {}
        try:
            version_counts = self.storage.get_entity_version_counts(orphan_fids)
        except Exception:
            pass  # 查询失败则保守不删

        # 只处理真正全新创建的孤立实体（版本数 == 1 且无关系）
        truly_new_orphans = [fid for fid in orphan_fids
                             if version_counts.get(fid, 1) <= 1]

        if not truly_new_orphans:
            return 0

        # ---- 补救阶段：尝试为孤立实体找关系 ----
        recovered = 0
        if window_text and all_entity_names and truly_new_orphans:
            recovered = self._recover_orphan_relations(
                truly_new_orphans, saved_entities, all_entity_names,
                window_text, episode_id, source_document, verbose,
            )

        # 补救后重新查询度数，只删除仍然孤立的
        if recovered > 0:
            try:
                degree_map = batch_fn(truly_new_orphans)
                truly_new_orphans = [fid for fid, deg in degree_map.items() if deg == 0]
            except Exception:
                pass  # 查询失败则保守不删

        if not truly_new_orphans:
            return 0

        # ---- 兜底阶段：为仍然孤立的实体创建共现关系 ----
        _fallback_count = self._create_fallback_cooccurrence_relations(
            truly_new_orphans, saved_entities,
            episode_id, source_document, verbose,
        )

        if _fallback_count > 0 or recovered > 0:
            try:
                self.storage._cache.invalidate_keys(["graph_stats"])
            except Exception:
                pass

        return 0  # 不再删除孤立实体

    def _create_fallback_cooccurrence_relations(
        self,
        orphan_fids: List[str],
        saved_entities: list,
        episode_id: str,
        source_document: str,
        verbose: bool,
    ) -> int:
        """为孤立实体创建兜底共现关系，确保每个实体至少有一个关系链接。"""
        if not orphan_fids:
            return 0

        # 构建 family_id → entity 映射
        fid_to_entity = {}
        for e in saved_entities:
            fid = getattr(e, 'family_id', None)
            if fid:
                fid_to_entity[fid] = e

        # 非孤立实体作为关系目标
        orphan_fid_set = set(orphan_fids)
        non_orphan_entities = [
            e for e in saved_entities
            if hasattr(e, 'family_id') and e.family_id
            and e.family_id not in orphan_fid_set
        ]

        if not non_orphan_entities:
            if verbose:
                wprint_info(f"  │  孤立实体兜底｜无法创建共现关系（无非孤立实体）")
            return 0

        relation_processor = getattr(self, 'relation_processor', None)
        if not relation_processor or not episode_id:
            return 0

        if verbose:
            _orphan_names = [getattr(fid_to_entity.get(fid), 'name', '?') for fid in orphan_fids]
            wprint_info(f"  │  孤立实体兜底｜为 {len(orphan_fids)} 个实体创建共现关系: {', '.join(_orphan_names[:5])}")

        fallback_count = 0
        for i, orphan_fid in enumerate(orphan_fids):
            orphan_entity = fid_to_entity.get(orphan_fid)
            if not orphan_entity:
                continue

            # 选择非孤立实体作为关系目标（轮询分配）
            target_entity = non_orphan_entities[i % len(non_orphan_entities)]

            try:
                rel = relation_processor._build_new_relation(
                    orphan_fid,
                    target_entity.family_id,
                    f"{orphan_entity.name}与{target_entity.name}在同一文本中出现",
                    episode_id,
                    entity1_name=orphan_entity.name,
                    entity2_name=target_entity.name,
                    verbose_relation=False,
                    source_document=source_document,
                    confidence=0.3,
                )
                if rel is not None:
                    relation_processor.storage.save_relation(rel)
                    fallback_count += 1
                    if verbose:
                        wprint_debug(f"  │  兜底共现关系: {orphan_entity.name} <-> {target_entity.name}")
            except Exception:
                pass

        if verbose:
            wprint_info(f"  │  孤立实体兜底｜{fallback_count}/{len(orphan_fids)} 个实体成功创建共现关系")
        return fallback_count

    def _recover_orphan_relations(
        self,
        orphan_fids: List[str],
        saved_entities: list,
        all_entity_names: List[str],
        window_text: str,
        episode_id: str,
        source_document: str,
        verbose: bool,
    ) -> int:
        """尝试为孤立实体找到并建立关系。

        Returns:
            成功补救的实体数量（度数从 0 变为 > 0）
        """
        # 构建 family_id → entity 映射
        fid_to_entity = {}
        for e in saved_entities:
            fid = getattr(e, 'family_id', None)
            if fid and fid in orphan_fids:
                fid_to_entity[fid] = e

        # 构建 entity_name → family_id 映射（所有实体，包括非孤儿）
        name_to_fid = {}
        for e in saved_entities:
            fid = getattr(e, 'family_id', None)
            name = getattr(e, 'name', None)
            if fid and name:
                name_to_fid[name] = fid

        orphan_names = [getattr(fid_to_entity[fid], 'name', '?') for fid in orphan_fids if fid in fid_to_entity]
        other_names = [n for n in all_entity_names if n not in orphan_names]

        if not orphan_names or not other_names:
            return 0

        if verbose:
            wprint_info(f"  │  孤立实体补救｜尝试为 {len(orphan_names)} 个实体找关系: {', '.join(orphan_names[:5])}")

        # 调用 LLM 寻找关系对
        try:
            user_prompt = ORPHAN_RECOVERY_USER.format(
                orphan_names="、".join(orphan_names),
                other_entity_names="、".join(other_names),
                window_text=window_text,
            )
            messages = [
                {"role": "system", "content": RELATION_DISCOVER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
            parsed, _ = self.llm_client.call_llm_until_json_parses(
                messages,
                parse_fn=self.llm_client._parse_pair_list,
                timeout=120,
            )
            raw_pairs = parsed or []
        except Exception as e:
            if verbose:
                wprint_debug(f"  │  孤立实体补救 LLM 调用失败: {e}")
            return 0

        if not raw_pairs:
            if verbose:
                wprint_info("  │  孤立实体补救｜LLM 未发现新关系")
            return 0

        # 解析并写入关系
        entity_name_set = set(all_entity_names)
        recovered_fids = set()
        relation_processor = getattr(self, 'relation_processor', None)
        from .steps import _ExtractionStepsMixin as _EPM

        # Phase 1: Resolve names + parallel LLM content writing
        _name_lookup = _EPM._build_name_lookup(entity_name_set)
        resolved_pairs = []
        for a, b in raw_pairs:
            resolved_a = _EPM._resolve_entity_name(a, entity_name_set, _lookup=_name_lookup)
            resolved_b = _EPM._resolve_entity_name(b, entity_name_set, _lookup=_name_lookup)
            if not resolved_a or not resolved_b or resolved_a == resolved_b:
                continue
            fid_a = name_to_fid.get(resolved_a)
            fid_b = name_to_fid.get(resolved_b)
            if not fid_a or not fid_b:
                continue
            resolved_pairs.append((resolved_a, resolved_b, fid_a, fid_b))

        # Batch LLM content writing (1 call instead of N parallel calls)
        batch_fn = getattr(self.llm_client, 'batch_write_relation_content', None)
        batch_results = {}
        if batch_fn and resolved_pairs:
            try:
                batch_results = batch_fn(
                    [(a, b) for a, b, _, _ in resolved_pairs], window_text,
                )
            except Exception:
                pass

        content_results = []
        for resolved_a, resolved_b, fid_a, fid_b in resolved_pairs:
            content = batch_results.get((resolved_a, resolved_b), "")
            if not content:
                content = batch_results.get((resolved_b, resolved_a), "")
            if not content:
                try:
                    content = self.llm_client.write_relation_content(resolved_a, resolved_b, window_text)
                except Exception:
                    content = ""
            content_results.append((resolved_a, resolved_b, fid_a, fid_b, content))

        # Phase 2: Build relations in batch, then bulk-save
        if relation_processor and episode_id:
            batch_relations = []
            batch_fids = []
            for resolved_a, resolved_b, fid_a, fid_b, content in content_results:
                try:
                    rel = relation_processor._build_new_relation(
                        fid_a, fid_b, content, episode_id,
                        entity1_name=resolved_a, entity2_name=resolved_b,
                        verbose_relation=False, source_document=source_document,
                    )
                    if rel is not None:
                        batch_relations.append(rel)
                        batch_fids.append((resolved_a, resolved_b, fid_a, fid_b))
                except Exception:
                    pass
            if batch_relations:
                try:
                    relation_processor.storage.bulk_save_relations(batch_relations)
                except Exception:
                    # Fallback: save individually
                    for rel in batch_relations:
                        try:
                            relation_processor.storage.save_relation(rel)
                        except Exception:
                            pass
                for resolved_a, resolved_b, fid_a, fid_b in batch_fids:
                    recovered_fids.add(fid_a)
                    recovered_fids.add(fid_b)
                    if verbose:
                        wprint_debug(f"  │  补救关系: {resolved_a} <-> {resolved_b}")

        recovered_count = len(recovered_fids & set(orphan_fids))
        if verbose:
            wprint_info(f"  │  孤立实体补救｜{recovered_count}/{len(orphan_names)} 个实体成功建立关系")
        return recovered_count
