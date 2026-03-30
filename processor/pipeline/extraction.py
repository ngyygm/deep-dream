"""抽取流水线 mixin：从 orchestrator.py 提取。"""
from __future__ import annotations

import hashlib
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
import threading

from ..models import MemoryCache, Entity
from ..debug_log import log as dbg, log_section as dbg_section
from ..utils import compute_doc_hash, wprint
from ..llm.client import (
    LLM_PRIORITY_STEP1, LLM_PRIORITY_STEP2, LLM_PRIORITY_STEP3,
    LLM_PRIORITY_STEP4, LLM_PRIORITY_STEP5, LLM_PRIORITY_STEP6, LLM_PRIORITY_STEP7,
)


def _normalize_pair_for_relation(e1: str, e2: str) -> Tuple[str, str]:
    """与 LLMClient._normalize_entity_pair 一致：无向边端点按字典序固定。"""
    a, b = (e1 or "").strip(), (e2 or "").strip()
    return (a, b) if a <= b else (b, a)


def dedupe_extracted_entities(entities: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """按实体 name（strip 后）去重；同名时保留 content 更长的条目。"""
    name_to_index: Dict[str, int] = {}
    out: List[Dict[str, str]] = []
    for e in entities or []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "").strip()
        if not name:
            continue
        content = str(e.get("content") or "").strip()
        existing_idx = name_to_index.get(name)
        if existing_idx is None:
            name_to_index[name] = len(out)
            out.append({"name": name, "content": content})
            continue
        if len(content) > len(out[existing_idx]["content"]):
            out[existing_idx] = {"name": name, "content": content}
    return out


def dedupe_extracted_relations(relations: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """关系去重：无向 (entity1, entity2) 字典序 + content（忽略大小写）。"""
    seen: set[Tuple[str, str, int]] = set()
    out: List[Dict[str, str]] = []
    for r in relations or []:
        if not isinstance(r, dict):
            continue
        e1 = str(r.get("entity1_name") or "").strip()
        e2 = str(r.get("entity2_name") or "").strip()
        content = str(r.get("content") or "").strip()
        if not e1 or not e2 or not content:
            continue
        n1, n2 = _normalize_pair_for_relation(e1, e2)
        key = (n1, n2, hash(content.lower()))
        if key in seen:
            continue
        seen.add(key)
        out.append({"entity1_name": n1, "entity2_name": n2, "content": content})
    return out


def dedupe_extraction_lists(
    entities: Optional[List[Dict[str, Any]]],
    relations: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """供缓存加载等场景：实体、关系各做一次列表级去重。"""
    return dedupe_extracted_entities(entities), dedupe_extracted_relations(relations)


@dataclass
class _AlignResult:
    """步骤6（实体对齐）的输出，供步骤7使用。"""
    entity_name_to_id: Dict[str, str] = field(default_factory=dict)
    pending_relations: List[Dict] = field(default_factory=list)
    unique_entities: List[Entity] = field(default_factory=list)
    unique_pending_relations: List[Dict] = field(default_factory=list)


class _ExtractionMixin:
    """抽取相关流水线步骤（mixin，通过 TemporalMemoryGraphProcessor 多继承使用）。"""

    def _update_cache(self, input_text: str, document_name: str,
                      text_start_pos: int = 0, text_end_pos: int = 0,
                      total_text_length: int = 0, verbose: bool = True,
                      verbose_steps: bool = True,
                      document_path: str = "",
                      event_time: Optional[datetime] = None,
                      window_index: int = 0, total_windows: int = 0) -> MemoryCache:
        """步骤1：更新记忆缓存。必须在 _cache_lock 下调用，保证 cache 链串行。"""
        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP1
        if verbose:
            wprint("【步骤1】缓存｜开始｜")
        elif verbose_steps:
            wprint("【步骤1】缓存｜开始｜")

        # 蒸馏数据准备：确保 task_id 在步骤1前生成
        if self.llm_client._distill_data_dir:
            if not self.llm_client._distill_task_id:
                self.llm_client._distill_task_id = f"{document_name}_{uuid.uuid4().hex[:8]}_{int(time.time() * 1000)}"
            self.llm_client._current_distill_step = "01_update_cache"

        new_memory_cache = self.llm_client.update_memory_cache(
            self.current_memory_cache,
            input_text,
            document_name=document_name,
            text_start_pos=text_start_pos,
            text_end_pos=text_end_pos,
            total_text_length=total_text_length,
            event_time=event_time,
            window_index=window_index,
            total_windows=total_windows,
        )

        self.llm_client._current_distill_step = None

        doc_hash = compute_doc_hash(input_text) if input_text else ""
        self.storage.save_memory_cache(new_memory_cache, text=input_text, document_path=document_path, doc_hash=doc_hash)
        self.current_memory_cache = new_memory_cache

        if verbose:
            wprint(f"【步骤1】缓存｜写入｜ID {new_memory_cache.absolute_id}")
        elif verbose_steps:
            wprint("【步骤1】缓存｜完成｜已更新")

        return new_memory_cache

    # =========================================================================
    # 步骤2-5：抽取实体/关系/补全/增强（只读，不写存储，可并行）
    # =========================================================================

    def _extract_only(self, new_memory_cache: MemoryCache, input_text: str,
                      document_name: str, verbose: bool = True,
                      verbose_steps: bool = True,
                      event_time: Optional[datetime] = None,
                      progress_callback=None,
                      progress_range: tuple = (0.1, 0.5),
                      window_index: int = 0,
                      total_windows: int = 1) -> Tuple[List[Dict], List[Dict]]:
        """步骤2-5：抽取实体/关系/补全/增强。不写存储，可在线程池中并行执行。

        Returns:
            (extracted_entities, extracted_relations) — 纯字典列表，不含 entity_id。
        """

        p_lo, p_hi = progress_range
        _win_label = f"窗口 {window_index + 1}/{total_windows}"
        _steps = 4  # 步骤2-5 共4步
        _step_size = (p_hi - p_lo) / _steps

        def _report_step(step_idx: int, label: str, message: str):
            """step_idx: 0-3 对应步骤2-5。"""
            if progress_callback:
                p = p_lo + _step_size * (step_idx + 1)
                progress_callback(p,
                    f"{_win_label} · 步骤{step_idx + 2}/7: {label}",
                    message)

        def _report_intermediate(step_idx: int, frac: float, label: str, message: str):
            if progress_callback:
                p = p_lo + _step_size * (step_idx + frac)
                progress_callback(p,
                    f"{_win_label} · 步骤{step_idx + 2}/7: {label}",
                    message)

        # ========== 步骤2：抽取实体 ==========
        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP2
        if verbose:
            wprint("【步骤2】实体｜开始｜")
        elif verbose_steps:
            wprint("【步骤2】实体｜开始｜")

        self.llm_client._current_distill_step = "02_extract_entities"

        _extraction_empty_retry = 3  # LLM返回空结果时最多重试次数

        for _step2_attempt in range(1 + _extraction_empty_retry):
            def _on_entity_round(round_done, total_rounds, cumulative):
                frac = round_done / max(1, total_rounds)
                _report_intermediate(0, frac,
                        f"抽取实体 ({round_done}/{total_rounds})",
                        f"实体抽取第 {round_done}/{total_rounds} 轮，累计 {cumulative} 个实体")

            extracted_entities = self.llm_client.extract_entities(
                new_memory_cache, input_text,
                rounds=self.entity_extraction_rounds,
                verbose=verbose,
                on_round_done=_on_entity_round,
                compress_multi_round=self.compress_multi_round_extraction,
            )

            if extracted_entities:
                break  # 正常退出
            if _step2_attempt < _extraction_empty_retry:
                wprint(
                    f"【步骤2】警告｜重试｜空结果 {_step2_attempt + 1}/{1 + _extraction_empty_retry}"
                )

        self.llm_client._current_distill_step = None

        _ent_count = len(extracted_entities)
        if verbose:
            wprint(f"【步骤2】实体｜完成｜{_ent_count}个")
        elif verbose_steps:
            wprint(f"【步骤2】实体｜完成｜{_ent_count}个")
        dbg(f"步骤2完成: 抽取到 {_ent_count} 个实体")
        for _ei, _e in enumerate(extracted_entities):
            dbg(f"  实体[{_ei}]: name='{_e.get('name', '')}'  content='{_e.get('content', '')[:80]}'")
        if verbose:
            if _ent_count == 0:
                wprint(f"【步骤2】警告｜跳过｜重试{_extraction_empty_retry}次仍空")
            else:
                wprint(f"【步骤2】实体｜小结｜共{_ent_count}个")
        _report_step(0, "抽取实体", f"抽取到 {len(extracted_entities)} 个实体")

        _ents_before_dedup = len(extracted_entities)
        extracted_entities = dedupe_extracted_entities(extracted_entities)
        if verbose and len(extracted_entities) != _ents_before_dedup:
            wprint(
                f"【步骤2】去重｜实体｜{_ents_before_dedup}→{len(extracted_entities)}"
            )
        dbg(
            f"步骤2后实体去重: {_ents_before_dedup} → {len(extracted_entities)}"
        )

        # ========== 步骤3：抽取关系 ==========
        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP3
        if verbose:
            wprint("【步骤3】关系｜开始｜")
        elif verbose_steps:
            wprint("【步骤3】关系｜开始｜")

        self.llm_client._current_distill_step = "03_extract_relations"

        for _step3_attempt in range(1 + _extraction_empty_retry):
            def _on_relation_round(round_done, total_rounds, cumulative):
                frac = round_done / max(1, total_rounds)
                _report_intermediate(1, frac,
                        f"抽取关系 ({round_done}/{total_rounds})",
                        f"关系抽取第 {round_done}/{total_rounds} 轮，累计 {cumulative} 个关系")

            extracted_relations = self.llm_client.extract_relations(
                new_memory_cache, input_text,
                entities=extracted_entities,
                rounds=self.relation_extraction_rounds,
                verbose=verbose,
                on_round_done=_on_relation_round,
                compress_multi_round=self.compress_multi_round_extraction,
            )

            if extracted_relations:
                break  # 正常退出
            if _step3_attempt < _extraction_empty_retry:
                wprint(
                    f"【步骤3】警告｜重试｜空结果 {_step3_attempt + 1}/{1 + _extraction_empty_retry}"
                )

        self.llm_client._current_distill_step = None

        _rel_count = len(extracted_relations)
        if verbose:
            wprint(f"【步骤3】关系｜完成｜{_rel_count}个")
        elif verbose_steps:
            wprint(f"【步骤3】关系｜完成｜{_rel_count}个")
        dbg(f"步骤3完成: 抽取到 {_rel_count} 个关系")
        for _ri, _r in enumerate(extracted_relations):
            dbg(f"  关系[{_ri}]: '{_r.get('entity1_name', '')}' <-> '{_r.get('entity2_name', '')}'  content='{_r.get('content', '')[:100]}'")
        if verbose:
            if _rel_count == 0:
                wprint(f"【步骤3】警告｜跳过｜重试{_extraction_empty_retry}次仍空")
            else:
                wprint(f"【步骤3】关系｜小结｜共{_rel_count}个")
        _report_step(1, "抽取关系", f"抽取到 {len(extracted_relations)} 个关系")

        _rels_before_dedup = len(extracted_relations)
        extracted_relations = dedupe_extracted_relations(extracted_relations)
        if verbose and len(extracted_relations) != _rels_before_dedup:
            wprint(
                f"【步骤3】去重｜关系｜{_rels_before_dedup}→{len(extracted_relations)}"
            )
        dbg(
            f"步骤3后关系去重: {_rels_before_dedup} → {len(extracted_relations)}"
        )

        # 步骤3 后：仅当有关系时，按关系端点裁剪步骤2实体（未出现在任一边的实体丢弃）
        if extracted_relations:
            _relation_endpoint_names: set[str] = set()
            for _rel in extracted_relations:
                _e1 = _rel.get('entity1_name', '').strip()
                _e2 = _rel.get('entity2_name', '').strip()
                if _e1:
                    _relation_endpoint_names.add(_e1)
                if _e2:
                    _relation_endpoint_names.add(_e2)
            _before_prune = len(extracted_entities)
            extracted_entities = [
                e for e in extracted_entities
                if e.get('name', '').strip() in _relation_endpoint_names
            ]
            if verbose and _before_prune != len(extracted_entities):
                wprint(
                    f"【步骤3】裁剪｜实体｜{_before_prune}→{len(extracted_entities)}"
                )
            elif verbose_steps and _before_prune != len(extracted_entities):
                wprint(
                    f"【步骤3】裁剪｜实体｜{_before_prune}→{len(extracted_entities)}"
                )
            dbg(
                f"步骤3后按关系裁剪实体: {_before_prune} → {len(extracted_entities)} "
                f"（端点种类 {len(_relation_endpoint_names)}）"
            )

        # ========== 步骤4：补全缺失实体 ==========
        extracted_entity_names = {e['name'] for e in extracted_entities}
        missing_entity_names = set()
        for rel in extracted_relations:
            e1 = rel.get('entity1_name', '').strip()
            e2 = rel.get('entity2_name', '').strip()
            if e1 and e1 not in extracted_entity_names:
                missing_entity_names.add(e1)
            if e2 and e2 not in extracted_entity_names:
                missing_entity_names.add(e2)

        if missing_entity_names:
            self.llm_client._priority_local.priority = LLM_PRIORITY_STEP4
            if verbose:
                wprint(f"【步骤4】补全｜开始｜缺{len(missing_entity_names)}个")
            elif verbose_steps:
                wprint(f"【步骤4】补全｜开始｜{len(missing_entity_names)}个")
            _report_intermediate(2, 0.0, f"补全实体 (0/{len(missing_entity_names)})",
                    f"开始补全 {len(missing_entity_names)} 个缺失实体")
            self.llm_client._current_distill_step = "04_supplement_entities"
            missing_entities = self.llm_client.extract_entities_by_names(
                new_memory_cache, input_text,
                entity_names=list(missing_entity_names),
                verbose=verbose,
            )
            self.llm_client._current_distill_step = None
            extracted_entities.extend(missing_entities)
            if verbose:
                wprint(f"【步骤4】补全｜完成｜新{len(missing_entities)}个")
            elif verbose_steps:
                wprint("【步骤4】补全｜完成｜")
        else:
            if verbose:
                wprint("【步骤4】补全｜跳过｜无缺失")
            elif verbose_steps:
                wprint("【步骤4】补全｜跳过｜无缺失")
        _report_step(2, "补全实体", f"补全完成")

        _ents_post_supp = len(extracted_entities)
        extracted_entities = dedupe_extracted_entities(extracted_entities)
        if verbose and len(extracted_entities) != _ents_post_supp:
            wprint(
                f"【步骤4】去重｜实体｜{_ents_post_supp}→{len(extracted_entities)}"
            )

        # ========== 步骤5：实体增强 ==========
        if self.entity_post_enhancement:
            self.llm_client._priority_local.priority = LLM_PRIORITY_STEP5
            if verbose:
                wprint("【步骤5】增强｜开始｜")
            elif verbose_steps:
                wprint("【步骤5】增强｜开始｜")

            self.llm_client._current_distill_step = "05_entity_enhancement"

            if self.llm_threads > 1 and len(extracted_entities) > 1:
                enhanced_entities = []
                _enhance_priority = self.llm_client._priority_local.priority
                _enhance_distill_step = self.llm_client._current_distill_step
                with ThreadPoolExecutor(max_workers=self.llm_threads, thread_name_prefix="tmg-llm") as executor:
                    def _enhance_task(entity):
                        # 将主线程的 distill step 和优先级传播到工作线程（threading.local）
                        self.llm_client._priority_local.priority = _enhance_priority
                        self.llm_client._current_distill_step = _enhance_distill_step
                        return self.llm_client.enhance_entity_content(
                            new_memory_cache, input_text, entity
                        )
                    future_entity2 = {
                        executor.submit(_enhance_task, entity): entity
                        for entity in extracted_entities
                    }

                    entity_results = {}
                    for future in as_completed(future_entity2):
                        entity = future_entity2[future]
                        try:
                            enhanced_content = future.result()
                            entity_results[entity['name']] = {
                                'name': entity['name'],
                                'content': enhanced_content
                            }
                        except Exception as e:
                            if verbose:
                                wprint(f"【步骤5】警告｜增强｜{entity['name']}: {e}")
                            entity_results[entity['name']] = {
                                'name': entity['name'],
                                'content': entity['content']
                            }

                    for entity in extracted_entities:
                        if entity['name'] in entity_results:
                            enhanced_entities.append(entity_results[entity['name']])
                        else:
                            enhanced_entities.append({
                                'name': entity['name'],
                                'content': entity['content']
                            })
            else:
                enhanced_entities = []
                for entity in extracted_entities:
                    enhanced_content = self.llm_client.enhance_entity_content(
                        new_memory_cache,
                        input_text,
                        entity
                    )
                    enhanced_entities.append({
                        'name': entity['name'],
                        'content': enhanced_content
                    })

            extracted_entities = enhanced_entities

            self.llm_client._current_distill_step = None

            if verbose:
                wprint(f"【步骤5】增强｜完成｜共{len(extracted_entities)}个")
            elif verbose_steps:
                wprint("【步骤5】增强｜完成｜")
        else:
            if verbose:
                wprint("【步骤5】增强｜跳过｜已禁用")
            elif verbose_steps:
                wprint("【步骤5】增强｜跳过｜已禁用")
        _report_step(3, "实体增强", f"增强完成，共 {len(extracted_entities)} 个实体")

        return extracted_entities, extracted_relations

    # =========================================================================
    # 步骤6：实体对齐（写存储，必须串行跨窗口）
    # =========================================================================

    def _build_step7_relation_inputs_from_align_result(
        self, align_result: _AlignResult
    ) -> Tuple[List[Dict[str, str]], Dict[str, str], List[Dict], List[Dict]]:
        """从步骤6输出构造步骤7批处理输入；与 _align_relations 内逻辑一致，供预取与步骤7共用。"""
        entity_name_to_id = dict(align_result.entity_name_to_id)
        pending_relations_from_entities = align_result.pending_relations
        updated_pending_relations = align_result.unique_pending_relations

        # 某些并行实体对齐分支可能留下只存在于内存中的临时 entity_id；
        # Step7 开始前按名称刷新一次，避免关系写入时再命中“entity_id 不存在”。
        for name, eid in list(entity_name_to_id.items()):
            if eid:
                entity_name_to_id[name] = self.storage.resolve_entity_id(eid)

        invalid_names = [
            name for name, eid in entity_name_to_id.items()
            if eid and self.storage.get_entity_by_entity_id(eid) is None
        ]
        if invalid_names:
            refreshed_map = self.storage.get_entity_ids_by_names(invalid_names)
            for name, refreshed_id in refreshed_map.items():
                if refreshed_id:
                    entity_name_to_id[name] = refreshed_id

        all_pending_relations = updated_pending_relations.copy()

        for rel_info in pending_relations_from_entities:
            entity1_name = rel_info.get("entity1_name", "")
            entity2_name = rel_info.get("entity2_name", "")
            content = rel_info.get("content", "")
            relation_type = rel_info.get("relation_type", "normal")

            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)

            if entity1_id and entity2_id:
                if entity1_id == entity2_id:
                    continue
                all_pending_relations.append({
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": content,
                    "relation_type": relation_type
                })

        seen_relations = set()
        unique_pending_relations = []
        for rel in all_pending_relations:
            entity1_id = rel.get("entity1_id")
            entity2_id = rel.get("entity2_id")
            content = rel.get("content", "")
            if entity1_id and entity2_id:
                pair_key = tuple(sorted([entity1_id, entity2_id]))
                content_hash = hash(content.strip().lower())
                relation_key = (pair_key, content_hash)
                if relation_key not in seen_relations:
                    seen_relations.add(relation_key)
                    unique_pending_relations.append(rel)

        relation_inputs = [
            {
                "entity1_name": rel_info.get("entity1_name", ""),
                "entity2_name": rel_info.get("entity2_name", ""),
                "content": rel_info.get("content", ""),
            }
            for rel_info in unique_pending_relations
        ]

        return relation_inputs, entity_name_to_id, unique_pending_relations, all_pending_relations

    def _align_entities(self, extracted_entities: List[Dict], extracted_relations: List[Dict],
                        new_memory_cache: MemoryCache, input_text: str,
                        document_name: str, verbose: bool = True,
                        verbose_steps: bool = True,
                        event_time: Optional[datetime] = None,
                        progress_callback=None,
                        progress_range: tuple = (0.5, 0.75),
                        window_index: int = 0,
                        total_windows: int = 1,
                        entity_embedding_prefetch: Optional[Future] = None) -> _AlignResult:
        """步骤6：实体对齐（搜索、合并、写入存储）。必须串行跨窗口。

        Returns:
            _AlignResult 包含 entity_name_to_id、pending_relations 等，供步骤7使用。
        """

        p_lo, p_hi = progress_range
        _win_label = f"窗口 {window_index + 1}/{total_windows}"

        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP6
        if verbose:
            wprint("【步骤6】实体｜开始｜对齐写入")
        elif verbose_steps:
            wprint("【步骤6】实体｜开始｜")

        self.llm_client._current_distill_step = "06_entity_alignment"

        # 记录原始实体名称列表（用于后续建立映射）
        original_entity_names = [e['name'] for e in extracted_entities]

        # 用于存储待处理的关系（使用实体名称）
        all_pending_relations_by_name = []
        if extracted_relations:
            for rel in extracted_relations:
                entity1_name = rel.get('entity1_name') or rel.get('from_entity_name', '').strip()
                entity2_name = rel.get('entity2_name') or rel.get('to_entity_name', '').strip()
                content = rel.get('content', '').strip()
                if entity1_name and entity2_name:
                    all_pending_relations_by_name.append({
                        "entity1_name": entity1_name,
                        "entity2_name": entity2_name,
                        "content": content,
                        "relation_type": "normal"
                    })

        entity_name_to_id_from_entities = {}
        _entity_total = len(extracted_entities)
        _entity_done = 0
        _step_size = p_hi - p_lo

        def on_entity_processed_callback(entity, current_entity_name_to_id, current_pending_relations):
            nonlocal all_pending_relations_by_name, entity_name_to_id_from_entities, _entity_done
            _entity_done += 1
            entity_name_to_id_from_entities.update(current_entity_name_to_id)
            all_pending_relations_by_name.extend(current_pending_relations)
            if progress_callback:
                frac = _entity_done / max(1, _entity_total)
                progress_callback(p_lo + _step_size * frac,
                    f"{_win_label} · 步骤6/7: 实体对齐 ({_entity_done}/{_entity_total})",
                    f"实体对齐 {_entity_done}/{_entity_total}")

        processed_entities, pending_relations_from_entities, entity_name_to_id_from_entities_final = self.entity_processor.process_entities(
            extracted_entities,
            new_memory_cache.absolute_id,
            self.similarity_threshold,
            memory_cache=new_memory_cache,
            source_document=document_name,
            context_text=input_text,
            extracted_relations=extracted_relations,
            jaccard_search_threshold=self.jaccard_search_threshold,
            embedding_name_search_threshold=self.embedding_name_search_threshold,
            embedding_full_search_threshold=self.embedding_full_search_threshold,
            on_entity_processed=on_entity_processed_callback,
            base_time=new_memory_cache.event_time,
            max_workers=self.llm_threads,
            verbose=verbose,
            entity_embedding_prefetch=entity_embedding_prefetch,
        )

        entity_name_to_id_from_entities.update(entity_name_to_id_from_entities_final)
        pending_relations_from_entities = all_pending_relations_by_name

        # 按entity_id去重，只保留最新版本
        unique_entities_dict = {}
        for entity in processed_entities:
            if entity.entity_id not in unique_entities_dict:
                unique_entities_dict[entity.entity_id] = entity
            else:
                if entity.processed_time > unique_entities_dict[entity.entity_id].processed_time:
                    unique_entities_dict[entity.entity_id] = entity

        unique_entities = list(unique_entities_dict.values())

        # 构建完整的实体名称到entity_id的映射
        entity_name_to_ids = {}
        for entity in unique_entities:
            if entity.name not in entity_name_to_ids:
                entity_name_to_ids[entity.name] = []
            if entity.entity_id not in entity_name_to_ids[entity.name]:
                entity_name_to_ids[entity.name].append(entity.entity_id)

        for name, entity_id in entity_name_to_id_from_entities.items():
            if name not in entity_name_to_ids:
                entity_name_to_ids[name] = []
            if entity_id not in entity_name_to_ids[name]:
                entity_name_to_ids[name].append(entity_id)

        for i, entity in enumerate(processed_entities):
            if i < len(original_entity_names):
                original_name = original_entity_names[i]
                if original_name not in entity_name_to_ids:
                    entity_name_to_ids[original_name] = []
                if entity.entity_id not in entity_name_to_ids[original_name]:
                    entity_name_to_ids[original_name].append(entity.entity_id)

        # 检测和处理同名实体冲突
        duplicate_names = {name: ids for name, ids in entity_name_to_ids.items() if len(ids) > 1}

        if duplicate_names:
            if verbose:
                wprint(f"【步骤6】警告｜同名｜{len(duplicate_names)}处")
                for name, ids in duplicate_names.items():
                    wprint(
                        f"【步骤6】冲突｜详情｜{name} {len(ids)}id {ids[:3]}{'...' if len(ids) > 3 else ''}"
                    )

            entity_name_to_id = {}
            for name, ids in entity_name_to_ids.items():
                if len(ids) > 1:
                    version_counts = {}
                    for eid in ids:
                        count = len(self.storage.get_entity_versions(eid))
                        version_counts[eid] = count
                    primary_id = max(ids, key=lambda eid: version_counts.get(eid, 0))
                    entity_name_to_id[name] = primary_id
                    for eid in ids:
                        if eid and eid != primary_id:
                            self.storage.register_entity_redirect(eid, primary_id)
                    if verbose:
                        wprint(
                            f"【步骤6】冲突｜主实体｜{name}->{primary_id} v{version_counts.get(primary_id, 0)}"
                        )
                else:
                    entity_name_to_id[name] = ids[0]
        else:
            entity_name_to_id = {name: ids[0] for name, ids in entity_name_to_ids.items()}

        merged_mappings = []
        for i, entity in enumerate(processed_entities):
            if i < len(original_entity_names):
                original_name = original_entity_names[i]
                if original_name != entity.name:
                    merged_mappings.append((original_name, entity.name, entity.entity_id))

        if verbose:
            if len(unique_entities) == 0:
                wprint(
                    f"【步骤6】小结｜实体｜无新·抽{len(original_entity_names)}个已存在"
                )
            else:
                wprint(
                    f"【步骤6】小结｜实体｜唯一{len(unique_entities)}·原{len(original_entity_names)}"
                )
            if merged_mappings:
                wprint(f"【步骤6】映射｜合并｜{len(merged_mappings)}个")

        # 步骤6.3：构建完整的实体名称→ID映射表，防止关系丢失
        # 收集关系中引用的所有实体名称
        _rel_entity_names = set()
        for rel_info in pending_relations_from_entities:
            n1 = rel_info.get("entity1_name", "")
            n2 = rel_info.get("entity2_name", "")
            if n1:
                _rel_entity_names.add(n1)
            if n2:
                _rel_entity_names.add(n2)
        # 补全：关系中引用但映射表里缺失的名称，从数据库查找
        _missing_names = [n for n in _rel_entity_names if n not in entity_name_to_id]
        _db_matched = 0
        if _missing_names:
            _db_map = self.storage.get_entity_ids_by_names(_missing_names)
            for name, eid in _db_map.items():
                if name not in entity_name_to_id:
                    entity_name_to_id[name] = eid
                    _db_matched += 1

        # 名称→ID转换
        updated_pending_relations = []
        _skipped_relations = []
        _self_relations = 0
        for rel_info in pending_relations_from_entities:
            entity1_name = rel_info.get("entity1_name", "")
            entity2_name = rel_info.get("entity2_name", "")
            content = rel_info.get("content", "")
            relation_type = rel_info.get("relation_type", "normal")

            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)

            if entity1_id and entity2_id:
                if entity1_id == entity2_id:
                    _self_relations += 1
                    continue
                updated_pending_relations.append({
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": content,
                    "relation_type": relation_type
                })
            else:
                _reason = []
                if not entity1_id:
                    _reason.append(f"entity1='{entity1_name}'")
                if not entity2_id:
                    _reason.append(f"entity2='{entity2_name}'")
                _skipped_relations.append(f"  {entity1_name} <-> {entity2_name} (无法解析: {', '.join(_reason)})")

        if _skipped_relations or _self_relations > 0:
            _parts = [f"成功解析 {len(updated_pending_relations)} 个"]
            if _db_matched > 0:
                _parts.append(f"数据库补全 {_db_matched} 个")
            if _self_relations > 0:
                _parts.append(f"自关系 {_self_relations} 个")
            if _skipped_relations:
                _parts.append(f"无法解析 {len(_skipped_relations)} 个")
            if verbose:
                wprint(
                    f"【步骤6】关系｜待处理｜{len(pending_relations_from_entities)}→{', '.join(_parts)}"
                )
                if _skipped_relations:
                    wprint(
                        f"【步骤6】映射｜表｜{len(entity_name_to_id)}名 "
                        f"{', '.join(list(entity_name_to_id.keys())[:15])}{'...' if len(entity_name_to_id) > 15 else ''}"
                    )
                    for _sr in _skipped_relations[:10]:
                        wprint(f"【步骤6】关系｜跳过｜{_sr}")
                    if len(_skipped_relations) > 10:
                        wprint(f"【步骤6】关系｜跳过｜余{len(_skipped_relations) - 10}条")
        else:
            if verbose:
                wprint(
                    f"【步骤6】关系｜待处理｜{len(pending_relations_from_entities)}→全解析"
                    + (f"·库补{_db_matched}" if _db_matched > 0 else "")
                )

        if verbose_steps and not verbose:
            wprint("【步骤6】实体｜完成｜映射")

        dbg_section("步骤6.3: 实体名称→ID映射")
        dbg(f"entity_name_to_id 映射 ({len(entity_name_to_id)} 个):")
        for _mn, _mid in entity_name_to_id.items():
            dbg(f"  '{_mn}' -> {_mid}")
        dbg(f"待处理关系 {len(pending_relations_from_entities)} 个 → 成功 {len(updated_pending_relations)}, 自关系 {_self_relations}, 跳过 {len(_skipped_relations)}")
        for _sr in _skipped_relations:
            dbg(f"  跳过: {_sr}")

        self.llm_client._current_distill_step = None

        if progress_callback:
            progress_callback(p_hi,
                f"{_win_label} · 步骤6/7: 实体对齐",
                f"实体对齐完成，共 {len(unique_entities)} 个实体")

        # Phase C: 记录 Episode → Entity MENTIONS
        if unique_entities and hasattr(self.storage, 'save_episode_mentions'):
            try:
                abs_ids = [e.absolute_id for e in unique_entities]
                self.storage.save_episode_mentions(new_memory_cache.absolute_id, abs_ids)
            except Exception as e:
                if verbose:
                    wprint(f"【步骤6】MENTIONS｜失败｜{e}")

        return _AlignResult(
            entity_name_to_id=entity_name_to_id,
            pending_relations=pending_relations_from_entities,
            unique_entities=unique_entities,
            unique_pending_relations=updated_pending_relations,
        )

    # =========================================================================
    # 步骤7：关系对齐（写存储，串行跨窗口）
    # =========================================================================

    def _align_relations(self, align_result: _AlignResult,
                         new_memory_cache: MemoryCache, input_text: str,
                         document_name: str, verbose: bool = True,
                         verbose_steps: bool = True,
                         event_time: Optional[datetime] = None,
                         progress_callback=None,
                         progress_range: tuple = (0.75, 1.0),
                         window_index: int = 0,
                         total_windows: int = 1,
                         prepared_relations_by_pair: Optional[Dict[Tuple[str, str], List[Dict[str, str]]]] = None,
                         step7_inputs_cache: Optional[Tuple[List[Dict[str, str]], Dict[str, str], List[Dict], List[Dict]]] = None,
                         ) -> List:
        """步骤7：关系对齐（搜索、合并、写入存储）。串行跨窗口。

        Args:
            align_result: 步骤6的输出，包含 entity_name_to_id 和 pending_relations。
            prepared_relations_by_pair: 可选，跨窗预取的按实体对分组结果（须在上一窗 step7 完成后读库）。
            step7_inputs_cache: 可选，与 _build_step7_relation_inputs_from_align_result 返回值一致，避免重复计算。
        """

        p_lo, p_hi = progress_range
        _win_label = f"窗口 {window_index + 1}/{total_windows}"
        _step_size = p_hi - p_lo

        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP7
        if verbose:
            wprint("【步骤7】关系｜开始｜对齐写入")
        elif verbose_steps:
            wprint("【步骤7】关系｜开始｜")

        self.llm_client._current_distill_step = "07_relation_alignment"

        unique_entities = align_result.unique_entities

        if step7_inputs_cache is not None:
            relation_inputs, entity_name_to_id, unique_pending_relations, all_pending_relations = step7_inputs_cache
        else:
            relation_inputs, entity_name_to_id, unique_pending_relations, all_pending_relations = (
                self._build_step7_relation_inputs_from_align_result(align_result)
            )

        if verbose:
            duplicate_count = len(all_pending_relations) - len(unique_pending_relations)
            if duplicate_count > 0:
                wprint(
                    f"【步骤7】关系｜待处理｜{len(all_pending_relations)}→去重{len(unique_pending_relations)}"
                )
            else:
                wprint(f"【步骤7】关系｜待处理｜{len(unique_pending_relations)}个")

        _upr_count = len(unique_pending_relations)
        if _upr_count == 0:
            if verbose:
                wprint("【步骤7】关系｜跳过｜无待处理")
        else:
            if verbose:
                wprint(
                    f"【步骤7】关系｜待处理｜去重{_upr_count}·原{len(all_pending_relations)}"
                )
        dbg(f"步骤7: 去重后待处理关系 {len(unique_pending_relations)} 个 (去重前 {len(all_pending_relations)} 个)")
        for _upr in unique_pending_relations:
            dbg(f"  待处理: '{_upr.get('entity1_name', '')}' <-> '{_upr.get('entity2_name', '')}' (e1_id={_upr.get('entity1_id', '?')}, e2_id={_upr.get('entity2_id', '?')})  content='{_upr.get('content', '')[:100]}'")

        _rel_done = [0]

        def _on_relation_pair_done(done, total):
            _rel_done[0] = done
            if progress_callback:
                frac = done / max(1, total)
                progress_callback(p_lo + _step_size * frac,
                    f"{_win_label} · 步骤7/7: 关系对齐 ({done}/{total})",
                    f"关系对齐 {done}/{total}")

        all_processed_relations = self.relation_processor.process_relations_batch(
            relation_inputs,
            entity_name_to_id,
            new_memory_cache.absolute_id,
            source_document=document_name,
            base_time=new_memory_cache.event_time,
            max_workers=self.llm_threads,
            on_relation_done=_on_relation_pair_done,
            # detail 模式常开 verbose、关 verbose_steps：避免逐条 [关系操作] 刷屏
            verbose_relation=bool(verbose and verbose_steps),
            prepared_relations_by_pair=prepared_relations_by_pair,
        )

        if verbose:
            if len(all_processed_relations) == 0:
                wprint("【步骤7】关系｜小结｜无新")
            else:
                wprint(f"【步骤7】关系｜小结｜{len(all_processed_relations)}个")
        elif verbose_steps:
            wprint("【步骤7】关系｜完成｜")

        if verbose:
            wprint("【窗口】流水｜结束｜")
        _final_ents = len(unique_entities)
        _final_rels = len(all_processed_relations)
        if verbose:
            if _final_ents == 0 and _final_rels == 0:
                wprint("【窗口】汇总｜空｜无新实体关系")
            else:
                wprint(
                    f"【窗口】汇总｜得｜实体{_final_ents} 关系{_final_rels}·待{len(unique_pending_relations)}"
                )
        elif verbose_steps:
            wprint(f"【窗口】汇总｜得｜实体{_final_ents} 关系{_final_rels}")
        dbg(f"窗口处理完成: {len(unique_entities)} 个实体, {len(all_processed_relations)} 个关系 (从 {len(unique_pending_relations)} 个待处理)")

        if progress_callback:
            progress_callback(p_hi,
                f"{_win_label} · 步骤7/7: 窗口完成",
                f"{len(unique_entities)} 个实体, {len(all_processed_relations)} 个关系")

        self.llm_client._current_distill_step = None
        self.llm_client._distill_task_id = None

        return all_processed_relations

    # =========================================================================
    # 兼容入口：串行执行步骤2-7（_process_window 旧路径使用）
    # =========================================================================

    def _process_extraction(self, new_memory_cache: MemoryCache, input_text: str,
                            document_name: str, verbose: bool = True,
                            verbose_steps: bool = True,
                            event_time: Optional[datetime] = None,
                            progress_callback=None,
                            progress_range: tuple = (0.1, 1.0),
                            window_index: int = 0,
                            total_windows: int = 1):
        """兼容入口：串行执行步骤2-7（_process_window 等旧路径使用）。"""

        # 步骤2-5 占 progress_range 的 5/7，步骤6 占 1/7，步骤7 占 1/7
        total_size = progress_range[1] - progress_range[0]
        p1_end = progress_range[0] + total_size * 5 / 7
        p2_end = progress_range[0] + total_size * 6 / 7

        extracted_entities, extracted_relations = self._extract_only(
            new_memory_cache, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(progress_range[0], p1_end),
            window_index=window_index, total_windows=total_windows,
        )

        align_result = self._align_entities(
            extracted_entities, extracted_relations,
            new_memory_cache, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(p1_end, p2_end),
            window_index=window_index, total_windows=total_windows,
        )

        self._align_relations(
            align_result,
            new_memory_cache, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(p2_end, progress_range[1]),
            window_index=window_index, total_windows=total_windows,
        )

    def _process_window(self, input_text: str, document_name: str,
                       is_new_document: bool, text_start_pos: int = 0,
                       text_end_pos: int = 0, total_text_length: int = 0,
                       verbose: bool = True, verbose_steps: bool = True,
                       document_path: str = "",
                       event_time: Optional[datetime] = None,
                       window_index: int = 0, total_windows: int = 1):
        """兼容入口：串行执行 cache 更新 + 抽取处理（process_documents 等旧路径使用）。"""
        if verbose:
            wprint(f"\n{'='*60}")
            wprint(f"处理窗口 (文档: {document_name}, 位置: {text_start_pos}-{text_end_pos}/{total_text_length})")
            wprint(f"输入文本长度: {len(input_text)} 字符")
            wprint(f"{'='*60}\n")
        elif verbose_steps:
            wprint(f"窗口开始 · {document_name}  [{text_start_pos}-{text_end_pos}/{total_text_length}]")

        with self._cache_lock:
            new_mc = self._update_cache(
                input_text, document_name,
                text_start_pos=text_start_pos, text_end_pos=text_end_pos,
                total_text_length=total_text_length, verbose=verbose,
                verbose_steps=verbose_steps,
                document_path=document_path, event_time=event_time,
                window_index=window_index, total_windows=total_windows,
            )
        self._process_extraction(new_mc, input_text, document_name,
                                 verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                                 window_index=window_index, total_windows=total_windows)
