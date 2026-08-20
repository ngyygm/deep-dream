"""strong-v1 profile 的窗口抽取步骤：单次 LLM 调用 + 规则后处理。

无存储写入，线程池安全；返回：
(extracted_entities, extracted_relations) — dict 列表，无 family_id。

复用的规则资产（与现管线同一实现，无行为分叉）：
- _normalize_and_dedup_entity_names / _validate_entity / _validate_relation
- prose 回退内容（_build_entity_fallback_content）与关系模板回退
- quality.filter_entity_names / cap_relation_pairs（profile 检查含 strong-v1）
- 控制流检查（pause/delete 可中断）、进度回调、窗口计时
"""
from __future__ import annotations

import time as _time
from typing import Dict, List, Optional, Tuple

from core.utils import wprint_info
from core.llm.client import LLM_PRIORITY_EXTRACT

from ._steps_helpers import (
    _pair_key,
    _normalize_and_dedup_entity_names, _validate_entity, _validate_relation,
    _prepare_prose_sentences, _ProseIndex, _build_entity_fallback_content,
    _MIN_ENTITY_CONTENT_LEN, _MIN_RELATION_CONTENT_LEN,
    _build_name_lookup, _resolve_entity_name,
)


def strong_extract_only(
    self,
    new_episode,
    input_text: str,
    document_name: str,
    verbose: bool = True,
    verbose_steps: bool = True,
    event_time=None,
    progress_callback=None,
    progress_range: tuple = (0.1, 0.5),
    window_index: int = 0,
    total_windows: int = 1,
    window_timings_ref: Optional[Dict[str, float]] = None,
    control_check_fn=None,
    early_entity_done_fn=None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """单遍抽取：1 次 LLM 调用 + 规则后处理（质量门/去重/回退）。"""
    from core.remember.orchestrator import RememberControlFlow

    def _check_control():
        if control_check_fn:
            action = control_check_fn()
            if action:
                raise RememberControlFlow(action)

    def _record_timing(key: str, elapsed: float):
        if window_timings_ref is not None:
            window_timings_ref[key] = elapsed

    p_lo, p_hi = progress_range
    _win = f"窗口 {window_index + 1}/{total_windows}"

    def _progress(frac, label, msg):
        if progress_callback:
            progress_callback(p_lo + (p_hi - p_lo) * frac, label, msg)

    _cancel_check_fn = (lambda: control_check_fn() is not None) if control_check_fn else None
    if _cancel_check_fn:
        self.llm_client.set_cancel_check(_cancel_check_fn)

    try:
        # ------------------------------------------------------------------
        # 1. 单遍 LLM 抽取（发现 + 内容一次完成）
        # ------------------------------------------------------------------
        _progress(0.10, f"{_win} · S2s: 单遍结构化抽取", "开始")
        _t = _time.time()
        previous_priority = getattr(self.llm_client._priority_local, "priority", None)
        self.llm_client._priority_local.priority = LLM_PRIORITY_EXTRACT
        try:
            structured = self.llm_client.extract_window_structured(
                input_text,
                max_entities=getattr(self, "remember_max_entities_per_window", 16) * 2,
                max_relations=getattr(self, "remember_max_relations_per_window", 24) * 2,
            )
        finally:
            if previous_priority is None:
                try:
                    del self.llm_client._priority_local.priority
                except AttributeError:
                    pass
            else:
                self.llm_client._priority_local.priority = previous_priority
        _elapsed_onepass = _time.time() - _t
        _record_timing("step2s_onepass_extract", _elapsed_onepass)
        _check_control()

        raw_entities = structured.get("entities") or []
        raw_relations = structured.get("relations") or []
        if verbose or verbose_steps:
            wprint_info(
                f"【步骤2s】单遍结构化抽取｜{len(raw_entities)}实体 {len(raw_relations)}关系"
                f"｜{_elapsed_onepass:.1f}s"
            )

        # ------------------------------------------------------------------
        # 2. 实体后处理：名称归一去重 + 内容回退 + 质量门 + 紧急保留
        # ------------------------------------------------------------------
        _progress(0.40, f"{_win} · S3s: 实体质量门", f"{len(raw_entities)} 个候选")
        _t = _time.time()
        _prose_index = None

        def _get_prose_index():
            nonlocal _prose_index
            if _prose_index is None:
                _prose_index = _ProseIndex(_prepare_prose_sentences(input_text))
            return _prose_index

        _fallback_cache: Dict[str, str] = {}

        def _cached_fallback(name: str) -> str:
            if name not in _fallback_cache:
                _fallback_cache[name] = _build_entity_fallback_content(name, _get_prose_index())
            return _fallback_cache[name]

        # 名称归一（复用现管线的别名/大小写合并）
        _content_by_name: Dict[str, str] = {}
        for e in raw_entities:
            _content_by_name.setdefault(e["name"], e.get("content", ""))
        entity_names = _normalize_and_dedup_entity_names(list(_content_by_name.keys()))
        if getattr(self, "remember_profile", "current") in ("quality-v1", "strong-v1"):
            from .quality import filter_entity_names
            # strong-v1：名称落地由 prompt 硬约束负责，这里不做逐字落地硬杀
            # （矩阵 C 教训：kimi 把英文概念译成中文名，落地检查把 4/6 实体杀光）
            entity_names = filter_entity_names(
                entity_names, input_text,
                limit=getattr(self, "remember_max_entities_per_window", 16),
                require_grounding=getattr(self, "remember_profile", "current") != "strong-v1",
            )

        extracted_entities = []
        valid_entities = []
        rejected_entities = []
        for name in entity_names:
            content = _content_by_name.get(name, "") or ""
            if not content or len(content) < _MIN_ENTITY_CONTENT_LEN:
                content = _cached_fallback(name)
            extracted_entities.append({"name": name, "content": content})
            if _validate_entity(name, content):
                valid_entities.append({"name": name, "content": content})
            else:
                rejected_entities.append((name, content))

        # 紧急保留：全部被过滤时宁可保留不完美实体，不丢窗口
        if not valid_entities and extracted_entities:
            for e in extracted_entities:
                valid_entities.append(e)

        _record_timing("step3s_entity_gate", _time.time() - _t)
        if verbose or verbose_steps:
            _rejected_n = len(extracted_entities) - len(valid_entities)
            wprint_info(f"【步骤3s】实体质量门｜{len(valid_entities)}通过"
                        + (f"，{_rejected_n}被过滤" if _rejected_n else "")
                        + f"｜{_time.time() - _t:.1f}s")

        if early_entity_done_fn:
            early_entity_done_fn(valid_entities)

        _check_control()

        # ------------------------------------------------------------------
        # 3. 关系后处理：端点归一 + 去重 + 内容模板回退 + 质量门
        # ------------------------------------------------------------------
        _progress(0.70, f"{_win} · S4s: 关系质量门", f"{len(raw_relations)} 条候选")
        _t = _time.time()
        _name_lookup = _build_name_lookup(set(entity_names))
        _entity_name_set = set(entity_names)
        relation_pairs: List[Tuple[str, str]] = []
        _pair_contents: Dict[Tuple[str, str], str] = {}
        seen_pairs = set()
        for r in raw_relations:
            a = _resolve_entity_name(
                r.get("entity1_name", ""), _entity_name_set, _lookup=_name_lookup)
            b = _resolve_entity_name(
                r.get("entity2_name", ""), _entity_name_set, _lookup=_name_lookup)
            if not a or not b or a == b:
                continue
            pk = _pair_key(a, b)
            if pk in seen_pairs:
                continue
            seen_pairs.add(pk)
            relation_pairs.append((a, b))
            _pair_contents[pk] = str(r.get("content", "") or "")

        if getattr(self, "remember_profile", "current") in ("quality-v1", "strong-v1"):
            from .quality import cap_relation_pairs
            relation_pairs = cap_relation_pairs(
                relation_pairs,
                limit=getattr(self, "remember_max_relations_per_window", 24),
            )

        valid_relations = []
        for a, b in relation_pairs:
            content = _pair_contents.get(_pair_key(a, b), "") or ""
            if not content or len(content) < _MIN_RELATION_CONTENT_LEN:
                content = (
                    f"{a} is related to {b}." if getattr(self, "preserve_source_language", False)
                    else f"{a}与{b}存在关联"
                )
            if _validate_relation(a, b, content, _entity_name_set):
                valid_relations.append({
                    "entity1_name": a, "entity2_name": b, "content": content,
                })

        _record_timing("step4s_relation_gate", _time.time() - _t)
        if verbose or verbose_steps:
            wprint_info(f"【步骤4s】关系质量门｜{len(valid_relations)}条通过｜{_time.time() - _t:.1f}s")

        _progress(0.95, f"{_win} · 完成",
                  f"{len(valid_entities)} 实体, {len(valid_relations)} 关系")
        return valid_entities, valid_relations
    finally:
        self.llm_client.clear_cancel_check()
