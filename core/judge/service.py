"""AlignmentJudgeService：对齐判断的 memo / single-flight / 攒批 门面。

接入方式：registry 惰性建一个库级实例，经 processor 传给所有 LLMClient
（`llm_client.judge_service`）。LLMClient 的三个判断方法入口先查本服务，
服务未启用（judge_service=None）时完全走原路径——矩阵 A/B 的开关即此。

三个 namespace：
- guard        实体对齐 guard（judge_entity_alignment）
- resolve_ent  实体候选批量裁决（resolve_entity_candidates_batch）
- resolve_rel  关系对批量裁决（resolve_relation_pair_batch）

错误语义：raw 调用返回带 "error" 键的 dict（现有降级路径）时照常返回但
不写 memo；抛异常时经 single-flight 传递，follower 收到 MISS 后自行直调。
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from . import models
from .collector import BatchCollector
from .memo import VerdictMemo
from .singleflight import SingleFlight

logger = logging.getLogger(__name__)

NS_GUARD = "guard"
NS_RESOLVE_ENT = "resolve_ent"
NS_RESOLVE_REL = "resolve_rel"


class AlignmentJudgeService:
    def __init__(self, memo: VerdictMemo,
                 *, batch_delay_ms: int = 200, batch_max: int = 32):
        self._memo = memo
        self._singleflight = SingleFlight()
        self._collector = BatchCollector(
            batch_delay_ms=batch_delay_ms, batch_max=batch_max)
        self._stats_lock = threading.Lock()
        self._stats = {
            "requests": 0, "memo_hits": 0, "llm_calls": 0,
            "singleflight_miss_fallbacks": 0, "invalidations": 0,
        }

    # ------------------------------------------------------------------
    # 三个判断入口（由 LLMClient 方法入口委托进来）
    # ------------------------------------------------------------------
    def judge_entity_alignment(self, llm_client, name_a: str, content_a: str,
                               name_b: str, content_b: str,
                               *, name_match_type: str = "none") -> Dict[str, Any]:
        key = models.guard_key(name_a, content_a, name_b, content_b, name_match_type)
        families = None  # guard 入参没有 family_id，失效依赖名称级失效与 TTL
        return self._run(NS_GUARD, key, lambda: llm_client._judge_entity_alignment_llm(
            name_a, content_a, name_b, content_b, name_match_type=name_match_type,
        ), families)

    def resolve_entity_candidates(self, llm_client, current_entity: Dict[str, Any],
                                  candidates: List[Dict[str, Any]],
                                  context_text: Optional[str] = None) -> Dict[str, Any]:
        key = models.resolve_entity_key(current_entity, candidates, context_text)
        families = models.families_touched(current_entity, candidates)
        return self._run(NS_RESOLVE_ENT, key, lambda: llm_client._resolve_entity_candidates_llm(
            current_entity, candidates, context_text=context_text,
        ), families)

    def resolve_relation_pair(self, llm_client, entity1_name: str,
                              entity2_name: str, new_relation_contents: List[str],
                              existing_relations: List[Dict[str, Any]],
                              new_source_document: str = "") -> Dict[str, Any]:
        key = models.resolve_relation_key(
            entity1_name, entity2_name, new_relation_contents,
            existing_relations, new_source_document)
        families = models.families_touched_for_relation(existing_relations)
        return self._run(NS_RESOLVE_REL, key, lambda: llm_client._resolve_relation_pair_llm(
            entity1_name, entity2_name, new_relation_contents,
            existing_relations, new_source_document=new_source_document,
        ), families)

    # ------------------------------------------------------------------
    def _run(self, ns: str, key: str, raw_fn, families: Optional[List[str]]) -> Dict[str, Any]:
        with self._stats_lock:
            self._stats["requests"] += 1
        cached = self._memo.get(ns, key)
        if cached is not None:
            with self._stats_lock:
                self._stats["memo_hits"] += 1
            return cached

        def _execute() -> Dict[str, Any]:
            with self._stats_lock:
                self._stats["llm_calls"] += 1
            return self._collector.submit(raw_fn)

        try:
            result = self._singleflight.execute(key, _execute)
        except Exception:
            # leader 失败直接上抛（raw 内部已有降级，走到这里的多为意外错误）；
            # follower 不会进这个分支——它们拿到 MISS_SENTINEL 返回值
            raise
        if result is SingleFlight.MISS_SENTINEL:
            # follower 且 leader 失败：直调一次兜底
            with self._stats_lock:
                self._stats["singleflight_miss_fallbacks"] += 1
                self._stats["llm_calls"] += 1
            result = raw_fn()
        if isinstance(result, dict) and "error" not in result:
            self._memo.put(ns, key, result, family_ids=families)
        return result

    # ------------------------------------------------------------------
    def invalidate_for_family(self, family_id: str, names: Optional[List[str]] = None) -> None:
        """实体合并后清除涉及该 family 的缓存判断。"""
        if not family_id:
            return
        with self._stats_lock:
            self._stats["invalidations"] += 1
        self._memo.invalidate_for_families([str(family_id)])

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            out = dict(self._stats)
        out["memo"] = self._memo.stats()
        out["collector"] = self._collector.stats()
        out["singleflight"] = self._singleflight.stats()
        return out

    def close(self) -> None:
        self._collector.close()
        self._memo.close()
