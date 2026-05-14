"""Contradiction detection and summary evolution sub-mixin for _PipelineExtractionMixin."""
from __future__ import annotations

import asyncio
import threading
from typing import List
from concurrent.futures import ThreadPoolExecutor

_HIGH_MEDIUM_SEVERITY = frozenset(("high", "medium"))

# Shared pool for contradiction detection and summary evolution (avoids per-call thread churn)
_alignment_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="align")

from core.utils import wprint_info


class _ContradictionMixin:
    """Auto contradiction detection + summary evolution methods."""

    def _detect_and_apply_contradictions(self, family_ids: List[str], verbose: bool = False,
                                          pre_fetched_versions=None):
        """对多版本实体运行矛盾检测，发现高严重性矛盾时自动降低置信度。

        这是 remember 流水线的自动矛盾检测步骤：
        1. 使用预获取或批量获取所有 family_id 的版本历史
        2. 并行调用 LLM detect_contradictions 检测矛盾
        3. 对 medium/high 严重性矛盾调用 adjust_confidence_on_contradiction
        """
        # 使用预获取的版本数据，或批量获取
        all_versions = pre_fetched_versions
        if all_versions is None:
            batch_fn = getattr(self.storage, 'get_entity_versions_batch', None)
            if batch_fn:
                try:
                    all_versions = batch_fn(family_ids)
                except Exception:
                    all_versions = None

        # 构建待检测列表（跳过版本不足的）
        to_check = []
        for fid in family_ids:
            versions = (all_versions or {}).get(fid) if all_versions is not None else None
            if versions is None:
                try:
                    versions = self.storage.get_entity_versions(fid)
                except Exception:
                    continue
            if len(versions) >= 2:
                to_check.append((fid, versions))

        if not to_check:
            return

        # 并行检测矛盾
        n_workers = min(len(to_check), getattr(self, 'llm_threads', 2))

        def _detect_one(item):
            fid, versions = item
            try:
                return (fid, self.llm_client.detect_contradictions(fid, versions))
            except Exception as e:
                if verbose:
                    wprint_info(f"【矛盾检测】{fid}: 检测失败 ({e})")
                return (fid, None)

        if n_workers > 1 and len(to_check) > 1:
            results = list(_alignment_pool.map(_detect_one, to_check))
        else:
            results = [_detect_one(item) for item in to_check]

        # Batch apply confidence adjustments
        _to_downgrade = []
        for fid, contradictions in results:
            if not contradictions:
                continue
            high_severity = [c for c in contradictions if c.get("severity") in _HIGH_MEDIUM_SEVERITY]
            if high_severity:
                _to_downgrade.append(fid)
                if verbose:
                    wprint_info(f"【矛盾检测】{fid}: 发现 {len(high_severity)} 个中/高严重性矛盾，降低置信度")
        if _to_downgrade:
            try:
                batch_fn = getattr(self.storage, 'adjust_confidence_on_contradiction_batch', None)
                if batch_fn:
                    batch_fn(_to_downgrade, source_type="entity")
                else:
                    for fid in _to_downgrade:
                        self.storage.adjust_confidence_on_contradiction(fid, source_type="entity")
            except Exception:
                pass

    # =========================================================================
    # 自动摘要进化
    # =========================================================================
    SUMMARY_EVOLVE_MIN_VERSIONS = 3  # 至少 3 个版本才触发摘要进化

    def _auto_evolve_summaries(self, family_ids: List[str], verbose: bool = False,
                               pre_fetched_versions=None):
        """对版本数足够的实体自动进化摘要。

        当实体积累了多个版本后，其 _extract_summary (首行截断) 已无法反映完整信息。
        此方法调用 LLM 生成综合性摘要，覆盖存储中的 summary 字段。

        阈值：version_count >= SUMMARY_EVOLVE_MIN_VERSIONS
        """

        # 使用预获取的版本数据，或批量获取
        all_versions_map = pre_fetched_versions
        if all_versions_map is None:
            batch_fn = getattr(self.storage, 'get_entity_versions_batch', None)
            if batch_fn:
                try:
                    all_versions_map = batch_fn(family_ids)
                except Exception:
                    all_versions_map = None

        # 收集需要进化的实体（过滤掉不需要的）
        to_evolve = []
        for fid in family_ids:
            try:
                if all_versions_map is not None:
                    versions = all_versions_map.get(fid, [])
                else:
                    versions = self.storage.get_entity_versions(fid)
                if len(versions) < self.SUMMARY_EVOLVE_MIN_VERSIONS:
                    continue

                # versions 按 processed_time ASC 排序，最新在末尾
                current = versions[-1] if versions else None
                if not current:
                    continue
                old_version = versions[-2] if len(versions) > 1 else None

                # 检查当前 summary 是否已经是 LLM 生成的高质量摘要
                existing_summary = getattr(current, 'summary', '') or ''
                if len(existing_summary) > 50:
                    if old_version and old_version.content == current.content:
                        continue  # 内容未变，无需进化

                to_evolve.append((fid, current, old_version))
            except Exception:
                continue

        if not to_evolve:
            return

        # 并行进化摘要
        n_workers = min(len(to_evolve), getattr(self, 'llm_threads', 2))

        # Per-thread persistent event loop — avoids asyncio.run() overhead
        # (asyncio.run creates/destroys an event loop per call)
        _local_loop = threading.local()

        def _get_loop():
            loop = getattr(_local_loop, 'loop', None)
            if loop is None or loop.is_closed():
                loop = asyncio.new_event_loop()
                _local_loop.loop = loop
            return loop

        def _evolve_one(item):
            fid, current, old_version = item
            try:
                loop = _get_loop()
                summary = loop.run_until_complete(
                    self.llm_client.evolve_entity_summary(current, old_version)
                )
                return (fid, current, summary)
            except Exception as e:
                if verbose:
                    wprint_info(f"【摘要进化】{fid}: 进化失败 ({e})")
                return (fid, current, None)

        if n_workers > 1 and len(to_evolve) > 1:
            results = list(_alignment_pool.map(_evolve_one, to_evolve))
        else:
            results = [_evolve_one(item) for item in to_evolve]

        # Batch write summaries to DB
        summary_updates = {}
        for fid, current, summary in results:
            if summary:
                _s = summary.strip()
                if _s:
                    summary_updates[fid] = _s
                if verbose:
                    wprint_info(f"【摘要进化】{fid} ({current.name}): 摘要已更新")
        if summary_updates:
            batch_fn = getattr(self.storage, 'batch_update_entity_summaries', None)
            if batch_fn:
                try:
                    batch_fn(summary_updates)
                except Exception:
                    for fid, summary in summary_updates.items():
                        try:
                            self.storage.update_entity_summary(fid, summary)
                        except Exception:
                            pass
            else:
                for fid, summary in summary_updates.items():
                    try:
                        self.storage.update_entity_summary(fid, summary)
                    except Exception:
                        pass
