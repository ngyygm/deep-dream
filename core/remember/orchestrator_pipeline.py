"""
Pipeline mixin: remember_text entry point, lifecycle helpers, and standalone main().
"""
from functools import wraps
from typing import Callable, Dict, Optional
from datetime import datetime
import hashlib
import logging
import threading
import time
import uuid

from core.log import info as _log_info
from core.storage.sqlite.repositories import pipeline as pipeline_repo
from core.text_chunking import apply_document_metadata_prefix
from core.utils import (
    clear_parallel_log_context,
    classify_episode_type,
    compute_doc_hash,
    set_pipeline_role,
    set_window_label,
    wprint_info,
)
from .helpers import dedupe_extraction_lists
from .strong_steps import strong_extract_only

logger = logging.getLogger(__name__)


def _clear_pipeline_context(method):
    """Ensure per-run storage context is cleared on every exit path.

    ``remember_text`` has several early returns and control-flow exceptions
    (pause/cancel/retry).  Keeping cleanup in a small wrapper prevents a
    failed run from leaking its run/document identifiers into the next task.
    """
    @wraps(method)
    def _wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            storage = getattr(self, "storage", None)
            if storage is not None:
                try:
                    storage._current_run_id = ""
                except Exception:
                    logger.debug("failed to clear pipeline run context", exc_info=True)
            # This attribute is consumed by save_episode while the method is
            # running; it must not survive an early return or exception.
            try:
                self._pipeline_override_doc_id = ""
            except Exception:
                pass
    return _wrapped


class _PipelineMixin:
    """Mixin that provides the remember_text pipeline and lifecycle methods."""

    def _recompute_document_publish_windows(self, *, last_episode_id: str,
                                            override_doc_id: str, total_chunks: int,
                                            failed_window_indices, successful_window_indices,
                                            default_document_id: str):
        """成功/暂停 epilogue 共用：从持久化 episodes 重算文档完整窗口集合。

        Recompute publication from the document's persisted active windows,
        rather than assuming that a targeted repair implies every other window
        already exists — a new document repaired at W0 must remain hidden until
        W1..Wn are actually present.  反过来，target_window_indices/start_chunk
        限定的一次 run 没碰到的历史窗口，只要已有 active episode 就不算
        missing：暂停/修复不得把已完整入库的文档整体降级出搜索。

        Returns:
            (document_id, active_window_indices)。本次 run 失败的窗口从
            active 集合剔除；持久化查询不可用时保守退回本 run 的成功窗口
            （绝不把未处理窗口当完整）。
        """
        _publish_document_id = default_document_id
        _active_window_indices = set()
        try:
            _doc_row = None
            if last_episode_id:
                _doc_row = self.storage._conn().execute(
                    "SELECT document_id, document_version_id FROM episodes "
                    "WHERE episode_id = ?", (last_episode_id,)
                ).fetchone()
            if _doc_row is None and override_doc_id:
                _doc_row = self.storage._conn().execute(
                    "SELECT document_id, current_version_id FROM documents "
                    "WHERE document_id = ?", (override_doc_id,)
                ).fetchone()
            if _doc_row:
                _publish_document_id = _doc_row[0] or _publish_document_id
                _ver_id = _doc_row[1]
                if _ver_id:
                    _rows = self.storage._conn().execute(
                        "SELECT chunk_index FROM episodes "
                        "WHERE document_version_id = ? AND status = 'active'",
                        (_ver_id,),
                    ).fetchall()
                    _active_window_indices = {
                        int(r[0]) for r in _rows
                        if r[0] is not None and 0 <= int(r[0]) < total_chunks
                    }
        except Exception:
            logger.debug("failed to recompute persisted ingestion windows", exc_info=True)

        _active_window_indices.difference_update(failed_window_indices or ())
        # If the storage lookup was unavailable, retain the conservative
        # local successes only; never claim unprocessed windows complete.
        if not _active_window_indices:
            _active_window_indices = set(successful_window_indices or ())
        return _publish_document_id, _active_window_indices

    def _publish_final_ingestion_state(self, *, set_publish_state, last_episode_id: str,
                                       override_doc_id: str, total_chunks: int,
                                       failed_window_indices, successful_window_indices,
                                       default_document_id: str):
        """成功/暂停 epilogue 共用：按重算的完整窗口集合写 ingestion state。"""
        if not set_publish_state:
            return
        _publish_document_id, _active_window_indices = self._recompute_document_publish_windows(
            last_episode_id=last_episode_id,
            override_doc_id=override_doc_id,
            total_chunks=total_chunks,
            failed_window_indices=failed_window_indices,
            successful_window_indices=successful_window_indices,
            default_document_id=default_document_id,
        )
        _expected_windows = set(range(total_chunks))
        _missing_windows = sorted(_expected_windows - _active_window_indices)
        _complete_windows = len(_expected_windows & _active_window_indices)
        _publish_state = "active" if not _missing_windows else "incomplete"
        set_publish_state(
            _publish_document_id, _publish_state, total_windows=total_chunks,
            complete_windows=_complete_windows,
            missing_windows=_missing_windows,
        )

    @_clear_pipeline_context
    def remember_text(self, text: str, doc_name: str = "", verbose: bool = False,
                      verbose_steps: bool = True,
                      load_cache_memory: Optional[bool] = None,
                      event_time: Optional[datetime] = None,
                      document_path: str = "",
                      progress_callback: Optional[Callable] = None,
                      control_callback: Optional[Callable[[], Optional[str]]] = None,
                      start_chunk: int = 0,
                      main_chunk_done_callback: Optional[Callable] = None,
                      step9_chunk_done_callback: Optional[Callable] = None,
                      chunk_done_callback: Optional[Callable] = None,
                      source_document: Optional[str] = None,
                      target_window_indices: Optional[list] = None,
                      override_doc_id: str = "") -> Dict:
        """
        将一段文本作为记忆入库：流水线式并行处理 step9（实体对齐）和 step10（关系对齐）。

        流水线架构：
        - 主线程：Phase A（step1 串行更新缓存）+ 提交 Phase B（step2-8 并行抽取）
        - step9 线程：等待当前窗口 step2-8 完成 + 前一窗口 step9 完成 → 实体对齐
        - step10 线程：等待当前窗口 step9 完成 + 前一窗口 step10 完成 → 关系对齐
        - step9 W(i+1) 可与 step10 W(i) 并行执行

        Args:
            text: 原始文本内容
            doc_name: 文档/来源名称
            verbose: 是否打印详细处理日志（步骤内细节、LLM 提示等）
            verbose_steps: 是否在控制台输出步骤级「开始/结束」汇报（verbose=True 时仍生效，但以详细日志为准）
                并行时控制台行格式为 [窗号][角色] 正文；角色为 主线程 / 抽取 / 步骤9 / 步骤10 之一。
            load_cache_memory: 是否在开始前加载最新缓存记忆再追加
            event_time: 事件实际发生时间
            document_path: 原文文件路径
            progress_callback: 进度回调 fn(progress, phase_label, message, chain_id)
            control_callback: 控制回调 fn() -> {"pause","cancel",None}，在窗口级安全点生效
            start_chunk: 从第几个窗口开始（关系链断点续传）
            main_chunk_done_callback: 步骤1–5 完成一个窗口后的回调 fn(processed_count)
            step9_chunk_done_callback: 步骤9 完成一个窗口后的回调 fn(processed_count)
            chunk_done_callback: 步骤10 完成一个窗口后的回调 fn(processed_count)
            source_document: 来源文档名称（优先于 doc_name）

        Returns:
            dict: episode_id, chunks_processed, storage_path
        """
        # Import here to avoid circular dependency — orchestrator exports these
        from .orchestrator import RememberControlFlow

        doc_name = source_document or doc_name

        # 每次入库独立计数：入口清零，结束随 result 返回（processor 共享计数器）
        self.reset_llm_call_stats()

        # Input validation: reject empty or whitespace-only text early.
        if not text or not text.strip():
            return {
                "episode_id": None,
                "chunks_processed": 0,
                "storage_path": str(self.storage.storage_path),
                "entities": 0,
                "relations": 0,
                "warnings": [{"phase": "input_validation", "error": "text is empty or whitespace-only"}],
            }

        use_load_cache = load_cache_memory if load_cache_memory is not None else self.load_cache_memory
        # 仅在真正的断点续传（start_chunk > 0）时加载已有缓存链；
        # start_chunk == 0 表示从头开始，加载旧缓存会导致 step1 重复处理已有内容
        if use_load_cache and start_chunk > 0:
            latest_metadata = self.storage.get_latest_episode_metadata(activity_type="文档处理")
            if latest_metadata:
                self.current_episode = self.storage.load_episode(latest_metadata["episode_id"])
                if verbose and self.current_episode:
                    _log_info("Remember",
                        f"已加载缓存记忆: {self.current_episode.absolute_id}，"
                        f"将在此链上追加（断点续传 start_chunk={start_chunk}）"
                    )
                elif verbose_steps and self.current_episode:
                    _log_info("Remember","已加载缓存记忆（断点续传）")
            else:
                self.current_episode = None
        else:
            self.current_episode = None
            if start_chunk == 0 and use_load_cache:
                if verbose:
                    _log_info("Remember","start_chunk=0，从头开始处理，不加载旧缓存链")
                elif verbose_steps:
                    _log_info("Remember","从头开始处理（不加载旧缓存链）")

        if not document_path:
            document_path = f"api://{uuid.uuid4().hex}"
        # Store override_doc_id for save_episode to use
        self._pipeline_override_doc_id = override_doc_id
        total_length = len(text)
        chunks = self.document_processor.chunk_text(text)
        total_chunks = len(chunks)
        # P3.6 窗口哈希只算一次：run 内已对每个窗口计算 chunk 哈希（缓存
        # 查找复用同一值），按绝对窗口索引收集后随 result 传给 task_queue
        # 的修复检测/完整性检查，避免同一文档被重复 chunk_text。
        _window_hashes: list = [None] * total_chunks

        # Generate a run_id for this pipeline invocation
        _run_id = uuid.uuid4().hex
        # Expose run_id on storage so save_entity/save_relation can pick it up
        # without threading it through 9+ layers of function calls.
        self.storage._current_run_id = _run_id

        # 所有窗口已处理完毕（断点续传恢复后无需重跑）
        if start_chunk >= total_chunks and not target_window_indices:
            return {
                "episode_id": getattr(self.current_episode, 'absolute_id', None),
                "chunks_processed": total_chunks,
                "storage_path": str(self.storage.storage_path),
            }

        # Publish atomically: save_episode may write intermediate rows, but all
        # search views keep this document invisible until every window finishes.
        _atomic_document_id = override_doc_id or (
            "doc_" + hashlib.sha256(
                (document_path or doc_name or text[:64]).encode()
            ).hexdigest()[:16]
        )
        _set_publish_state = getattr(self.storage, "set_document_ingestion_state", None)
        if _set_publish_state:
            _set_publish_state(
                _atomic_document_id, "processing", total_windows=total_chunks,
                complete_windows=0, missing_windows=range(total_chunks),
            )

        # Targeted mode: only process specific window indices
        if target_window_indices is not None:
            _sorted_targets = sorted(target_window_indices)
            N = len(_sorted_targets)
            def _local_to_abs(i):
                return _sorted_targets[i]
            start_chunk = 0  # targeting handles absolute index mapping
        else:
            _sorted_targets = None
            def _local_to_abs(i):
                return start_chunk + i
            N = total_chunks - start_chunk  # 待处理窗口数
        last_episode_id = None
        # Pre-compute absolute window indices for workers
        _window_abs_indices = [_local_to_abs(i) for i in range(N)]
        clear_parallel_log_context()

        # Record pipeline run
        try:
            pipeline_repo.insert_pipeline_run(
                self.storage._conn(), _run_id, "remember", "running",
                started_at=datetime.now().isoformat(),
            )
            self.storage._commit_if_not_batched(self.storage._conn())
        except Exception:
            logger.debug("pipeline_runs insert failed (non-critical)", exc_info=True)

        # 预分配共享状态
        state = self._init_remember_shared_state(N)

        # 暴露 state 引用供 get_pipeline_snapshot() 读取
        with self._current_state_lock:
            self._current_state = state

        # 启动 step9 / step10 线程
        t9 = threading.Thread(target=self._run_step9_worker, name="tmg-step9-chain", daemon=True,
                              args=(state, _window_abs_indices, total_chunks, doc_name, verbose, verbose_steps,
                                    event_time, progress_callback, step9_chunk_done_callback,
                                    control_callback))
        t10 = threading.Thread(target=self._run_step10_worker, name="tmg-step10-chain", daemon=True,
                              args=(state, _window_abs_indices, total_chunks, doc_name, verbose, verbose_steps,
                                    event_time, progress_callback, chunk_done_callback,
                                    control_callback))
        t9.start()
        t10.start()

        if verbose or verbose_steps:
            _log_info("Remember",
                f"流水线启动｜{total_chunks}窗口×{N}待处理｜并发={self._max_concurrent_windows}｜"
                f"step1串行→step2-8并行→step9/10链式｜"
                f"{'注意: window_workers=1 时流水线完全串行，窗口2必须等窗口1全部完成' if self._max_concurrent_windows <= 1 else ''}"
            )

        # ========== 主线程：Phase A（step1 串行）+ 提交 Phase B（step2-8）==========
        try:
            for ci in range(N):
                # Pipeline depth gate: wait for an earlier window's step10 to
                # finish so that at most ``_max_concurrent_windows`` windows are
                # active across *all* pipeline stages (step1 through step10).
                if ci >= self._max_concurrent_windows:
                    state.step10_done_ev[ci - self._max_concurrent_windows].wait()

                _action = self._poll_control(state, control_callback)
                if _action:
                    self._signal_control_stop(state, _action, ci)
                    break
                self._acquire_window_slot()
                _slot_acquired = True

                try:
                    _action = self._poll_control(state, control_callback)
                    if _action:
                        self._signal_control_stop(state, _action, ci)
                        self._release_window_slot()
                        _slot_acquired = False
                        break

                    _wi = _local_to_abs(ci)
                    _chunk_tuple = chunks[_wi]
                    if len(_chunk_tuple) >= 4:
                        chunk, start, end, _heading_path = _chunk_tuple[:4]
                    else:
                        chunk, start, end = _chunk_tuple[:3]
                        _heading_path = ""
                    # 不变式 a：window-0 元数据前缀与 task_queue 共享同一 helper，保持字节一致
                    chunk = apply_document_metadata_prefix(doc_name, chunk, _wi)

                    _wlabel = f"W{_wi + 1}/{total_chunks}"
                    if verbose:
                        set_window_label(_wlabel)
                        set_pipeline_role("主线程")
                        wprint_info(
                            f"【窗口】{_wlabel}｜{doc_name}｜[{start}-{end}/{total_length}] {len(chunk)}字"
                        )
                    elif verbose_steps:
                        set_window_label(_wlabel)
                        set_pipeline_role("主线程")
                        wprint_info(
                            f"【窗口】{_wlabel}｜{doc_name}｜[{start}-{end}/{total_length}]"
                        )

                    _g_lo = _wi / total_chunks
                    _g_hi = (_wi + 1) / total_chunks
                    _span = _g_hi - _g_lo
                    _p_after_step1 = _g_lo + _span * (1.0 / 10.0)
                    _p_end_main = _g_lo + _span * (8.0 / 10.0)
                    if progress_callback:
                        self._safe_progress(progress_callback,
                            _g_lo + _span * 0.02,
                            f"窗口 {_wi + 1}/{total_chunks} · 步骤1/10 进行中",
                            "", "main",
                        )

                    # Step1: 更新缓存
                    _t_step1_start = time.time()
                    _chunk_hash = compute_doc_hash(chunk)
                    _window_hashes[_wi] = _chunk_hash
                    _t_cache_lookup = time.time()
                    existing_mc, _saved_extraction = (
                        self.storage.find_cache_and_extraction_by_doc_hash(_chunk_hash, document_path=document_path)
                        if _chunk_hash else (None, None)
                    )
                    state.window_timings[ci]["step1-cache_lookup"] = time.time() - _t_cache_lookup
                    if existing_mc:
                        new_mc = existing_mc
                        self.current_episode = existing_mc
                        state.window_timings[ci]["step1-cache_hit"] = 1e-6
                        if _saved_extraction is None:
                            if verbose:
                                wprint_info("【步骤1】缓存｜命中｜跳过生成")
                            elif verbose_steps:
                                wprint_info("【步骤1】缓存｜命中｜跳过生成")
                    else:
                        with self._cache_lock:
                            def _run_step1():
                                return self._update_cache(
                                    chunk, doc_name,
                                    text_start_pos=start, text_end_pos=end,
                                    total_text_length=total_length, verbose=verbose,
                                    verbose_steps=verbose_steps,
                                    document_path=document_path, event_time=event_time,
                                    window_index=_wi + 1, total_windows=total_chunks,
                                    doc_hash=_chunk_hash,
                                    heading_path=_heading_path,
                                    episode_type=classify_episode_type(chunk),
                                    run_id=_run_id,
                                )

                            _t_cache_write = time.time()
                            new_mc = self._run_with_progress_heartbeat(
                                _run_step1,
                                chain_id="main",
                                base_progress=_g_lo + _span * 0.02,
                                phase_label=f"窗口 {_wi + 1}/{total_chunks} · 步骤1/10 进行中",
                                message="步骤1 更新记忆缓存",
                                window_label=_wlabel,
                                pipeline_role="主线程",
                                progress_callback=progress_callback,
                            )
                            state.window_timings[ci]["step1-update_cache"] = time.time() - _t_cache_write
                    _step1_elapsed = time.time() - _t_step1_start
                    state.window_timings[ci]["step1"] = _step1_elapsed
                    if verbose or verbose_steps:
                        wprint_info(f"【步骤1】完成｜{_step1_elapsed:.1f}s")
                    state.episodes[ci] = new_mc
                    state.input_texts[ci] = chunk
                    last_episode_id = new_mc.absolute_id

                    _action = self._poll_control(state, control_callback)
                    if _action:
                        self._signal_control_stop(state, _action, ci + 1)
                        state.entity_content_done[ci].set()
                        state.extract_done[ci].set()
                        state.step9_done_ev[ci].set()
                        state.step10_done_ev[ci].set()
                        self._release_window_slot()
                        _slot_acquired = False
                        break

                    # 提交 step2-5
                    if _saved_extraction is not None:
                        _dedup_ents, _dedup_rels = dedupe_extraction_lists(
                            _saved_extraction[0], _saved_extraction[1]
                        )
                        state.extract_results[ci] = (_dedup_ents, _dedup_rels)
                        state.early_entity_results[ci] = _dedup_ents
                        state.window_timings[ci]["step2-8"] = 0.0
                        state.entity_content_done[ci].set()
                        state.extract_done[ci].set()
                        if main_chunk_done_callback:
                            main_chunk_done_callback(_wi + 1)
                        self._release_window_slot()
                        _slot_acquired = False
                        if progress_callback:
                            self._safe_progress(progress_callback,
                                _p_end_main,
                                f"窗口 {_wi + 1}/{total_chunks} · 步骤1–8/10 已完成(缓存)",
                                "", "main",
                            )
                        if verbose:
                            _ents_count = len(_dedup_ents)
                            _rels_count = len(_dedup_rels)
                            if existing_mc:
                                wprint_info(
                                    f"【步骤1–5】缓存｜命中｜实体{_ents_count} 关系{_rels_count}→步骤9"
                                )
                            else:
                                wprint_info(
                                    f"【步骤2–8】缓存｜命中｜实体{_ents_count} 关系{_rels_count}"
                                )
                        elif verbose_steps:
                            if existing_mc:
                                wprint_info(
                                    f"窗口 {_wi + 1}/{total_chunks} · 步骤1–8 已缓存跳过 → 步骤9/10"
                                )
                            else:
                                wprint_info("【步骤2–8】缓存｜跳过｜抽取已存在")
                    else:
                        if progress_callback:
                            self._safe_progress(progress_callback,
                                _p_after_step1,
                                f"窗口 {_wi + 1}/{total_chunks} · 步骤1/10 完成",
                                "", "main",
                            )

                        def _do_extract(idx=ci, mc=new_mc, chunk_text=chunk, __hash=_chunk_hash):
                            _abs_idx = _local_to_abs(idx)
                            _wlabel = f"W{_abs_idx + 1}/{total_chunks}"
                            set_window_label(_wlabel)
                            set_pipeline_role("抽取")
                            _success_main = False
                            _t_extract_start = time.time()
                            with self._runtime_lock:
                                self._active_window_extractions += 1
                                self._peak_window_extractions = max(
                                    self._peak_window_extractions,
                                    self._active_window_extractions,
                                )
                            def _early_entity_cb(valid_entities):
                                state.early_entity_results[idx] = valid_entities
                                state.entity_content_done[idx].set()
                            try:
                                _idx_lo = _abs_idx / total_chunks
                                _idx_hi = (_abs_idx + 1) / total_chunks
                                _idx_span = _idx_hi - _idx_lo
                                ents, rels = strong_extract_only(
                                    self, mc, chunk_text, doc_name,
                                    verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                                    progress_callback=lambda p, label, m: self._safe_progress(progress_callback, p, label, m, "main"),
                                    progress_range=(
                                        _idx_lo + _idx_span * (1.0 / 10.0),
                                        _idx_lo + _idx_span * (8.0 / 10.0),
                                    ),
                                    window_index=_abs_idx, total_windows=total_chunks,
                                    window_timings_ref=state.window_timings[idx],
                                    control_check_fn=lambda _s=state, _cb=control_callback: self._poll_control(_s, _cb),
                                    early_entity_done_fn=_early_entity_cb,
                                )
                                state.extract_results[idx] = (ents, rels)
                                self.storage.save_extraction_result(__hash, ents, rels, document_path=document_path)
                                _success_main = True
                                _extract_elapsed = time.time() - _t_extract_start
                                state.window_timings[idx]["step2-8"] = _extract_elapsed
                                if verbose or verbose_steps:
                                    wprint_info(f"【步骤2–8】完成｜{_extract_elapsed:.1f}s")
                            except Exception as e:
                                if isinstance(e, RememberControlFlow):
                                    self._signal_control_stop(state, e.remember_control_action, idx)
                                    # pause/cancel 是正常控制流，不记录为错误
                                elif self._record_window_error(state, "extract", idx, e):
                                    logger.error("extract window %d error: %s", idx, e, exc_info=True)
                            finally:
                                with self._runtime_lock:
                                    self._active_window_extractions = max(0, self._active_window_extractions - 1)
                                # Ensure entity_content_done is always set to prevent step9 deadlock
                                state.entity_content_done[idx].set()
                                state.extract_done[idx].set()
                                # 无论 extraction 成功或失败都推进 main 进度：
                                # 失败时 step9 仍会用 partial data 处理，前端需要看到正确位置
                                if main_chunk_done_callback:
                                    main_chunk_done_callback(_abs_idx + 1)
                                self._release_window_slot()
                                clear_parallel_log_context()

                        try:
                            self._extraction_executor.submit(_do_extract)
                        except RuntimeError:
                            _do_extract()
                        _slot_acquired = False

                finally:
                    if _slot_acquired:
                        self._release_window_slot()
        except Exception as e:
            with state.errors_lock:
                state.errors.append(("main", 0, e))
            logger.error("main pipeline error: %s", e, exc_info=True)
            # Signal remaining windows so step9/10 threads and the wait loop
            # below don't hang forever when the main pipeline dies mid-way.
            _crash_ci = ci if 'ci' in dir() else 0
            self._signal_control_stop(state, None, _crash_ci)
            _main_pipeline_exc = e
        else:
            _main_pipeline_exc = None
        finally:
            clear_parallel_log_context()

        # 等待所有窗口 step10 完成（异常后仍需等待，否则 step9/10 线程可能写已释放的 state）
        for i in range(N):
            state.step10_done_ev[i].wait()

        # P3.3：run 结束释放候选表 run 级投影缓存（所有 step9 已完成，此后不再
        # 构建候选表；成功/失败/控制流中断路径都经过这里）
        try:
            self.entity_processor.release_candidate_run_cache()
        except Exception:
            logger.debug("release candidate run cache failed (non-critical)", exc_info=True)

        # 无论成功还是异常，都清理 _current_state，避免残留上一个任务的快照
        with self._current_state_lock:
            self._current_state = None

        # Clean shutdown of prefetch executor with proper timeout
        try:
            state.prefetch_executor.shutdown(wait=True)
        except Exception as e:
            logger.warning("Prefetch executor shutdown failed: %s", e)
            try:
                state.prefetch_executor.shutdown(wait=False)
            except Exception:
                pass

        t9.join(timeout=60)
        if t9.is_alive():
            _log_info("Remember","警告: step9 线程在 join(60s) 超时后仍在运行")

        t10.join(timeout=60)
        if t10.is_alive():
            _log_info("Remember","警告: step10 线程在 join(60s) 超时后仍在运行")

        if state.control_state["action"] is not None:
            _action = state.control_state["action"]
            # Control-flow exceptions intentionally skip the normal success
            # epilogue.  Close the persistence loop here so a paused/cancelled
            # document is not left permanently in ``processing`` and its
            # pipeline_run does not remain ``running`` forever.
            try:
                # 完整性口径与成功 epilogue 相同：从持久化 episodes 重算。
                # 只按本次 run 的目标窗口断言完整性——run 之外的历史窗口
                # 只要有 active episode 就不算 missing，暂停/修复中的文档
                # 不会因本次 run 未跑完全库而被整体降级出搜索。
                _successful_window_indices = [
                    _local_to_abs(i) for i in range(N)
                    if state.step10_results[i] is not None
                ]
                _failed_window_indices = [
                    _local_to_abs(i) for i in range(N)
                    if state.step10_results[i] is None and state.window_failures[i] is not None
                ]
                self._publish_final_ingestion_state(
                    set_publish_state=_set_publish_state,
                    last_episode_id=last_episode_id,
                    override_doc_id=override_doc_id,
                    total_chunks=total_chunks,
                    failed_window_indices=_failed_window_indices,
                    successful_window_indices=_successful_window_indices,
                    default_document_id=_atomic_document_id,
                )
                pipeline_repo.update_pipeline_run_status(
                    self.storage._conn(), _run_id,
                    "paused" if _action == "pause" else "cancelled",
                    finished_at=datetime.now().isoformat(),
                    error=f"pipeline { _action } by control request",
                )
                self.storage._commit_if_not_batched(self.storage._conn())
            except Exception:
                logger.debug("failed to finalize controlled pipeline run", exc_info=True)
            raise RememberControlFlow(_action)

        # If the main pipeline crashed (not just individual window errors),
        # propagate the exception so the worker can retry.
        if _main_pipeline_exc is not None:
            try:
                pipeline_repo.update_pipeline_run_status(
                    self.storage._conn(), _run_id, "failed",
                    finished_at=datetime.now().isoformat(),
                    error=str(_main_pipeline_exc),
                )
                self.storage._commit_if_not_batched(self.storage._conn())
            except Exception:
                pass
            self.storage._current_run_id = ""
            if _set_publish_state:
                _set_publish_state(
                    _atomic_document_id, "failed", total_windows=total_chunks,
                    complete_windows=0, missing_windows=range(total_chunks),
                )
            raise _main_pipeline_exc

        # ========== Post-window cross-window dedup (always runs, even for N=1) ==========
        # Run cross-window dedup even when some windows failed -- partial results are valuable.
        _dedup_exc = None
        try:
            self._cross_window_dedup(state.align_results, verbose=verbose)
        except Exception as e:
            _dedup_exc = e
            logger.error("Cross-window dedup failed: %s", e, exc_info=True)
            _log_info("Remember",f"后处理｜跨窗口去重失败: {e}")

        # ========== ALIGN-V2：文档末全库收敛扫描（同名/别名 family 簇合并） ==========
        from core.remember.align_v2 import align_v2_enabled, doc_end_library_sweep
        if align_v2_enabled():
            try:
                _sweep_stats = doc_end_library_sweep(self, verbose=verbose or verbose_steps)
                if _sweep_stats.get("pairs_judged"):
                    _log_info("Remember",
                              f"align-v2 收敛扫描｜判定 {_sweep_stats['pairs_judged']} 对｜"
                              f"合并 {_sweep_stats['merged']} family｜{_sweep_stats['seconds']}s")
            except Exception as e:
                logger.error("align-v2 library sweep failed: %s", e, exc_info=True)
                _log_info("Remember", f"align-v2 收敛扫描失败: {e}")

        # ========== 计时汇总 ==========
        timing_summary = self._summarize_window_timings(state.window_timings)

        storage_path = str(self.storage.storage_path)
        total_entities = sum(state.aligned_entity_counts)
        total_relations = sum(
            len(rl) for rl in state.step10_results if rl is not None
        )
        _contiguous_done = 0
        for i in range(N):
            if state.step10_results[i] is None:
                break
            _contiguous_done += 1

        # Per-window absolute index tracking for targeted retry
        _abs_of = _local_to_abs  # set during main loop (normal or targeted)
        _successful_window_indices = []
        _failed_window_indices = []
        for i in range(N):
            _aidx = _abs_of(i) if callable(_abs_of) else (start_chunk + i)
            if state.step10_results[i] is not None:
                _successful_window_indices.append(_aidx)
            elif state.window_failures[i] is not None:
                _failed_window_indices.append(_aidx)

        # Collect partial results even when some windows failed.
        _successful_windows = sum(
            1 for i in range(N)
            if state.align_results[i] is not None or state.step10_results[i] is not None
        )
        _failed_windows = len(state.errors)
        _window_errors_detail = [
            {"phase": phase, "window_index": _abs_of(idx) if callable(_abs_of) else start_chunk + idx, "error": str(exc)}
            for phase, idx, exc in state.errors
        ]

        if _set_publish_state:
            # 完整性口径与暂停 epilogue 共用：从持久化 active episodes 重算
            #（见 _recompute_document_publish_windows docstring）
            self._publish_final_ingestion_state(
                set_publish_state=_set_publish_state,
                last_episode_id=last_episode_id,
                override_doc_id=override_doc_id,
                total_chunks=total_chunks,
                failed_window_indices=_failed_window_indices,
                successful_window_indices=_successful_window_indices,
                default_document_id=_atomic_document_id,
            )

        # Resolve document_version_id from the last episode
        document_version_id = ""
        if last_episode_id:
            try:
                document_version_id = self.storage._document_version_for_episode(last_episode_id)
            except Exception:
                pass

        # Update pipeline_run status to 'succeeded'
        try:
            pipeline_repo.update_pipeline_run_status(
                self.storage._conn(), _run_id, "succeeded",
                finished_at=datetime.now().isoformat(),
                episode_count=_contiguous_done,
                entity_count=total_entities,
                relation_count=total_relations,
            )
            self.storage._commit_if_not_batched(self.storage._conn())
        except Exception:
            logger.debug("pipeline_runs status update failed (non-critical)", exc_info=True)

        # Update documents.last_indexed_at for the processed document
        if document_version_id:
            try:
                _dv = self.storage._conn().execute(
                    "SELECT document_id FROM document_versions WHERE document_version_id = ?",
                    (document_version_id,),
                ).fetchone()
                if _dv:
                    _doc_id = _dv[0]
                    self.storage._conn().execute(
                        "UPDATE documents SET last_indexed_at = ?, updated_at = ? WHERE document_id = ?",
                        (datetime.now().isoformat(), datetime.now().isoformat(), _doc_id),
                    )
                    self.storage._commit_if_not_batched(self.storage._conn())
            except Exception:
                logger.debug("last_indexed_at update failed (non-critical)", exc_info=True)

        result = {
            "episode_id": last_episode_id,
            "document_version_id": document_version_id,
            "run_id": _run_id,
            "chunks_processed": _contiguous_done if _contiguous_done < N else N,
            "storage_path": storage_path,
            "entities": total_entities,
            "relations": total_relations,
            "window_timings": state.window_timings,
            "timing_summary": timing_summary or self._build_timing_summary(state.window_timings),
            "llm_call_stats": self.get_llm_call_stats(),
            # P3.6：按绝对窗口索引的 chunk 哈希（未到达的窗口为 None），
            # 供 task_queue 修复检测/完整性检查复用，免掉重复 chunk_text。
            "window_hashes": _window_hashes,
            "total_chunks": total_chunks,
        }

        if _failed_windows > 0:
            # Graceful degradation: log errors but return partial results instead of raising.
            # This ensures successful windows are persisted even when some fail.
            _error_summary = "; ".join(
                f"{phase}[W{idx}]: {exc}" for phase, idx, exc in state.errors[:5]
            )
            logger.error(
                "remember_text completed with %d/%d window failures: %s%s",
                _failed_windows, N, _error_summary,
                " (+ cross-window dedup failed)" if _dedup_exc else "",
            )
            _log_info("Remember",
                f"完成｜成功{_successful_windows}/{N}窗 "
                f"实体{total_entities} 关系{total_relations} "
                f"| {_failed_windows}窗失败: {_error_summary}"
            )
            result["warnings"] = _window_errors_detail
            result["failed_windows"] = _failed_windows
            result["successful_windows"] = _successful_windows
            result["failed_window_indices"] = sorted(_failed_window_indices)
            result["successful_window_indices"] = sorted(_successful_window_indices)
            result["failed_window_errors"] = _window_errors_detail
        elif _dedup_exc:
            result["warnings"] = [{"phase": "cross_window_dedup", "error": str(_dedup_exc)}]

        # Only raise if ALL windows failed -- partial results are still valuable.
        if _failed_windows >= N:
            _phase, _idx, exc = state.errors[0]
            if _set_publish_state:
                _set_publish_state(
                    _atomic_document_id, "failed", total_windows=total_chunks,
                    complete_windows=0, missing_windows=range(total_chunks),
                )
            raise exc

        # Clear the run_id from storage now that pipeline is done
        self.storage._current_run_id = ""

        return result

    def get_statistics(self) -> dict:
        """获取处理统计信息"""
        stats = self.storage.get_stats()
        return {
            "episodes": stats.get("episodes", 0),
            "entities": stats.get("entities", 0),
            "relations": stats.get("relations", 0),
            "storage_path": str(self.storage.storage_path)
        }

    def close(self):
        """释放资源：关闭线程池和存储连接。"""
        if hasattr(self, '_extraction_executor') and self._extraction_executor:
            self._extraction_executor.shutdown(wait=False)
        if hasattr(self, 'storage') and self.storage and hasattr(self.storage, 'close'):
            self.storage.close()

    def __del__(self):
        try:
            import sys
            if sys.is_finalizing():
                # Interpreter shutting down — don't touch executor, just close storage
                if hasattr(self, 'storage') and self.storage and hasattr(self.storage, 'close'):
                    try:
                        self.storage.close()
                    except Exception:
                        pass
                return
            self.close()
        except Exception:
            pass
