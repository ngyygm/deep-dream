"""
Pipeline mixin: remember_text entry point, lifecycle helpers, and standalone main().
"""
from typing import Any, Callable, Dict, Optional
from datetime import datetime
import sys
import logging
import threading
import time
import uuid

from core.log import info as _log_info
from core.utils import (
    clear_parallel_log_context,
    compute_doc_hash,
    set_pipeline_role,
    set_window_label,
    wprint_info,
)
from .helpers import dedupe_extraction_lists

logger = logging.getLogger(__name__)


class _PipelineMixin:
    """Mixin that provides the remember_text pipeline and lifecycle methods."""

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
                      source_document: Optional[str] = None) -> Dict:
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
                self.current_episode = self.storage.load_episode(latest_metadata["absolute_id"])
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
        total_length = len(text)
        chunks = self.document_processor.chunk_text(text)
        total_chunks = len(chunks)

        # 所有窗口已处理完毕（断点续传恢复后无需重跑）
        if start_chunk >= total_chunks:
            return {
                "episode_id": getattr(self.current_episode, 'absolute_id', None),
                "chunks_processed": total_chunks,
                "storage_path": str(self.storage.storage_path),
            }

        N = total_chunks - start_chunk  # 待处理窗口数
        last_episode_id = None
        clear_parallel_log_context()

        # 预分配共享状态
        state = self._init_remember_shared_state(N)

        # 暴露 state 引用供 get_pipeline_snapshot() 读取
        with self._current_state_lock:
            self._current_state = state

        # 启动 step9 / step10 线程
        t9 = threading.Thread(target=self._run_step9_worker, name="tmg-step9-chain", daemon=True,
                              args=(state, start_chunk, total_chunks, doc_name, verbose, verbose_steps,
                                    event_time, progress_callback, step9_chunk_done_callback))
        t10 = threading.Thread(target=self._run_step10_worker, name="tmg-step10-chain", daemon=True,
                              args=(state, start_chunk, total_chunks, doc_name, verbose, verbose_steps,
                                    event_time, progress_callback, chunk_done_callback))
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

                    _wi = start_chunk + ci
                    chunk, start, end = chunks[_wi]
                    if _wi == 0 and doc_name and not doc_name.startswith(("auto_", "api:")):
                        chunk = f"[文档元数据] 文档名：{doc_name} [/文档元数据]\n\n{chunk}"

                    _wlabel = f"W{start_chunk + ci + 1}/{total_chunks}"
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
                            f"窗口 {start_chunk + ci + 1}/{total_chunks} · 步骤1/10 进行中",
                            "", "main",
                        )

                    # Step1: 更新缓存
                    _t_step1_start = time.time()
                    _chunk_hash = compute_doc_hash(chunk)
                    existing_mc, _saved_extraction = (
                        self.storage.find_cache_and_extraction_by_doc_hash(_chunk_hash, document_path=document_path)
                        if _chunk_hash else (None, None)
                    )
                    if existing_mc:
                        new_mc = existing_mc
                        self.current_episode = existing_mc
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
                                )

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
                            main_chunk_done_callback(start_chunk + ci + 1)
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
                                    f"窗口 {start_chunk + ci + 1}/{total_chunks} · 步骤1–8 已缓存跳过 → 步骤9/10"
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
                            _wlabel = f"W{start_chunk + idx + 1}/{total_chunks}"
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
                                _idx_lo = (start_chunk + idx) / total_chunks
                                _idx_hi = (start_chunk + idx + 1) / total_chunks
                                _idx_span = _idx_hi - _idx_lo
                                ents, rels = self._extract_only(
                                    mc, chunk_text, doc_name,
                                    verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                                    progress_callback=lambda p, l, m: self._safe_progress(progress_callback, p, l, m, "main"),
                                    progress_range=(
                                        _idx_lo + _idx_span * (1.0 / 10.0),
                                        _idx_lo + _idx_span * (8.0 / 10.0),
                                    ),
                                    window_index=start_chunk + idx, total_windows=total_chunks,
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
                                if self._record_window_error(state, "extract", idx, e):
                                    logger.error("extract window %d error: %s", idx, e, exc_info=True)
                            finally:
                                with self._runtime_lock:
                                    self._active_window_extractions = max(0, self._active_window_extractions - 1)
                                # Ensure entity_content_done is always set to prevent step9 deadlock
                                state.entity_content_done[idx].set()
                                state.extract_done[idx].set()
                                if _success_main and main_chunk_done_callback:
                                    main_chunk_done_callback(start_chunk + idx + 1)
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

        # 无论成功还是异常，都清理 _current_state，避免残留上一个任务的快照
        with self._current_state_lock:
            self._current_state = None

        # Clean shutdown of prefetch executor with proper timeout
        try:
            state.prefetch_executor.shutdown(wait=True, timeout=5)
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
            raise RememberControlFlow(state.control_state["action"])

        # If the main pipeline crashed (not just individual window errors),
        # propagate the exception so the worker can retry.
        if _main_pipeline_exc is not None:
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

        # ========== 计时汇总 ==========
        self._summarize_window_timings(state.window_timings)

        storage_path = str(self.storage.storage_path)
        total_entities = sum(state.aligned_entity_counts)
        total_relations = sum(
            len(rl) for rl in state.step10_results if rl is not None
        )

        # Collect partial results even when some windows failed.
        _successful_windows = sum(
            1 for i in range(N)
            if state.align_results[i] is not None or state.step10_results[i] is not None
        )
        _failed_windows = len(state.errors)
        _window_errors_detail = [
            {"phase": phase, "window_index": idx, "error": str(exc)}
            for phase, idx, exc in state.errors
        ]

        result = {
            "episode_id": last_episode_id,
            "chunks_processed": total_chunks,
            "storage_path": storage_path,
            "entities": total_entities,
            "relations": total_relations,
            "window_timings": state.window_timings,
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
        elif _dedup_exc:
            result["warnings"] = [{"phase": "cross_window_dedup", "error": str(_dedup_exc)}]

        # Only raise if ALL windows failed -- partial results are still valuable.
        if _failed_windows >= N:
            _phase, _idx, exc = state.errors[0]
            raise exc

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


def main():
    """示例使用"""
    # Late import to avoid circular dependency at module load time
    from .orchestrator import TemporalMemoryGraphProcessor

    # 配置
    storage_path = "./tmg_storage"
    document_paths = sys.argv[1:] if len(sys.argv) > 1 else []

    if not document_paths:
        wprint_info("用法: python -m Temporal_Memory_Graph.processor <文档路径1> [文档路径2] ...")
        wprint_info("示例: python -m Temporal_Memory_Graph.processor doc1.txt doc2.txt")
        return

    # 创建处理器
    processor = TemporalMemoryGraphProcessor(
        storage_path=storage_path,
        window_size=1000,
        overlap=200,
        # llm_api_key="your-api-key",  # 如果需要，取消注释并填入
        # llm_model="gpt-4",
        # llm_base_url="https://api.openai.com/v1",  # 可自定义LLM API URL
        # embedding_model_path="/path/to/local/model",  # 本地embedding模型路径
        # embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # 或使用HuggingFace模型
    )

    # 处理文档
    processor.process_documents(document_paths, verbose=True)

    # 输出统计信息
    stats = processor.get_statistics()
    wprint_info("\n处理完成！")
    wprint_info(f"统计信息: {stats}")
