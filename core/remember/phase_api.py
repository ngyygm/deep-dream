"""
Two-phase remember API: phase1 (overall cache) and phase2 (window-by-window extraction).

Extracted from orchestrator.py.  Functions receive the processor instance
(``TemporalMemoryGraphProcessor``) as the first argument so they can call
mixin methods and access attributes without circular imports.
"""
import uuid
from concurrent.futures import Future
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from core.utils import wprint_info, set_window_label, set_pipeline_role

from .pipeline_workers import acquire_window_slot, run_extraction_job


# ------------------------------------------------------------------
# Phase 1 — overall document memory
# ------------------------------------------------------------------

def remember_phase1_overall(
    processor,
    text: str,
    doc_name: str = "api_input",
    event_time: Optional[datetime] = None,
    document_path: str = "",
    previous_overall_cache=None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """
    Phase 1: generate a document-level overall memory (describes what is about to be processed).
    The result can serve as the initial cache for the next document B without waiting for
    the last window of the current document.
    """
    text_preview = (text[:2000] + "…") if len(text) > 2000 else text
    prev_content = previous_overall_cache.content if previous_overall_cache else None
    overall = processor.llm_client.create_document_overall_memory(
        text_preview=text_preview,
        document_name=doc_name,
        event_time=event_time,
        previous_overall_content=prev_content,
    )
    if progress_callback is not None:
        progress_callback({
            "phase": "phase1",
            "phase_label": "整体记忆已生成",
            "completed": 1,
            "total": 1,
            "message": f"文档整体记忆已生成: {doc_name}",
        })
    if verbose:
        wprint_info(f"[Phase1] 文档整体记忆已生成: {overall.absolute_id[:20]}…, doc_name={doc_name!r}")
    return overall


# ------------------------------------------------------------------
# Phase 2 — window-by-window extraction
# ------------------------------------------------------------------

def remember_phase2_windows(
    processor,
    text: str,
    doc_name: str = "api_input",
    verbose: bool = False,
    verbose_steps: bool = True,
    event_time: Optional[datetime] = None,
    document_path: str = "",
    overall_cache=None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict:
    """
    Phase 2: run all sliding windows starting from *overall_cache*, updating the
    cache and extracting entities/relations per window.
    """
    if not document_path:
        document_path = f"api://{uuid.uuid4().hex}"
    processor.current_episode = overall_cache  # first window's _update_cache will build on this
    window_size = processor.document_processor.window_size
    overlap = processor.document_processor.overlap
    total_length = len(text)
    start = 0
    chunk_idx = 0
    last_episode_id = None
    futures: List[Future] = []
    total_chunks = 1
    if total_length > 0:
        stride = max(1, window_size - overlap)
        total_chunks = 1 + max(0, (max(total_length - window_size, 0) + stride - 1) // stride)
    if progress_callback is not None:
        progress_callback({
            "phase": "phase2",
            "phase_label": "准备滑窗处理",
            "completed": 0,
            "total": total_chunks,
            "message": f"准备处理 {total_chunks} 个窗口",
        })

    while start < total_length:
        # Wait for concurrency slot: same pattern as remember_text
        acquire_window_slot(processor)

        end = min(start + window_size, total_length)
        chunk = text[start:end]
        if start == 0:
            chunk = f"[文档元数据] 文档名：{doc_name} [/文档元数据]\n\n{chunk}"

        if verbose:
            wprint_info(f"\n{'='*60}")
            wprint_info(f"处理窗口 (文档: {doc_name}, 位置: {start}-{end}/{total_length})")
            wprint_info(f"输入文本长度: {len(chunk)} 字符")
            wprint_info(f"{'='*60}\n")
        elif verbose_steps:
            wprint_info(f"窗口 {chunk_idx + 1}/{total_chunks} 开始 · {doc_name} [{start}-{end}/{total_length}]")

        with processor._cache_lock:
            new_mc = processor._update_cache(
                chunk, doc_name,
                text_start_pos=start, text_end_pos=end,
                total_text_length=total_length, verbose=verbose,
                verbose_steps=verbose_steps,
                document_path=document_path, event_time=event_time,
            )

        fut = processor._extraction_executor.submit(
            run_extraction_job,
            processor,
            new_mc, chunk, doc_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
        )
        futures.append(fut)
        last_episode_id = new_mc.absolute_id
        chunk_idx += 1
        if progress_callback is not None:
            progress_callback({
                "phase": "phase2",
                "phase_label": "滑窗处理进行中",
                "completed": chunk_idx,
                "total": total_chunks,
                "message": f"窗口 {chunk_idx}/{total_chunks} ({start}-{end}/{total_length})",
                "window_start": start,
                "window_end": end,
                "text_length": total_length,
            })
        if end >= total_length:
            break
        start = end - overlap

    for fut in futures:
        fut.result()

    return {
        "episode_id": last_episode_id,
        "chunks_processed": chunk_idx,
        "storage_path": str(processor.storage.storage_path),
    }
