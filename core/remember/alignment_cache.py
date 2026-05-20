"""Cache update and debug directory sub-mixin for _PipelineExtractionMixin."""
from __future__ import annotations

import time as _time
import uuid
from typing import Optional
from datetime import datetime
from pathlib import Path

from core.models import Episode
from core.utils import compute_doc_hash, wprint_info
from core.llm.client import LLM_PRIORITY_STEP1


class _CacheMixin:
    """Step 1 cache update and debug directory helpers."""

    def _update_cache(self, input_text: str, document_name: str,
                      text_start_pos: int = 0, text_end_pos: int = 0,
                      total_text_length: int = 0, verbose: bool = True,
                      verbose_steps: bool = True,
                      document_path: str = "",
                      event_time: Optional[datetime] = None,
                      window_index: int = 0, total_windows: int = 0,
                      doc_hash: str = "") -> Episode:
        """步骤1：更新记忆缓存。必须在 _cache_lock 下调用，保证 cache 链串行。"""
        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP1
        if verbose:
            wprint_info("【步骤1】缓存｜开始｜")
        elif verbose_steps:
            wprint_info("【步骤1】缓存｜开始｜")

        # 蒸馏数据准备：确保 task_id 在步骤1前生成
        if self.llm_client._distill_data_dir:
            if not self.llm_client._distill_task_id:
                self.llm_client._distill_task_id = f"{document_name}_{uuid.uuid4().hex[:8]}_{int(_time.time() * 1000)}"
            self.llm_client._current_distill_step = "01_update_cache"

        new_episode = self.llm_client.update_episode(
            self.current_episode,
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

        doc_hash = doc_hash or (compute_doc_hash(input_text) if input_text else "")
        self.storage.save_episode(
            new_episode,
            text=input_text,
            document_path=document_path,
            doc_hash=doc_hash,
            start_offset=text_start_pos,
            end_offset=text_end_pos,
        )
        self.current_episode = new_episode

        if verbose:
            wprint_info(f"【步骤1】缓存｜写入｜ID {new_episode.absolute_id}")
        elif verbose_steps:
            wprint_info("【步骤1】缓存｜完成｜已更新")

        return new_episode

    def _remember_debug_base_dir(self, document_name: str) -> Optional[Path]:
        root = getattr(self.llm_client, "_distill_data_dir", None)
        if not root:
            return None
        task_id = getattr(self.llm_client, "_distill_task_id", None) or f"adhoc_{document_name}"
        return Path(root) / "remember_debug" / task_id
