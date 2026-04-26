"""LLM客户端 - 记忆缓存相关操作。"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from ..models import Episode
from ..utils import clean_markdown_code_blocks
from .prompts import (
    UPDATE_MEMORY_CACHE_SYSTEM_PROMPT,
    CREATE_DOCUMENT_OVERALL_MEMORY_SYSTEM_PROMPT,
)


import re

# Pre-compiled regex for stripping injected system status sections
_SYSTEM_STATUS_RE = re.compile(r'\n*## 系统状态\s*\n.*$', re.DOTALL)


def _append_system_status(content: str, doc_name: str,
                         text_start_pos: int = 0, text_end_pos: int = 0,
                         total_text_length: int = 0,
                         window_index: int = 0, total_windows: int = 0) -> str:
    """在 LLM 生成的记忆缓存内容末尾追加「系统状态」section（代码注入，不依赖 LLM）。

    追加格式：
        ## 系统状态
        - 文档名：xxx
        - 处理进度：[3/10]（字符位置：1000-2000/10000）

    如果 doc_name 为空且窗口参数全为零，则跳过追加。
    """
    # 先移除之前可能注入的 ## 系统状态 段落，避免重复
    content = _SYSTEM_STATUS_RE.sub('', content).rstrip()

    if not doc_name and text_start_pos == 0 and text_end_pos == 0 and window_index == 0:
        return content

    parts = ["\n## 系统状态"]
    if doc_name:
        parts.append(f"- 文档名：{doc_name}")

    # 窗口进度
    if total_windows > 0:
        parts.append(f"- 处理进度：[{window_index}/{total_windows}]")
    elif total_text_length > 0:
        parts.append(
            f"- 处理进度：[{text_start_pos}-{text_end_pos}/{total_text_length}]"
        )
    elif text_end_pos > 0:
        parts.append(f"- 处理进度：字符位置 {text_start_pos}-{text_end_pos}")

    return content + "\n" + "\n".join(parts)


class _MemoryOpsMixin:
    """记忆缓存相关的 LLM 操作（mixin，通过 LLMClient 多继承使用）。"""

    def update_episode(self, current_cache: Optional[Episode], input_text: str,
                           document_name: str = "", text_start_pos: int = 0,
                           text_end_pos: int = 0, total_text_length: int = 0,
                           event_time: Optional[datetime] = None,
                           window_index: int = 0, total_windows: int = 0) -> Episode:
        """
        任务1：更新记忆缓存

        Args:
            current_cache: 当前的记忆缓存（如果存在）
            input_text: 当前窗口的输入文本
            document_name: 当前文档名称
            event_time: 事件实际发生时间；若提供则用于 event_time，否则 datetime.now()
            text_start_pos: 当前文本在文档中的起始位置（字符位置）
            text_end_pos: 当前文本在文档中的结束位置（字符位置）
            total_text_length: 文档总长度（字符数）
            window_index: 当前窗口索引（从1开始）
            total_windows: 总窗口数

        Returns:
            更新后的新记忆缓存
        """
        system_prompt = UPDATE_MEMORY_CACHE_SYSTEM_PROMPT

        if current_cache:
            # 喂给 LLM 之前，移除上次代码注入的系统状态段落
            cache_for_prompt = _SYSTEM_STATUS_RE.sub('', current_cache.content).rstrip()
            prompt = f"""<记忆缓存>
{cache_for_prompt}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>

请根据新的输入文本更新记忆缓存：保留有用信息，添加新信息，删除过期信息。"""
        else:
            prompt = f"""<输入文本>
{input_text}
</输入文本>

请创建初始记忆缓存。"""

        new_content = self._call_llm(prompt, system_prompt)

        if not new_content:
            # LLM 返回空，使用已有缓存内容或原始输入作为兜底
            if current_cache and current_cache.content:
                new_content = current_cache.content
            else:
                new_content = input_text

        # 清理 markdown 代码块标识符
        new_content = clean_markdown_code_blocks(new_content)
        # XML 分隔符标签已在 _call_llm 中统一清理

        # 代码注入系统状态：文档名 + 窗口进度（不依赖 LLM 生成）
        source_document_only = document_name.split('/')[-1] if document_name else ""
        new_content = _append_system_status(new_content, source_document_only,
                                           text_start_pos, text_end_pos, total_text_length,
                                           window_index, total_windows)

        _now = datetime.now()
        base_time = event_time if event_time is not None else _now
        new_cache_id = f"cache_{base_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        return Episode(
            absolute_id=new_cache_id,
            content=new_content,
            event_time=base_time,
            processed_time=_now,
            source_document=source_document_only,
            activity_type="文档处理"
        )

    def create_document_overall_memory(self, text_preview: str, document_name: str = "",
                                       event_time: Optional[datetime] = None,
                                       previous_overall_content: Optional[str] = None) -> Episode:
        """
        生成文档整体记忆：描述「即将处理的内容」是什么，供下一文档作为初始背景。
        与窗口链分离，生成后即可作为 B 的初始记忆，无需等 A 的最后一窗。

        Args:
            text_preview: 文档开头预览（如前 2000 字符）
            document_name: 文档/来源名称
            event_time: 事件时间
            previous_overall_content: 上一文档的整体记忆（Markdown），可选，用于衔接

        Returns:
            Episode，activity_type="文档整体"
        """
        system_prompt = CREATE_DOCUMENT_OVERALL_MEMORY_SYSTEM_PROMPT
        if previous_overall_content:
            prompt = f"""<上一文档记忆>
{previous_overall_content[:1500]}
</上一文档记忆>

<当前文档>
文档名：{document_name}

文档内容预览：
{text_preview[:2000]}
</当前文档>

请生成当前文档的「文档整体记忆」："""
        else:
            prompt = f"""<当前文档>
文档名：{document_name}

文档内容预览：
{text_preview[:2000]}
</当前文档>

请生成该文档的「文档整体记忆」："""
        new_content = self._call_llm(prompt, system_prompt)
        new_content = clean_markdown_code_blocks(new_content or "")
        # XML 分隔符标签已在 _call_llm 中统一清理
        _now = datetime.now()
        base_time = event_time if event_time is not None else _now
        new_cache_id = f"overall_{base_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        source_document_only = document_name.split("/")[-1] if document_name else ""
        return Episode(
            absolute_id=new_cache_id,
            content=new_content,
            event_time=base_time,
            processed_time=_now,
            source_document=source_document_only,
            activity_type="文档整体",
        )
