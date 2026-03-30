"""LLM客户端 - 记忆缓存相关操作。"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from ..models import MemoryCache
from ..utils import clean_markdown_code_blocks
from .prompts import (
    UPDATE_MEMORY_CACHE_SYSTEM_PROMPT,
    CREATE_DOCUMENT_OVERALL_MEMORY_SYSTEM_PROMPT,
    GENERATE_CONSOLIDATION_SUMMARY_SYSTEM_PROMPT,
    GENERATE_RELATION_MEMORY_CACHE_SYSTEM_PROMPT,
)


import re


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
    content = re.sub(r'\n*## 系统状态\s*\n.*$', '', content, flags=re.DOTALL).rstrip()

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

    def update_memory_cache(self, current_cache: Optional[MemoryCache], input_text: str,
                           document_name: str = "", text_start_pos: int = 0,
                           text_end_pos: int = 0, total_text_length: int = 0,
                           event_time: Optional[datetime] = None,
                           window_index: int = 0, total_windows: int = 0) -> MemoryCache:
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
            cache_for_prompt = re.sub(
                r'\n*## 系统状态\s*\n.*$', '', current_cache.content, flags=re.DOTALL
            ).rstrip()
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

        base_time = event_time if event_time is not None else datetime.now()
        new_cache_id = f"cache_{base_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        return MemoryCache(
            absolute_id=new_cache_id,
            content=new_content,
            event_time=base_time,
            source_document=source_document_only,
            activity_type="文档处理"
        )

    def create_document_overall_memory(self, text_preview: str, document_name: str = "",
                                       event_time: Optional[datetime] = None,
                                       previous_overall_content: Optional[str] = None) -> MemoryCache:
        """
        生成文档整体记忆：描述「即将处理的内容」是什么，供下一文档作为初始背景。
        与窗口链分离，生成后即可作为 B 的初始记忆，无需等 A 的最后一窗。

        Args:
            text_preview: 文档开头预览（如前 2000 字符）
            document_name: 文档/来源名称
            event_time: 事件时间
            previous_overall_content: 上一文档的整体记忆（Markdown），可选，用于衔接

        Returns:
            MemoryCache，activity_type="文档整体"
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
        base_time = event_time if event_time is not None else datetime.now()
        new_cache_id = f"overall_{base_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        source_document_only = document_name.split("/")[-1] if document_name else ""
        return MemoryCache(
            absolute_id=new_cache_id,
            content=new_content,
            event_time=base_time,
            source_document=source_document_only,
            activity_type="文档整体",
        )

    def generate_consolidation_summary(self, merge_results: list,
                                       alias_results: list,
                                       analyzed_groups_count: int) -> str:
        """
        生成知识图谱整理的摘要

        Args:
            merge_results: 合并操作结果列表
            alias_results: 别名关系创建结果列表
            analyzed_groups_count: 分析的实体组数量

        Returns:
            整理摘要的Markdown格式文本
        """
        system_prompt = GENERATE_CONSOLIDATION_SUMMARY_SYSTEM_PROMPT

        # 构建操作详情
        operations_str = f"分析的相似实体组数量: {analyzed_groups_count}\n\n"

        if merge_results:
            operations_str += "## 实体合并操作\n\n"
            for i, merge in enumerate(merge_results, 1):
                operations_str += f"{i}. 将 {merge.get('merged_source_ids', [])} 合并到 {merge.get('target_entity_id', '')}\n"
                operations_str += f"   - 更新的实体记录数: {merge.get('entities_updated', 0)}\n"
                if merge.get('reason'):
                    operations_str += f"   - 原因: {merge.get('reason', '')}\n"
        else:
            operations_str += "## 实体合并操作\n\n无需合并的实体。\n"

        if alias_results:
            operations_str += "\n## 别名关系创建\n\n"
            for i, alias in enumerate(alias_results, 1):
                operations_str += f"{i}. {alias.get('entity1_name', '')} -> {alias.get('entity2_name', '')}\n"
                operations_str += f"   - 关系描述: {alias.get('content', '')}\n"
        else:
            operations_str += "\n## 别名关系创建\n\n无需创建的别名关系。\n"

        prompt = f"""<操作详情>
{operations_str}
</操作详情>

请生成一个简洁的Markdown格式摘要，包括：
1. 整理概述
2. 主要操作
3. 整理效果

直接输出Markdown内容，不要包含代码块标记："""

        response = self._call_llm(prompt, system_prompt)

        # 清理可能的代码块标记
        response = clean_markdown_code_blocks(response)

        return response

    def generate_relation_memory_cache(self, relations: list,
                                       entities_info: list,
                                       entity_memory_caches: dict) -> str:
        """
        根据关系和实体生成记忆缓存内容

        Args:
            relations: 关系列表，每个包含 entity1_id, entity2_id, entity1_name, entity2_name, content, relation_id, is_new, is_updated
            entities_info: 实体信息列表，每个包含 entity_id, name, content
            entity_memory_caches: 实体ID到其记忆缓存内容的映射

        Returns:
            生成的记忆缓存内容（Markdown格式）
        """
        system_prompt = GENERATE_RELATION_MEMORY_CACHE_SYSTEM_PROMPT

        # 构建实体信息
        entities_str = ""
        for entity_info in entities_info:
            entity_id = entity_info.get("entity_id", "")
            name = entity_info.get("name", "")
            content = entity_info.get("content", "")
            memory_cache_content = entity_memory_caches.get(entity_id, "")

            entities_str += f"\n### 实体: {name}\n"
            entities_str += f"- 实体描述: {content}\n"
            if memory_cache_content:
                entities_str += f"- 相关记忆缓存: {memory_cache_content[:200]}...\n"

        # 构建关系信息
        relations_str = ""
        for relation in relations:
            entity1_name = relation.get("entity1_name", "")
            entity2_name = relation.get("entity2_name", "")
            content = relation.get("content", "")
            is_new = relation.get("is_new", False)
            is_updated = relation.get("is_updated", False)

            status = "新建" if is_new else ("更新" if is_updated else "已存在")
            relations_str += f"\n- {entity1_name} -> {entity2_name} ({status})\n"
            relations_str += f"  关系描述: {content}\n"

        prompt = f"""<实体信息>
{entities_str}
</实体信息>

<关系信息>
{relations_str}
</关系信息>

请生成一份记忆缓存文档，说明：
1. 系统当前正在进行的操作
2. 正在处理哪些实体和关系
3. 这些实体和关系的内容和意义
4. 处理的目的和结果

直接输出Markdown内容，不要包含代码块标记："""

        response = self._call_llm(prompt, system_prompt)

        # 清理可能的代码块标记
        response = clean_markdown_code_blocks(response)

        return response.strip()
