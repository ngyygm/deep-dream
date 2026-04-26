"""LLM 摘要进化 - 基于新旧版本生成进化后的实体/关系摘要。"""
from __future__ import annotations

import json
from typing import Optional

from ..models import Entity
from ..utils import wprint_info

EVOLVE_ENTITY_SUMMARY_SYSTEM_PROMPT = """你是一个知识图谱维护助手。你的任务是基于实体的新旧版本信息，生成一个简洁、准确的进化后摘要。

摘要应该：
1. 捕捉实体的核心特征和最新状态
2. 保留重要的历史信息
3. 简洁明了，不超过 200 字
4. 使用第三人称描述

请只输出一个 ```json ... ``` 代码块，包含键 "summary"（值为进化后的摘要字符串）。"""


class SummaryEvolutionMixin:
    """实体/关系摘要进化 mixin，通过 LLMClient 多继承使用。"""

    async def evolve_entity_summary(self, entity: Entity, old_version: Optional[Entity] = None) -> str:
        """基于实体新旧版本生成进化后的摘要。

        Args:
            entity: 当前最新版实体
            old_version: 上一个版本（可选，用于对比进化）

        Returns:
            进化后的摘要字符串
        """
        old_info = ""
        if old_version and old_version.content != entity.content:
            old_info = f"\n<旧版本内容>\n{old_version.content[:500]}\n</旧版本内容>"

        prompt = f"""<实体名称>
{entity.name}
</实体名称>

<当前内容>
{entity.content[:800]}
</当前内容>
{old_info}

请为该实体生成一个进化后的摘要："""

        messages = [
            {"role": "system", "content": EVOLVE_ENTITY_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_summary_response(r),
                json_parse_retries=2,
            )
            return result
        except Exception as e:
            wprint_info(f"摘要进化失败，使用截断内容: {e}")
            return entity.content[:200]

    def _parse_summary_response(self, response: str) -> str:
        """解析摘要进化的 LLM 响应。"""
        result = self._parse_json_response(response)
        if not isinstance(result, dict) or "summary" not in result:
            raise json.JSONDecodeError("summary: 需要含 summary 的 JSON 对象", response, 0)
        summary = result["summary"].strip()
        if not summary:
            raise json.JSONDecodeError("summary: summary 为空", response, 0)
        return summary
