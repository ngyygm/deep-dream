"""LLM 矛盾检测 - 检测同一实体不同版本之间的矛盾。"""
from __future__ import annotations

import json
from typing import List, Optional

from ..models import Entity
from ..utils import wprint

DETECT_CONTRADICTIONS_SYSTEM_PROMPT = """你是一个知识图谱一致性检查助手。你的任务是检测同一实体不同版本之间是否存在矛盾。

矛盾的定义：
1. 同一属性在不同版本中描述了不同的值（如"在北京工作" vs "在上海工作"）
2. 状态发生了变化但未说明原因（如"已婚" → "未婚"）
3. 时间线上不合理的快速变化

不是矛盾的情况：
1. 信息补充（新版本添加了旧版本没有的信息）
2. 信息的细化或修正
3. 时间线上的合理演变

请只输出一个 ```json ... ``` 代码块，包含键 "contradictions"（值为矛盾列表数组，每个矛盾包含 "description" 和 "severity"（"high"/"medium"/"low"））。"""

RESOLVE_CONTRADICTION_SYSTEM_PROMPT = """你是一个知识图谱矛盾裁决助手。你的任务是裁决检测到的矛盾，给出解决方案。

裁决方案可以是：
- "keep_new": 保留新版本的信息（新版本更准确）
- "keep_old": 保留旧版本的信息（旧版本更准确）
- "merge": 合并两个版本的信息
- "flag": 标记为需要人工确认

请只输出一个 ```json ... ``` 代码块，包含键 "resolution"（值为裁决方案对象，包含 "decision" 和 "reason"）。"""


class ContradictionDetectionMixin:
    """矛盾检测 mixin，通过 LLMClient 多继承使用。"""

    async def detect_contradictions(
        self,
        entity_id: str,
        versions: List[Entity],
    ) -> List[dict]:
        """检测同一实体不同版本之间的矛盾。

        Args:
            entity_id: 实体 ID
            versions: 实体的多个版本列表

        Returns:
            矛盾列表，每个包含 description 和 severity
        """
        if len(versions) < 2:
            return []

        versions_text = "\n".join(
            f"版本 {i + 1} (时间: {v.event_time.isoformat() if v.event_time else '未知'}):\n{v.content[:300]}"
            for i, v in enumerate(versions[:5])  # 限制版本数量
        )

        prompt = f"""<实体 ID>
{entity_id}
</实体 ID>

<实体版本>
{versions_text}
</实体版本>

请检测上述版本之间是否存在矛盾："""

        messages = [
            {"role": "system", "content": DETECT_CONTRADICTIONS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_contradictions_response(r),
                json_parse_retries=2,
            )
            return result
        except Exception as e:
            wprint(f"矛盾检测失败: {e}")
            return []

    async def resolve_contradiction(self, contradiction: dict) -> dict:
        """LLM 裁决矛盾，返回解决方案。

        Args:
            contradiction: 矛盾信息，包含 description 等字段

        Returns:
            裁决方案，包含 decision 和 reason
        """
        prompt = f"""<矛盾描述>
{contradiction.get('description', '未知矛盾')}
</矛盾描述>

请裁决上述矛盾："""

        messages = [
            {"role": "system", "content": RESOLVE_CONTRADICTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_resolution_response(r),
                json_parse_retries=2,
            )
            return result
        except Exception as e:
            wprint(f"矛盾裁决失败: {e}")
            return {"decision": "flag", "reason": f"自动裁决失败: {e}"}

    def _parse_contradictions_response(self, response: str) -> List[dict]:
        """解析矛盾检测的 LLM 响应。"""
        result = self._parse_json_response(response)
        if not isinstance(result, dict):
            return []
        contradictions = result.get("contradictions", [])
        if not isinstance(contradictions, list):
            return []
        valid = []
        for c in contradictions:
            if isinstance(c, dict) and "description" in c:
                valid.append({
                    "description": str(c["description"]),
                    "severity": str(c.get("severity", "medium")),
                })
        return valid

    def _parse_resolution_response(self, response: str) -> dict:
        """解析矛盾裁决的 LLM 响应。"""
        result = self._parse_json_response(response)
        if not isinstance(result, dict):
            raise json.JSONDecodeError("resolution: 需要 JSON 对象", response, 0)
        resolution = result.get("resolution", result)
        if not isinstance(resolution, dict):
            resolution = {"decision": "flag", "reason": str(resolution)}
        return {
            "decision": str(resolution.get("decision", "flag")),
            "reason": str(resolution.get("reason", "")),
        }
