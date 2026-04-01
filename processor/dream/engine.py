"""DeepDream 记忆巩固引擎 - 像人做梦一样复习、重新连接、压缩记忆。

包含两种梦境模式：
1. DeepDreamEngine — 传统 4 阶段梦境周期（review → reconnect → consolidate → narrative）
2. DreamAgent — 基于 LLM 工具调用的自主梦境代理（agent loop 模式）
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..models import Entity, Relation
from .models import DreamConfig, DreamReport

logger = logging.getLogger(__name__)

DREAM_REVIEW_SYSTEM_PROMPT = """你是一个知识图谱记忆巩固助手。你的任务是复习近期记忆，标记异常或过时的信息。

请分析以下实体和关系，找出：
1. 可能过时的信息（与当前状态不符）
2. 异常或不一致的事实
3. 可能冗余的重复信息

请只输出一个 ```json ... ``` 代码块，包含键 "insights"（值为洞察列表数组，每个包含 "type"、"description" 和 "entity_id"）。"""

DREAM_RECONNECT_SYSTEM_PROMPT = """你是一个知识图谱记忆巩固助手。你的任务是发现跨领域、跨时间的新连接。

基于以下洞察和实体信息，找出：
1. 跨领域的新关联
2. 时间线上的演变模式
3. 隐藏的间接关系

请只输出一个 ```json ... ``` 代码块，包含键 "connections"（值为新连接列表数组，每个包含 "entity1_id"、"entity2_id"、"reason" 和 "confidence"）。"""

DREAM_CONSOLIDATE_SYSTEM_PROMPT = """你是一个知识图谱记忆巩固助手。你的任务是巩固记忆：更新摘要、合并建议。

基于以下新发现和实体信息，给出：
1. 建议合并的相似实体（entity_ids 列表）
2. 建议压缩的冗余关系（relation_ids 列表）
3. 建议更新的实体摘要

请只输出一个 ```json ... ``` 代码块，包含键 "consolidations"（值为巩固建议列表数组，每个包含 "action"、"target_id" 和 "reason"）。"""

DREAM_NARRATIVE_SYSTEM_PROMPT = """你是一个知识图谱梦境叙述生成器。你的任务是将记忆巩固的结果转化为人类可读的梦境叙述。

梦境叙述应该：
1. 以第一人称描述巩固过程
2. 使用比喻和意象（如"在记忆的海洋中漫游"）
3. 总结关键发现和行动
4. 保持简洁，200-500 字

请直接输出梦境叙述文本，不要使用 JSON 格式。"""


class DeepDreamEngine:
    """记忆巩固引擎 - 像人做梦一样复习、重新连接、压缩记忆。"""

    def __init__(self, storage: Any, llm_client: Any):
        self.storage = storage
        self.llm = llm_client
        self._current_cycle: Optional[DreamReport] = None

    async def run_dream_cycle(
        self,
        graph_id: str,
        config: Optional[DreamConfig] = None,
    ) -> DreamReport:
        """执行一次完整的梦境周期。

        Args:
            graph_id: 图谱 ID
            config: 梦境配置（可选）

        Returns:
            梦境报告
        """
        if config is None:
            config = DreamConfig()

        report = DreamReport(
            cycle_id=uuid.uuid4().hex,
            graph_id=graph_id,
            start_time=datetime.now(),
            status="running",
        )
        self._current_cycle = report

        try:
            # 获取近期实体和关系
            cutoff = datetime.now() - timedelta(days=config.review_window_days)
            entities = self._get_recent_entities(cutoff, config.max_entities_per_cycle)
            relations = self._get_recent_relations(cutoff, config.max_entities_per_cycle * 3)

            if not entities:
                report.status = "completed"
                report.narrative = "梦境空无：没有找到近期的记忆可供复习。"
                report.end_time = datetime.now()
                self._save_report(report)
                return report

            # 阶段 1：复习
            insights = await self._stage1_review(entities, relations)
            report.insights = insights

            # 阶段 2：重新连接
            connections = await self._stage2_reconnect(insights, entities)
            report.new_connections = connections

            # 阶段 3：巩固
            consolidations = await self._stage3_consolidate(connections, entities)
            report.consolidations = consolidations

            # 阶段 4：生成梦境叙述
            narrative = await self._stage4_dream_narrative(report)
            report.narrative = narrative

            report.status = "completed"
        except Exception as e:
            logger.error("梦境周期失败: %s", e)
            report.status = "failed"
            report.narrative = f"梦境中断：{str(e)}"
        finally:
            report.end_time = datetime.now()
            self._save_report(report)
            self._current_cycle = None

        return report

    def _get_recent_entities(self, cutoff: datetime, limit: int) -> List[Entity]:
        """获取近期的实体。"""
        try:
            entities = self.storage.get_all_entities(limit=limit, exclude_embedding=True)
            return [e for e in entities if e.event_time and e.event_time >= cutoff]
        except Exception:
            return self.storage.get_all_entities(limit=limit, exclude_embedding=True)

    def _get_recent_relations(self, cutoff: datetime, limit: int) -> List[Relation]:
        """获取近期的关系。"""
        try:
            relations = self.storage.get_all_relations(limit=limit)
            return [r for r in relations if r.event_time and r.event_time >= cutoff]
        except Exception:
            return self.storage.get_all_relations(limit=limit) if hasattr(self.storage, 'get_all_relations') else []

    async def _stage1_review(self, entities: List[Entity], relations: List[Relation]) -> List[dict]:
        """阶段1：复习近期记忆，标记异常/过时。"""
        entities_text = "\n".join(
            f"- {e.name}: {e.content[:200]}"
            for e in entities[:30]
        )
        relations_text = "\n".join(
            f"- {r.content[:200]}"
            for r in relations[:30]
        )

        prompt = f"""<近期实体（{len(entities)} 个）>
{entities_text}
</近期实体>

<近期关系（{len(relations)} 条）>
{relations_text}
</近期关系>

请分析上述记忆，标记异常或过时的信息："""

        messages = [
            {"role": "system", "content": DREAM_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result, _ = self.llm.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_insights(r),
                json_parse_retries=2,
            )
            return result
        except Exception as e:
            logger.warning("梦境阶段1失败: %s", e)
            return []

    async def _stage2_reconnect(self, insights: List[dict], entities: List[Entity]) -> List[dict]:
        """阶段2：发现跨领域/跨时间的新连接。"""
        if not insights:
            return []

        insights_text = "\n".join(
            f"- [{i.get('type', 'unknown')}] {i.get('description', '')} (实体: {i.get('entity_id', 'N/A')})"
            for i in insights[:20]
        )

        entities_text = "\n".join(
            f"- {e.entity_id}: {e.name}"
            for e in entities[:30]
        )

        prompt = f"""<洞察>
{insights_text}
</洞察>

<可用实体>
{entities_text}
</可用实体>

请发现跨领域的新连接："""

        messages = [
            {"role": "system", "content": DREAM_RECONNECT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result, _ = self.llm.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_connections(r),
                json_parse_retries=2,
            )
            return result
        except Exception as e:
            logger.warning("梦境阶段2失败: %s", e)
            return []

    async def _stage3_consolidate(self, connections: List[dict], entities: List[Entity]) -> List[dict]:
        """阶段3：巩固 - 更新摘要、合并相似实体、压缩冗余关系。"""
        if not connections:
            return []

        connections_text = "\n".join(
            f"- {c.get('entity1_id', '')} <-> {c.get('entity2_id', '')}: {c.get('reason', '')}"
            for c in connections[:20]
        )

        prompt = f"""<新发现的连接>
{connections_text}
</新发现的连接>

请给出巩固建议："""

        messages = [
            {"role": "system", "content": DREAM_CONSOLIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result, _ = self.llm.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_consolidations(r),
                json_parse_retries=2,
            )
            return result
        except Exception as e:
            logger.warning("梦境阶段3失败: %s", e)
            return []

    async def _stage4_dream_narrative(self, report: DreamReport) -> str:
        """阶段4：生成梦境叙述（人类可读的巩固报告）。"""
        summary = f"""梦境周期 {report.cycle_id}
- 洞察数量: {len(report.insights)}
- 新连接数量: {len(report.new_connections)}
- 巩固建议数量: {len(report.consolidations)}
"""

        if report.insights:
            summary += "\n关键洞察:\n" + "\n".join(
                f"- {i.get('description', '')}" for i in report.insights[:5]
            )
        if report.new_connections:
            summary += "\n新发现连接:\n" + "\n".join(
                f"- {c.get('entity1_id', '')} <-> {c.get('entity2_id', '')}" for c in report.new_connections[:5]
            )

        prompt = f"""<梦境摘要>
{summary}
</梦境摘要>

请将上述梦境巩固结果转化为梦境叙述："""

        messages = [
            {"role": "system", "content": DREAM_NARRATIVE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            narrative, _ = self.llm.call_llm(messages)
            return (narrative or "").strip()
        except Exception as e:
            logger.warning("梦境叙述生成失败: %s", e)
            return summary

    def _save_report(self, report: DreamReport):
        """保存梦境报告到存储层。"""
        try:
            if hasattr(self.storage, 'save_dream_log'):
                self.storage.save_dream_log(report)
        except Exception as e:
            logger.error("保存梦境报告失败: %s", e)

    @staticmethod
    def _parse_insights(response: str) -> List[dict]:
        """解析洞察列表响应。"""
        import json as _json
        # 去掉 markdown 代码块
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
        result = _json.loads(text.strip())
        if isinstance(result, dict):
            return result.get("insights", [])
        return []

    @staticmethod
    def _parse_connections(response: str) -> List[dict]:
        """解析连接列表响应。"""
        import json as _json
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
        result = _json.loads(text.strip())
        if isinstance(result, dict):
            return result.get("connections", [])
        return []

    @staticmethod
    def _parse_consolidations(response: str) -> List[dict]:
        """解析巩固建议响应。"""
        import json as _json
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
        result = _json.loads(text.strip())
        if isinstance(result, dict):
            return result.get("consolidations", [])
        return []


# ============================================================
# Dream Agent 入口 — 工具驱动的自主梦境
# ============================================================

async def run_agent_dream(
    storage: Any,
    llm_client: Any,
    config: Optional["DreamAgentConfig"] = None,
) -> "DreamAgentState":
    """运行 Dream Agent 自主梦境会话。

    Args:
        storage: 存储层实例（需支持 dream 相关方法）
        llm_client: LLM 客户端实例
        config: Dream Agent 配置（可选）

    Returns:
        DreamAgentState 包含完整的梦境会话状态和结果
    """
    from .agent import DreamAgent
    from .models import DreamAgentConfig as _DAC

    if config is None:
        config = _DAC()

    agent = DreamAgent(storage, llm_client, config)
    return await agent.run()
