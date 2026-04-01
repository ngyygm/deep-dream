"""Dream Agent — 自主梦境代理，基于工具调用循环发现知识图谱中的隐藏关系。

设计参考 claude-code-rev 的 AgentTool 模式：
  工具定义 → LLM 推理 → 工具执行 → 观察 → 迭代
"""
from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    DREAM_STRATEGIES,
    VALID_DREAM_STRATEGIES,
    DreamAgentConfig,
    DreamAgentState,
    DreamCycleResult,
    DreamToolCall,
)
from .prompts import (
    DREAM_AGENT_INITIAL_PROMPT,
    DREAM_AGENT_PLAN_PROMPT,
    DREAM_AGENT_REFLECT_PROMPT,
    DREAM_AGENT_SUMMARY_PROMPT,
    DREAM_AGENT_SYSTEM_PROMPT,
)
from .tools import DreamToolExecutor, format_tool_descriptions

logger = logging.getLogger(__name__)


class DreamAgent:
    """自主梦境代理 — 像人做梦一样在知识图谱中发现隐藏关系。

    工作流程：
    1. 选择策略 → 获取种子实体
    2. LLM 规划工具调用 → 执行 → 收集观察
    3. LLM 反思观察 → 提出新关系
    4. 保存发现 → 进入下一个周期
    """

    def __init__(self, storage: Any, llm_client: Any, config: Optional[DreamAgentConfig] = None):
        self.storage = storage
        self.llm = llm_client
        self.config = config or DreamAgentConfig()
        self.tool_descriptions = format_tool_descriptions()
        self.state = DreamAgentState(
            session_id=f"dream_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            graph_id=self.config.graph_id,
        )

    async def run(self) -> DreamAgentState:
        """运行完整的 Dream Agent 会话。"""
        self.state.status = "running"
        self.state.start_time = datetime.now()

        try:
            for cycle in range(self.config.max_cycles):
                self.state.current_cycle = cycle + 1
                strategy = self._select_strategy(cycle)

                logger.info(
                    "Dream Agent [%s] 周期 %d/%d 策略=%s",
                    self.state.session_id, cycle + 1, self.config.max_cycles, strategy,
                )

                result = await self._run_cycle(strategy)
                self.state.cycle_results.append(result)
                self.state.total_entities_examined += result.entities_examined
                self.state.total_relations_discovered += result.relations_discovered
                self.state.total_relations_saved += result.relations_saved
                self.state.total_tool_calls += result.tool_calls_made

                if result.error:
                    logger.warning("Dream Agent 周期 %d 错误: %s", cycle + 1, result.error)

            # 生成整体梦境叙述
            self._generate_narrative()
            self.state.status = "completed"

        except Exception as e:
            logger.error("Dream Agent 运行失败: %s", e)
            self.state.status = "failed"
            self.state.narrative = f"梦境中断：{str(e)}"
        finally:
            self.state.end_time = datetime.now()

        return self.state

    # ============================================================
    # 策略选择
    # ============================================================

    def _select_strategy(self, cycle: int) -> str:
        """根据 strategy_mode 选择下一个策略。"""
        strategies = self.config.strategies
        if not strategies:
            strategies = VALID_DREAM_STRATEGIES

        if self.config.strategy_mode == "round_robin":
            return strategies[cycle % len(strategies)]
        elif self.config.strategy_mode == "random":
            return random.choice(strategies)
        else:  # adaptive — 优先选择 yield 低的策略
            return strategies[cycle % len(strategies)]

    # ============================================================
    # 单周期执行
    # ============================================================

    async def _run_cycle(self, strategy: str) -> DreamCycleResult:
        """执行单个策略周期。"""
        result = DreamCycleResult(strategy=strategy)
        executor = DreamToolExecutor(
            self.storage, self.state.session_id, self.config,
        )

        try:
            # Phase 1: LLM 驱动的工具调用循环
            system_prompt = DREAM_AGENT_SYSTEM_PROMPT.format(
                tool_descriptions=self.tool_descriptions,
            )

            # 初始提示
            initial_msg = DREAM_AGENT_INITIAL_PROMPT.format(
                strategy_name=strategy,
                strategy_description=DREAM_STRATEGIES.get(strategy, ""),
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_msg},
            ]

            for step in range(self.config.max_tool_calls_per_cycle):
                # LLM 推理
                llm_response = self._call_llm(messages)
                if not llm_response:
                    break

                # 解析 LLM 输出
                parsed = self._parse_llm_response(llm_response)
                if parsed is None:
                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({
                        "role": "user",
                        "content": "请输出合法的 JSON 格式。参考系统提示中的输出格式。",
                    })
                    result.tool_calls_made += 1
                    continue

                messages.append({"role": "assistant", "content": llm_response})

                # 检查是否完成
                if parsed.get("done"):
                    # 处理最终输出的关系
                    final_relations = parsed.get("relations", [])
                    for rel in final_relations:
                        result.proposed_relations.append(rel)
                        save_result = self._save_relation(executor, rel, result)
                        if save_result:
                            result.relations_saved += 1

                    # 保存 episode
                    episode_content = parsed.get("episode_content", "")
                    if episode_content or result.relations_saved > 0:
                        self._save_episode(executor, result, episode_content)

                    # 收集观察
                    thought = parsed.get("thought", "")
                    if thought:
                        result.observations.append(thought)
                    break

                # 执行工具调用
                tool_calls = parsed.get("tool_calls", [])
                if not tool_calls:
                    # 思考阶段，记录观察
                    thought = parsed.get("thought", "")
                    if thought:
                        result.observations.append(thought)
                    continue

                tool_results_text = []
                for tc in tool_calls:
                    tool_name = tc.get("tool", "")
                    arguments = tc.get("arguments", {})
                    result.tool_calls_made += 1

                    exec_result = executor.execute(tool_name, arguments)

                    if exec_result.success:
                        tool_results_text.append(
                            f"[{tool_name}] 成功: {json.dumps(exec_result.data, ensure_ascii=False, default=str)[:500]}"
                        )
                        # 跟踪检查的实体
                        if tool_name in ("get_seeds", "search_similar", "search_bm25"):
                            data = exec_result.data
                            items = data if isinstance(data, list) else data.get("results", data.get("seeds", []))
                            if isinstance(items, list):
                                for item in items:
                                    eid = item.get("entity_id", "") if isinstance(item, dict) else ""
                                    if eid:
                                        self.state.examined_entity_ids.add(eid)
                                        result.entities_examined += 1
                        elif tool_name == "get_entity":
                            data = exec_result.data
                            ent = data.get("entity", {}) if isinstance(data, dict) else {}
                            eid = ent.get("entity_id", "")
                            if eid:
                                self.state.examined_entity_ids.add(eid)
                                result.entities_examined += 1
                        elif tool_name == "traverse":
                            data = exec_result.data
                            ents = data.get("entities", []) if isinstance(data, dict) else []
                            for ent in ents:
                                eid = ent.get("entity_id", "")
                                if eid:
                                    self.state.examined_entity_ids.add(eid)
                                    result.entities_examined += 1
                        elif tool_name == "create_relation":
                            result.relations_saved += 1
                    else:
                        tool_results_text.append(
                            f"[{tool_name}] 失败: {exec_result.error}"
                        )

                # 将工具结果反馈给 LLM
                if tool_results_text:
                    result.observations.extend(tool_results_text)
                    messages.append({
                        "role": "user",
                        "content": "工具执行结果：\n" + "\n".join(tool_results_text),
                    })

            result.relations_discovered = len(result.proposed_relations)
            result.saved_relations = executor.saved_relations
            result.episode_id = executor.saved_episode_ids[-1] if executor.saved_episode_ids else None

        except Exception as e:
            result.error = str(e)
            logger.error("Dream Agent 周期错误 (strategy=%s): %s", strategy, e)

        return result

    # ============================================================
    # 辅助方法
    # ============================================================

    def _call_llm(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """调用 LLM，返回响应文本。"""
        try:
            response = self.llm._call_llm(
                prompt="",
                messages=messages,
                timeout=120,
            )
            return response.strip() if response else None
        except Exception as e:
            logger.warning("Dream Agent LLM 调用失败: %s", e)
            return None

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 输出为结构化数据。"""
        text = response.strip()

        # 尝试提取 JSON 代码块
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            # 跳过可能的语言标记
            if text[start:start+4].strip() and not text[start].isspace():
                nl = text.find("\n", start)
                if nl > start:
                    start = nl + 1
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # 尝试在整个响应中找 JSON
        for i in range(len(response)):
            if response[i] == '{':
                try:
                    parsed = json.loads(response[i:])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

        return None

    def _save_relation(self, executor: DreamToolExecutor, rel: Dict[str, Any],
                       result: DreamCycleResult) -> bool:
        """保存一条发现的关系。"""
        try:
            exec_result = executor.execute("create_relation", {
                "entity1_id": rel.get("entity1_id", ""),
                "entity2_id": rel.get("entity2_id", ""),
                "content": rel.get("content", ""),
                "confidence": rel.get("confidence", 0.5),
                "reasoning": rel.get("reasoning", ""),
            })
            return exec_result.success
        except Exception as e:
            logger.warning("Dream Agent 保存关系失败: %s", e)
            return False

    def _save_episode(self, executor: DreamToolExecutor, result: DreamCycleResult,
                      extra_content: str = ""):
        """保存梦境 episode。"""
        try:
            content_parts = []
            if extra_content:
                content_parts.append(extra_content)
            if result.observations:
                content_parts.append("观察: " + "; ".join(result.observations[-5:]))
            if result.saved_relations:
                rel_text = ", ".join(
                    r.get("entity1_name", "") + " ↔ " + r.get("entity2_name", "")
                    for r in result.saved_relations[:5]
                )
                content_parts.append(f"发现关系: {rel_text}")

            content = "\n".join(content_parts) if content_parts else f"策略 {result.strategy} 周期完成"

            executor.execute("create_episode", {
                "content": content,
                "entities_examined": list(self.state.examined_entity_ids)[-20:],
                "relations_created": result.saved_relations,
            })
        except Exception as e:
            logger.warning("Dream Agent 保存 episode 失败: %s", e)

    def _generate_narrative(self):
        """生成整个梦境会话的叙述。"""
        if not self.state.cycle_results:
            self.state.narrative = "空梦：无记忆可供复习。"
            return

        relations_text = []
        for cr in self.state.cycle_results:
            for r in cr.saved_relations:
                relations_text.append(
                    f"- {r.get('entity1_name', '?')} ↔ {r.get('entity2_name', '?')}"
                )

        observations_text = []
        for cr in self.state.cycle_results:
            observations_text.extend(cr.observations[:3])

        prompt = DREAM_AGENT_SUMMARY_PROMPT.format(
            strategy=", ".join(set(cr.strategy for cr in self.state.cycle_results)),
            entities_examined=self.state.total_entities_examined,
            relations_discovered=self.state.total_relations_discovered,
            relations_saved=self.state.total_relations_saved,
            tool_calls=self.state.total_tool_calls,
            relations_text="\n".join(relations_text[:20]) if relations_text else "无",
            observations_text="\n".join(observations_text[:10]) if observations_text else "无",
        )

        try:
            messages = [
                {"role": "user", "content": prompt},
            ]
            narrative = self.llm._call_llm(prompt="", messages=messages, timeout=60)
            self.state.narrative = (narrative or "").strip()
        except Exception as e:
            logger.warning("Dream Agent 叙述生成失败: %s", e)
            self.state.narrative = (
                f"梦境完成：检查了 {self.state.total_entities_examined} 个实体，"
                f"发现 {self.state.total_relations_discovered} 条关系，"
                f"保存 {self.state.total_relations_saved} 条新关系。"
            )
