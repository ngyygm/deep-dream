"""
Planner 规划器

负责解析用户问题，生成工具调用计划
"""
import json
import re
from typing import List, Dict, Any, Optional

from ..llm.base import BaseLLMClient, LLMResponse
from ..models import ToolCall, PlanStep
from ..logger import AgentLogger, get_logger
from .prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_REQUEST_TEMPLATE,
    PLANNER_REQUEST_TEMPLATE_SIMPLE,
    NO_TOOL_NEEDED_PROMPT,
    format_tools_description,
    format_collected_info
)


class Planner:
    """规划器 - 解析问题，生成工具调用计划"""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        tools: Dict[str, Any],
        logger: Optional[AgentLogger] = None
    ):
        self.llm_client = llm_client
        self.tools = tools
        self.logger = logger or get_logger()
        tools_desc = format_tools_description(tools)
        self.system_prompt = PLANNER_SYSTEM_PROMPT.format(tools_description=tools_desc)
    
    def plan(
        self,
        question: str,
        collected_info: List[Dict[str, Any]] = None,
        conversation_history: List[Dict[str, str]] = None,
        reasoning_state: str = None
    ) -> Dict[str, Any]:
        """
        生成工具调用计划
        
        Args:
            question: 用户问题
            collected_info: 已收集的信息
            conversation_history: 对话历史
            reasoning_state: 推理状态摘要（来自 ReasoningCache.get_state_summary()）
        """
        collected_info = collected_info or []
        collected_info_str = format_collected_info(collected_info)
        
        # 根据是否有推理状态选择模板
        if reasoning_state:
            request = PLANNER_REQUEST_TEMPLATE.format(
                question=question,
                collected_info=collected_info_str,
                reasoning_state=reasoning_state
            )
        else:
            request = PLANNER_REQUEST_TEMPLATE_SIMPLE.format(
                question=question,
                collected_info=collected_info_str
            )
        
        if collected_info:
            request += "\n\n" + NO_TOOL_NEEDED_PROMPT
        
        messages = [{"role": "system", "content": self.system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": request})
        
        self.logger.debug(f"Planner request: {request}")
        response = self.llm_client.chat(messages)
        self.logger.debug(f"Planner response: {response.content}")
        
        return self._parse_response(response.content)
    
    async def aplan(
        self,
        question: str,
        collected_info: List[Dict[str, Any]] = None,
        conversation_history: List[Dict[str, str]] = None,
        reasoning_state: str = None
    ) -> Dict[str, Any]:
        """异步版本的规划"""
        collected_info = collected_info or []
        collected_info_str = format_collected_info(collected_info)
        
        if reasoning_state:
            request = PLANNER_REQUEST_TEMPLATE.format(
                question=question,
                collected_info=collected_info_str,
                reasoning_state=reasoning_state
            )
        else:
            request = PLANNER_REQUEST_TEMPLATE_SIMPLE.format(
                question=question,
                collected_info=collected_info_str
            )
        
        if collected_info:
            request += "\n\n" + NO_TOOL_NEEDED_PROMPT
        
        messages = [{"role": "system", "content": self.system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": request})
        
        response = await self.llm_client.achat(messages)
        return self._parse_response(response.content)
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = content.strip()
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
                result = json.loads(json_str)
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse planner response: {content[:200]}")
                return {
                    "analysis": content,
                    "tool_calls": [],
                    "is_complete": False,
                    "parse_error": True
                }
        standardized = {
            "analysis": result.get("analysis", ""),
            "tool_calls": [],
            "is_complete": result.get("is_complete", False),
            "next_steps": result.get("next_steps", ""),
            "summary": result.get("summary", "")
        }
        raw_calls = result.get("tool_calls", [])
        for call in raw_calls:
            if isinstance(call, dict) and "tool_name" in call:
                tool_call = ToolCall(
                    tool_name=call["tool_name"],
                    parameters=call.get("parameters", {})
                )
                standardized["tool_calls"].append(tool_call)
        return standardized
    
    def create_initial_plan(self, question: str) -> Dict[str, Any]:
        return self.plan(question, collected_info=[])
