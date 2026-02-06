"""
LLM 客户端基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    role: str = "assistant"
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None
    
    @property
    def has_tool_calls(self) -> bool:
        return self.tool_calls is not None and len(self.tool_calls) > 0


class BaseLLMClient(ABC):
    """LLM 客户端基类"""
    
    def __init__(self, api_key: str, base_url: str, model: str, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 4096)
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        同步聊天接口
        
        Args:
            messages: 消息列表，OpenAI 格式
            tools: 工具定义列表（可选）
            tool_choice: 工具选择策略 ("auto", "required", "none")
            **kwargs: 其他参数
            
        Returns:
            LLMResponse 对象
        """
        pass
    
    @abstractmethod
    async def achat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        异步聊天接口
        """
        pass
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式聊天接口（可选实现）
        """
        response = self.chat(messages, **kwargs)
        yield response.content
    
    def format_tool_definitions(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        格式化工具定义为 OpenAI 函数调用格式
        
        Args:
            tools: 工具定义列表
            
        Returns:
            OpenAI 格式的函数定义
        """
        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                }
            })
        return formatted
    
    def parse_tool_calls(self, response: LLMResponse) -> List[Dict[str, Any]]:
        """
        解析工具调用
        
        Args:
            response: LLM 响应
            
        Returns:
            工具调用列表，每个包含 name 和 arguments
        """
        if not response.has_tool_calls:
            return []
        
        calls = []
        for tool_call in response.tool_calls:
            calls.append({
                "id": tool_call.get("id", ""),
                "name": tool_call.get("function", {}).get("name", ""),
                "arguments": tool_call.get("function", {}).get("arguments", "{}")
            })
        return calls
