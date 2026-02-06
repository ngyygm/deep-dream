"""
OpenAI 兼容的 LLM 客户端

支持所有 OpenAI 兼容的 API，包括：
- OpenAI 官方 API
- Azure OpenAI
- 本地部署的模型（如 Ollama、vLLM）
- 其他兼容 API（如智谱、DeepSeek 等）
"""
import json
import httpx
from typing import List, Dict, Any, Optional
import asyncio

from .base import BaseLLMClient, LLMResponse


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI 兼容的 LLM 客户端"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4",
        **kwargs
    ):
        super().__init__(api_key, base_url, model, **kwargs)
        self.timeout = kwargs.get("timeout", 60.0)
        
        # 确保 base_url 格式正确
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            # 某些 API 可能已经包含完整路径
            pass
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _build_request_body(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """构建请求体"""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }
        
        # 添加工具定义
        if tools:
            body["tools"] = self.format_tool_definitions(tools)
            if tool_choice:
                body["tool_choice"] = tool_choice
        
        return body
    
    def _parse_response(self, response_data: Dict[str, Any]) -> LLMResponse:
        """解析响应"""
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        # 解析工具调用
        tool_calls = None
        if "tool_calls" in message:
            tool_calls = message["tool_calls"]
        
        return LLMResponse(
            content=message.get("content", "") or "",
            role=message.get("role", "assistant"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=response_data.get("usage")
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """同步聊天接口"""
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        body = self._build_request_body(messages, tools, tool_choice, **kwargs)
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return self._parse_response(response.json())
    
    async def achat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """异步聊天接口"""
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        body = self._build_request_body(messages, tools, tool_choice, **kwargs)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return self._parse_response(response.json())
    
    def chat_with_retry(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3,
        **kwargs
    ) -> LLMResponse:
        """带重试的聊天接口"""
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.chat(messages, tools, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # 指数退避
        raise last_error


class MockLLMClient(BaseLLMClient):
    """
    模拟 LLM 客户端，用于测试
    
    可以预设响应或使用简单的规则生成响应
    """
    
    def __init__(self, **kwargs):
        super().__init__("mock-key", "mock-url", "mock-model", **kwargs)
        self.responses: List[LLMResponse] = []
        self.call_count = 0
    
    def add_response(self, response: LLMResponse):
        """添加预设响应"""
        self.responses.append(response)
    
    def add_tool_call_response(self, tool_name: str, arguments: Dict[str, Any]):
        """添加工具调用响应"""
        self.responses.append(LLMResponse(
            content="",
            tool_calls=[{
                "id": f"call_{self.call_count}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments)
                }
            }]
        ))
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """返回预设响应或生成简单响应"""
        self.call_count += 1
        
        if self.responses:
            return self.responses.pop(0)
        
        # 默认响应
        return LLMResponse(
            content="这是一个模拟响应。",
            role="assistant"
        )
    
    async def achat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """异步版本"""
        return self.chat(messages, tools, tool_choice, **kwargs)
