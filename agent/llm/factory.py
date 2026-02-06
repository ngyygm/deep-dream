"""
LLM 客户端工厂
"""
from typing import Dict, Any, Optional
from .base import BaseLLMClient
from .openai_client import OpenAICompatibleClient, MockLLMClient


def create_llm_client(
    provider: str = "openai",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    **kwargs
) -> BaseLLMClient:
    """
    创建 LLM 客户端
    
    Args:
        provider: 提供者类型
            - "openai": OpenAI 官方 API
            - "azure": Azure OpenAI
            - "ollama": 本地 Ollama
            - "zhipu": 智谱 AI
            - "deepseek": DeepSeek
            - "custom": 自定义 OpenAI 兼容 API
            - "mock": 模拟客户端（用于测试）
        api_key: API 密钥
        base_url: API 基础 URL
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        LLM 客户端实例
    """
    # 预设配置
    presets = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4"
        },
        "azure": {
            "base_url": "",  # 需要用户提供
            "model": "gpt-4"
        },
        "ollama": {
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "llama3"
        },
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "model": "glm-4"
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat"
        }
    }
    
    # 模拟客户端
    if provider == "mock":
        return MockLLMClient(**kwargs)
    
    # 获取预设配置
    preset = presets.get(provider, {})
    
    # 合并配置（用户提供的优先）
    final_base_url = base_url or preset.get("base_url", "https://api.openai.com/v1")
    final_model = model or preset.get("model", "gpt-4")
    
    # 所有提供者都使用 OpenAI 兼容客户端
    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=final_base_url,
        model=final_model,
        **kwargs
    )


def create_llm_client_from_config(config: Dict[str, Any]) -> BaseLLMClient:
    """
    从配置字典创建 LLM 客户端
    
    Args:
        config: 配置字典，包含:
            - provider: 提供者类型（可选，默认 "custom"）
            - api_key: API 密钥
            - base_url: API 基础 URL
            - model: 模型名称
            - temperature: 温度（可选）
            - max_tokens: 最大 token 数（可选）
            - timeout: 超时时间（可选）
            
    Returns:
        LLM 客户端实例
    """
    provider = config.get("provider", "custom")
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "")
    model = config.get("model", "")
    
    # 提取其他参数
    extra_kwargs = {
        k: v for k, v in config.items()
        if k not in ["provider", "api_key", "base_url", "model"]
    }
    
    return create_llm_client(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        **extra_kwargs
    )
