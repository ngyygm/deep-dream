"""llm.protocol 显式端点协议（P5.4）：替代 URL 嗅探，嗅探保留为兜底。"""
import pytest

from core.exceptions import ConfigError
from core.llm.client import LLMClient
from core.server.config import _validate_config


def _client(**kw):
    return LLMClient(
        api_key=kw.pop("api_key", "sk-test"),
        model_name="test-model",
        base_url=kw.pop("base_url", "http://127.0.0.1:9000"),
        context_window_tokens=1000,
        **kw,
    )


def test_explicit_openai_overrides_sniff():
    """自建网关：无 /v1 后缀、不在已知域名表——嗅探判 False，显式 openai 判 True。"""
    c = _client(protocol="openai")
    assert c._use_openai_compatible_url("http://gateway.internal:9000/chat", "sk-x") is True


def test_explicit_ollama_overrides_sniff():
    c = _client(protocol="ollama")
    assert c._use_openai_compatible_url("https://api.openai.com/v1", "sk-x") is False


def test_none_keeps_sniffing_fallback():
    c = _client()
    assert c.protocol is None
    # 嗅探规则保持不变：本地 11434 → Ollama；/v1 结尾 → OpenAI 兼容
    assert c._use_openai_compatible_url("http://127.0.0.1:11434", "ollama") is False
    assert c._use_openai_compatible_url("http://gateway.internal:9000/v1", "sk-x") is True


def test_invalid_protocol_value_rejected():
    with pytest.raises(ConfigError):
        _validate_config({
            "llm": {"api_key": "k", "mock": True, "protocol": "grpc"},
        })
