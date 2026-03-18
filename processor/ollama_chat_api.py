"""
统一的 LLM Chat 调用封装（Python SDK 版本）。

- Ollama：通过 OpenAI 兼容接口（`<ollama>/v1/chat/completions`）访问；
- OpenAI/智谱等：通过 OpenAI 兼容接口访问。

项目内统一走 `openai>=1.0` 的 Python SDK，不再依赖 curl 或手写 urllib 请求。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI


@dataclass
class OllamaChatResponse:
    """统一的非流式响应结构（兼容旧接口）。"""

    content: str
    thinking: Optional[str] = None
    model: Optional[str] = None
    done: Optional[bool] = None
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None


def _as_dict(obj: Any) -> Dict[str, Any]:
    """兼容 openai>=1.0 返回对象与字典两种形态。"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    try:
        return dict(obj)  # type: ignore[arg-type]
    except Exception:
        return {}


def _ollama_v1_base_url(base_url: str) -> str:
    base = (base_url or "http://localhost:11434").rstrip("/")
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def ollama_chat(
    messages: List[Dict[str, str]],
    *,
    model: str = "qwen3.5:4b",
    base_url: str = "http://localhost:11434",
    think: bool = False,
    timeout: int = 300,
) -> OllamaChatResponse:
    """Ollama 非流式 chat（OpenAI 兼容接口）。"""
    client = OpenAI(base_url=_ollama_v1_base_url(base_url), api_key="ollama")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=timeout,
        extra_body={"think": think},
    )
    data = _as_dict(resp)
    choices = data.get("choices") or []
    content = ""
    thinking_out: Optional[str] = None
    if choices:
        c0 = choices[0] if isinstance(choices[0], dict) else _as_dict(choices[0])
        msg = c0.get("message") or {}
        content = msg.get("content") or ""
        thinking_out = msg.get("thinking") or msg.get("reasoning")
    return OllamaChatResponse(content=content, thinking=thinking_out, raw=data)


def ollama_chat_stream(
    messages: List[Dict[str, str]],
    *,
    model: str = "qwen3.5:4b",
    base_url: str = "http://localhost:11434",
    think: bool = False,
    timeout: int = 300,
) -> Iterator[Dict[str, Any]]:
    """Ollama 流式 chat（OpenAI 兼容接口），逐块产出 chunk 字典。"""
    client = OpenAI(base_url=_ollama_v1_base_url(base_url), api_key="ollama")
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        timeout=timeout,
        extra_body={"think": think},
    )
    for chunk in stream:
        yield _as_dict(chunk)


def ollama_chat_stream_content(
    messages: List[Dict[str, str]],
    *,
    model: str = "qwen3.5:4b",
    base_url: str = "http://localhost:11434",
    think: bool = False,
    timeout: int = 300,
) -> Iterator[str]:
    """Ollama 流式 chat：仅产出 content 增量字符串。"""
    for chunk in ollama_chat_stream(
        messages, model=model, base_url=base_url, think=think, timeout=timeout
    ):
        choices = chunk.get("choices") or []
        if not choices:
            continue
        c0 = choices[0] if isinstance(choices[0], dict) else _as_dict(choices[0])
        delta = (c0.get("delta") or {}).get("content") or ""
        if delta:
            yield delta


def openai_compatible_chat(
    messages: List[Dict[str, str]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int = 300,
) -> OllamaChatResponse:
    """OpenAI 兼容 chat（非流式）。"""
    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=timeout,
    )
    data = _as_dict(resp)
    choices = data.get("choices") or []
    content = ""
    if choices:
        c0 = choices[0] if isinstance(choices[0], dict) else _as_dict(choices[0])
        msg = c0.get("message") or {}
        content = msg.get("content") or ""
    return OllamaChatResponse(content=content, raw=data)
