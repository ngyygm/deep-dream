"""
统一的 LLM Chat 调用封装。

- Ollama：通过原生接口 `POST <ollama>/api/chat` 访问；
- OpenAI/智谱等：通过 OpenAI 兼容接口访问。

关键约束：`think` 仅在 Ollama 原生协议下生效，不能走 `/v1/chat/completions`。
"""
from __future__ import annotations

import logging as _logging

import atexit
from dataclasses import dataclass
import json
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib import request
from urllib.error import HTTPError, URLError

from openai import OpenAI
import httpx

# 每个 LLM 请求若都 new OpenAI()，会在高并发下为每个实例挂一套 httpx 连接池，迅速耗尽 fd（Errno 24）。
_openai_singleton_lock = threading.Lock()
_openai_singletons: Dict[Tuple[str, str], OpenAI] = {}


def _openai_shared_client(base_url: str, api_key: str) -> OpenAI:
    bu = (base_url or "").rstrip("/")
    key = api_key if api_key is not None else ""
    cache_key = (bu, key)
    with _openai_singleton_lock:
        client = _openai_singletons.get(cache_key)
        if client is None:
            http_client = httpx.Client(
                limits=httpx.Limits(max_connections=128, max_keepalive_connections=32),
                timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0),
            )
            client = OpenAI(base_url=bu, api_key=key or None, http_client=http_client)
            _openai_singletons[cache_key] = client
        return client


def _close_all_openai_shared_clients() -> None:
    with _openai_singleton_lock:
        for c in list(_openai_singletons.values()):
            try:
                close = getattr(c, "close", None)
                if callable(close):
                    close()
            except Exception as _e:
                _logging.getLogger(__name__).debug("关闭 OpenAI 客户端失败: %s", _e)
        _openai_singletons.clear()


atexit.register(_close_all_openai_shared_clients)


@dataclass(slots=True)
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
    except Exception as _e:
        _logging.getLogger(__name__).debug("_as_dict 转换失败: %s", _e)
        return {}


def _ollama_native_base_url(base_url: str) -> str:
    base = (base_url or "http://localhost:11434").rstrip("/")
    if base.endswith("/api/chat"):
        base = base[:-9]
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _ollama_chat_url(base_url: str) -> str:
    return _ollama_native_base_url(base_url) + "/api/chat"


def _read_json_response(resp: Any) -> Dict[str, Any]:
    raw = resp.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _extract_ollama_message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return content or ""
    return ""


def _extract_ollama_thinking(data: Dict[str, Any]) -> Optional[str]:
    message = data.get("message") or {}
    thinking = None
    if isinstance(message, dict):
        thinking = message.get("thinking") or message.get("reasoning")
    return thinking or data.get("thinking") or data.get("reasoning")


def ollama_chat(
    messages: List[Dict[str, str]],
    *,
    model: str = "qwen3.5:4b",
    base_url: str = "http://localhost:11434",
    think: bool = False,
    timeout: int = 300,
    num_predict: Optional[int] = None,
    json_format: bool = False,
) -> OllamaChatResponse:
    """Ollama 非流式 chat（原生 /api/chat 接口）。"""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
    }
    if json_format:
        payload["format"] = "json"
    if num_predict is not None:
        payload["num_predict"] = num_predict
    req = request.Request(
        _ollama_chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = _read_json_response(resp)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama /api/chat HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"Ollama /api/chat 连接失败: {e}") from e

    message = data.get("message") or {}
    return OllamaChatResponse(
        content=_extract_ollama_message_content(message),
        thinking=_extract_ollama_thinking(data),
        model=data.get("model"),
        done=data.get("done"),
        done_reason=data.get("done_reason"),
        total_duration=data.get("total_duration"),
        prompt_eval_count=data.get("prompt_eval_count"),
        eval_count=data.get("eval_count"),
        raw=data,
    )


def ollama_chat_stream(
    messages: List[Dict[str, str]],
    *,
    model: str = "qwen3.5:4b",
    base_url: str = "http://localhost:11434",
    think: bool = False,
    timeout: int = 300,
) -> Iterator[Dict[str, Any]]:
    """Ollama 流式 chat（原生 /api/chat 接口），逐块产出 chunk 字典。"""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": think,
    }
    req = request.Request(
        _ollama_chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if not line:
                    continue
                text = line.decode("utf-8").strip()
                if not text:
                    continue
                yield json.loads(text)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama /api/chat HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"Ollama /api/chat 连接失败: {e}") from e


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
        delta = _extract_ollama_message_content(chunk.get("message") or {})
        if delta:
            yield delta


def openai_compatible_chat(
    messages: List[Dict[str, str]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int = 300,
    max_tokens: Optional[int] = None,
) -> OllamaChatResponse:
    """OpenAI 兼容 chat（非流式）。"""
    client = _openai_shared_client(base_url, api_key)
    kwargs: Dict[str, Any] = dict(model=model, messages=messages, timeout=timeout)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    data = _as_dict(resp)
    choices = data.get("choices") or []
    content = ""
    finish_reason = None
    if choices:
        c0 = choices[0] if isinstance(choices[0], dict) else _as_dict(choices[0])
        msg = c0.get("message") or {}
        content = msg.get("content") or ""
        finish_reason = c0.get("finish_reason")
    return OllamaChatResponse(content=content, done_reason=finish_reason, raw=data)
