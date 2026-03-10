"""
Ollama 原生 /api/chat 接口封装。

等价于 curl 调用，例如：
  curl http://localhost:11434/api/chat -d '{"model":"qwen3.5:4b","messages":[...],"think":false,"stream":false}'

提供非流式与流式两种调用方式，返回统一结构（content、thinking、用量等）。
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

# ---------------------------------------------------------------------------
# 请求/响应类型
# ---------------------------------------------------------------------------


@dataclass
class OllamaChatResponse:
    """Ollama /api/chat 非流式响应的结构化结果。"""

    content: str
    """助手回复正文。"""

    thinking: Optional[str] = None
    """思考过程（仅当请求中 think=True 时可能有值）。"""

    model: Optional[str] = None
    done: Optional[bool] = None
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None
    """原始 JSON 响应，便于扩展。"""

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "OllamaChatResponse":
        msg = data.get("message") or {}
        return cls(
            content=(msg.get("content") or ""),
            thinking=msg.get("thinking"),
            model=data.get("model"),
            done=data.get("done"),
            done_reason=data.get("done_reason"),
            total_duration=data.get("total_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
            raw=data,
        )


# ---------------------------------------------------------------------------
# 非流式调用
# ---------------------------------------------------------------------------


def ollama_chat(
    messages: List[Dict[str, str]],
    *,
    model: str = "qwen3.5:4b",
    base_url: str = "http://localhost:11434",
    think: bool = False,
    timeout: int = 300,
) -> OllamaChatResponse:
    """
    调用 Ollama 原生 /api/chat 接口（非流式）。

    等价于：
      curl <base_url>/api/chat -d '{"model":"...","messages":[...],"think":false,"stream":false}'

    Args:
        messages: 对话消息列表，每项为 {"role": "user"|"system"|"assistant", "content": "..."}。
        model: 模型名称。
        base_url: Ollama 服务地址（不含路径）。
        think: 是否开启思考链（think 模式）。
        timeout: 请求超时秒数。

    Returns:
        OllamaChatResponse: 包含 content、thinking、用量等字段。
    """
    url = base_url.rstrip("/") + "/api/chat"
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "think": think,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=timeout) as f:
        resp = json.loads(f.read().decode("utf-8"))

    return OllamaChatResponse.from_api_response(resp)


# ---------------------------------------------------------------------------
# 流式调用
# ---------------------------------------------------------------------------


def ollama_chat_stream(
    messages: List[Dict[str, str]],
    *,
    model: str = "qwen3.5:4b",
    base_url: str = "http://localhost:11434",
    think: bool = False,
    timeout: int = 300,
) -> Iterator[Dict[str, Any]]:
    """
    调用 Ollama 原生 /api/chat 接口（流式），按行迭代每块 JSON。

    等价于：
      curl <base_url>/api/chat -d '{"model":"...","messages":[...],"think":false,"stream":true}'

    Args:
        messages: 对话消息列表。
        model: 模型名称。
        base_url: Ollama 服务地址。
        think: 是否开启思考链。
        timeout: 请求超时秒数。

    Yields:
        每行解析后的 JSON 字典，通常包含 "message" 等字段；最后一块 "done" 为 True。
    """
    url = base_url.rstrip("/") + "/api/chat"
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "think": think,
        "stream": True,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=timeout) as f:
        for line in f:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def ollama_chat_stream_content(
    messages: List[Dict[str, str]],
    *,
    model: str = "qwen3.5:4b",
    base_url: str = "http://localhost:11434",
    think: bool = False,
    timeout: int = 300,
) -> Iterator[str]:
    """
    流式调用 /api/chat，仅逐块产出 content 增量文本（便于拼接成完整回复）。

    Args/Yields:
        同 ollama_chat_stream，但只 yield message.content 的增量字符串。
    """
    for chunk in ollama_chat_stream(
        messages, model=model, base_url=base_url, think=think, timeout=timeout
    ):
        msg = chunk.get("message") or {}
        delta = msg.get("content") or ""
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# OpenAI 兼容接口（智谱 GLM、OpenAI 等）
# ---------------------------------------------------------------------------


def openai_compatible_chat(
    messages: List[Dict[str, str]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int = 300,
) -> OllamaChatResponse:
    """
    调用 OpenAI 兼容的 /chat/completions 接口（如智谱 open.bigmodel.cn、OpenAI）。

    Args:
        messages: 对话消息列表，每项为 {"role": "user"|"system"|"assistant", "content": "..."}。
        model: 模型名称（如 glm-4-flash、glm-4.7-flash、gpt-4）。
        base_url: API 基础 URL，如 https://open.bigmodel.cn/api/paas/v4 或 .../api/coding/paas/v4。
        api_key: API Key，请求头为 Authorization: Bearer <api_key>。
        timeout: 请求超时秒数。

    Returns:
        OllamaChatResponse: content 为助手回复正文，与 ollama_chat 返回结构一致。
    """
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + api_key)

    with urllib.request.urlopen(req, timeout=timeout) as f:
        resp = json.loads(f.read().decode("utf-8"))

    choices = resp.get("choices") or []
    content = ""
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
    return OllamaChatResponse(content=content, raw=resp)
