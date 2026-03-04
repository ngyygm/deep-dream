#!/usr/bin/env python3
"""
Ollama 原生 /api/chat 接口封装示例。

等价于终端中的 curl 调用，例如：
  curl http://localhost:11434/api/chat -d '{
    "model": "qwen3.5:4b",
    "messages": [{"role": "user", "content": "how many r in the word strawberry?"}],
    "think": false,
    "stream": false
  }'
"""
import sys

# 可从包入口或直接模块导入
try:
    from processor import ollama_chat, ollama_chat_stream_content, OllamaChatResponse
except ImportError:
    from processor.ollama_chat_api import (
        ollama_chat,
        ollama_chat_stream_content,
        OllamaChatResponse,
    )


def main():
    base_url = "http://localhost:11434"
    model = "qwen3.5:4b"
    user_content = "how many r in the word strawberry?"

    print("Ollama /api/chat 接口封装示例")
    print("=" * 50)
    print(f"URL: {base_url}/api/chat")
    print(f"model: {model}")
    print(f"user: {user_content!r}")
    print()

    # 非流式调用（与终端 curl 行为一致）
    print("--- 非流式 (think=True) ---")
    try:
        resp = ollama_chat(
            [{"role": "user", "content": user_content}],
            model=model,
            base_url=base_url,
            think=True,
            timeout=120,
        )
        assert isinstance(resp, OllamaChatResponse)
        print("content:", resp.content[:200] + "..." if len(resp.content) > 200 else resp.content)
        if resp.thinking:
            print("thinking:", resp.thinking[:150], "...")
        print("done:", resp.done, "eval_count:", resp.eval_count)
    except Exception as e:
        print("错误:", e)
        sys.exit(1)

    # 非流式调用（与终端 curl 行为一致）
    print("--- 非流式 (think=False) ---")
    try:
        resp = ollama_chat(
            [{"role": "user", "content": user_content}],
            model=model,
            base_url=base_url,
            think=False,
            timeout=120,
        )
        assert isinstance(resp, OllamaChatResponse)
        print("content:", resp.content[:200] + "..." if len(resp.content) > 200 else resp.content)
        if resp.thinking:
            print("thinking:", resp.thinking[:150], "...")
        print("done:", resp.done, "eval_count:", resp.eval_count)
    except Exception as e:
        print("错误:", e)
        sys.exit(1)

    # 流式调用（仅逐块打印 content）
    print("\n--- 流式 (仅 content 增量) ---")
    try:
        full = []
        for delta in ollama_chat_stream_content(
            [{"role": "user", "content": "Say hello in one short sentence."}],
            model=model,
            base_url=base_url,
            think=False,
            timeout=60,
        ):
            full.append(delta)
            print(delta, end="", flush=True)
        print()
        print("(完整长度:", len("".join(full)), ")")
    except Exception as e:
        print("错误:", e)

    print("\n完成。")


if __name__ == "__main__":
    main()
