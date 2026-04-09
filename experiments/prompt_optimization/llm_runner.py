"""Lightweight LLM caller for experiments."""
import time
from typing import Any, Dict, List, Optional

from .config import LLM_CONFIG


class ExperimentLLMRunner:
    """Direct LLM caller bypassing LLMClient complexity."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or LLM_CONFIG
        try:
            from openai import OpenAI
            import httpx
            self.client = OpenAI(
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
                http_client=httpx.Client(timeout=httpx.Timeout(cfg.get("timeout", 120))),
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self.model = cfg["model"]
        self.max_tokens = cfg.get("max_tokens", 5000)

    def call(self, system_prompt: str, user_prompt: str, max_retries: int = 2) -> str:
        """Call LLM with system + user. Returns raw response text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.call_messages(messages, max_retries)

    def call_messages(self, messages: List[Dict], max_retries: int = 2) -> str:
        """Call LLM with arbitrary message list."""
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~1.5 chars per token for Chinese."""
        return int(len(text) * 0.7)
