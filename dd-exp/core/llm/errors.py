"""LLM 客户端异常类型（供 pipeline 与各 mixin 按类型捕获）。"""

from __future__ import annotations

from typing import Optional


class LLMContextBudgetExceeded(RuntimeError):
    """Input tokens exceed the model's context window.

    This is a permanent error for the given input; retrying won't help.
    """

    def __init__(self, message: str, *, context_window: Optional[int] = None,
                 estimated_tokens: Optional[int] = None):
        super().__init__(message)
        self.context_window = context_window
        self.estimated_tokens = estimated_tokens
