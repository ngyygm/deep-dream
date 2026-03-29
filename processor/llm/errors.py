"""LLM 客户端异常类型（供 pipeline 与各 mixin 按类型捕获）。"""


class LLMContextBudgetExceeded(RuntimeError):
    """估算输入已超过 llm.context_window_tokens，无法再为本请求预留输出额度。"""
