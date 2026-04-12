"""共享 LLM 工具函数：指数退避重试、LLM 可用性检查。"""

import logging
import random
import time
from typing import Optional

logger = logging.getLogger(__name__)


def call_llm_with_backoff(
    processor,
    prompt: str,
    timeout: int = 60,
    max_waits: int = 5,
    backoff_base_seconds: int = 2,
) -> str:
    """调用 LLM（指数退避重试 + 抖动）。

    Args:
        processor: 含 llm_client 属性的处理器实例
        prompt: 发给 LLM 的 prompt
        timeout: 单次调用超时（秒）
        max_waits: 最大重试等待次数
        backoff_base_seconds: 退避基数（秒）

    Returns:
        LLM 响应文本

    Raises:
        RuntimeError: 重试耗尽后仍失败
    """
    last_error: Optional[str] = None
    max_attempts = max_waits + 1
    for attempt in range(1, max_attempts + 1):
        try:
            response = processor.llm_client._call_llm(
                prompt,
                max_retries=0,
                timeout=timeout,
                allow_mock_fallback=False,
            )
            if response is not None and isinstance(response, str) and len(response.strip()) > 0:
                return response
            last_error = "大模型未返回有效结果"
        except Exception as e:
            last_error = str(e)

        if attempt <= max_waits:
            wait_seconds = min(backoff_base_seconds ** attempt, 64)
            jitter = wait_seconds * (0.75 + random.random() * 0.5)
            print(f"[LLM] 访问失败，第 {attempt} 次重试前等待 {jitter:.1f}s；错误: {last_error}")
            time.sleep(jitter)

    raise RuntimeError(f"重试 {max_attempts} 次仍失败: {last_error or '未知错误'}")


def check_llm_available(processor, *, priority_steps=None) -> tuple[bool, Optional[str]]:
    """启动前握手：检查上游 LLM；若启用 alignment 专用通道，再按步骤优先级检查对齐端点。

    Args:
        processor: 处理器实例
        priority_steps: 可选的 LLM 优先级步骤列表（如 [6]），依次检查对应通道

    Returns:
        (成功与否, 错误信息或 None)
    """
    try:
        _ = call_llm_with_backoff(
            processor,
            "请只回复一个词：OK",
            timeout=60,
        )
        lc = processor.llm_client
        if getattr(lc, "alignment_enabled", False) and priority_steps:
            from processor.llm.client import LLM_PRIORITY_STEP6
            _old_pri = getattr(lc._priority_local, "priority", None)
            lc._priority_local.priority = LLM_PRIORITY_STEP6
            try:
                _ = call_llm_with_backoff(
                    processor,
                    "请只回复一个词：OK",
                    timeout=60,
                )
            finally:
                if _old_pri is not None:
                    lc._priority_local.priority = _old_pri
                else:
                    try:
                        del lc._priority_local.priority
                    except AttributeError:
                        pass
        return True, None
    except Exception as e:
        return False, f"大模型不可用: {e}"
