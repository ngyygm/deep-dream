"""
DeepDream 自定义异常层次结构。

使用方式：
    from core.exceptions import DeepDreamError, StorageError, LLMError

    raise StorageError("数据库写入失败")
"""
from __future__ import annotations


class DeepDreamError(Exception):
    """DeepDream 基础异常。"""


class StorageError(DeepDreamError):
    """存储层异常（Neo4j、文件 I/O）。"""


class LLMError(DeepDreamError):
    """LLM 调用异常（API 错误、超时、响应解析失败）。"""


class QueueError(DeepDreamError):
    """任务队列异常（提交失败、重试耗尽）。"""


class ConfigError(DeepDreamError):
    """配置校验异常（缺少必填字段、值超出范围）。"""
