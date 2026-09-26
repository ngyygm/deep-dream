"""
DeepDream 自定义异常。

使用方式：
    from core.exceptions import ConfigError

    raise ConfigError("配置校验失败")
"""
from __future__ import annotations


class ConfigError(Exception):
    """配置校验异常（缺少必填字段、值超出范围）。"""
