"""
LLM 输入传输层清理（P5.5 收敛）。

只做不改写语义的传输级清理：截断超长、剔除控制字符。此前本模块对文档
内容做提示注入模式改写（[REDACTED] 替换 "ignore previous instructions"
等字样、折叠连续空格/换行）——但这类字样在文献、小说、安全研究语料中
是正常内容，改写会破坏"原始文件即 Source of Truth"原则：清洗后的文本
与用户上传文件不再一致，chunk 哈希与语义都会漂移。

防注入边界收窄到指令构造层（core/llm/prompts.py）：文档内容作为数据
进入 prompt，抽取指令不依赖内容中的任何指令性文字。本模块不再承担
内容改写职责。
"""
from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# 剔除 C0 控制字符与 DEL，保留 \t(9) \n(10) \r(13)（正常文档排版的一部分；
# 保留 \r 使 CRLF 文件不被静默改写为 LF）。
# 注意：str.translate 的 dict 键必须是整数序号——单字符字符串键会静默不命中。
_STRIP_TABLE = {i: None for i in range(32) if i not in (9, 10, 13)}
_STRIP_TABLE[127] = None


def clean_document_text(text: str, max_length: int = 100_000) -> Tuple[str, bool]:
    """
    传输级清理：截断超长 + 剔除控制字符。不改写任何可见内容。

    Returns:
        (cleaned_text, was_modified) 元组；was_modified=True 仅表示发生过
        截断或控制字符剔除，正文文字永远原样保留。
    """
    if not text:
        return "", False

    was_modified = False

    # 1. 剔除控制字符（含 null bytes）
    cleaned = text.translate(_STRIP_TABLE)
    if cleaned != text:
        text = cleaned
        was_modified = True
        logger.info("Document input contained control characters; they were stripped")

    # 2. 截断超长输入
    if len(text) > max_length:
        original_len = len(text)
        text = text[:max_length]
        was_modified = True
        logger.warning(
            "LLM input truncated from %d to %d characters",
            original_len, max_length
        )

    return text, was_modified
