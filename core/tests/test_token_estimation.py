"""P2.10：token 估算对 CJK 文本的偏差修复。

旧实现按 len(text) 逐字符计数：英文被高估约 4 倍（纯英文窗口易被误拒），
中文恰好 ≈1 token/字。统一改走 estimate_tokens：CJK 计 1 token/字，
其余 ASCII 可打印计 0.25 token/字符，混合文本线性叠加。
纯函数测试，不依赖真实 LLM/网络。
"""

import pytest

from core.llm.client import estimate_tokens
from core.llm.prompts import estimate_messages_token_count, estimate_text_token_count


def test_estimate_tokens_pure_chinese():
    """纯中文 N 字 → ≈N tokens（±20%）。"""
    text = "深度梦境是文档优先的概念图谱记忆服务器"
    n = len(text)
    assert n == 19
    got = estimate_tokens(text)
    assert got == pytest.approx(n, rel=0.2)
    assert got == n  # CJK 每字恰计 1 token


def test_estimate_tokens_pure_english():
    """纯英文 100 字符 → ≈25 tokens（4 chars/token）。"""
    text = "word " * 20  # 100 chars（含空格）
    assert len(text) == 100
    assert estimate_tokens(text) == 25


def test_estimate_tokens_mixed_linear():
    """混合文本 = CJK 部分 + 非 CJK 部分线性叠加。"""
    cjk_part = "红楼梦与曹雪芹"          # 7 个 CJK 字
    ascii_part = "classic Chinese novel"  # 21 个 ASCII 字符
    text = f"{cjk_part} {ascii_part}"
    assert estimate_tokens(text) == estimate_tokens(cjk_part) + estimate_tokens(" " + ascii_part)


def test_estimate_tokens_fullwidth_punct_counts_as_cjk():
    """全角标点（，。「」（）ＡＢ 等 U+FF00-U+FFEF）按 CJK 计 1 token/字。"""
    assert estimate_tokens("，。！？「」『』（）＝ＡＢ") == 13
    # CJK 标点 U+3000-U+303F
    assert estimate_tokens("　〈〉《》【】") == 7


def test_estimate_tokens_edge_cases():
    """None / 空串 / 非字符串输入。"""
    assert estimate_tokens(None) == 0
    assert estimate_tokens("") == 0
    assert estimate_tokens(12345) == estimate_tokens("12345")


def test_estimate_tokens_old_entrypoint_delegates():
    """旧入口 estimate_text_token_count 统一委托 estimate_tokens。"""
    assert estimate_text_token_count("你好abc") == estimate_tokens("你好abc")


def test_estimate_messages_token_count_uses_estimator():
    """messages 估算（含多段 content）各部分统一走 estimate_tokens。"""
    zh = "知识图谱抽取引擎"
    en = "knowledge graph extraction"
    messages = [
        {"role": "system", "content": zh},
        {"role": "user", "content": [
            {"type": "text", "text": en},
            "plain string part",
            {"type": "image_url", "image_url": {"url": "http://example.com/a.png"}},
        ]},
    ]
    # 固定开销：每条 message 8 + 尾部 16；role 与每个 part 的所有字符串值都走估算
    text_part = {"type": "text", "text": en}
    image_part = {"type": "image_url", "image_url": {"url": "http://example.com/a.png"}}
    expected = (
        8 + estimate_tokens("system") + estimate_tokens(zh)
        + 8 + estimate_tokens("user")
        + sum(estimate_tokens(v) for v in text_part.values() if isinstance(v, str))
        + estimate_tokens("plain string part")
        + sum(estimate_tokens(v) for v in image_part.values() if isinstance(v, str))
        + 16
    )
    assert estimate_messages_token_count(messages) == expected


def test_english_no_longer_overestimated_fourfold():
    """回归：英文预算预检不再按 len() 高估 4 倍导致误拒。"""
    english_1k_chars = "abcdefghij" * 100  # 1000 chars → 250 tokens
    assert estimate_tokens(english_1k_chars) == 250
    # 旧实现会返回 1000，超过 512 的窗口预算；现在 250 可以通过
    assert estimate_tokens(english_1k_chars) < 512
