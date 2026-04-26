"""
Deep-Dream Find — 搜索功能

提供混合搜索（BM25 + 向量 + 图遍历）能力。
"""

from .hybrid import HybridSearcher

__all__ = ["HybridSearcher"]
