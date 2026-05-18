"""Shared utilities for the remember package."""

import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

from .helpers import _PAREN_ANNOTATION_RE

# Shared thread pool — reused across entity processing calls within a session
_ENTITY_POOL: list = [None]
_ENTITY_POOL_MAX: list = [1]

# Supplement pool for candidate enrichment (entity/relation batch fetches)
_SUPP_POOL: list = [None]
_SUPP_POOL_MAX: list = [1]

# BM25 pool for concept search parallelism
_BM25_POOL: list = [None]
_BM25_POOL_MAX: list = [2]
BM25_POOL_MAX = 4


def _get_entity_pool(max_workers: int) -> ThreadPoolExecutor:
    """Return (and lazily create) the shared entity ThreadPoolExecutor."""
    return _get_or_create_pool(_ENTITY_POOL, max_workers, _ENTITY_POOL_MAX, "entity")


def _get_supp_pool(max_workers: int = 2) -> ThreadPoolExecutor:
    """Return (and lazily create) the supplement ThreadPoolExecutor."""
    return _get_or_create_pool(_SUPP_POOL, max_workers, _SUPP_POOL_MAX, "supp")


def _get_bm25_pool(max_workers: int = 2) -> ThreadPoolExecutor:
    """Return (and lazily create) the BM25 search ThreadPoolExecutor."""
    return _get_or_create_pool(_BM25_POOL, max_workers, _BM25_POOL_MAX, "bm25")


def _doc_basename(source_document: str) -> str:
    """Extract basename from source_document path using rpartition."""
    return source_document.rpartition('/')[-1] if source_document else ""


# ---------------------------------------------------------------------------
# Name normalization (shared between entity_candidates.py and enrich mixin)
# ---------------------------------------------------------------------------

_TITLE_SUFFIXES_RE = re.compile(
    r'(?:教授|博士|先生|女士|同学|老师|工程师|经理|总监|院长|所长|主任|校长|站长|馆长|主编|首席|总裁'
    r'|部长|省长|市长|县长|区长|镇长|村长|将军|上校|中校|少校|大校|司令|参谋|政委|舰长|机长)$'
)


@lru_cache(maxsize=4096)
def normalize_entity_name_for_matching(name: str) -> str:
    """Strip parenthetical annotations and title suffixes for matching."""
    core = _PAREN_ANNOTATION_RE.sub('', name).strip()
    core = _TITLE_SUFFIXES_RE.sub('', core).strip()
    return core


def _get_or_create_pool(
    pool_ref: list,  # [ThreadPoolExecutor | None]
    max_workers: int,
    max_ref: list,   # [int]
    thread_prefix: str,
) -> ThreadPoolExecutor:
    """Return (and lazily create/upgrade) a shared ThreadPoolExecutor.

    Args:
        pool_ref: mutable container holding the current pool (or None)
        max_workers: requested max_workers
        max_ref: mutable container holding the current max_workers
        thread_prefix: thread name prefix
    """
    current = pool_ref[0]
    if current is not None:
        if max_workers > max_ref[0]:
            try:
                current.shutdown(wait=False)
            except Exception:
                pass
            pool_ref[0] = None
        else:
            return current
    max_ref[0] = max(max_workers, max_ref[0])
    pool_ref[0] = ThreadPoolExecutor(
        max_workers=max_ref[0],
        thread_name_prefix=thread_prefix,
    )
    return pool_ref[0]
