"""Shared utilities for the remember package."""

from concurrent.futures import ThreadPoolExecutor

from core.utils import entity_match_key

# Shared thread pool — reused across entity processing calls within a session
_ENTITY_POOL: list = [None]
_ENTITY_POOL_MAX: list = [1]

def _get_entity_pool(max_workers: int) -> ThreadPoolExecutor:
    """Return (and lazily create) the shared entity ThreadPoolExecutor."""
    return _get_or_create_pool(_ENTITY_POOL, max_workers, _ENTITY_POOL_MAX, "entity")


def _doc_basename(source_document: str) -> str:
    """Extract basename from source_document path using rpartition."""
    return source_document.rpartition('/')[-1] if source_document else ""


# ---------------------------------------------------------------------------
# Name normalization — P2 起委托 core.utils.entity_match_key（全库唯一语义）
# ---------------------------------------------------------------------------

def normalize_entity_name_for_matching(name: str) -> str:
    """实体名匹配键（括号注记/称号后缀剥离 + casefold）。

    历史实现不做 casefold——P2 统一后 'IBM' 与 'ibm' 在候选匹配层也视为同名。
    """
    return entity_match_key(name)


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
