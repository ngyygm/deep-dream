"""
Embedding客户端：支持自定义embedding模型，内置内容哈希缓存
"""
import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils import wprint_info


class _EmbeddingCache:
    """Thread-safe LRU cache keyed by content hash.

    Avoids re-encoding identical text across the same remember() call or
    across multiple requests. Entries expire after TTL to prevent unbounded
    growth and to allow model warm-up differences to be discarded.

    Statistics are maintained lock-free via atomic counters where possible;
    size() and stats() acquire the lock for a consistent snapshot.
    """

    def __init__(self, max_size: int = 8192, default_ttl: float = 300.0):
        self._cache: OrderedDict[str, Tuple[float, np.ndarray]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.Lock()

        # Statistics -- simple integers, updated under _lock
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def _content_hash(text: str) -> str:
        """SHA-256 of UTF-8 encoded text, truncated to 16 hex chars."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get(self, text: str) -> Optional[np.ndarray]:
        """Look up a single text. Returns None on miss (caller should encode)."""
        key = self._content_hash(text)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            expire_at, value = entry
            if time.monotonic() > expire_at:
                del self._cache[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def get_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Look up multiple texts. Returns list parallel to input; None means cache miss."""
        results: List[Optional[np.ndarray]] = []
        with self._lock:
            for text in texts:
                key = self._content_hash(text)
                entry = self._cache.get(key)
                if entry is not None:
                    expire_at, value = entry
                    if time.monotonic() <= expire_at:
                        self._cache.move_to_end(key)
                        self._hits += 1
                        results.append(value)
                        continue
                    # Expired
                    del self._cache[key]
                self._misses += 1
                results.append(None)
        return results

    def set(self, text: str, embedding: np.ndarray, ttl: Optional[float] = None) -> None:
        """Store a single text -> embedding mapping."""
        key = self._content_hash(text)
        with self._lock:
            if len(self._cache) >= self._max_size:
                self._evict_locked()
            self._cache[key] = (time.monotonic() + (ttl or self._default_ttl), embedding)
            self._cache.move_to_end(key)

    def set_batch(self, texts: List[str], embeddings: np.ndarray,
                  ttl: Optional[float] = None) -> None:
        """Store multiple text -> embedding mappings in one locked section."""
        if not texts:
            return
        expire_at = time.monotonic() + (ttl or self._default_ttl)
        with self._lock:
            for i, text in enumerate(texts):
                key = self._content_hash(text)
                if len(self._cache) >= self._max_size:
                    self._evict_locked()
                self._cache[key] = (expire_at, embeddings[i])
                self._cache.move_to_end(key)

    def _evict_locked(self) -> None:
        """Evict expired entries, then oldest 10% if still over capacity. Caller holds lock."""
        now = time.monotonic()
        while self._cache:
            key = next(iter(self._cache))
            expire_at, _ = self._cache[key]
            if now > expire_at:
                self._cache.popitem(last=False)
            else:
                break
        if len(self._cache) >= self._max_size:
            remove_count = max(1, self._max_size // 10)
            for _ in range(remove_count):
                if self._cache:
                    self._cache.popitem(last=False)

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def stats(self) -> Dict[str, int]:
        """Return hit/miss/size statistics."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_size": self._max_size,
            }

    def invalidate(self) -> None:
        """Clear all entries and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


class EmbeddingClient:
    """Embedding客户端 - 支持多种embedding模型，内置内容哈希缓存"""

    def __init__(self, model_path: Optional[str] = None, model_name: Optional[str] = None,
                 device: str = "cpu", use_local: bool = True,
                 cache_max_size: int = 8192, cache_ttl: float = 300.0):
        """
        初始化Embedding客户端

        Args:
            model_path: 本地模型路径（优先使用）
            model_name: 模型名称（如果使用HuggingFace模型）
            device: 计算设备 ("cpu" 或 "cuda")
            use_local: 是否优先使用本地模型
            cache_max_size: 嵌入缓存最大条目数（默认8192）
            cache_ttl: 缓存条目TTL秒数（默认300秒/5分钟）
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.use_local = use_local
        self.model = None
        # Semaphore allows parallel batch encodes while bounding concurrency.
        # CPU-aware scaling: min(cpu_count, 8) for optimal throughput without oversubscription.
        # sentence-transformers is thread-safe for inference.
        _sem_value = min(os.cpu_count() or 4, 8)
        self._encode_semaphore = threading.Semaphore(_sem_value)

        # Content-hash embedding cache
        self._cache = _EmbeddingCache(max_size=cache_max_size, default_ttl=cache_ttl)

        self._init_model()

    def _init_model(self):
        """初始化embedding模型"""
        try:
            from sentence_transformers import SentenceTransformer

            if self.model_path and self.use_local:
                # 使用本地模型路径
                wprint_info(f"加载本地embedding模型: {self.model_path}")
                self.model = SentenceTransformer(
                    self.model_path,
                    device=self.device,
                    trust_remote_code=True
                )
            elif self.model_name:
                # 使用HuggingFace模型名称
                wprint_info(f"加载HuggingFace embedding模型: {self.model_name}")
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=True
                )
            else:
                # 使用默认模型
                wprint_info("使用默认embedding模型: all-MiniLM-L6-v2")
                self.model = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    device=self.device
                )
        except ImportError:
            self.model = None
            wprint_info("警告：未安装sentence-transformers库，将使用文本相似度搜索")
            wprint_info("安装命令: pip install sentence-transformers")
        except Exception as e:
            self.model = None
            wprint_info(f"警告：embedding 模型加载失败，将使用文本相似度搜索: {e}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        编码文本为向量（线程安全，带缓存）

        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小

        Returns:
            向量数组（numpy array）
        """
        if self.model is None:
            return None

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        if not texts:
            return None

        # --- Cache lookup: partition into hits and misses ---
        cached_results = self._cache.get_batch(texts)
        miss_indices = [i for i, v in enumerate(cached_results) if v is None]
        miss_texts = [texts[i] for i in miss_indices]

        # All cache hits
        if not miss_texts:
            return np.stack(cached_results)

        # --- Encode only the misses ---
        miss_embeddings = self._encode_uncached(miss_texts, batch_size)
        if miss_embeddings is None:
            # Encode failed -- return whatever we have from cache, or None
            hit_results = [r for r in cached_results if r is not None]
            return np.stack(hit_results) if hit_results else None

        # --- Store misses in cache ---
        self._cache.set_batch(miss_texts, miss_embeddings)

        # --- Merge cached + freshly encoded ---
        # Build result array in input order
        dim = miss_embeddings.shape[1]
        if single_input:
            # Optimization: single text, return 1-D or 1-row array matching old behavior
            if cached_results[0] is not None:
                return cached_results[0]
            return miss_embeddings[0]

        result = np.empty((len(texts), dim), dtype=np.float32)
        miss_idx = 0
        for i in range(len(texts)):
            if cached_results[i] is not None:
                result[i] = cached_results[i]
            else:
                result[i] = miss_embeddings[miss_idx]
                miss_idx += 1
        return result

    def _encode_uncached(self, texts: List[str], batch_size: int) -> Optional[np.ndarray]:
        """Encode texts that are not in cache. Internal method."""
        if len(texts) > batch_size:
            chunks = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            results = []
            for chunk in chunks:
                emb = self._encode_chunk(chunk, batch_size)
                if emb is None:
                    return None
                results.append(emb)
            return np.concatenate(results, axis=0)

        return self._encode_chunk(texts, batch_size)

    def _encode_chunk(self, texts: List[str], batch_size: int) -> Optional[np.ndarray]:
        """编码单批文本，使用信号量控制并发。"""
        with self._encode_semaphore:
            try:
                return self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            except Exception as e:
                wprint_info(f"Embedding编码错误: {e}")
                return None

    def encode_uncached(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        编码文本为向量，绕过缓存（用于需要确保结果不共享的场景）。

        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小

        Returns:
            向量数组（numpy array）
        """
        if self.model is None:
            return None

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return None

        return self._encode_uncached(texts, batch_size)

    def is_available(self) -> bool:
        """检查embedding模型是否可用"""
        return self.model is not None

    # ------------------------------------------------------------------
    # Cache management & statistics
    # ------------------------------------------------------------------

    def cache_stats(self) -> Dict[str, int]:
        """Return embedding cache statistics: hits, misses, size, max_size."""
        return self._cache.stats()

    def cache_invalidate(self) -> None:
        """Clear the embedding cache entirely."""
        self._cache.invalidate()

    def cache_size(self) -> int:
        """Return current number of entries in the embedding cache."""
        return self._cache.size()
