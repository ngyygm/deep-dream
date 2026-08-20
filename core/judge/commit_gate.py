"""FamilyWriteGate：family 写入的库级互斥 + 名称重验。

并发正确性核心（P4 已接入 entity 创建路径）：LLM/embedding 判断全部在门外
并行完成，创建新 family 前进入 write_txn 短临界区重验 `resolve_name` ——
若其他 worker 在本次候选检索之后新建了同名 family，则改在该 family 下创建
新版本，以此消灭并发下重复建 family 的竞态。入口见
`core/remember/entity.py::_gate_create_entity`（对已在候选列表中出现、
已被裁决为"同名不同概念"的名字，gate 不介入）。

注意 register 发生在 save 落盘之前：缓存命中的 fid 可能尚未提交。此时
`_gate_create_entity` 直接在该 pending fid 下建版本（不读库、不另建 family），
配合 `upsert_entity_family` 的原子 UPSERT（INSERT ... ON CONFLICT），两个
worker 无论保存顺序如何都收敛到单一 family 行。
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Optional, Set


class FamilyWriteGate:
    def __init__(self, storage=None, *, resolve_from_storage=None):
        """
        Args:
            storage: 存储管理器（提供 find_family_id_by_name 能力）
            resolve_from_storage: 可注入的名称解析函数 (原文名 -> fid|None)，
                                  测试/registry 用（registry 注入短只读连接版本）；
                                  缺省用 storage.find_family_id_by_name
        """
        self._lock = threading.RLock()
        self._names: dict = {}
        self._storage = storage
        if resolve_from_storage is not None:
            self._resolve_from_storage = resolve_from_storage
        else:
            self._resolve_from_storage = self._default_resolve

    def _default_resolve(self, name: str) -> Optional[str]:
        if self._storage is None:
            return None
        finder = getattr(self._storage, "find_family_id_by_name", None)
        if finder is None:
            return None
        try:
            return finder(name) or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    @contextmanager
    def write_txn(self) -> Iterator[None]:
        """family 写入临界区。进程内互斥；临界区内禁止任何 LLM/embedding 调用。"""
        with self._lock:
            yield

    def resolve_name(self, name: str) -> Optional[str]:
        """名称 -> 现存 fid（内存缓存优先，未命中查存储并缓存）。

        存储腿收原文名——解析器内部用 entity_name_variants 做变体召回，
        归一化后的键无法还原原文/核心名变体。
        """
        from .models import norm_name
        norm = norm_name(name)
        if not norm:
            return None
        with self._lock:
            if norm in self._names:
                return self._names[norm]
        fid = self._resolve_from_storage(name)
        with self._lock:
            # double-check：等待期间可能有人 register 了同名
            if norm in self._names:
                return self._names[norm]
            if fid:
                self._names[norm] = fid
        return fid or None

    def register(self, name: str, family_id: str) -> None:
        from .models import norm_name
        norm = norm_name(name)
        if not norm or not family_id:
            return
        with self._lock:
            self._names[norm] = family_id

    def invalidate(self, name: Optional[str] = None, family_id: Optional[str] = None) -> None:
        """合并/改名后清除缓存条目。两者都给时按 family_id 清。"""
        with self._lock:
            if family_id:
                for k in [k for k, v in self._names.items() if v == family_id]:
                    del self._names[k]
            elif name:
                from .models import norm_name as _nn
                self._names.pop(_nn(name), None)

    def known_names(self) -> Set[str]:
        with self._lock:
            return set(self._names.keys())
