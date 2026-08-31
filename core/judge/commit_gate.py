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
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional


class FamilyWriteGate:
    def __init__(self, storage=None, *, resolve_from_storage=None,
                 pending_trust_seconds: float = 900.0,
                 clock: Optional[Callable[[], float]] = None):
        """
        Args:
            storage: 存储管理器（提供 find_family_id_by_name 能力）
            resolve_from_storage: 可注入的名称解析函数 (原文名 -> fid|None)，
                                  测试/registry 用（registry 注入短只读连接版本）；
                                  缺省用 storage.find_family_id_by_name
            pending_trust_seconds: register 后的 pending 信任窗（秒）。
            clock: 可注入的单调时钟（测试用），缺省 time.monotonic
        """
        self._lock = threading.RLock()
        self._names: dict = {}
        # register 登记但尚未在库中确认过的 fid -> 登记时刻（monotonic）。
        # 信任窗内视作并发在途（不读库判死，配合 UPSERT 收敛）；超窗后
        # save 早已落盘或已失败，恢复"读库验死"的兜底——否则跨进程合并
        # 删掉的 fid 会因永久 pending 而被复活（f2 残留）。窗口取 15 min：
        # 窗口批量对齐的 LLM 调用可能把 build→save 间隔拖到分钟级。
        self._pending: dict = {}
        self._pending_trust_seconds = pending_trust_seconds
        self._clock = clock or time.monotonic
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
        """登记新建 family 的名称映射，并标记 fid 为信任窗内 pending。

        register 发生在 save 落盘之前：窗口内的 pending fid 不做读库验死，
        直接在其下建版本、依赖 UPSERT 收敛。对存储中已确认存在的 family
        请用 remember()（只更新缓存），反复 register 会让 pending 永久
        膨胀、死 fid 兜底被永久跳过。
        """
        from .models import norm_name
        norm = norm_name(name)
        if not norm or not family_id:
            return
        with self._lock:
            self._names[norm] = family_id
            self._pending[family_id] = self._clock()

    def remember(self, name: str, family_id: str) -> None:
        """只更新名称缓存，不标记 pending——用于存储中已确认的 family。"""
        from .models import norm_name
        norm = norm_name(name)
        if not norm or not family_id:
            return
        with self._lock:
            self._names[norm] = family_id

    def is_pending(self, family_id: str) -> bool:
        """该 fid 是否处于 register 后的信任窗内（save 可能还在门外）。

        超窗即不再按"并发在途"信任——save 已落盘（或已失败），读库验死
        的死 fid 兜底恢复生效。过期条目顺手清理。
        """
        with self._lock:
            ts = self._pending.get(family_id)
            if ts is None:
                return False
            if (self._clock() - ts) >= self._pending_trust_seconds:
                del self._pending[family_id]
                return False
            return True

    def invalidate(self, name: Optional[str] = None, family_id: Optional[str] = None) -> None:
        """合并/改名后清除缓存条目。两者都给时按 family_id 清。

        被清除的 fid 同时退出 pending 集合——合并/删除路径失效的 fid
        已不可作为写入目标，不得再按"并发在途"信任。
        """
        with self._lock:
            if family_id:
                for k in [k for k, v in self._names.items() if v == family_id]:
                    del self._names[k]
                self._pending.pop(family_id, None)
            elif name:
                from .models import norm_name as _nn
                fid = self._names.pop(_nn(name), None)
                if fid:
                    self._pending.pop(fid, None)

    def known_names(self) -> set:
        with self._lock:
            return set(self._names.keys())
