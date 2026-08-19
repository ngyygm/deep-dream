"""裁决结果持久化 memo：LRU 前置 + SQLite 落盘 + TTL。

- 进程内 LRU（默认 4096）挡住热路径；未命中再查 SQLite（judge_verdicts 表）
- 写穿：put 同时进 LRU 与 SQLite 批量缓冲，缓冲满或 close 时单事务落盘
- TTL 默认 7 天；按 family_id 失效用于实体合并后清除受污染的判断
- 线程安全：单连接 + 锁串行化（judge 调用频率远低于 embedding，可接受）
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 7 * 24 * 3600
_FLUSH_THRESHOLD = 64


def ensure_judge_tables(conn: sqlite3.Connection) -> None:
    """建表（幂等）。schema_v15.init_schema_v15 也会建同样的表；这里兜底
    旧库升级路径与独立 db 文件场景。"""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS judge_verdicts (
        ns TEXT NOT NULL,
        k TEXT NOT NULL,
        verdict_json TEXT NOT NULL,
        family_ids_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        expires_at REAL NOT NULL,
        PRIMARY KEY (ns, k)
    )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_judge_verdicts_expiry "
        "ON judge_verdicts(expires_at)"
    )
    conn.commit()


class VerdictMemo:
    def __init__(self, db_path: Optional[str] = None,
                 *, ttl_seconds: int = _DEFAULT_TTL,
                 lru_size: int = 4096,
                 conn: Optional[sqlite3.Connection] = None):
        self._ttl = max(60, int(ttl_seconds))
        self._lru_size = max(64, int(lru_size))
        self._lru: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._pending: List[Tuple[str, str, str, str, float]] = []
        self._own_conn = conn is None
        if conn is not None:
            self._conn = conn
        else:
            path = Path(db_path) if db_path else Path(":memory:")
            if str(path) != ":memory:":
                path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(path), check_same_thread=False, timeout=10.0)
            if str(path) != ":memory:":
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA busy_timeout=5000")
                except sqlite3.Error:
                    pass
            ensure_judge_tables(self._conn)
        self._stats = {"hits_lru": 0, "hits_db": 0, "misses": 0, "puts": 0}

    # ------------------------------------------------------------------
    def get(self, ns: str, key: str) -> Optional[Dict[str, Any]]:
        ck = (ns, key)
        with self._lock:
            if ck in self._lru:
                self._lru.move_to_end(ck)
                self._stats["hits_lru"] += 1
                return self._lru[ck]
        try:
            row = self._conn.execute(
                "SELECT verdict_json, expires_at FROM judge_verdicts WHERE ns=? AND k=?",
                (ns, key),
            ).fetchone()
        except sqlite3.Error as exc:
            logger.debug("judge memo read failed: %s", exc)
            row = None
        if row is None:
            with self._lock:
                self._stats["misses"] += 1
            return None
        verdict_json, expires_at = row
        if expires_at is not None and float(expires_at) <= time.time():
            # 过期条目惰性清除
            try:
                self._conn.execute(
                    "DELETE FROM judge_verdicts WHERE ns=? AND k=?", (ns, key))
                self._conn.commit()
            except sqlite3.Error:
                pass
            with self._lock:
                self._stats["misses"] += 1
            return None
        try:
            verdict = json.loads(verdict_json)
        except (ValueError, TypeError):
            with self._lock:
                self._stats["misses"] += 1
            return None
        if not isinstance(verdict, dict):
            with self._lock:
                self._stats["misses"] += 1
            return None
        with self._lock:
            self._lru[ck] = verdict
            self._lru.move_to_end(ck)
            if len(self._lru) > self._lru_size:
                self._lru.popitem(last=False)
            self._stats["hits_db"] += 1
        return verdict

    def put(self, ns: str, key: str, verdict: Dict[str, Any],
            family_ids: Optional[List[str]] = None) -> None:
        if not isinstance(verdict, dict):
            return
        ck = (ns, key)
        now = time.time()
        expires = now + self._ttl
        payload = json.dumps(verdict, ensure_ascii=False)
        fams = json.dumps(sorted(set(family_ids or [])), ensure_ascii=False)
        with self._lock:
            self._lru[ck] = verdict
            self._lru.move_to_end(ck)
            if len(self._lru) > self._lru_size:
                self._lru.popitem(last=False)
            self._pending.append((ns, key, payload, fams, expires))
            self._stats["puts"] += 1
            should_flush = len(self._pending) >= _FLUSH_THRESHOLD
        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            batch = self._pending
            self._pending = []
        if not batch:
            return
        try:
            self._conn.executemany(
                "INSERT OR REPLACE INTO judge_verdicts "
                "(ns, k, verdict_json, family_ids_json, created_at, expires_at) "
                "VALUES (?,?,?,?,datetime('now'),?)",
                batch,
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("judge memo flush failed (%d items dropped): %s",
                           len(batch), exc)

    def invalidate_for_families(self, family_ids: List[str]) -> int:
        """删除所有涉及给定 family 的记录（合并后判断已失真）。"""
        fids = [str(f) for f in family_ids if f]
        if not fids:
            return 0
        removed = 0
        try:
            for fid in fids:
                cur = self._conn.execute(
                    "DELETE FROM judge_verdicts WHERE family_ids_json LIKE ?",
                    (f'%"{fid}"%',),
                )
                removed += cur.rowcount or 0
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("judge memo invalidate failed: %s", exc)
        # LRU 值不带 family 索引——失效很少发生（仅在合并后），整体清空保守但正确
        with self._lock:
            self._lru.clear()
        return removed

    def purge_expired(self) -> int:
        try:
            cur = self._conn.execute(
                "DELETE FROM judge_verdicts WHERE expires_at <= ?", (time.time(),))
            self._conn.commit()
            return cur.rowcount or 0
        except sqlite3.Error as exc:
            logger.warning("judge memo purge failed: %s", exc)
            return 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            out = dict(self._stats)
            out["pending"] = len(self._pending)
            out["lru_size"] = len(self._lru)
        return out

    def close(self) -> None:
        self.flush()
        if self._own_conn:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
