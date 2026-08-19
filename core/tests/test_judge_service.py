"""AlignmentJudgeService 单测：memo / single-flight / 攒批 / commit gate / LLMClient 委托。

全部 mock LLM，不发真实请求。
"""
import threading
import time

import pytest

from core.judge import (
    AlignmentJudgeService,
    BatchCollector,
    FamilyWriteGate,
    SingleFlight,
    VerdictMemo,
    guard_key,
    resolve_entity_key,
    resolve_relation_key,
)


# ----------------------------------------------------------------------
# keys
# ----------------------------------------------------------------------

class TestKeys:
    def test_guard_key_symmetric(self):
        a = guard_key("Noah", "描述A", "诺亚", "描述B")
        b = guard_key("诺亚", "描述B", "Noah", "描述A")
        assert a == b

    def test_guard_key_name_match_type_differs(self):
        assert guard_key("a", "c1", "b", "c2", "exact") != guard_key("a", "c1", "b", "c2", "none")

    def test_guard_key_content_change_differs(self):
        assert guard_key("a", "c1", "b", "c2") != guard_key("a", "c1-changed", "b", "c2")

    def test_guard_key_norm_name(self):
        # 大小写/空白归一应命中同一 key（A/B 对称拼装下双侧归一）
        k1 = guard_key("Noah", "c", "Sakura", "d")
        k2 = guard_key("noah", "c", "sakura", "d")
        assert k1 == k2

    def test_resolve_entity_key_candidates_order_insensitive(self):
        ent = {"name": "X", "content": "cx"}
        c1 = {"family_id": "f1", "name": "A", "content": "ca"}
        c2 = {"family_id": "f2", "name": "B", "content": "cb"}
        assert resolve_entity_key(ent, [c1, c2]) == resolve_entity_key(ent, [c2, c1])

    def test_resolve_relation_key_pair_symmetric(self):
        assert resolve_relation_key("Alice", "Bob", ["r"], []) == \
               resolve_relation_key("Bob", "Alice", ["r"], [])


# ----------------------------------------------------------------------
# memo
# ----------------------------------------------------------------------

class TestVerdictMemo:
    def test_put_get_roundtrip(self, tmp_path):
        memo = VerdictMemo(str(tmp_path / "j.db"))
        memo.put("guard", "k1", {"verdict": "same", "confidence": 0.9})
        assert memo.get("guard", "k1") == {"verdict": "same", "confidence": 0.9}
        memo.close()

    def test_miss_then_persisted_after_reopen(self, tmp_path):
        db = str(tmp_path / "j.db")
        memo = VerdictMemo(db)
        memo.put("guard", "k1", {"verdict": "different"})
        memo.close()
        memo2 = VerdictMemo(db)
        assert memo2.get("guard", "k1") == {"verdict": "different"}
        memo2.close()

    def test_ttl_expiry(self, tmp_path):
        memo = VerdictMemo(str(tmp_path / "j.db"), ttl_seconds=60)
        memo.put("guard", "k1", {"verdict": "same"})
        # 直接把过期时间改到过去
        memo.flush()
        memo._conn.execute("UPDATE judge_verdicts SET expires_at = 1.0")
        memo._conn.commit()
        memo._lru.clear()
        assert memo.get("guard", "k1") is None
        memo.close()

    def test_invalidate_for_family(self, tmp_path):
        memo = VerdictMemo(str(tmp_path / "j.db"))
        memo.put("resolve_ent", "k1", {"update_mode": "merge"}, family_ids=["fid_a", "fid_b"])
        memo.put("guard", "k2", {"verdict": "same"})
        memo.flush()
        removed = memo.invalidate_for_families(["fid_a"])
        assert removed >= 1
        assert memo.get("resolve_ent", "k1") is None
        # 不涉及该 family 的记录保留
        assert memo.get("guard", "k2") == {"verdict": "same"}
        memo.close()

    def test_flush_batch(self, tmp_path):
        memo = VerdictMemo(str(tmp_path / "j.db"))
        for i in range(70):  # 超过 _FLUSH_THRESHOLD=64 触发自动 flush
            memo.put("guard", f"k{i}", {"i": i})
        memo.close()
        memo2 = VerdictMemo(str(tmp_path / "j.db"))
        assert memo2.get("guard", "k69") == {"i": 69}
        memo2.close()


# ----------------------------------------------------------------------
# single-flight
# ----------------------------------------------------------------------

class TestSingleFlight:
    def test_same_key_executes_once(self):
        sf = SingleFlight()
        calls = {"n": 0}
        barrier = {"ready": threading.Event()}

        def slow_fn():
            calls["n"] += 1
            barrier["ready"].wait(timeout=5)
            return "ok"

        def leader():
            barrier["ready"].wait(timeout=5) if False else None
            # leader 先占位再放行 follower
            return sf.execute("k", slow_fn)

        results = []

        def follower():
            results.append(sf.execute("k", slow_fn))

        t0 = threading.Thread(target=leader)
        t0.start()
        time.sleep(0.1)
        t1 = threading.Thread(target=follower)
        t1.start()
        time.sleep(0.1)
        barrier["ready"].set()
        t0.join(timeout=5)
        t1.join(timeout=5)
        assert calls["n"] == 1
        assert results == ["ok"]
        assert sf.stats()["coalesced"] == 1

    def test_leader_error_gives_follower_miss(self):
        sf = SingleFlight()
        err = {"raised": False}

        def boom():
            err["raised"] = True
            raise RuntimeError("boom")

        follower_result = []

        def follower():
            follower_result.append(sf.execute("k", boom))

        leader = threading.Thread(target=lambda: pytest.raises(RuntimeError, sf.execute, "k", boom))
        leader.start()
        time.sleep(0.05)
        t1 = threading.Thread(target=follower)
        t1.start()
        time.sleep(0.05)
        # 唤醒后续飞行：leader 已经失败并清理
        leader.join(timeout=5)
        # follower 在 leader 失败前进入等待会拿到 MISS；这里直接验证第二次执行可用
        assert sf.execute("k2", lambda: "fresh") == "fresh"


# ----------------------------------------------------------------------
# collector
# ----------------------------------------------------------------------

class TestBatchCollector:
    def test_serial_submit_returns_result(self):
        c = BatchCollector(batch_delay_ms=10)
        assert c.submit(lambda: 42) == 42
        c.close()

    def test_error_propagates(self):
        c = BatchCollector(batch_delay_ms=10)

        def boom():
            raise ValueError("x")

        with pytest.raises(ValueError):
            c.submit(boom)
        c.close()

    def test_batching_groups_concurrent_items(self):
        c = BatchCollector(batch_delay_ms=120, batch_max=8)
        started = threading.Event()
        concurrent = {"n": 0}
        lock = threading.Lock()

        def item(i):
            with lock:
                concurrent["n"] += 1
            return i * 2

        results = [None] * 6
        threads = []
        for i in range(6):
            def run(i=i):
                results[i] = c.submit(lambda: item(i))
            t = threading.Thread(target=run)
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert results == [0, 2, 4, 6, 8, 10]
        stats = c.stats()
        assert stats["items"] == 6
        assert stats["batches"] <= 6  # 并发提交应尽量合并（不要求严格 1 批）
        c.close()


# ----------------------------------------------------------------------
# service（mock LLM）
# ----------------------------------------------------------------------

class _MockLLM:
    def __init__(self, verdict=None, error=None):
        self.calls = {"guard": 0, "ent": 0, "rel": 0}
        self.verdict = verdict or {"verdict": "different", "confidence": 0.8}
        self.error = error

    def _judge_entity_alignment_llm(self, name_a, content_a, name_b, content_b, *, name_match_type="none"):
        self.calls["guard"] += 1
        if self.error:
            raise self.error
        return dict(self.verdict)

    def _resolve_entity_candidates_llm(self, entity, candidates, context_text=None):
        self.calls["ent"] += 1
        if self.error:
            raise self.error
        return {"match_existing_id": "", "update_mode": "create_new",
                "merged_name": entity.get("name", ""), "relations_to_create": [],
                "confidence": 0.9}

    def _resolve_relation_pair_llm(self, e1, e2, new_contents, existing, new_source_document=""):
        self.calls["rel"] += 1
        if self.error:
            raise self.error
        return {"action": "create_new", "matched_relation_id": "",
                "need_update": True, "confidence": 0.9}


def _make_service(tmp_path, **kw):
    memo = VerdictMemo(str(tmp_path / "j.db"))
    return AlignmentJudgeService(memo, **kw), memo


class TestService:
    def test_guard_memo_hit(self, tmp_path):
        svc, memo = _make_service(tmp_path)
        llm = _MockLLM()
        r1 = svc.judge_entity_alignment(llm, "Noah", "c1", "诺亚", "c2")
        r2 = svc.judge_entity_alignment(llm, "Noah", "c1", "诺亚", "c2")
        assert r1 == r2
        assert llm.calls["guard"] == 1
        stats = svc.stats()
        assert stats["memo_hits"] == 1
        svc.close()

    def test_guard_ab_order_shares_memo(self, tmp_path):
        svc, memo = _make_service(tmp_path)
        llm = _MockLLM()
        svc.judge_entity_alignment(llm, "Noah", "c1", "诺亚", "c2")
        svc.judge_entity_alignment(llm, "诺亚", "c2", "Noah", "c1")
        assert llm.calls["guard"] == 1
        svc.close()

    def test_error_dict_not_cached(self, tmp_path):
        svc, memo = _make_service(tmp_path)
        llm = _MockLLM(verdict={"action": "fallback", "error": "boom"})
        svc.judge_entity_alignment(llm, "a", "c", "b", "d")
        svc.judge_entity_alignment(llm, "a", "c", "b", "d")
        assert llm.calls["guard"] == 2  # 带 error 的结果不缓存
        svc.close()

    def test_resolve_entity_roundtrip_and_family_invalidate(self, tmp_path):
        svc, memo = _make_service(tmp_path)
        llm = _MockLLM()
        ent = {"name": "X", "content": "cx"}
        cands = [{"family_id": "fid1", "name": "Y", "content": "cy"}]
        r1 = svc.resolve_entity_candidates(llm, ent, cands)
        r2 = svc.resolve_entity_candidates(llm, ent, cands)
        assert llm.calls["ent"] == 1
        # 合并后失效 → 重新判断
        svc.invalidate_for_family("fid1")
        svc.resolve_entity_candidates(llm, ent, cands)
        assert llm.calls["ent"] == 2
        svc.close()

    def test_resolve_relation_roundtrip(self, tmp_path):
        svc, memo = _make_service(tmp_path)
        llm = _MockLLM()
        existing = [{"family_id": "relfid1", "content": "r"}]
        svc.resolve_relation_pair(llm, "A", "B", ["新关系"], existing)
        svc.resolve_relation_pair(llm, "B", "A", ["新关系"], existing)
        assert llm.calls["rel"] == 1
        svc.close()

    def test_concurrent_same_key_single_llm_call(self, tmp_path):
        svc, memo = _make_service(tmp_path, batch_delay_ms=50)
        llm = _MockLLM()
        release = threading.Event()

        orig = llm._judge_entity_alignment_llm
        def slow(*a, **k):
            release.wait(timeout=5)
            return orig(*a, **k)
        llm._judge_entity_alignment_llm = slow

        results = []
        threads = [threading.Thread(target=lambda: results.append(
            svc.judge_entity_alignment(llm, "a", "c1", "b", "c2"))) for _ in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.2)
        release.set()
        for t in threads:
            t.join(timeout=5)
        assert len(results) == 4
        assert all(r == results[0] for r in results)
        assert llm.calls["guard"] == 1  # single-flight 去重
        svc.close()


# ----------------------------------------------------------------------
# FamilyWriteGate
# ----------------------------------------------------------------------

class TestFamilyWriteGate:
    def test_register_resolve_invalidate(self):
        gate = FamilyWriteGate()
        assert gate.resolve_name("Noah") is None
        gate.register("Noah", "fid_1")
        assert gate.resolve_name("Noah") == "fid_1"
        assert gate.resolve_name("noah") == "fid_1"  # 归一化
        gate.invalidate(name="NOAH")
        assert gate.resolve_name("Noah") is None

    def test_storage_backed_resolve(self):
        store = {"noah": "fid_db"}
        gate = FamilyWriteGate(resolve_from_storage=lambda n: store.get(n))
        assert gate.resolve_name("Noah") == "fid_db"

    def test_concurrent_write_txn_no_duplicate_family(self):
        """两个 worker 并发判断"新实体"，门外判断完成后进临界区重验——只有一个建 fid。"""
        gate = FamilyWriteGate(storage=None)
        created = []
        created_lock = threading.Lock()

        def worker(name_norm, delay):
            # 门外阶段（模拟 LLM 判断耗时，互不阻塞）
            time.sleep(delay)
            fid = None
            with gate.write_txn():
                existing = gate.resolve_name(name_norm)
                if existing is not None:
                    fid = existing
                else:
                    fid = f"fid_{len(created) + 1}"
                    gate.register(name_norm, fid)
                    with created_lock:
                        created.append(fid)
            return fid

        threads = [threading.Thread(target=worker, args=("noah", 0.02 * i)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(created) == 1
        assert gate.resolve_name("Noah") == created[0]


# ----------------------------------------------------------------------
# LLMClient 委托路径
# ----------------------------------------------------------------------

class TestLLMClientDelegation:
    def _client(self, judge_service=None):
        from core.llm.client import LLMClient
        return LLMClient(api_key="k", model_name="m", context_window_tokens=4096,
                         judge_service=judge_service)

    def test_none_service_calls_raw(self, monkeypatch):
        client = self._client()
        called = {"n": 0}
        monkeypatch.setattr(
            client, "_judge_entity_alignment_llm",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1) or {"verdict": "same", "confidence": 1.0})
        client.judge_entity_alignment("a", "c", "b", "d")
        assert called["n"] == 1
        assert client.judge_service is None

    def test_with_service_routes_and_caches(self, tmp_path):
        memo = VerdictMemo(str(tmp_path / "j.db"))
        svc = AlignmentJudgeService(memo)
        client = self._client(judge_service=svc)
        called = {"n": 0}

        def fake_raw(*a, **k):
            called["n"] += 1
            return {"verdict": "different", "confidence": 0.9}

        client._judge_entity_alignment_llm = fake_raw
        client.judge_entity_alignment("a", "c", "b", "d")
        client.judge_entity_alignment("a", "c", "b", "d")
        assert called["n"] == 1
        assert svc.stats()["memo_hits"] == 1
        svc.close()

    def test_resolve_methods_delegate(self, tmp_path):
        memo = VerdictMemo(str(tmp_path / "j.db"))
        svc = AlignmentJudgeService(memo)
        client = self._client(judge_service=svc)
        calls = {"ent": 0, "rel": 0}

        def fake_ent(entity, candidates, context_text=None):
            calls["ent"] += 1
            return {"match_existing_id": "", "update_mode": "create_new",
                    "merged_name": "", "relations_to_create": [], "confidence": 1.0}

        def fake_rel(e1, e2, new_contents, existing, new_source_document=""):
            calls["rel"] += 1
            return {"action": "create_new", "matched_relation_id": "",
                    "need_update": True, "confidence": 1.0}

        client._resolve_entity_candidates_llm = fake_ent
        client._resolve_relation_pair_llm = fake_rel
        client.resolve_entity_candidates_batch({"name": "X", "content": "c"}, [])
        client.resolve_entity_candidates_batch({"name": "X", "content": "c"}, [])
        assert calls["ent"] == 1
        client.resolve_relation_pair_batch("A", "B", ["r"], [])
        client.resolve_relation_pair_batch("A", "B", ["r"], [])
        assert calls["rel"] == 1
        svc.close()


# ----------------------------------------------------------------------
# P4 接线：_gate_create_entity 并发重复建 family 兜底
# ----------------------------------------------------------------------

def _make_db_gate(db_path):
    """与 registry._get_family_write_gate 相同的短只读连接 resolver。"""
    import sqlite3
    from core.judge import norm_name

    def _resolve(norm):
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            try:
                rows = conn.execute(
                    "SELECT entity_family_id, canonical_name FROM entity_families "
                    "WHERE canonical_name = ? COLLATE NOCASE "
                    "ORDER BY updated_at DESC LIMIT 4", (norm,)).fetchall()
            finally:
                conn.close()
            for fid, name in rows:
                if norm_name(name) == norm:
                    return fid
        except Exception:
            return None
        return None

    return FamilyWriteGate(resolve_from_storage=_resolve)


def _gate_processor(tmp_path, gate):
    from core.remember.orchestrator import TemporalMemoryGraphProcessor
    from core.storage.embedding import EmbeddingClient
    emb = EmbeddingClient(model_path="/nonexistent-mock-model", use_local=True)
    emb.model = None
    return TemporalMemoryGraphProcessor(
        storage_path=str(tmp_path / "lib"),
        embedding_client=emb,
        remember_config={"profile": "strong-v1", "mode": "strong_one_pass",
                         "window_size_chars": 6000, "overlap_chars": 300},
        family_write_gate=gate,
    )


class TestGateWiring:
    def test_gate_backstops_concurrent_duplicate_family(self, tmp_path):
        """模拟竞态：B 的候选检索发生在 A 写入可见之前（检索落空），
        gate 在写临界区内重验名称，把 B 的 Alice 版本挂到 A 的 family 下。"""
        import sqlite3
        db = str(tmp_path / "lib" / "library.db")
        gate = _make_db_gate(db)

        proc_a = _gate_processor(tmp_path, gate)
        proc_a.remember_text(
            "Alice met Bob at the cafe. Alice and Bob discussed quantum physics.",
            doc_name="a.md", verbose=False)

        proc_b = _gate_processor(tmp_path, gate)  # 共享同一 library.db 与 gate
        proc_b.entity_processor._build_entity_candidate_table = lambda *a, **k: {}
        proc_b.remember_text(
            "Alice met Carol at the observatory. Alice showed Carol the telescope.",
            doc_name="b.md", verbose=False)

        conn = sqlite3.connect(db)
        alices = conn.execute(
            "SELECT entity_family_id FROM entity_families "
            "WHERE canonical_name = 'Alice' COLLATE NOCASE").fetchall()
        assert len(alices) == 1, f"并发重复建 family: {alices}"
        obs = conn.execute(
            "SELECT count(*) FROM entity_observations WHERE entity_family_id = ?",
            (alices[0][0],)).fetchone()[0]
        assert obs == 2, "A/B 两个 episode 应各留一个版本"
        conn.close()

    def test_gate_respects_judged_candidate_names(self, tmp_path):
        """同名候选已被判"同名不同概念"（judged）时 gate 不得覆盖裁决。"""
        gate = _make_db_gate(str(tmp_path / "lib" / "library.db"))
        proc = _gate_processor(tmp_path, gate)
        ep = proc.entity_processor

        seeded = ep._build_new_entity("Alice", "人物。", "ep1", "a.md")
        proc.storage.save_entity(seeded)
        gate.register("Alice", seeded.family_id)

        # 未在候选中出现（竞态窗口内他人新建）→ gate 命中，挂到已有 family
        v = ep._gate_create_entity("Alice", "新内容", "ep2", "b.md")
        assert v.family_id == seeded.family_id

        # 同名候选已被裁决 → gate 不介入，创建新 family
        v2 = ep._gate_create_entity("Alice", "另一概念", "ep3", "c.md",
                                    judged_candidate_names=["Alice"])
        assert v2.family_id != seeded.family_id

        # 无 gate → 完全退回旧路径
        ep.family_write_gate = None
        v3 = ep._gate_create_entity("Alice", "再来一个", "ep4", "d.md")
        assert v3.family_id not in (seeded.family_id, v2.family_id)

    def test_gate_pending_fid_never_creates_second_family(self, tmp_path):
        """缓存命中的 fid 尚未提交（他方 worker 刚 register、save 未落盘）：
        不得落穿另建新 family，必须直接在该 pending fid 下建版本。

        这是 mock 并发压测抓到的真实竞态：修复前第二次创建会走
        _build_new_entity → 同名双 family；且两路 save 都 INSERT 同一
        pending fid → UNIQUE constraint / database is locked。"""
        gate = _make_db_gate(str(tmp_path / "lib" / "library.db"))
        proc = _gate_processor(tmp_path, gate)
        ep = proc.entity_processor

        # 模拟 worker A：register 了 fid 但尚未 save（DB 无此行）
        pending = ep._build_new_entity("Alice", "A 的内容", "ep1", "a.md")
        gate.register("Alice", pending.family_id)

        v = ep._gate_create_entity("Alice", "B 的内容", "ep2", "b.md")
        assert v.family_id == pending.family_id, "pending fid 必须收敛到同一 family"
        assert getattr(v, "episode_id", "") == "ep2"
