"""core.judge 单测：models keys / FamilyWriteGate / gate 接线。

全部 mock LLM，不发真实请求。
"""
import threading
import time

from core.judge import (
    FamilyWriteGate,
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
        remember_config={"profile": "strong-v1",
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
