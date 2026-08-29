"""code review 修复回归：b-gate-merge 组。

- f2 FamilyWriteGate 死缓存：合并删除 family 后 gate 内存缓存仍指向死
  fid，pending 分支在其下建版本、save 的 UPSERT 复活已删 family。
  修法两层：pipeline 合并调用点失效缓存 + _gate_create_entity 死 fid 兜底。
- f3 FK 爆炸：delete_entity_all_versions 删 entity_observations 前未清
  其他 relation family 中引用这些 observations 的 relation_assertions
  （subject/object_entity_id 观察锚点），PRAGMA foreign_keys=ON 下 DELETE
  炸 FK、整个 dedup/merge 批次回滚。
"""
import pathlib
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from core.judge import FamilyWriteGate
from core.models import Entity, Episode, Relation
from core.remember.cross_window import _CrossWindowDedupMixin
from core.remember.entity import EntityProcessor
from core.storage.sqlite import SQLiteGraphStorageManager


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _store(tmp_path):
    return SQLiteGraphStorageManager(
        storage_path=str(tmp_path / "graphs" / "g"), graph_id="g")


def _episode(store, episode_id, text="# Doc\nX", source="Doc.md"):
    doc_dir = pathlib.Path(store.storage_path) / "content"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_file = doc_dir / source
    doc_file.write_text(text, encoding="utf-8")
    ep = Episode(absolute_id=episode_id, content=text,
                 event_time=datetime.now(), processed_time=datetime.now(),
                 source_document=source)
    store.save_episode(ep, text=text, document_path=str(doc_file), doc_hash=episode_id)
    return ep


def _save_entity(store, obs_id, fid, name, content,
                 episode_id="ep1", source="Doc.md", dt=None):
    now = dt or datetime.now()
    ent = Entity(obs_id, fid, name, content, now, now, episode_id, source)
    store.save_entity(ent)
    return ent


def _family_ids(store):
    return {r[0] for r in store._conn().execute(
        "SELECT entity_family_id FROM entity_families")}


class _DedupHost(_CrossWindowDedupMixin):
    """cross_window mixin 的最小宿主：orchestrator 暴露的同名属性子集。"""

    def __init__(self, store, gate):
        self.storage = store
        self.llm_client = None
        self.family_write_gate = gate


# ----------------------------------------------------------------------
# f2: FamilyWriteGate 死缓存
# ----------------------------------------------------------------------

class TestGatePendingTracking:
    def test_register_marks_pending_and_invalidate_clears(self):
        gate = FamilyWriteGate()
        gate.register("Noah", "fid_p")
        assert gate.is_pending("fid_p")

        gate2 = FamilyWriteGate(
            resolve_from_storage=lambda n: {"Noah2": "fid_db"}.get(n))
        assert gate2.resolve_name("Noah2") == "fid_db"
        # 存储腿缓存的 fid 已在库中确认，不标记 pending
        assert not gate2.is_pending("fid_db")

        gate2.register("Noah3", "fid_p3")
        assert gate2.is_pending("fid_p3")
        gate2.invalidate(family_id="fid_p3")
        assert not gate2.is_pending("fid_p3")
        assert gate2.resolve_name("Noah3") is None


class TestGateStaleFid:
    def test_gate_create_entity_skips_dead_fid_after_merge(self, tmp_path):
        """缓存指向被合并删除的 fid 时不得在其下建版本（不复活已删 family）。"""
        store = _store(tmp_path)
        _episode(store, "ep1")
        # 先 fam_2 后 fam_1：fam_1 的 updated_at 更新，存储解析腿先返回 fam_1
        _save_entity(store, "obs_2", "fam_2", "Docker", "container tech")
        _save_entity(store, "obs_1", "fam_1", "Docker", "container tech")

        gate = FamilyWriteGate(storage=store)
        assert gate.resolve_name("Docker") == "fam_1"
        assert not gate.is_pending("fam_1")  # 来自存储腿，非并发在途

        # 同名合并把 fam_1 折进 fam_2（此处不经 pipeline，模拟跨进程合并
        # 未通知 gate 的情形——_gate_create_entity 的死 fid 兜底须生效）
        assert store.dedup_merge_batch([("fam_1", "fam_2")]) == 1
        assert "fam_1" not in _family_ids(store)
        assert gate.resolve_name("Docker") == "fam_1"  # 内存缓存仍是死 fid

        proc = EntityProcessor(store, None)
        proc.family_write_gate = gate
        ent = proc._gate_create_entity("Docker", "new content", "ep1", "Doc.md")
        assert ent.family_id == "fam_2"  # 收敛到合并幸存者，而非死 fid
        store.save_entity(ent)
        assert "fam_1" not in _family_ids(store)  # 已删 family 不复活

    def test_gate_create_entity_pending_fid_still_converges(self, tmp_path):
        """并发在途 fid（register 后 save 未落盘）的收敛语义不受兜底影响。"""
        store = _store(tmp_path)
        _episode(store, "ep1")
        gate = FamilyWriteGate(storage=store)
        gate.register("Pending", "fid_pending")

        proc = EntityProcessor(store, None)
        proc.family_write_gate = gate
        ent = proc._gate_create_entity("Pending", "content", "ep1", "Doc.md")
        assert ent.family_id == "fid_pending"
        # 落盘后 family 行出现——两 worker 收敛同一家族的设计语义
        store.save_entity(ent)
        assert "fid_pending" in _family_ids(store)


class TestPendingTrustWindow:
    """f2 残留：pending 无限累积放大死 fid 复活窗口。

    修法双管：(a) 存储已确认的 family 只 remember（不标 pending），
    消灭 :1229 的再登记放大器；(b) pending 加信任窗（默认 900s），
    超窗后读库验死兜底恢复生效——跨进程合并删掉的 fid 不再永久豁免。
    """

    def test_pending_expires_after_trust_window(self):
        t = {"now": 1000.0}
        gate = FamilyWriteGate(clock=lambda: t["now"])
        gate.register("Alice", "fid_w")
        assert gate.is_pending("fid_w")  # 窗口内：并发在途
        t["now"] += 899.0
        assert gate.is_pending("fid_w")
        t["now"] += 2.0  # 越过 900s 信任窗
        assert not gate.is_pending("fid_w")
        assert gate.resolve_name("Alice") == "fid_w"  # 名称缓存仍在

    def test_remember_is_cache_only(self):
        gate = FamilyWriteGate()
        gate.remember("Confirmed", "fid_c")
        assert gate.resolve_name("Confirmed") == "fid_c"
        assert not gate.is_pending("fid_c")

    def test_version_existing_family_not_marked_pending(self, tmp_path):
        """在已确认 family 下建版本（remember 路径）不得把 fid 重新标 pending。"""
        store = _store(tmp_path)
        try:
            _episode(store, "ep1")
            _save_entity(store, "obs_1", "fam_1", "Docker", "container tech")
            gate = FamilyWriteGate(storage=store)
            proc = EntityProcessor(store, None)
            proc.family_write_gate = gate
            ent = proc._gate_create_entity("Docker", "new version", "ep1", "Doc.md")
            assert ent.family_id == "fam_1"
            assert not gate.is_pending("fam_1")  # 修前：register 会把它标成 pending
        finally:
            store.close()

    def test_expired_pending_dead_fid_does_not_resurrect(self, tmp_path):
        """信任窗外，跨进程合并删掉的 pending-origin fid 不再复活。"""
        store = _store(tmp_path)
        try:
            _episode(store, "ep1")
            _save_entity(store, "obs_2", "fam_2", "Docker", "container tech")
            _save_entity(store, "obs_1", "fam_1", "Docker", "container tech")

            t = {"now": 1000.0}
            gate = FamilyWriteGate(storage=store, clock=lambda: t["now"])
            # 模拟：本进程曾创建 fam_1（register 标 pending），save 已落盘
            gate.register("Docker", "fam_1")
            assert gate.resolve_name("Docker") == "fam_1"  # 内存缓存优先

            # 跨进程合并删 fam_1（未通知 gate）+ 时间越过信任窗
            assert store.dedup_merge_batch([("fam_1", "fam_2")]) == 1
            t["now"] += 901.0
            assert not gate.is_pending("fam_1")

            proc = EntityProcessor(store, None)
            proc.family_write_gate = gate
            ent = proc._gate_create_entity("Docker", "again", "ep1", "Doc.md")
            assert ent.family_id == "fam_2"  # 收敛到合并幸存者
            store.save_entity(ent)
            assert "fam_1" not in _family_ids(store)  # 已删 family 不复活
        finally:
            store.close()


class TestCrossWindowInvalidation:
    def test_same_name_dedup_batch_invalidates_gate(self, tmp_path):
        """cross_window 批量合并后 gate 缓存不得继续指向被删 fid。"""
        store = _store(tmp_path)
        _episode(store, "ep1")
        _save_entity(store, "obs_2", "fam_2", "Docker", "container tech")
        _save_entity(store, "obs_1", "fam_1", "Docker", "container tech")

        gate = FamilyWriteGate(storage=store)
        assert gate.resolve_name("Docker") == "fam_1"

        # 列表首位的 family 作为 primary：合并对为 (fam_1 -> fam_2)，fam_1 被删
        host = _DedupHost(store, gate)
        ar = SimpleNamespace(unique_entities=[
            Entity("obs_2b", "fam_2", "Docker", "container tech",
                   datetime.now(), datetime.now(), "ep1", "Doc.md"),
            Entity("obs_1b", "fam_1", "Docker", "container tech",
                   datetime.now(), datetime.now(), "ep1", "Doc.md"),
        ])
        host._cross_window_dedup([ar], verbose=False)

        assert "fam_1" not in _family_ids(store)
        assert "fam_2" in _family_ids(store)
        # 缓存已失效：同名解析走存储腿拿到幸存者，而非内存里的死 fid
        assert gate.resolve_name("Docker") == "fam_2"
        assert not gate.is_pending("fam_1")

        # 失效后再走创建路径，不复活 fam_1
        proc = EntityProcessor(store, None)
        proc.family_write_gate = gate
        ent = proc._gate_create_entity("Docker", "again", "ep1", "Doc.md")
        store.save_entity(ent)
        assert "fam_1" not in _family_ids(store)

    def test_sequential_fallback_wrapper_plumbs_gate_invalidation(self, tmp_path, monkeypatch):
        """EntityProcessor 包装层把 gate 失效回调传进 sequential fallback。"""
        import core.remember.entity as entity_mod

        captured = {}

        def _fake_fallback(**kwargs):
            captured.update(kwargs)
            return None, [], {}

        monkeypatch.setattr(entity_mod, "_process_entity_sequential_fallback", _fake_fallback)
        proc = EntityProcessor(_store(tmp_path), None)
        proc.family_write_gate = FamilyWriteGate()
        proc._process_entity_sequential_fallback({"name": "X"}, "ep1", 0.7)
        assert captured["gate_invalidate_fn"] == proc._invalidate_gate_fid

        # 失效回调本身：清缓存 + 退 pending
        proc.family_write_gate.register("Dead", "fam_dead")
        assert proc.family_write_gate.is_pending("fam_dead")
        proc._invalidate_gate_fid("fam_dead")
        assert proc.family_write_gate.resolve_name("Dead") is None
        assert not proc.family_write_gate.is_pending("fam_dead")

    def test_invalidate_gate_fid_noop_without_gate(self, tmp_path):
        proc = EntityProcessor(_store(tmp_path), None)
        proc.family_write_gate = None
        proc._invalidate_gate_fid("fam_x")  # 不抛异常


# ----------------------------------------------------------------------
# f3: delete_entity_all_versions FK 爆炸
# ----------------------------------------------------------------------

class TestFkSafeDelete:
    def test_foreign_keys_enforced(self, tmp_path):
        """回归前提：测试连接确实开着 PRAGMA foreign_keys=ON。"""
        store = _store(tmp_path)
        try:
            assert store._conn().execute("PRAGMA foreign_keys").fetchone()[0] == 1
        finally:
            store.close()

    def test_dedup_merge_batch_survives_cross_anchored_assertions(self, tmp_path):
        """其他 relation family 的断言锚定待删 observations 时合并不得炸 FK。"""
        store = _store(tmp_path)
        try:
            _episode(store, "ep1")
            now = datetime.now()
            _save_entity(store, "obs_old", "fam_old", "X", "old")
            _save_entity(store, "obs_b", "fam_b", "B", "b")
            _save_entity(store, "obs_keep", "fam_keep", "K", "keep")
            store.save_relation(Relation("rel_1", "relfam_1", "obs_b", "obs_keep",
                                         "B knows K", now, now, "ep1", "Doc.md"))
            conn = store._conn()
            # 对齐产出形态：断言端点 family 与 fam_old 无关，但观察锚点
            # （subject_entity_id）被重挂到 fam_old 的 observation 上
            conn.execute(
                "UPDATE relation_assertions SET subject_entity_id='obs_old' "
                "WHERE relation_id='rel_1'")
            store._commit_if_not_batched(conn)

            # PRAGMA foreign_keys=ON：修前此处 IntegrityError、整批回滚
            assert store.dedup_merge_batch([("fam_old", "fam_keep")]) == 1

            fams = _family_ids(store)
            assert "fam_old" not in fams
            assert {"fam_b", "fam_keep"} <= fams
            # 待删观察已清，且不再有断言引用它们
            dangling = conn.execute(
                "SELECT COUNT(*) FROM relation_assertions ra "
                "LEFT JOIN entity_observations o "
                "ON o.entity_id IN (ra.subject_entity_id, ra.object_entity_id) "
                "WHERE (ra.subject_entity_id='obs_old' OR ra.object_entity_id='obs_old')"
            ).fetchone()[0]
            assert dangling == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM entity_observations WHERE entity_family_id='fam_old'"
            ).fetchone()[0] == 0
        finally:
            store.close()

    def test_dedup_merge_batch_multi_pair_not_rolled_back(self, tmp_path):
        """FK 陷阱在批次后段时，前段已完成的对不得随之回滚。"""
        store = _store(tmp_path)
        try:
            _episode(store, "ep1")
            now = datetime.now()
            _save_entity(store, "obs_keep", "fam_keep", "K", "keep")
            # 对 1：干净合并
            _save_entity(store, "obs_c1", "fam_c1", "C", "c1")
            # 对 2：带 FK 陷阱（无关 relation family 的断言锚定待删观察）
            _save_entity(store, "obs_old", "fam_old", "X", "old")
            _save_entity(store, "obs_b", "fam_b", "B", "b")
            store.save_relation(Relation("rel_1", "relfam_1", "obs_b", "obs_keep",
                                         "B knows K", now, now, "ep1", "Doc.md"))
            conn = store._conn()
            conn.execute(
                "UPDATE relation_assertions SET subject_entity_id='obs_old' "
                "WHERE relation_id='rel_1'")
            store._commit_if_not_batched(conn)

            total = store.dedup_merge_batch([("fam_c1", "fam_keep"),
                                             ("fam_old", "fam_keep")])
            assert total == 2
            fams = _family_ids(store)
            assert "fam_c1" not in fams
            assert "fam_old" not in fams
            # 两对的重定向都已登记
            for old in ("fam_c1", "fam_old"):
                row = conn.execute(
                    "SELECT target_family_id FROM entity_redirects WHERE source_family_id=?",
                    (old,)).fetchone()
                assert row is not None and row[0] == "fam_keep"
        finally:
            store.close()

    def test_delete_entity_all_versions_direct_with_cross_mention(self, tmp_path):
        """跨 family 锚定待删观察的 entity_mentions 同样不得挡住删除。"""
        store = _store(tmp_path)
        try:
            _episode(store, "ep1")
            now = datetime.now()
            _save_entity(store, "obs_old", "fam_old", "X", "old", dt=now)
            _save_entity(store, "obs_b", "fam_b", "B", "b",
                         dt=now + timedelta(seconds=1))
            conn = store._conn()
            # 对齐产出形态：fam_b 的 mention 锚点挂在 fam_old 的观察上
            conn.execute(
                "INSERT INTO entity_mentions (mention_id, entity_id, entity_family_id, "
                "episode_id, surface_text, created_at) "
                "VALUES ('m_x', 'obs_old', 'fam_b', 'ep1', 'X', ?)",
                (now.isoformat(),))
            store._commit_if_not_batched(conn)

            deleted = store.delete_entity_all_versions("fam_old")
            assert deleted >= 1
            assert "fam_old" not in _family_ids(store)
            assert conn.execute(
                "SELECT COUNT(*) FROM entity_mentions WHERE entity_id='obs_old'"
            ).fetchone()[0] == 0
        finally:
            store.close()
