"""P3 效率修复定向测试（P3.4 索引 / P3.5 N+1 收敛 / P3.8 size 缓存 / P3.11 单遍多词）。

只测行为等价与查询数收敛；不依赖真实 LLM / 网络。
"""
import pathlib
from datetime import datetime, timedelta

from core.cli._helpers import expand_query_terms, search_document_terms
from core.models import Entity, Episode, Relation
from core.storage.sqlite import SQLiteGraphStorageManager


def _store(tmp_path):
    return SQLiteGraphStorageManager(
        storage_path=str(tmp_path / "graphs" / "g"), graph_id="g")


def _episode(store, episode_id, text="# Doc\nAlice knows Bob", source="Doc.md", doc_hash=None):
    doc_dir = pathlib.Path(store.storage_path) / "content"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_file = doc_dir / source
    doc_file.write_text(text, encoding="utf-8")
    ep = Episode(
        absolute_id=episode_id,
        content=text,
        event_time=datetime.now(),
        processed_time=datetime.now(),
        source_document=source,
    )
    store.save_episode(ep, text=text, document_path=str(doc_file),
                       doc_hash=doc_hash or episode_id)
    return ep


class _StmtCounter:
    """统计连接上实际执行的顶层 SQL 数（滤掉 FTS5 内部 `--` 辅助语句）。"""

    def __init__(self, store):
        self._store = store
        self.count = 0

    def _cb(self, sql):
        if sql.lstrip().startswith("--"):
            return
        self.count += 1

    def __enter__(self):
        self._store._conn().set_trace_callback(self._cb)
        return self

    def __exit__(self, *exc):
        self._store._conn().set_trace_callback(None)
        return False


def _seed_graph(store, n_docs=5):
    """n_docs 个文档，各带 2 实体（Alice/Bob 同族）+ 1 关系。

    注意：同实体对的关系会按 pair 合并进一个 relation family，各版本
    processed_time 必须互不相同（与现有测试一致），"最新断言"才确定。
    """
    base = datetime.now()
    for i in range(n_docs):
        now = base + timedelta(seconds=i)
        ep_id = f"epver_{i}"
        _episode(store, ep_id, text=f"# Doc {i}\nAlice {i} knows Bob", source=f"Doc{i}.md")
        alice = Entity(f"conver_a_{i}", "confam_alice", "Alice", "A person",
                       now, now, ep_id, f"Doc{i}.md")
        bob = Entity(f"conver_b_{i}", "confam_bob", "Bob", "A person",
                       now, now, ep_id, f"Doc{i}.md")
        store.save_entity(alice)
        store.save_entity(bob)
        store.save_episode_mentions(ep_id, [alice.absolute_id, bob.absolute_id])
        store.save_relation(Relation(
            f"conver_r_{i}", f"confam_rel_{i}", alice.absolute_id, bob.absolute_id,
            f"Alice knows Bob {i}", now, now, ep_id, f"Doc{i}.md"))


# ── P3.4 episodes(chunk_hash) 索引 ────────────────────────────

def test_p34_chunk_hash_index_present_and_used(tmp_path):
    store = _store(tmp_path)
    try:
        _episode(store, "epver_idx", text="# Idx\nAlice", source="Idx.md",
                 doc_hash="hash_idx_1")
        conn = store._conn()
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()}
        assert "idx_episodes_chunk_hash" in names
        # find_cache_by_doc_hash 的查询点必须走索引而非全表扫
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM episodes "
            "WHERE chunk_hash = ? AND status = 'active' "
            "ORDER BY processed_at DESC LIMIT 1",
            ("hash_idx_1",)).fetchall()
        detail = " ".join(str(p[-1]) for p in plan)
        assert "USING INDEX idx_episodes_chunk_hash" in detail
        assert "SCAN" not in detail
        assert store.find_cache_by_doc_hash("hash_idx_1") is not None
    finally:
        store.close()


def test_p34_index_backfills_existing_library(tmp_path):
    # 存量库：索引被删后，重新打开经 init 路径的 IF NOT EXISTS 自愈补建
    store = _store(tmp_path)
    try:
        _episode(store, "epver_mig", text="# Mig\nAlice", source="Mig.md",
                 doc_hash="hash_mig_1")
        conn = store._conn()
        conn.execute("DROP INDEX idx_episodes_chunk_hash")
        conn.commit()
        assert store.find_cache_by_doc_hash("hash_mig_1") is not None  # 查询本身不依赖索引存在
    finally:
        store.close()
    store2 = _store(tmp_path)
    try:
        row = store2._conn().execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_episodes_chunk_hash'").fetchone()
        assert row is not None
    finally:
        store2.close()


# ── P3.5 N+1 收敛 ─────────────────────────────────────────────

def test_p35_list_documents_counts_with_constant_queries(tmp_path):
    store = _store(tmp_path)
    try:
        _seed_graph(store, n_docs=5)
        with _StmtCounter(store) as ctr:
            docs = store.list_documents(limit=10)
        # 旧实现：1 + 3×5 = 16 条；批量后恒定个位数（含 P3.7 并入的 episode_count 批量）
        assert ctr.count <= 6, f"list_documents 语句数未收敛: {ctr.count}"
        assert len(docs) == 5
        for d in docs:
            assert d["entity_count"] == 2      # confam_alice + confam_bob
            assert d["relation_count"] == 1
            assert d["size"] > 0
            assert d["char_count"] > 0
            assert d["role"] == "document"
    finally:
        store.close()


def test_p35_list_documents_empty_and_missing_version(tmp_path):
    store = _store(tmp_path)
    try:
        assert store.list_documents(limit=10) == []
        with _StmtCounter(store) as ctr:
            docs = store.list_documents(limit=10)
        assert ctr.count <= 2  # 空列表时只有基础文档查询（+列探测），无批量开销
        assert docs == []
    finally:
        store.close()


def test_p35_get_relations_by_family_ids_matches_single_path(tmp_path):
    store = _store(tmp_path)
    try:
        _seed_graph(store, n_docs=3)
        with _StmtCounter(store) as ctr:
            batch = store.get_relations_by_family_ids(["confam_alice", "confam_bob"], limit=10)
        # 旧实现：DISTINCT 查询 + 逐 fid 5-6 条 ≈ 10+；批量后 ≤ 5 条
        assert ctr.count <= 5, f"get_relations_by_family_ids 语句数未收敛: {ctr.count}"
        assert batch, "应找到关系"
        for rel in batch:
            single = store.get_relation_by_family_id(rel.family_id)
            assert single is not None
            assert rel.absolute_id == single.absolute_id
            assert rel.content == single.content
            assert rel.entity1_absolute_id == single.entity1_absolute_id
            assert rel.entity2_absolute_id == single.entity2_absolute_id
            assert rel.embedding == single.embedding
        assert store.get_relations_by_family_ids([], limit=10) == []
    finally:
        store.close()


def test_p35_bm25_hydration_constant_queries(tmp_path):
    store = _store(tmp_path)
    try:
        _seed_graph(store, n_docs=3)
        with _StmtCounter(store) as ctr:
            ents = store.search_entities_by_bm25("Alice", limit=10)
        # 旧实现：FTS 1 条 + 逐结果 1 条观测查询；批量后只剩 FTS + 1 条 IN()
        assert ctr.count <= 4, f"entities_by_bm25 语句数未收敛: {ctr.count}"
        assert ents and all(e.family_id == "confam_alice" for e in ents)

        with _StmtCounter(store) as ctr:
            rels = store.search_relations_by_bm25("Alice knows Bob", limit=10)
        assert ctr.count <= 5, f"relations_by_bm25 语句数未收敛: {ctr.count}"
        assert rels and all(r.content.startswith("Alice knows Bob") for r in rels)
    finally:
        store.close()


def test_p35_similarity_hydration_constant_queries(tmp_path):
    import numpy as np

    class FakeEmb:
        model_name = "fake"

        def is_available(self):
            return True

        def encode(self, text):
            v = np.zeros(8, dtype=np.float32)
            v[0] = 1.0
            return v

    store = _store(tmp_path)
    try:
        _seed_graph(store, n_docs=3)
        # 先挂 client 再存 embedding：_store_embedding_if_available 无 client 时直接跳过
        store.embedding_client = FakeEmb()
        vec = np.zeros(8, dtype=np.float32)
        vec[0] = 1.0
        for i in range(3):
            store._store_embedding_if_available("entity_obs", f"conver_a_{i}", "content", "x", vec.tobytes())
            store._store_embedding_if_available("relation_assert", f"conver_r_{i}", "content", "x", vec.tobytes())
        with _StmtCounter(store) as ctr:
            ents = store.search_entities_by_similarity("Alice", max_results=10)
        assert ctr.count <= 3, f"entities_by_similarity 语句数未收敛: {ctr.count}"
        assert ents and ents[0].embedding is not None

        with _StmtCounter(store) as ctr:
            rels = store.search_relations_by_similarity("Alice", max_results=10)
        assert ctr.count <= 5, f"relations_by_similarity 语句数未收敛: {ctr.count}"
        assert rels
        for rel in rels:
            single = store.get_relation_by_absolute_id(rel.absolute_id)
            assert single is not None
            assert rel.content == single.content
            assert rel.embedding == single.embedding
            assert rel.entity1_absolute_id == single.entity1_absolute_id
            assert rel.entity2_absolute_id == single.entity2_absolute_id
    finally:
        store.close()


def test_p35_concept_names_by_family_ids(tmp_path):
    store = _store(tmp_path)
    try:
        _seed_graph(store, n_docs=1)
        rel_fam = store.get_relation_by_absolute_id("conver_r_0").family_id
        names = store.get_concept_names_by_family_ids(
            ["confam_alice", rel_fam, "epver_0", "missing_fid"])
        # 与 get_concept_by_family_id 的取名语义一致
        assert names["confam_alice"] == "Alice"
        assert names[rel_fam] == rel_fam           # 关系族无名称列 → fid 本身
        assert names["epver_0"]                     # episode → heading_path 或 name
        assert "missing_fid" not in names           # 未命中不出现在结果里
        assert store.get_concept_names_by_family_ids([]) == {}
        assert store.get_concept_names_by_family_ids(["", None]) == {}
        with _StmtCounter(store) as ctr:
            store.get_concept_names_by_family_ids(
                ["confam_alice", rel_fam, "epver_0", "missing_fid"])
        assert ctr.count <= 3  # 三类表各一批（entity 命中后不再查后两类则更少）
    finally:
        store.close()


def test_p35_batch_and_single_agree_on_processed_at_ties(tmp_path):
    """对抗回归：processed_at 完全并列时，批量与单条路径必须选同一行。

    批量 helper 用 (processed_at, rowid) 反连接决并列（后插入者胜），
    单条版若不显式 rowid 决并列会退化为索引扫描序（先插入者胜），
    两条 API 路径对同一 family 会给出不同内容——此处锁定二者一致。
    """
    store = _store(tmp_path)
    try:
        _episode(store, "ep_tie1", text="# Tie\nX knows Y", source="Tie.md")
        _episode(store, "ep_tie2", text="# Tie2\nX knows Y again", source="Tie2.md")
        base = datetime.now()
        store.save_entity(Entity("obs_tx", "confam_tx", "X", "c", base, base,
                                 "ep_tie1", "Tie.md"))
        store.save_entity(Entity("obs_ty", "confam_ty", "Y", "c", base, base,
                                 "ep_tie1", "Tie.md"))
        store.save_episode_mentions("ep_tie1", ["obs_tx", "obs_ty"])
        store.save_relation(Relation(
            "convr_old", "confam_tie", "obs_tx", "obs_ty",
            "OLD content", base, base, "ep_tie1", "Tie.md"))
        conn = store._conn()
        fam_id = store.get_relation_by_absolute_id("convr_old").family_id
        tie_ts = conn.execute(
            "SELECT processed_at FROM relation_assertions WHERE relation_id = ?",
            ("convr_old",)).fetchone()[0]
        # 手工并列：同 family 两条 active 断言 / 两条 active 观测，processed_at 相同
        conn.execute(
            "INSERT INTO relation_assertions(relation_id, relation_family_id, "
            "subject_entity_id, object_entity_id, subject_entity_family_id, "
            "object_entity_family_id, content, status, processed_at, episode_id, extra_json) "
            "VALUES('convr_new', ?, 'obs_tx', 'obs_ty', 'confam_tx', 'confam_ty', "
            "'NEW content', 'active', ?, 'ep_tie2', '{}')", (fam_id, tie_ts))
        conn.execute(
            "INSERT INTO entity_observations(entity_id, entity_family_id, episode_id, "
            "name, content, status, processed_at, extra_json) "
            "VALUES('obs_tx2', 'confam_tx', 'ep_tie2', 'X', 'c2', 'active', ?, '{}')",
            (tie_ts,))
        conn.commit()

        single = store.get_relation_by_family_id(fam_id)
        batch = store.get_relations_by_family_ids(["confam_tx"], limit=10)
        assert single is not None and len(batch) == 1
        # 决并列方向统一：后插入者（rowid 更大）胜，与 _VECTOR_ROLE_CONFIG 约定一致
        assert single.absolute_id == "convr_new"
        assert single.content == batch[0].content == "NEW content"
        assert single.entity1_absolute_id == batch[0].entity1_absolute_id == "obs_tx2"
    finally:
        store.close()


# ── P3.8 document_size_bytes 只算一次 ─────────────────────────

def test_p38_document_size_bytes_computed_once():
    from core.server.task_journal import RememberTask
    from core.server.task_queue import RememberTaskQueue

    q = RememberTaskQueue.__new__(RememberTaskQueue)  # 不启动 worker 线程
    task = RememberTask(
        task_id="t1", text="记忆内容 abc", source_name="doc.md",
        load_cache=None, control_action=None,
        event_time=datetime.now(), original_path="/tmp/x.txt",
    )
    d1 = RememberTaskQueue._task_to_dict(q, task)
    assert d1["document_size_bytes"] == len("记忆内容 abc".encode("utf-8"))
    assert task.document_size_bytes == d1["document_size_bytes"]
    # 二次序列化直读缓存：text 变化也不再重算（首次之后以缓存为准）
    task.text = "changed"
    d2 = RememberTaskQueue._task_to_dict(q, task)
    assert d2["document_size_bytes"] == d1["document_size_bytes"]

    # None 文本 / journal 反序列化路径：默认 None → 懒计算
    task2 = RememberTask(
        task_id="t2", text=None, source_name="doc.md",
        load_cache=None, control_action=None,
        event_time=datetime.now(), original_path="/tmp/x.txt",
    )
    d3 = RememberTaskQueue._task_to_dict(q, task2)
    assert d3["document_size_bytes"] == 0


# ── P3.11 单遍多词搜索 ────────────────────────────────────────

class _FakeFileStore:
    """document_rows/read_sql 的最小桩：按 processed_time 升序返回文件行。"""

    def __init__(self, root, paths):
        self.storage_path = str(root)
        self._paths = paths

    def read_sql(self, sql, params=None, limit=200, **kw):
        rows = [{
            "document_version_id": f"docver_{i:02d}",
            "document_family_id": f"doc_{i:02d}",
            "title": p.name,
            "source_mode": "external",
            "absolute_path": str(p),
            "managed_path": "",
            "snapshot_path": "",
            "relative_path": p.name,
            "vault_root": "",
            "read_path": str(p),
            "content_hash": f"h{i}",
            "byte_size": 1, "char_count": 1, "line_count": 5,
            "processed_time": f"2026-08-0{i + 1}",
            "complete_windows": 1, "total_windows": 1, "missing_windows": "[]",
        } for i, p in enumerate(self._paths)]
        return {"rows": rows[:limit]}


def test_p311_single_pass_multi_term(tmp_path, monkeypatch):
    paths = []
    for i in range(6):
        p = tmp_path / f"f{i}.md"
        lines = [f"doc{i} line{j} filler" for j in range(5)]
        if i == 5:
            lines += ["alpha needle one", "beta needle two", "alpha needle three",
                      "beta needle four", "alpha needle five"]
        p.write_text("\n".join(lines), encoding="utf-8")
        paths.append(p)

    reads = {"n": 0}
    orig = pathlib.Path.read_text

    def counting(self, *a, **kw):
        reads["n"] += 1
        return orig(self, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "read_text", counting)

    terms = expand_query_terms("alpha", "alpha,beta")
    hits = search_document_terms(_FakeFileStore(tmp_path, paths), terms,
                                 per_term_limit=2, total_limit=10)
    # 单遍：6 个文件各读一次（旧实现 2 词 = 12 次全库扫描）
    assert reads["n"] == 6
    # 语义保持：每词配额 2、词序合并、matched_term 归属、term_source 标记
    assert len(hits) == 4
    assert [h["matched_term"] for h in hits] == ["alpha", "alpha", "beta", "beta"]
    assert all("needle" in h["text"] for h in hits)
    assert hits[0]["term_source"] == "original"
    assert hits[2]["term_source"] == "expanded"
    for h in hits:
        assert h["episode"] is None
        assert h["document"]["line_start"] == h["document"]["line_end"]


def test_p311_cross_term_dedupe_and_total_limit(tmp_path):
    # 每行同时含 alpha 和 beta：alpha 先收，beta 对同一行的命中被跨词去重
    p = tmp_path / "both.md"
    p.write_text("\n".join(f"alpha beta both {i}" for i in range(3)), encoding="utf-8")
    terms = expand_query_terms("alpha", "alpha,beta")
    hits = search_document_terms(_FakeFileStore(tmp_path, [p]), terms,
                                 per_term_limit=5, total_limit=10)
    assert [h["matched_term"] for h in hits] == ["alpha", "alpha", "alpha"]

    # total_limit 截断优先于词配额
    hits2 = search_document_terms(_FakeFileStore(tmp_path, [p]), terms,
                                  per_term_limit=5, total_limit=2)
    assert len(hits2) == 2


def test_p311_common_term_early_exit_and_empty(tmp_path, monkeypatch):
    # 常见词在前几个文件就收满配额 → 满额后不再读剩余文件
    paths = []
    for i in range(4):
        p = tmp_path / f"c{i}.md"
        lines = [f"common hit {j}" for j in range(3)]
        p.write_text("\n".join(lines), encoding="utf-8")
        paths.append(p)
    reads = {"n": 0}
    orig = pathlib.Path.read_text

    def counting(self, *a, **kw):
        reads["n"] += 1
        return orig(self, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "read_text", counting)
    hits = search_document_terms(_FakeFileStore(tmp_path, paths),
                                 expand_query_terms("common"), per_term_limit=3)
    assert reads["n"] == 1  # 第一个文件收满 3 条后提前退出
    assert len(hits) == 3

    # 空词列表 / 全空词：直接返回（不再对空 pattern 抛 ValueError）
    assert search_document_terms(_FakeFileStore(tmp_path, paths), []) == []
    assert search_document_terms(_FakeFileStore(tmp_path, paths),
                                 [{"term": "", "source": "expanded"}]) == []
