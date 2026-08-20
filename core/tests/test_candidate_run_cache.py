"""P3.3 候选表 run 级缓存单测：全 mock LLM/embedding，真实 SQLite（tmp_path）。

覆盖：
- 同 run 内多窗口只做一次全量投影扫描，后续窗口增量并入
- 本 run 新建实体可被后续窗口的精确名匹配命中
- 结构性变更（family 合并/重定向 → 向量缓存代数变化）触发整体重建
- run 结束释放（release_run_cache）与 run 边界（token 变化）重建
"""
from datetime import datetime

import pytest

from core.remember.entity_candidates import EntityCandidateBuilder
from core.storage.sqlite.manager import SQLiteGraphStorageManager
from core.storage.sqlite.repositories import entities as ent_repo


class _StubLLMClient:
    """只提供候选构建所需的最小接口（切片长度），不触网。"""

    def effective_entity_snippet_length(self) -> int:
        return 300


def _seed_entities(storage, names, start_idx=0):
    """直插 family+observation（绕开 save_entity 的 episode FK 语义）。"""
    conn = storage._conn()
    now = datetime.now().isoformat()
    fids = []
    for i, name in enumerate(names):
        fid = f"seedfam_{start_idx + i:06d}"
        ent_repo.upsert_entity_family(conn, fid, name, f"{name}的描述。",
                                      created_at=now, updated_at=now)
        ent_repo.insert_entity_observation(
            conn, f"seedobs_{start_idx + i:06d}", fid, None,
            name=name, content=f"{name}的描述。", processed_at=now)
        fids.append(fid)
    conn.commit()
    return fids


def ent_repo_upsert(conn, fid, name, updated_at, idx=0):
    """按给定 updated_at 直插（或更新）family，并每次追加一条新 observation 行
    （模拟该 family 获得新版本——新行触发 rowid 增量刷新；family 无 observation
    行时不会出现在投影里，因此初始插入也必须带行）。"""
    now = datetime.now().isoformat()
    ent_repo.upsert_entity_family(conn, fid, name, f"{name}的内容。",
                                  created_at=now, updated_at=updated_at)
    ent_repo.insert_entity_observation(
        conn, f"obs_{fid}_{idx}", fid, None,
        name=name, content=f"{name}的内容。", processed_at=now)


@pytest.fixture
def storage(tmp_path):
    mgr = SQLiteGraphStorageManager(library_path=str(tmp_path / "lib"))
    yield mgr
    mgr.close()


@pytest.fixture
def builder(storage):
    return EntityCandidateBuilder(storage, _StubLLMClient(), verbose=False)


def _counting_full_scan(storage):
    """包装 get_latest_entities_projection 计数全量扫描次数。"""
    original = storage.get_latest_entities_projection
    counter = {"n": 0}

    def wrapper(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    storage.get_latest_entities_projection = wrapper
    return counter


def _build(builder, names):
    extracted = [{"name": n, "content": "窗口抽取描述。"} for n in names]
    return builder.build_candidate_table(extracted, similarity_threshold=0.7)


class TestRunCandidateCache:
    def test_same_run_single_full_scan_and_stable_results(self, storage, builder):
        _seed_entities(storage, [f"实体_{i}" for i in range(50)])
        counter = _counting_full_scan(storage)
        t1 = _build(builder, ["实体_7", "全新实体"])
        t2 = _build(builder, ["实体_7", "全新实体"])
        assert counter["n"] == 1  # 同 run 内第二次构建复用缓存，不重扫全库
        # 两次构建候选一致（精确名命中）
        assert [c["family_id"] for c in t1[0]] == [c["family_id"] for c in t2[0]]
        assert t1[0][0]["name"] == "实体_7"
        assert t2[1] == []  # 无同名、无 embedding → 无候选（与逐窗口重扫行为一致）

    def test_new_entities_merged_into_cache(self, storage, builder):
        _seed_entities(storage, ["老实体"])
        _build(builder, ["查询实体"])
        # 模拟本 run 前一窗口新建实体
        _seed_entities(storage, ["新实体"], start_idx=100)
        table = _build(builder, ["新实体"])
        assert table[0] and table[0][0]["name"] == "新实体"
        assert table[0][0]["name_match_type"] == "exact"

    def test_core_winner_switch_matches_full_rescan(self, storage, builder):
        """同 core 两个 family：旧 family 在 run 内获得新版本后，core 键赢家应与
        全量重扫一致地切换到它（core 语义 = updated_at 最新赢），而非保留旧赢家。"""
        from core.utils import entity_match_key
        n1, n2 = "Alice (科学家)", "alice"
        core = entity_match_key(n1)
        assert core == entity_match_key(n2)
        conn = storage._conn()
        for fid, name, updated in [("fam_old", n1, "2024-01-01"), ("fam_new", n2, "2026-01-01")]:
            ent_repo_upsert(conn, fid, name, updated, idx=0)
        conn.commit()
        _build(builder, [n1])
        assert builder._run_projection_cache["core_to_proj"][core]["family_id"] == "fam_new"
        # run 内 fam_old 获得新版本（新 observation 行 + updated_at 变最新）
        ent_repo_upsert(storage._conn(), "fam_old", n1, "2027-01-01", idx=1)
        storage._conn().commit()
        table = _build(builder, [n1])
        inc = builder._run_projection_cache["core_to_proj"][core]["family_id"]
        builder.release_run_cache()
        _build(builder, [n1])
        full = builder._run_projection_cache["core_to_proj"][core]["family_id"]
        assert inc == full == "fam_old"
        # 候选表按 core 匹配命中切换后的赢家
        assert table[0] and table[0][0]["family_id"] == "fam_old"

    def test_rename_triggers_rebuild_and_restores_fallback_mapping(self, storage, builder):
        """同名（name 键）赢家改名后，次级同名 family 的回退映射应经整体重建恢复，
        与全量重扫语义一致。"""
        conn = storage._conn()
        for fid, updated in [("fam_a", "2024-01-01"), ("fam_c", "2026-01-01")]:
            ent_repo_upsert(conn, fid, "SameName", updated)
        conn.commit()
        _build(builder, ["SameName"])
        cache = builder._run_projection_cache
        assert cache["name_to_proj"]["SameName"]["family_id"] == "fam_a"  # 最旧赢
        # run 内 fam_a 改名（新 observation 行 + canonical_name 变更）
        ent_repo_upsert(storage._conn(), "fam_a", "Renamed", "2027-01-01", idx=1)
        storage._conn().commit()
        _build(builder, ["SameName"])
        cache = builder._run_projection_cache
        # fam_a 改名后 "SameName" 应回退到 fam_c（与全量重扫一致）
        assert cache["name_to_proj"]["SameName"]["family_id"] == "fam_c"
        assert cache["name_to_proj"]["Renamed"]["family_id"] == "fam_a"

    def test_structural_change_triggers_rebuild(self, storage, builder):
        fids = _seed_entities(storage, ["实体A", "实体B"])
        _build(builder, ["实体A"])
        # 结构性变更：合并 A→B（内部触发 invalidate_vector_caches → 代数递增）
        storage.merge_entity_families(fids[1], [fids[0]])
        table = _build(builder, ["实体A"])
        # 被合并走的 family 不再作为候选（重定向后投影排除）
        assert table[0] == []

    def test_release_and_token_change_rebuild(self, storage, builder):
        _seed_entities(storage, ["实体X"])
        counter = _counting_full_scan(storage)
        _build(builder, ["实体X"])
        assert counter["n"] == 1
        builder.release_run_cache()  # run 结束释放
        _build(builder, ["实体X"])
        assert counter["n"] == 2  # 缓存已释放 → 重扫
        storage._current_run_id = "another_run"  # 新 run 边界
        _build(builder, ["实体X"])
        assert counter["n"] == 3

    def test_cache_released_after_remember_run(self, tmp_path):
        """端到端：多窗口 run 中缓存生效（窗口2 经增量并入归并窗口1 同名实体），
        run 结束后 run 级缓存被释放。"""
        from core.remember.orchestrator import TemporalMemoryGraphProcessor
        from core.storage.embedding import EmbeddingClient
        emb = EmbeddingClient(model_path="/nonexistent/mock-model", use_local=True)
        emb.model = None
        proc = TemporalMemoryGraphProcessor(
            storage_path=str(tmp_path / "lib2"), embedding_client=emb,
            remember_config={"profile": "strong-v1", "window_size_chars": 500,
                             "overlap_chars": 50},
        )
        text = "\n\n".join([
            "Alice met Bob at the cafe one sunny morning in spring.",
            "Alice and Bob discussed quantum physics for hours over coffee.",
            "Carol joined later and told Alice about the new telescope at the observatory.",
            "Bob said the telescope could help their research on dark matter.",
            "Alice wrote careful notes about dark matter while Carol watched quietly.",
            "Later that week Carol visited the observatory with Bob and Alice together.",
            "Alice and Carol talked about the telescope again during the visit.",
            "Bob mentioned that dark matter research needed more telescope time.",
        ])
        assert len(proc.document_processor.chunk_text(text)) >= 2  # 确认多窗口
        result = proc.remember_text(text, doc_name="run_cache_doc.md", verbose=False)
        assert result.get("chunks_processed", 0) >= 2
        # 跨窗口同名实体归并到同一 family（窗口2 的候选表靠 run 缓存命中窗口1 新实体）
        fams = {}
        for e in proc.storage.get_all_entities():
            name = e.get("name") if isinstance(e, dict) else e.name
            fid = e.get("family_id") if isinstance(e, dict) else e.family_id
            fams.setdefault(name, set()).add(fid)
        for name, fids in fams.items():
            assert len(fids) == 1, f"跨窗口同名实体未归并: {name} -> {fids}"
        # run 结束释放
        assert proc.entity_processor._candidate_builder._run_projection_cache is None
        proc.close()
