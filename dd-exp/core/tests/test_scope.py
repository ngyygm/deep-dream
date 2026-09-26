"""Scope 沙箱（core/find/scope.py）测试。

模式参考 test_hybrid_search.py：LibraryManager + 真实 SQLite，repo 层灌
documents/versions/episodes/FTS，实体与关系用裸 SQL 补全 FK 三件套
（family + observation + mention / relation family + assertion）。

覆盖：
- 概念回溯命中正确文档（含 vault 文档路径解析、摄取态隐藏、无关文档排除）；
- LibraryManager.concept_source_documents 的裸数据形状（偏移、去重、过滤）；
- max_docs 截断与 stats；
- materialize_scope 生成 symlink + manifest（幂等、sha256、缺失目标跳过）；
- role='episode' 的 BM25 兜底纳入与 include_episode_rank=False 关闭；
- 空结果不炸。
"""

import hashlib
import json
from pathlib import Path

import pytest

from core.storage.sqlite.repositories import (
    documents as doc_repo,
    episodes as ep_repo,
)
from core.find.scope import build_document_scope, materialize_scope

NOW = "2026-08-01T00:00:00Z"


# ── Fixtures / 灌数据 helpers ──────────────────────────────

@pytest.fixture
def mgr(tmp_path):
    from core.storage.sqlite.library_manager import LibraryManager
    manager = LibraryManager(str(tmp_path / "lib"))
    yield manager
    manager.close()


def _add_doc(mgr, doc_id, title, *, ver_id, managed_path="",
             source_mode="managed", absolute_path=""):
    conn = mgr._conn()
    doc_repo.insert_document(conn, doc_id, title=title, managed_path=managed_path,
                             source_mode=source_mode, absolute_path=absolute_path,
                             created_at=NOW, updated_at=NOW)
    doc_repo.insert_document_version(conn, ver_id, doc_id, f"hash-{doc_id}",
                                     processed_at=NOW)
    doc_repo.update_current_version(conn, doc_id, ver_id, updated_at=NOW)


def _add_episode(mgr, ep_id, doc_id, ver_id, source_text, *, name="",
                 start_offset=0, end_offset=0, chunk_index=0):
    conn = mgr._conn()
    ep_repo.insert_episode(conn, ep_id, f"fam-{ep_id}", doc_id, ver_id,
                           source_text=source_text, memory_text=source_text,
                           name=name, start_offset=start_offset,
                           end_offset=end_offset, chunk_index=chunk_index,
                           chunk_hash=f"ch-{ep_id}", processed_at=NOW)
    ep_repo.fts_sync_episode(conn, ep_id, doc_id, ver_id,
                             source_text=source_text, memory_text=source_text)


def _obs_id(fam_id, ep_id):
    return f"{fam_id}-obs-{ep_id}"


def _ensure_family(mgr, fam_id, name):
    mgr._conn().execute(
        "INSERT OR IGNORE INTO entity_families "
        "(entity_family_id, canonical_name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)", (fam_id, name, NOW, NOW))


def _link_entity(mgr, fam_id, name, episode_ids, *, with_mention=True):
    """为已存在的 family 在指定 episode 追加 observation(+mention)。"""
    conn = mgr._conn()
    for ep_id in episode_ids:
        conn.execute(
            "INSERT INTO entity_observations "
            "(entity_id, entity_family_id, episode_id, name, content, status, processed_at) "
            "VALUES (?, ?, ?, ?, '', 'active', ?)",
            (_obs_id(fam_id, ep_id), fam_id, ep_id, name, NOW))
        if with_mention:
            conn.execute(
                "INSERT INTO entity_mentions "
                "(mention_id, entity_id, entity_family_id, episode_id, surface_text, "
                " start_offset, end_offset, created_at) "
                "VALUES (?, ?, ?, ?, ?, 10, 20, ?)",
                (f"{fam_id}-ment-{ep_id}", _obs_id(fam_id, ep_id), fam_id,
                 ep_id, name, NOW))


def _add_entity(mgr, fam_id, name, episode_ids):
    _ensure_family(mgr, fam_id, name)
    _link_entity(mgr, fam_id, name, episode_ids)


def _add_relation(mgr, rel_fam_id, sub_fam, obj_fam, ep_id, content, *,
                  sub_obs_ep=None, obj_obs_ep=None):
    """断言挂在 ep_id；subject/object 观测可来自其它 episode（FK 只要求存在）。"""
    conn = mgr._conn()
    conn.execute(
        "INSERT INTO relation_families "
        "(relation_family_id, subject_entity_family_id, object_entity_family_id, "
        " canonical_content, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (rel_fam_id, sub_fam, obj_fam, content, NOW, NOW))
    conn.execute(
        "INSERT INTO relation_assertions "
        "(relation_id, relation_family_id, episode_id, subject_entity_id, "
        " object_entity_id, subject_entity_family_id, object_entity_family_id, "
        " content, evidence_text, evidence_start_offset, evidence_end_offset, "
        " status, processed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 5, 25, 'active', ?)",
        (f"{rel_fam_id}-ra1", rel_fam_id, ep_id,
         _obs_id(sub_fam, sub_obs_ep or ep_id), _obs_id(obj_fam, obj_obs_ep or ep_id),
         sub_fam, obj_fam, content, content, NOW))


@pytest.fixture
def seeded(mgr, tmp_path):
    """种子语料：doc1 主文档 / doc2 无关对照 / doc3 vault / doc4 摄取中。"""
    lib = tmp_path / "lib"

    # doc1（managed）：量子退火主文档，实体 + 关系都锚在 ep1
    d1 = lib / "content" / "current" / "doc1.md"
    d1.parent.mkdir(parents=True, exist_ok=True)
    d1.write_text("# 量子计算导论\n量子退火是量子计算的绝热实现……", encoding="utf-8")
    _add_doc(mgr, "doc1", "量子计算导论", ver_id="ver1",
             managed_path="content/current/doc1.md")
    _add_episode(mgr, "ep1", "doc1", "ver1", "量子退火利用绝热演化求解组合优化问题",
                 name="退火章节", start_offset=0, end_offset=40, chunk_index=0)
    _add_episode(mgr, "ep2", "doc1", "ver1", "量子比特的相干时间受噪声影响",
                 name="比特章节", start_offset=40, end_offset=80, chunk_index=1)

    # doc2（managed）：与查询无关的对照文档
    _add_doc(mgr, "doc2", "经典物理讲义", ver_id="ver2",
             managed_path="content/current/doc2.md")
    _add_episode(mgr, "ep3", "doc2", "ver2", "张三在讲义里讲牛顿三定律",
                 name="力学章节", chunk_index=0)

    # doc3（vault）：同样命中量子退火，验证 absolute_path 解析
    vault_file = tmp_path / "vault" / "note.md"
    vault_file.parent.mkdir(parents=True, exist_ok=True)
    vault_file.write_text("vault 笔记：量子退火硬件", encoding="utf-8")
    _add_doc(mgr, "doc3", "外部笔记", ver_id="ver3", source_mode="vault",
             absolute_path=str(vault_file))
    _add_episode(mgr, "ep4", "doc3", "ver3", "量子退火需要极低温环境",
                 name="硬件章节", chunk_index=0)

    # doc4（processing，摄取态非 active）：检索与回溯都必须隐藏
    _add_doc(mgr, "doc4", "处理中文档", ver_id="ver4",
             managed_path="content/current/doc4.md")
    _add_episode(mgr, "ep5", "doc4", "ver4", "量子退火的另一份手稿",
                 name="手稿章节", chunk_index=0)
    mgr.set_document_ingestion_state("doc4", "processing")

    # 实体 / 关系（fam-q 在 ep5 的挂点应随 doc4 一起被过滤）
    _add_entity(mgr, "fam-q", "量子退火", ["ep1", "ep4", "ep5"])
    _add_entity(mgr, "fam-n", "牛顿", ["ep3"])
    _add_relation(mgr, "rel-qa", "fam-q", "fam-n", "ep1", "量子退火优于经典模拟",
                  obj_obs_ep="ep3")
    mgr._conn().commit()
    return mgr


# ── build_document_scope：回溯命中 ─────────────────────────

def test_scope_backtrace_hits_expected_documents(seeded, tmp_path):
    scope = build_document_scope(seeded, "量子退火", mode="bm25")

    doc_ids = [d["document_id"] for d in scope["documents"]]
    # doc1 命中实体+关系排第一；doc3 命中同一实体排第二；
    # doc2 与查询无关、doc4 摄取态 processing——都不进沙箱
    assert doc_ids == ["doc1", "doc3"]

    doc1, doc3 = scope["documents"]
    assert set(doc1["matched_concepts"]) == {"fam-q", "rel-qa"}
    assert set(doc3["matched_concepts"]) == {"fam-q"}
    assert doc1["score"] > doc3["score"]  # 2 个概念 > 1 个概念

    # doc1：managed → 相对库根 managed_path；file_path 指向真实文件
    assert doc1["path"] == "content/current/doc1.md"
    assert Path(doc1["file_path"]).is_file()
    # doc3：vault → absolute_path
    assert doc3["path"] == str(tmp_path / "vault" / "note.md")
    assert doc3["file_path"] == doc3["path"]

    # episode 级证据：ep1 的 snippet 与 matched
    assert len(doc1["episodes"]) == 1
    ep1 = doc1["episodes"][0]
    assert ep1["episode_id"] == "ep1"
    assert ep1["name"] == "退火章节"
    assert ep1["snippet"].startswith("量子退火利用")
    assert set(ep1["matched"]) == {"fam-q", "rel-qa"}
    assert doc3["episodes"][0]["episode_id"] == "ep4"

    # 概念列表带分数与角色
    fams = {c["family_id"]: c for c in scope["concepts"]}
    assert "fam-q" in fams and fams["fam-q"]["role"] == "entity"
    assert "rel-qa" in fams and fams["rel-qa"]["role"] == "relation"


def test_scope_stats_shape(seeded):
    scope = build_document_scope(seeded, "量子退火")  # 默认 hybrid
    assert scope["mode"] == "hybrid"
    stats = scope["stats"]
    for key in ("seed_concepts", "episodes_found", "documents_total",
                "documents_returned", "documents_missing_path"):
        assert key in stats
    assert stats["documents_total"] == stats["documents_returned"] == 2
    assert stats["episodes_found"] == 2  # ep1（doc1）+ ep4（doc3）
    assert stats["documents_missing_path"] == 0


# ── LibraryManager.concept_source_documents：裸数据形状 ─────

def test_concept_source_documents_filters_and_offsets(seeded):
    rows = seeded.concept_source_documents(["fam-q", "rel-qa"])
    assert {r["document_id"] for r in rows} == {"doc1", "doc3"}  # doc4 被过滤

    fam_q_rows = [r for r in rows if r["family_id"] == "fam-q"]
    assert {r["episode_id"] for r in fam_q_rows} == {"ep1", "ep4"}
    ep1_row = next(r for r in fam_q_rows if r["episode_id"] == "ep1")
    # mention 行优先于 observation 行：带证据偏移
    assert ep1_row["evidence_start_offset"] == 10
    assert ep1_row["evidence_end_offset"] == 20
    assert ep1_row["role"] == "entity"
    assert ep1_row["managed_path"] == "content/current/doc1.md"

    rel_rows = [r for r in rows if r["family_id"] == "rel-qa"]
    assert rel_rows and rel_rows[0]["role"] == "relation"
    assert rel_rows[0]["evidence_start_offset"] == 5
    assert rel_rows[0]["episode_id"] == "ep1"

    # include_offsets=False：不带任何偏移键
    rows2 = seeded.concept_source_documents(["fam-q"], include_offsets=False)
    assert rows2
    for r in rows2:
        for key in ("evidence_start_offset", "evidence_end_offset",
                    "episode_start_offset", "episode_end_offset"):
            assert key not in r

    # 空入参不炸
    assert seeded.concept_source_documents([]) == []


def test_concept_source_documents_observation_only_anchor(seeded):
    """只有 observation（无 mention）也能锚定 episode，证据偏移为 None。"""
    conn = seeded._conn()
    conn.execute(
        "INSERT INTO entity_families "
        "(entity_family_id, canonical_name, created_at, updated_at) "
        "VALUES ('fam-obs-only', '退火观测', ?, ?)", (NOW, NOW))
    conn.execute(
        "INSERT INTO entity_observations "
        "(entity_id, entity_family_id, episode_id, name, content, status, processed_at) "
        "VALUES ('fam-obs-only-obs', 'fam-obs-only', 'ep2', '退火观测', '', 'active', ?)",
        (NOW,))
    conn.commit()
    rows = seeded.concept_source_documents(["fam-obs-only"])
    assert len(rows) == 1
    assert rows[0]["episode_id"] == "ep2"
    assert rows[0]["document_id"] == "doc1"
    assert rows[0]["evidence_start_offset"] is None
    assert rows[0]["episode_start_offset"] == 40  # ep2 在文档中的偏移


# ── max_docs 截断 ──────────────────────────────────────────

def test_scope_max_docs_truncates(seeded):
    # 追加 4 个只命中单概念的文档 → 命中文档总数 6
    for i in range(5, 9):
        _add_doc(seeded, f"doc{i}", f"量子杂记{i}", ver_id=f"ver{i}",
                 managed_path=f"content/current/doc{i}.md")
        _add_episode(seeded, f"ep-{i}", f"doc{i}", f"ver{i}",
                     f"量子退火杂记{i}", chunk_index=0)
    _link_entity(seeded, "fam-q", "量子退火", [f"ep-{i}" for i in range(5, 9)])
    seeded._conn().commit()

    scope = build_document_scope(seeded, "量子退火", mode="bm25", max_docs=2)
    assert len(scope["documents"]) == 2
    assert scope["stats"]["documents_returned"] == 2
    assert scope["stats"]["documents_total"] == 6
    # top1 = doc1（双概念）；其余同分按 document_id 稳定排序 → doc3 次之
    assert [d["document_id"] for d in scope["documents"]] == ["doc1", "doc3"]


# ── materialize_scope：symlink + manifest ─────────────────

def test_materialize_creates_symlinks_and_manifest(seeded, tmp_path):
    scope = build_document_scope(seeded, "量子退火", mode="bm25")
    sandbox = tmp_path / "sandbox"
    out = materialize_scope(scope, sandbox)

    assert len(out["files"]) == 2
    scope_dir = Path(out["path"])
    assert scope_dir.is_dir()
    assert out["scope_id"] == scope_dir.name
    for f in out["files"]:
        link = Path(f)
        assert link.is_symlink()
        assert link.resolve().is_file()

    manifest = json.loads(Path(out["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["query"] == "量子退火"
    assert manifest["mode"] == "bm25"
    assert manifest["symlinked"] == 2
    assert len(manifest["files"]) == 2

    first = manifest["files"][0]
    assert first["document_id"] == "doc1"
    assert first["filename"].startswith("01-") and first["filename"].endswith(".md")
    # sha256 与目标文件实际内容一致
    digest = hashlib.sha256(Path(first["target"]).read_bytes()).hexdigest()
    assert first["sha256"] == digest
    assert first["matched_concepts"]
    assert first["episodes"][0]["episode_id"] == "ep1"
    assert first["episodes"][0]["start_offset"] == 0
    assert first["episodes"][0]["end_offset"] == 40

    # 幂等：同 scope 重复物化 → 同 scope_id，链接重建，manifest 可读
    out2 = materialize_scope(scope, sandbox)
    assert out2["scope_id"] == out["scope_id"]
    manifest2 = json.loads(Path(out2["manifest_path"]).read_text(encoding="utf-8"))
    assert len(manifest2["files"]) == 2
    assert all(Path(f).is_symlink() for f in out2["files"])


def test_materialize_skips_missing_targets(seeded, tmp_path):
    # doc7 无 managed_path / absolute_path → path 空串并计数，物化跳过链接
    _add_doc(seeded, "doc7", "无路径文档", ver_id="ver7")
    _add_episode(seeded, "ep7", "doc7", "ver7", "无路径文档也谈量子退火",
                 chunk_index=0)
    _link_entity(seeded, "fam-q", "量子退火", ["ep7"])
    seeded._conn().commit()

    scope = build_document_scope(seeded, "量子退火", mode="bm25")
    assert scope["stats"]["documents_missing_path"] == 1
    doc7 = next(d for d in scope["documents"] if d["document_id"] == "doc7")
    assert doc7["path"] == "" and doc7["file_path"] == ""

    out = materialize_scope(scope, tmp_path / "sandbox")
    assert len(out["files"]) == len(scope["documents"]) - 1
    manifest = json.loads(Path(out["manifest_path"]).read_text(encoding="utf-8"))
    entry7 = next(f for f in manifest["files"] if f["document_id"] == "doc7")
    assert entry7["target_exists"] is False
    assert entry7["filename"] == ""


def test_materialize_explicit_scope_id(seeded, tmp_path):
    scope = build_document_scope(seeded, "量子退火", mode="bm25")
    out = materialize_scope(scope, tmp_path / "sandbox", scope_id="manual-id")
    assert out["scope_id"] == "manual-id"
    assert Path(out["manifest_path"]).name == "manifest.json"


def test_materialize_rejects_unsafe_scope_id(seeded, tmp_path):
    """scope_id 白名单校验：路径逃逸与非法字符直接 ValueError。"""
    scope = build_document_scope(seeded, "量子退火", mode="bm25")
    sandbox = tmp_path / "sandbox"
    for bad in ("../escaped", "..", "a/b", "sub/../../escaped",
                "id with space", "id;rm", "  ../escaped  "):
        with pytest.raises(ValueError):
            materialize_scope(scope, sandbox, scope_id=bad)
    # sandbox_root 之外没有建出任何目录
    assert not (tmp_path / "escaped").exists()
    if sandbox.exists():
        assert list(sandbox.iterdir()) == []


# ── role='episode' 兜底 ────────────────────────────────────

def test_scope_episode_fallback(seeded):
    # doc6 文本可被 FTS 命中但没有任何概念 → role='episode' 兜底行
    _add_doc(seeded, "doc6", "冷聚变记录", ver_id="ver6",
             managed_path="content/current/doc6.md")
    _add_episode(seeded, "ep6", "doc6", "ver6", "冷聚变实验的现场记录",
                 name="实验章节", start_offset=100, end_offset=200, chunk_index=0)
    seeded._conn().commit()

    scope = build_document_scope(seeded, "冷聚变实验", mode="bm25")
    assert [d["document_id"] for d in scope["documents"]] == ["doc6"]
    doc6 = scope["documents"][0]
    assert doc6["matched_concepts"] == ["ep6"]  # 兜底行 family_id 即 episode_id
    assert doc6["episodes"][0]["episode_id"] == "ep6"
    assert doc6["episodes"][0]["start_offset"] == 100
    assert scope["stats"]["episode_seeds"] >= 1
    assert scope["concepts"] == []  # 兜底行不进 concepts 列表

    # include_episode_rank=False：关闭兜底 → 空范围
    scope_off = build_document_scope(seeded, "冷聚变实验", mode="bm25",
                                     include_episode_rank=False)
    assert scope_off["documents"] == []
    assert scope_off["stats"]["episode_seeds"] == 0


# ── 空结果不炸 ─────────────────────────────────────────────

def test_scope_empty_result_and_materialize(seeded, tmp_path):
    scope = build_document_scope(seeded, "完全不存在的查询词组", mode="hybrid")
    assert scope["concepts"] == []
    assert scope["documents"] == []
    assert scope["stats"]["documents_total"] == 0
    assert scope["stats"]["episodes_found"] == 0

    out = materialize_scope(scope, tmp_path / "sandbox")
    assert out["files"] == []
    manifest = json.loads(Path(out["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["files"] == []
    assert manifest["query"] == "完全不存在的查询词组"


def test_scope_invalid_mode_raises(seeded):
    with pytest.raises(ValueError):
        build_document_scope(seeded, "量子退火", mode="nope")
