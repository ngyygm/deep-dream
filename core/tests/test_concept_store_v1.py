from datetime import datetime, timedelta
import json
from pathlib import Path

from core.models import Entity, Episode, Relation
from core.server.api import create_app
from core.storage.sqlite import SQLiteGraphStorageManager


def _store(path: Path, graph_id: str):
    return SQLiteGraphStorageManager(storage_path=str(path / "graphs" / graph_id), graph_id=graph_id)


def _episode(store, episode_id: str, text: str = "# Doc\nAlice knows Bob"):
    ep = Episode(
        absolute_id=episode_id,
        content=text,
        event_time=datetime.now(),
        processed_time=datetime.now(),
        source_document="Doc.md",
    )
    store.save_episode(ep, text=text, document_path="", doc_hash=episode_id)
    return ep


def test_physical_graph_isolation_and_blob_scope(tmp_path):
    graph_a = _store(tmp_path, "a")
    graph_b = _store(tmp_path, "b")
    try:
        _episode(graph_a, "epver_a", "# Same\nContent")
        same_family = "confam_same"
        graph_a.save_entity(Entity("conver_a", same_family, "Alice", "Same", datetime.now(), datetime.now(), "epver_a", "Doc.md"))
        graph_b.save_entity(Entity("conver_b", same_family, "Alice", "Same", datetime.now(), datetime.now(), "missing", "Doc.md"))

        assert graph_a.get_entity_by_family_id(same_family).absolute_id == "conver_a"
        assert graph_b.get_entity_by_family_id(same_family).absolute_id == "conver_b"
        assert (tmp_path / "graphs" / "a" / "graph.db").exists()
        assert (tmp_path / "graphs" / "b" / "graph.db").exists()
        assert graph_a.blobs_dir != graph_b.blobs_dir
        assert len(list((tmp_path / "graphs" / "a" / "blobs").rglob("*.md"))) == 1
        assert len(list((tmp_path / "graphs" / "b" / "blobs").rglob("*.md"))) == 0
    finally:
        graph_a.close()
        graph_b.close()


def test_concept_version_rules_same_episode_dedup_cross_episode_append(tmp_path):
    store = _store(tmp_path, "g")
    try:
        _episode(store, "epver_1")
        now = datetime.now()
        e1 = Entity("conver_1", "confam_alice", "Alice", "Same content", now, now, "epver_1", "Doc.md")
        e1_dup = Entity("conver_1_dup", "confam_alice", "Alice", "Same content", now, now, "epver_1", "Doc.md")
        store.save_entity(e1)
        store.save_entity(e1_dup)
        versions = store.get_concept_versions("confam_alice")
        assert len(versions) == 1
        assert e1_dup.absolute_id == "conver_1"

        _episode(store, "epver_2")
        e2 = Entity("conver_2", "confam_alice", "Alice", "Same content", now + timedelta(seconds=1), now + timedelta(seconds=1), "epver_2", "Doc.md")
        store.save_entity(e2)
        versions = store.get_concept_versions("confam_alice")
        assert len(versions) == 2
        assert versions[1]["content_changed"] is False
    finally:
        store.close()


def test_remember_chunks_share_one_document_version(tmp_path):
    store = _store(tmp_path, "g")
    original_dir = tmp_path / "graphs" / "g" / "tasks" / "originals"
    original_dir.mkdir(parents=True)
    original_path = original_dir / "task.txt"
    full_text = "# One Upload\n\nAlice knows Bob.\n\n## Details\nBob uses SQLite."
    original_path.write_text(full_text, encoding="utf-8")
    try:
        ep1 = Episode("epver_chunk_1", "Alice knows Bob.", datetime.now(), "upload.md", datetime.now())
        ep2 = Episode("epver_chunk_2", "Bob uses SQLite.", datetime.now(), "upload.md", datetime.now())
        store.save_episode(ep1, text=ep1.content, document_path=str(original_path), doc_hash="chunk_a")
        store.save_episode(ep2, text=ep2.content, document_path=str(original_path), doc_hash="chunk_b")

        docs = store.list_documents(limit=10)
        assert len(docs) == 1

        conn = store._connect()
        assert conn.execute("SELECT COUNT(*) FROM document_version WHERE graph_id = 'g'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM concept_version WHERE graph_id = 'g' AND role = 'episode'").fetchone()[0] == 2

        edges = conn.execute(
            "SELECT source_version_id, target_version_id FROM concept_edge WHERE graph_id = 'g' AND edge_type = 'HAS_EPISODE'"
        ).fetchall()
        assert len(edges) == 2
        assert len({row["source_version_id"] for row in edges}) == 1
    finally:
        store.close()


def test_cache_lookup_is_scoped_by_document_path(tmp_path):
    store = _store(tmp_path, "g")
    original_dir = tmp_path / "graphs" / "g" / "tasks" / "originals"
    original_dir.mkdir(parents=True)
    path_b = original_dir / "task_b.txt"
    path_a = original_dir / "task_a.txt"
    path_b.write_text("# B\nSame chunk text", encoding="utf-8")
    path_a.write_text("# A\nSame chunk text", encoding="utf-8")
    try:
        ep_b = Episode("epver_doc_b", "Same chunk text", datetime.now(), "same-name.md", datetime.now())
        ep_a = Episode("epver_doc_a", "Same chunk text", datetime.now(), "same-name.md", datetime.now())
        store.save_episode(ep_b, text=ep_b.content, document_path=str(path_b), doc_hash="same_chunk")
        store.save_episode(ep_a, text=ep_a.content, document_path=str(path_a), doc_hash="same_chunk")

        found = store.find_cache_by_doc_hash("same_chunk", document_path=str(path_b))
        assert found is not None
        assert found.absolute_id == "epver_doc_b"
    finally:
        store.close()


def test_relation_is_concept_with_asserts_and_connects(tmp_path):
    store = _store(tmp_path, "g")
    try:
        _episode(store, "epver_1")
        now = datetime.now()
        alice = Entity("conver_alice", "confam_alice", "Alice", "A", now, now, "epver_1", "Doc.md")
        bob = Entity("conver_bob", "confam_bob", "Bob", "B", now, now, "epver_1", "Doc.md")
        store.save_entity(alice)
        store.save_entity(bob)
        rel = Relation("conver_rel", "confam_rel", alice.absolute_id, bob.absolute_id, "Alice knows Bob", now, now, "epver_1", "Doc.md")
        store.save_relation(rel)

        concept = store.get_concept_by_family_id("confam_rel")
        assert concept["role"] == "relation"
        provenance = store.get_concept_provenance("confam_rel")
        assert any(p["edge_type"] == "ASSERTS" for p in provenance)
        connected = store.get_relations_by_entities("confam_alice", "confam_bob")
        assert connected and connected[0].family_id == "confam_rel"
    finally:
        store.close()


def test_mentions_edge_records_sentence_evidence(tmp_path):
    store = _store(tmp_path, "g")
    try:
        ep = Episode(
            absolute_id="epver_evidence",
            content="Alice knows Bob. Deep-Dream uses SQLite. Project Atlas stores memory.\n\nCarol appears later.",
            event_time=datetime.now(),
            processed_time=datetime.now(),
            source_document="Doc.md",
        )
        store.save_episode(
            ep,
            text=ep.content,
            document_path="",
            doc_hash="evidence_doc",
            start_offset=100,
            end_offset=100 + len(ep.content),
        )
        now = datetime.now()
        alice = Entity("conver_alice", "confam_alice", "Alice", "A person", now, now, "epver_evidence", "Doc.md")
        deep_dream = Entity("conver_deep_dream", "confam_deep_dream", "Deep Dream", "A project", now, now, "epver_evidence", "Doc.md")
        store.save_entity(alice)
        store.save_entity(deep_dream)
        store.save_episode_mentions("epver_evidence", [alice.absolute_id, deep_dream.absolute_id])

        rows = store._connect().execute(
            """
            SELECT target_family_id, provenance
            FROM concept_edge
            WHERE graph_id = 'g' AND edge_type = 'MENTIONS'
            ORDER BY target_family_id
            """
        ).fetchall()
        provenance = {row["target_family_id"]: json.loads(row["provenance"]) for row in rows}

        alice_ev = provenance["confam_alice"]["evidence"][0]
        assert alice_ev["sentence"] == "Alice knows Bob."
        assert alice_ev["quote"] == "Alice"
        assert alice_ev["start_offset"] == 100

        deep_ev = provenance["confam_deep_dream"]["evidence"][0]
        assert deep_ev["sentence"] == "Deep-Dream uses SQLite."
        assert deep_ev["quote"] == "Deep-Dream"
        assert deep_ev["match_type"] == "normalized"

        atlas = Entity("conver_atlas", "confam_atlas", "Project Atlas 系统", "A project", now, now, "epver_evidence", "Doc.md")
        store.save_entity(atlas)
        store.save_episode_mentions("epver_evidence", [atlas.absolute_id])
        row = store._connect().execute(
            """
            SELECT provenance
            FROM concept_edge
            WHERE graph_id = 'g' AND edge_type = 'MENTIONS' AND target_family_id = 'confam_atlas'
            ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()
        atlas_ev = json.loads(row["provenance"])["evidence"][0]
        assert atlas_ev["match_type"] == "similar_substring"
        assert atlas_ev["confidence"] >= 0.78
    finally:
        store.close()


def test_vault_index_parses_obsidian_links_and_heading_episodes(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "A.md").write_text(
        "---\ntags: [project]\naliases:\n- Alpha\n---\n# A\nSee [[B#Head]] and ![[B]].\n## Detail\n#tag body",
        encoding="utf-8",
    )
    (vault / "B.md").write_text("# Head\nBody", encoding="utf-8")
    store = _store(tmp_path, "g")
    try:
        result = store.index_vault(str(vault))
        assert result["files"] == 2
        assert result["indexed"] == 2
        docs = store.list_documents()
        assert len(docs) == 2
        episodes = store.search_concepts_by_bm25("Detail", role="episode", limit=10)
        assert episodes
        edges = store._connect().execute(
            "SELECT edge_type FROM concept_edge WHERE edge_type = 'DOCUMENT_LINK'"
        ).fetchall()
        assert len(edges) >= 1
    finally:
        store.close()


def test_document_graph_returns_document_episode_concept_subgraph(tmp_path):
    store = _store(tmp_path, "g")
    try:
        _episode(store, "epver_doc_a", "# A\nAlice uses SQLite with Deep-Dream")
        _episode(store, "epver_doc_b", "# B\nBob also uses SQLite")
        now = datetime.now()

        alice = Entity("conver_alice_a", "confam_alice", "Alice", "A person", now, now, "epver_doc_a", "DocA.md")
        sqlite_a = Entity("conver_sqlite_a", "confam_sqlite", "SQLite", "Database", now, now, "epver_doc_a", "DocA.md")
        sqlite_b = Entity("conver_sqlite_b", "confam_sqlite", "SQLite", "Database", now + timedelta(seconds=1), now + timedelta(seconds=1), "epver_doc_b", "DocB.md")
        store.save_entity(alice)
        store.save_entity(sqlite_a)
        store.save_entity(sqlite_b)
        store.save_episode_mentions("epver_doc_a", [alice.absolute_id, sqlite_a.absolute_id])
        store.save_episode_mentions("epver_doc_b", [sqlite_b.absolute_id])
        rel = Relation("conver_rel_a", "confam_rel_uses", alice.absolute_id, sqlite_a.absolute_id, "Alice uses SQLite", now, now, "epver_doc_a", "DocA.md")
        store.save_relation(rel)

        docs = store.list_documents(limit=10)
        selected = [d["document_version_id"] for d in docs]
        graph = store.get_document_graph(document_version_ids=selected)

        assert graph["counts"]["documents"] == 2
        assert graph["counts"]["episodes"] == 2
        assert {c["family_id"] for c in graph["concepts"]} >= {"confam_alice", "confam_sqlite", "confam_rel_uses"}
        assert sum(1 for c in graph["concepts"] if c["family_id"] == "confam_sqlite") == 1
        assert any(e["edge_type"] == "HAS_EPISODE" for e in graph["edges"])
        assert any(e["edge_type"] == "MENTIONS" for e in graph["edges"])
        assert any(e["edge_type"] == "ASSERTS" for e in graph["edges"])
        assert any(e["edge_type"] == "CONNECTS" for e in graph["edges"])
        assert graph["versions"]["confam_sqlite"]["total"] == 2
    finally:
        store.close()


def test_document_graph_outline_and_chunk_are_progressive(tmp_path):
    store = _store(tmp_path, "g")
    try:
        _episode(store, "epver_prog_a", "# A\nAlice uses SQLite with Deep-Dream")
        _episode(store, "epver_prog_b", "# B\nBob also uses SQLite")
        now = datetime.now()

        alice = Entity("conver_prog_alice", "confam_prog_alice", "Alice", "A person", now, now, "epver_prog_a", "ProgA.md")
        sqlite_a = Entity("conver_prog_sqlite_a", "confam_prog_sqlite", "SQLite", "Database", now, now, "epver_prog_a", "ProgA.md")
        sqlite_b = Entity("conver_prog_sqlite_b", "confam_prog_sqlite", "SQLite", "Database", now + timedelta(seconds=1), now + timedelta(seconds=1), "epver_prog_b", "ProgB.md")
        store.save_entity(alice)
        store.save_entity(sqlite_a)
        store.save_entity(sqlite_b)
        store.save_episode_mentions("epver_prog_a", [alice.absolute_id, sqlite_a.absolute_id])
        store.save_episode_mentions("epver_prog_b", [sqlite_b.absolute_id])
        rel = Relation("conver_prog_rel", "confam_prog_rel_uses", alice.absolute_id, sqlite_a.absolute_id, "Alice uses SQLite", now, now, "epver_prog_a", "ProgA.md")
        store.save_relation(rel)

        selected = [d["document_version_id"] for d in store.list_documents(limit=10)]
        outline = store.get_document_graph_outline(document_version_ids=selected)
        assert outline["counts"]["documents"] == 2
        assert outline["counts"]["episodes"] == 2
        assert outline["concepts"] == []
        assert any(e["edge_type"] == "HAS_EPISODE" for e in outline["edges"])
        assert outline["next_cursor"] == 0

        chunk1 = store.get_document_graph_chunk(document_version_ids=selected, cursor=0, limit=2)
        assert chunk1["cursor"] == 0
        assert chunk1["next_cursor"] is None
        assert len(chunk1["episodes"]) == 2
        assert any(c["family_id"] == "confam_prog_rel_uses" for c in chunk1["concepts"])
        assert any(e["edge_type"] == "MENTIONS" for e in chunk1["edges"])
        assert any(e["edge_type"] == "ASSERTS" for e in chunk1["edges"])
        assert any(e["edge_type"] == "CONNECTS" for e in chunk1["edges"])
        assert any(c["family_id"] == "confam_prog_sqlite" for c in chunk1["concepts"])
    finally:
        store.close()


def test_dream_routes_are_not_registered():
    class Registry:
        pass

    app = create_app(Registry(), config={"auth": {"enabled": False}, "rate_limit_per_minute": 0})
    client = app.test_client()
    assert client.get("/api/v1/find/dream/status?graph_id=g").status_code == 404
    assert client.get("/api/v1/dream/candidates?graph_id=g").status_code == 404
