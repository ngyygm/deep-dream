"""Focused regression tests for the 2026-06-14 P0/P1 retrieval fixes.

These tests directly exercise the storage/embedding/text helpers (no LLM
required, except where explicitly guarded) to prove:

* Vector search ranks by descending cosine similarity to the query, NOT rowid.
* ``find_text_evidence`` returns document-absolute ``line_start``/``line_end`` > 0.
* BM25 attribution fans an episode's score out to ALL its mentioned entities.
* Relations expose non-empty ``entity1_name``/``entity2_name`` endpoint names.
* Hybrid search applies three RRF channels (bm25 / semantic / graph).
* ``v_latest_concept`` returns rows with multiple roles (entity AND relation).

The shared helper ``_build_populated_conn`` materializes a tiny but real v15
graph (document -> version -> episode -> entities/relations + embeddings) so
the storage manager and repository functions operate on real rows.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import pytest

from core.storage.sqlite import schema_v15
from core.storage.sqlite.library_manager import LibraryManager
from core.storage.sqlite.repositories import embeddings as emb_repo
from core.text_chunking import find_text_evidence


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _vec(arr):
    """Encode a python list of floats as a float32 BLOB (matches storage)."""
    import numpy as np
    return np.asarray(arr, dtype="float32").tobytes()


def _build_populated_conn(tmp_path) -> sqlite3.Connection:
    """Create an initialized v15 DB and populate it with a small graph.

    The graph contains:
      * 1 active document + 1 active version
      * 1 active episode
      * 2 entity families (Alice, Bob) + active observations in the episode
      * 1 relation family (Alice -> Bob) + active assertion in the episode
      * entity_obs + relation_assert embeddings with KNOWN vectors
    """
    db_path = str(tmp_path / "fixes.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    schema_v15.init_schema_v15(conn)
    conn.row_factory = sqlite3.Row

    now = "2026-06-14T00:00:00Z"
    # document + version
    conn.execute(
        "INSERT INTO documents (document_id, status, created_at, updated_at) "
        "VALUES ('doc1', 'active', ?, ?)",
        (now, now),
    )
    conn.execute(
        "INSERT INTO document_versions "
        "(document_version_id, document_id, content_hash, status, processed_at) "
        "VALUES ('ver1', 'doc1', 'h1', 'active', ?)",
        (now,),
    )
    # episode
    conn.execute(
        "INSERT INTO episodes "
        "(episode_id, episode_family_id, document_id, document_version_id, "
        " name, source_text, status, processed_at) "
        "VALUES ('ep1', 'epfam1', 'doc1', 'ver1', 'Introduction', "
        "        'Alice and Bob are friends.', 'active', ?)",
        (now,),
    )
    # index the episode into the FTS table (search_fts reads episodes_fts)
    conn.execute(
        "INSERT INTO episodes_fts "
        "(episode_id, document_id, document_version_id, name, heading_path, "
        " source_text, memory_text) "
        "VALUES ('ep1', 'doc1', 'ver1', 'Introduction', '', "
        "        'Alice and Bob are friends.', '')",
    )
    # entity families
    conn.executemany(
        "INSERT INTO entity_families "
        "(entity_family_id, canonical_name, canonical_content, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            ("fam_alice", "Alice", "Alice is a person.", now, now),
            ("fam_bob", "Bob", "Bob is a person.", now, now),
        ],
    )
    # entity observations in the episode
    conn.executemany(
        "INSERT INTO entity_observations "
        "(entity_id, entity_family_id, episode_id, name, content, status, processed_at) "
        "VALUES (?, ?, 'ep1', ?, ?, 'active', ?)",
        [
            ("obs_alice", "fam_alice", "Alice", "Alice is a person.", now),
            ("obs_bob", "fam_bob", "Bob", "Bob is a person.", now),
        ],
    )
    # relation family + assertion
    conn.execute(
        "INSERT INTO relation_families "
        "(relation_family_id, subject_entity_family_id, object_entity_family_id, "
        " is_directed, canonical_content, created_at, updated_at) "
        "VALUES ('relfam1', 'fam_alice', 'fam_bob', 1, 'is friends with', ?, ?)",
        (now, now),
    )
    conn.execute(
        "INSERT INTO relation_assertions "
        "(relation_id, relation_family_id, episode_id, "
        " subject_entity_id, object_entity_id, "
        " subject_entity_family_id, object_entity_family_id, "
        " content, status, processed_at) "
        "VALUES ('rel1', 'relfam1', 'ep1', "
        "        'obs_alice', 'obs_bob', 'fam_alice', 'fam_bob', "
        "        'Alice is friends with Bob', 'active', ?)",
        (now,),
    )
    conn.commit()
    return conn


def _make_manager(conn: sqlite3.Connection) -> LibraryManager:
    """Wrap a pre-built connection in a LibraryManager without re-init."""
    mgr = LibraryManager.__new__(LibraryManager)
    import threading
    mgr._db_path = None
    mgr._conn = lambda c=conn: c
    mgr._conn_lock = threading.Lock()
    mgr.embedding_client = None
    mgr.entity_content_snippet_length = 50
    mgr.relation_content_snippet_length = 50
    return mgr


@pytest.fixture
def populated_conn(tmp_path):
    conn = _build_populated_conn(tmp_path)
    yield conn
    conn.close()


@pytest.fixture
def populated_manager(tmp_path):
    conn = _build_populated_conn(tmp_path)
    mgr = _make_manager(conn)
    yield mgr, conn
    conn.close()


# ---------------------------------------------------------------------------
# 1. Vector fix — ranks by cosine similarity to query, NOT rowid
# ---------------------------------------------------------------------------

class TestVectorSearchRanksBySimilarity:
    """Prove search_entity_embeddings orders by descending cosine similarity.

    The pre-fix implementation ignored ``query_vector`` and returned rows in
    rowid order. We construct a candidate set where rowid order != similarity
    order and assert similarity wins.
    """

    def test_similarity_order_overrides_rowid_order(self, populated_conn):
        # Embeddings with KNOWN, distinct directions.
        # obs_alice is inserted FIRST (lower rowid) but points AWAY from query.
        # obs_bob   is inserted SECOND (higher rowid) but points AT the query.
        # rowid order => [obs_alice, obs_bob]
        # similarity order (query ~ +x) => [obs_bob, obs_alice]
        embeddings = [
            ("emb_alice", "obs_alice", [0.0, 1.0, 0.0, 0.0]),   # rowid 1, sim 0
            ("emb_bob", "obs_bob", [1.0, 0.0, 0.0, 0.0]),        # rowid 2, sim 1
        ]
        for eid, owner, vec in embeddings:
            emb_repo.insert_embedding(
                populated_conn, eid, "entity_obs", owner, "content",
                "hash_" + owner, "test-model", 4, _vec(vec), "", "2026-06-14",
            )
        populated_conn.commit()

        query = _vec([1.0, 0.0, 0.0, 0.0])  # identical to obs_bob
        results = emb_repo.search_entity_embeddings(
            populated_conn, query, "test-model", limit=10,
        )

        assert len(results) >= 2
        # The most similar row must be obs_bob (similarity ~ 1.0), NOT the
        # first-inserted obs_alice. This is exactly the rowid-vs-similarity
        # inversion the bug caused.
        assert results[0]["owner_id"] == "obs_bob", (
            "vector search must rank by similarity, not rowid; got "
            f"{[r['owner_id'] for r in results]}"
        )
        assert results[1]["owner_id"] == "obs_alice"

    def test_partial_match_ranks_by_similarity(self, populated_conn):
        # Three vectors: insert in an order where rowid != similarity ranking,
        # and confirm the full ordering matches descending cosine similarity.
        # obs_alice2 reuses an existing family but is attached to a second
        # episode (each (episode_id, entity_family_id) active obs is unique).
        populated_conn.execute(
            "INSERT INTO episodes "
            "(episode_id, episode_family_id, document_id, document_version_id, "
            " name, source_text, status, processed_at, chunk_index) "
            "VALUES ('ep2', 'epfam1', 'doc1', 'ver1', 'More', '', 'active', '2026-06-14', 1)"
        )
        populated_conn.execute(
            "INSERT INTO entity_observations "
            "(entity_id, entity_family_id, episode_id, name, content, status, processed_at) "
            "VALUES ('obs_alice2', 'fam_alice', 'ep2', 'Alice2', 'x', 'active', '2026-06-14')"
        )
        populated_conn.commit()
        embeddings = [
            ("e1", "obs_alice", [1.0, 0.0, 0.0, 0.0]),   # sim to query = 1.0
            ("e2", "obs_bob", [0.6, 0.8, 0.0, 0.0]),     # sim = 0.6
            ("e3", "obs_alice2", [0.0, 1.0, 0.0, 0.0]),  # sim = 0.0
        ]
        for eid, owner, vec in embeddings:
            emb_repo.insert_embedding(
                populated_conn, eid, "entity_obs", owner, "content",
                "h_" + owner, "test-model", 4, _vec(vec), "", "2026-06-14",
            )
        populated_conn.commit()

        query = _vec([1.0, 0.0, 0.0, 0.0])
        results = emb_repo.search_entity_embeddings(
            populated_conn, query, "test-model", limit=10,
        )
        owners = [r["owner_id"] for r in results]
        # Descending similarity: alice(1.0) > bob(0.6) > alice2(0.0)
        assert owners == ["obs_alice", "obs_bob", "obs_alice2"], owners

    def test_relation_embeddings_also_rank_by_similarity(self, populated_conn):
        """The relation vector channel got the same fix; verify it too."""
        # second relation assertion to vary similarity
        populated_conn.execute(
            "INSERT INTO relation_families "
            "(relation_family_id, subject_entity_family_id, object_entity_family_id, "
            " is_directed, canonical_content, created_at, updated_at) "
            "VALUES ('relfam2', 'fam_bob', 'fam_alice', 1, 'enemy of', "
            "        '2026-06-14', '2026-06-14')"
        )
        populated_conn.execute(
            "INSERT INTO relation_assertions "
            "(relation_id, relation_family_id, episode_id, "
            " subject_entity_id, object_entity_id, "
            " subject_entity_family_id, object_entity_family_id, "
            " content, status, processed_at) "
            "VALUES ('rel2', 'relfam2', 'ep1', 'obs_bob', 'obs_alice', "
            "        'fam_bob', 'fam_alice', 'Bob enemy of Alice', 'active', '2026-06-14')"
        )
        populated_conn.commit()
        # rel1 first (lower rowid) sim=0; rel2 second sim=1
        emb_repo.insert_embedding(
            populated_conn, "er1", "relation_assert", "rel1", "content",
            "hr1", "test-model", 4, _vec([0.0, 1.0, 0.0, 0.0]), "", "2026-06-14",
        )
        emb_repo.insert_embedding(
            populated_conn, "er2", "relation_assert", "rel2", "content",
            "hr2", "test-model", 4, _vec([1.0, 0.0, 0.0, 0.0]), "", "2026-06-14",
        )
        populated_conn.commit()

        query = _vec([1.0, 0.0, 0.0, 0.0])
        results = emb_repo.search_relation_embeddings(
            populated_conn, query, "test-model", limit=10,
        )
        assert results[0]["relation_id"] == "rel2", (
            "relation vector search must rank by similarity; got "
            f"{[r['relation_id'] for r in results]}"
        )


# ---------------------------------------------------------------------------
# 2. Provenance — find_text_evidence returns line_start/line_end > 0
# ---------------------------------------------------------------------------

class TestProvenanceLineNumbers:
    def test_line_start_end_positive_and_correct(self):
        text = "line one about Alice.\nline two mentions Bob.\nline three."
        evidence = find_text_evidence(text, ["Alice"], base_line=1)
        assert evidence, "expected at least one match"
        ev = evidence[0]
        assert ev["line_start"] > 0, ev
        assert ev["line_end"] > 0, ev
        # Alice is on the first line
        assert ev["line_start"] == 1
        assert ev["line_end"] == 1
        assert "Alice" in ev["quote"]

    def test_line_numbers_are_absolute_across_lines(self):
        text = "intro.\nsecond.\nAlice appears here.\nBob next."
        # base_line=1 keeps document-absolute numbering
        evidence = find_text_evidence(text, ["Alice"], base_line=1)
        assert evidence
        ev = evidence[0]
        # Alice is on the 3rd line
        assert ev["line_start"] == 3, ev
        assert ev["line_end"] == 3, ev

    def test_base_line_lifts_chunk_local_to_document_absolute(self):
        chunk = "Alice is here."
        # Simulate a chunk that starts at document line 41
        evidence = find_text_evidence(chunk, ["Alice"], base_line=41)
        assert evidence
        ev = evidence[0]
        assert ev["line_start"] == 41, ev
        assert ev["line_end"] == 41, ev


# ---------------------------------------------------------------------------
# 3. BM25 attribution — episode score fans out to ALL mentioned entities
# ---------------------------------------------------------------------------

class TestBM25Attribution:
    def test_episode_score_attributes_to_all_entities(self, populated_manager):
        mgr, conn = populated_manager
        # The single episode mentions BOTH Alice and Bob. Pre-fix only ONE
        # arbitrary entity received the score; now both must.
        results = mgr.search_concepts_by_bm25("Alice Bob friends", role="entity", limit=10)
        names = {r["name"] for r in results}
        assert {"Alice", "Bob"}.issubset(names), (
            f"BM25 must attribute the episode score to ALL mentioned entities; "
            f"got names={names}"
        )
        # Both should carry a non-zero score
        scored = {r["name"]: r.get("_score", 0.0) for r in results}
        assert scored["Alice"] > 0.0 and scored["Bob"] > 0.0, scored

    def test_expand_entities_returns_both_families(self, populated_manager):
        mgr, conn = populated_manager
        # Drive the helper directly with a synthesized episode hit.
        episode_hits = [{
            "episode_id": "ep1",
            "_score": 0.9,
        }]
        out = mgr._expand_entities_from_episodes(episode_hits, limit=10)
        fids = {r["family_id"] for r in out}
        assert {"fam_alice", "fam_bob"}.issubset(fids), (
            f"_expand_entities_from_episodes must return every active entity of "
            f"a matched episode; got fids={fids}"
        )


# ---------------------------------------------------------------------------
# 4. Relation endpoint names — non-empty entity1_name/entity2_name
# ---------------------------------------------------------------------------

class TestRelationEndpointNames:
    def test_get_concept_by_family_id_relation_has_endpoint_names(self, populated_manager):
        mgr, conn = populated_manager
        concept = mgr.get_concept_by_family_id("relfam1")
        assert concept is not None
        assert concept["role"] == "relation"
        # The bug returned empty strings here.
        assert concept["entity1_name"] == "Alice", concept
        assert concept["entity2_name"] == "Bob", concept
        assert concept.get("name"), "relation should have a synthesized name"

    def test_search_concepts_by_similarity_relation_endpoint_names(self, populated_manager):
        """When an embedding client is unavailable, the semantic channel is
        empty — so we exercise the display-name logic via the BM25 relation
        expansion which shares _relation_display_name."""
        mgr, conn = populated_manager
        results = mgr._expand_relations_from_episodes(
            [{"episode_id": "ep1", "_score": 0.8}], limit=10,
        )
        assert results, "expected the relation to be expanded"
        rel = results[0]
        assert rel["role"] == "relation"
        assert rel["entity1_name"] == "Alice", rel
        assert rel["entity2_name"] == "Bob", rel


# ---------------------------------------------------------------------------
# 5. Hybrid 3-channel — three RRF weights applied
# ---------------------------------------------------------------------------

class TestHybridThreeChannel:
    def test_hybrid_applies_three_rrf_weights(self, monkeypatch):
        """Assert _hybrid_concept_search uses three distinct RRF contributions
        (bm25 / semantic / graph) by patching the storage channels and a
        sentinel graph result that could ONLY score via the graph weight.
        """
        from core.server.routes import concepts as concepts_mod

        class FakeStorage:
            def __init__(self):
                self.role = None

            def search_concepts_by_bm25(self, query, role=None, limit=20,
                                        time_point=None, source_document=None):
                return [{
                    "role": "entity", "family_id": "bm25_only",
                    "id": "bm25_only", "name": "B", "content": "",
                    "_score": 1.0,
                }]

            def search_concepts_by_similarity(self, query_text, role=None,
                                              threshold=0.3, max_results=20,
                                              time_point=None,
                                              source_document=None):
                return [{
                    "role": "entity", "family_id": "sem_only",
                    "id": "sem_only", "name": "S", "content": "",
                    "_score": 1.0,
                }]

            def get_concept_neighbors(self, fid, max_depth=1,
                                      time_point=None, max_results=20):
                # Only the graph channel can surface this family_id.
                return [{
                    "role": "entity", "family_id": "graph_only",
                    "id": "graph_only", "name": "G", "content": "",
                }]

            def get_episode_concepts(self, ep_id):
                return []

        fs = FakeStorage()
        # _hybrid_concept_search seeds graph BFS from bm25+semantic family_ids.
        # bm25_only / sem_only are valid seeds and will produce graph_only.
        results, meta = concepts_mod._hybrid_concept_search(
            fs, "query", role="entity", limit=10, threshold=0.3,
        )
        fids = {r["family_id"] for r in results}
        # All three channels contributed distinct concepts.
        assert "bm25_only" in fids, "BM25 channel contribution missing"
        assert "sem_only" in fids, "semantic channel contribution missing"
        assert "graph_only" in fids, (
            "graph BFS channel contribution missing — hybrid is NOT 3-channel"
        )
        # meta must report three channels populated
        assert meta["bm25_results"] >= 1
        assert meta["semantic_results"] >= 1
        assert meta["graph_results"] >= 1


# ---------------------------------------------------------------------------
# 6. Unified view — v_latest_concept returns multiple roles
# ---------------------------------------------------------------------------

class TestUnifiedView:
    def test_v_latest_concept_has_entity_and_relation_roles(self, populated_conn):
        rows = populated_conn.execute(
            "SELECT DISTINCT role FROM v_latest_concept"
        ).fetchall()
        roles = {r[0] for r in rows}
        assert "entity" in roles, roles
        assert "relation" in roles, (
            f"v_latest_concept must expose relation rows too; got roles={roles}"
        )

    def test_v_latest_concept_relation_row_shape(self, populated_conn):
        # The seeded relation assertion should appear with role='relation'.
        rows = populated_conn.execute(
            "SELECT role, family_id, content FROM v_latest_concept "
            "WHERE role = 'relation' AND family_id = 'relfam1'"
        ).fetchall()
        assert rows, "expected the seeded relation to appear in v_latest_concept"
        assert rows[0][0] == "relation"
        assert rows[0][1] == "relfam1"


# ---------------------------------------------------------------------------
# 7. Document-level semantic search — search_concept_embeddings(role='document')
# ---------------------------------------------------------------------------

class TestDocumentSearch:
    """Wire document-level semantic search through the document_version owner.

    Documents embed at owner_type='document_version'. The unified Concept entry
    must dispatch role='document' to search_document_embeddings, which ranks
    candidates by descending cosine similarity via the shared vectorized ranker.
    """

    def _seed_two_documents(self, conn):
        """Seed a second document + active version alongside doc1/ver1.

        The shared fixture already seeds doc1/ver1 (active). We add doc2/ver2
        so we can construct a candidate set where rowid order != similarity
        order, exactly mirroring the entity/relation search tests.
        """
        now = "2026-06-14T00:00:00Z"
        conn.execute(
            "INSERT INTO documents (document_id, status, created_at, updated_at) "
            "VALUES ('doc2', 'active', ?, ?)",
            (now, now),
        )
        conn.execute(
            "INSERT INTO document_versions "
            "(document_version_id, document_id, content_hash, status, title, "
            " processed_at) "
            "VALUES ('ver2', 'doc2', 'h2', 'active', 'Doc Two', ?)",
            (now,),
        )
        # Two document_version embeddings with KNOWN, distinct directions.
        # ver1 (lower rowid) points AWAY from the query; ver2 points AT it.
        emb_repo.insert_embedding(
            conn, "emb_doc1", "document_version", "ver1", "content",
            "hash_ver1", "test-model", 4, _vec([0.0, 1.0, 0.0, 0.0]),
            "", "2026-06-14",
        )
        emb_repo.insert_embedding(
            conn, "emb_doc2", "document_version", "ver2", "content",
            "hash_ver2", "test-model", 4, _vec([1.0, 0.0, 0.0, 0.0]),
            "", "2026-06-14",
        )
        conn.commit()

    def test_document_search_ranks_by_similarity(self, populated_conn):
        """role='document' ranks by cosine similarity, not rowid.

        ver1 is inserted FIRST (lower rowid) but points away from the query;
        ver2 is inserted SECOND (higher rowid) but is identical to the query.
        A correct ranker must place ver2 (doc2) at #1.
        """
        self._seed_two_documents(populated_conn)

        query = _vec([1.0, 0.0, 0.0, 0.0])  # identical to ver2
        results = emb_repo.search_concept_embeddings(
            populated_conn, "document", query, "test-model", limit=10,
        )

        assert len(results) >= 2
        # The most similar document must be doc2 (similarity ~ 1.0), NOT the
        # first-inserted doc1. This is exactly the rowid-vs-similarity
        # inversion the bug would cause if ranking were broken.
        assert results[0]["document_version_id"] == "ver2", (
            "document vector search must rank by similarity, not rowid; got "
            f"{[r['document_version_id'] for r in results]}"
        )
        assert results[0]["document_id"] == "doc2", results[0]
        assert results[1]["document_version_id"] == "ver1"

    def test_unified_entry_matches_per_role_function(self, populated_conn):
        """search_concept_embeddings(role='document') returns the identical
        ordered set as search_document_embeddings — pure dispatch, no
        re-ranking."""
        self._seed_two_documents(populated_conn)

        query = _vec([1.0, 0.0, 0.0, 0.0])
        per_role = emb_repo.search_document_embeddings(
            populated_conn, query, "test-model", limit=10,
        )
        unified = emb_repo.search_concept_embeddings(
            populated_conn, "document", query, "test-model", limit=10,
        )

        # Same ordered set of embedding_ids (ordered by similarity).
        assert [r["embedding_id"] for r in per_role] == \
               [r["embedding_id"] for r in unified], (
            "unified document dispatch must match search_document_embeddings "
            f"exactly; got per_role={[r['embedding_id'] for r in per_role]} "
            f"unified={[r['embedding_id'] for r in unified]}"
        )
        # And the candidate dict shapes agree.
        assert per_role == unified
