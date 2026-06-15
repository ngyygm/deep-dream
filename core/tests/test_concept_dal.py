"""Option B verification — the unified NL Concept primitive works through ONE interface.

These tests prove the ACL-2026 thesis claim ("entity/relation/episode/document =
one Concept, different role") is honest at the code surface:

* The Concept DAL (``core.storage.sqlite.repositories.concepts``) exposes a single
  role-parameterized interface. Seeding an entity concept and a relation concept
  goes through the SAME ``upsert_concept_family`` / ``insert_concept_version``
  entry points, just with different ``role``.
* Reading them back yields a unified ``Concept`` DTO whose ``role`` distinguishes
  how the name / endpoints are interpreted (``entity`` -> ``name`` set;
  ``relation`` -> ``subject_family_id`` / ``object_family_id`` populated plus a
  synthesized name).
* Supersession, listing, role->owner_type mapping, and the unified vector
  dispatch (``search_concept_embeddings``) all behave identically for both roles.

底层物理存储仍是双轨制（entity_families / relation_families 没有合并），本测试
只验证"一个门面、按 role 分派"的语义，不触及物理表形状。
"""
from __future__ import annotations

import sqlite3

import pytest

from core.models import Concept
from core.storage.sqlite import schema_v15
from core.storage.sqlite.library_manager import LibraryManager
from core.storage.sqlite.repositories import concepts as concept_dal
from core.storage.sqlite.repositories import embeddings as emb_repo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec(arr):
    """Encode a python list of floats as a float32 BLOB (matches storage)."""
    import numpy as np
    return np.asarray(arr, dtype="float32").tobytes()


def _build_conn(tmp_path) -> sqlite3.Connection:
    """Create a fresh initialized v15 in-memory-backed DB on disk.

    We use a real on-disk sqlite file (under tmp_path) so WAL / foreign_keys
    behave like production, but the data is tiny and throwaway.
    """
    db_path = str(tmp_path / "concept_dal.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    schema_v15.init_schema_v15(conn)
    return conn


def _seed_skeleton(conn) -> str:
    """Seed the document/version/episode scaffolding a concept needs to attach.

    Entity observations and relation assertions both FK to an episode, so we
    need one active document -> version -> episode before writing concepts.
    Returns the episode_id the concepts will live under.
    """
    now = "2026-06-14T00:00:00Z"
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
    conn.execute(
        "INSERT INTO episodes "
        "(episode_id, episode_family_id, document_id, document_version_id, "
        " name, source_text, status, processed_at) "
        "VALUES ('ep1', 'epfam1', 'doc1', 'ver1', 'Intro', "
        "        'Alice and Bob are friends.', 'active', ?)",
        (now,),
    )
    conn.commit()
    return "ep1"


def _seed_concepts(conn) -> dict:
    """Seed an entity concept AND a relation concept through the ONE DAL surface.

    Both go through ``concepts.upsert_concept_family`` + ``insert_concept_version``
    — the only difference is the ``role`` argument. Returns a dict of the IDs.
    """
    episode_id = _seed_skeleton(conn)
    now = "2026-06-14T00:00:00Z"

    # ── entity concept (two endpoint entities, then the relation between them) ──
    # entity A: "Alice"
    concept_dal.upsert_concept_family(
        conn, "entity", "fam_alice",
        canonical_name="Alice",
        canonical_content="Alice is a person.",
        created_at=now, updated_at=now,
    )
    concept_dal.insert_concept_version(
        conn, "entity", "obs_alice", "fam_alice", episode_id,
        name="Alice", content="Alice is a person.",
        processed_at=now,
    )
    # entity B: "Bob"
    concept_dal.upsert_concept_family(
        conn, "entity", "fam_bob",
        canonical_name="Bob",
        canonical_content="Bob is a person.",
        created_at=now, updated_at=now,
    )
    concept_dal.insert_concept_version(
        conn, "entity", "obs_bob", "fam_bob", episode_id,
        name="Bob", content="Bob is a person.",
        processed_at=now,
    )

    # ── relation concept: Alice is friends with Bob ──
    concept_dal.upsert_concept_family(
        conn, "relation", "relfam1",
        subject_entity_family_id="fam_alice",
        object_entity_family_id="fam_bob",
        canonical_content="is friends with",
        created_at=now, updated_at=now,
    )
    concept_dal.insert_concept_version(
        conn, "relation", "rel1", "relfam1", episode_id,
        subject_entity_id="obs_alice", object_entity_id="obs_bob",
        subject_entity_family_id="fam_alice", object_entity_family_id="fam_bob",
        content="Alice is friends with Bob",
        processed_at=now,
    )
    conn.commit()
    return {"alice": "fam_alice", "bob": "fam_bob", "rel": "relfam1",
            "episode": episode_id}


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
def seeded_conn(tmp_path):
    conn = _build_conn(tmp_path)
    ids = _seed_concepts(conn)
    yield conn, ids
    conn.close()


# ---------------------------------------------------------------------------
# 1. get_concept_family returns a role-aware Concept DTO for BOTH roles
# ---------------------------------------------------------------------------

class TestGetConceptFamilyRoleAware:
    def test_entity_role_dto(self, seeded_conn):
        conn, ids = seeded_conn
        c = concept_dal.get_concept_family(conn, "entity", ids["alice"])
        assert c is not None
        assert isinstance(c, Concept)
        assert c.role == "entity"
        assert c.family_id == "fam_alice"
        # entity concepts carry a real name; endpoints are None
        assert c.name == "Alice", c
        assert c.subject_family_id is None
        assert c.object_family_id is None

    def test_relation_role_dto_has_endpoints_and_nl_content(self, seeded_conn):
        conn, ids = seeded_conn
        c = concept_dal.get_concept_family(conn, "relation", ids["rel"])
        assert c is not None
        assert isinstance(c, Concept)
        assert c.role == "relation"
        assert c.family_id == "relfam1"
        # relation endpoints are populated from the family row
        assert c.subject_family_id == "fam_alice", c
        assert c.object_family_id == "fam_bob", c
        # The relation's NL predicate lives in `content` (canonical_content).
        assert c.content, "relation concept must surface its NL predicate content"
        assert "friend" in c.content, c.content
        # The DAL itself synthesizes a human-readable display name (the ACL
        # thesis: a Concept has a synthesized name). It must match the pure
        # formatter for the seeded endpoints, be NON-EMPTY, and read the
        # predicate between the two endpoint names.
        expected_name = concept_dal.format_relation_display_name(
            "Alice", "Bob", c.content,
        )
        assert c.name, "DAL relation Concept must carry a synthesized name"
        assert c.name == expected_name, (c.name, expected_name)
        assert c.name == "Alice is friends with Bob", c.name
        # Endpoint names travel with the DTO (parity with manager display dict).
        assert c.extra.get("entity1_name") == "Alice", c.extra
        assert c.extra.get("entity2_name") == "Bob", c.extra

    def test_missing_family_returns_none(self, seeded_conn):
        conn, _ids = seeded_conn
        assert concept_dal.get_concept_family(conn, "entity", "nope") is None
        assert concept_dal.get_concept_family(conn, "relation", "nope") is None

    def test_unknown_role_raises(self, seeded_conn):
        conn, _ids = seeded_conn
        with pytest.raises(ValueError):
            concept_dal.get_concept_family(conn, "observation", "fam_alice")


# ---------------------------------------------------------------------------
# 2. supersede_by_episodes works for BOTH roles and returns counts
# ---------------------------------------------------------------------------

class TestSupersedeByEpisodes:
    def test_supersede_entity_returns_count(self, seeded_conn):
        conn, ids = seeded_conn
        n = concept_dal.supersede_by_episodes(conn, "entity", [ids["episode"]])
        assert n >= 2, n  # Alice + Bob observations
        # The active observation for Alice is now gone
        active = conn.execute(
            "SELECT COUNT(*) FROM entity_observations "
            "WHERE episode_id = ? AND status = 'active'",
            (ids["episode"],),
        ).fetchone()[0]
        assert active == 0

    def test_supersede_relation_returns_count(self, seeded_conn):
        conn, ids = seeded_conn
        n = concept_dal.supersede_by_episodes(conn, "relation", [ids["episode"]])
        assert n >= 1, n  # the friends-with assertion
        active = conn.execute(
            "SELECT COUNT(*) FROM relation_assertions "
            "WHERE episode_id = ? AND status = 'active'",
            (ids["episode"],),
        ).fetchone()[0]
        assert active == 0

    def test_supersede_empty_episode_list_is_noop(self, seeded_conn):
        conn, _ids = seeded_conn
        assert concept_dal.supersede_by_episodes(conn, "entity", []) == 0
        assert concept_dal.supersede_by_episodes(conn, "relation", []) == 0


# ---------------------------------------------------------------------------
# 3. list_concept_families: role=None returns both, role='entity' only entities
# ---------------------------------------------------------------------------

class TestListConceptFamilies:
    def test_role_none_returns_both_roles(self, seeded_conn):
        conn, _ids = seeded_conn
        items = concept_dal.list_concept_families(conn, role=None)
        roles = {c.role for c in items}
        assert roles == {"entity", "relation"}, roles
        fids = {c.family_id for c in items}
        assert {"fam_alice", "fam_bob", "relfam1"}.issubset(fids), fids
        # cross-role list enriches ONLY relation concepts (entity names come
        # from their own family row, untouched).
        rel = next(c for c in items if c.role == "relation")
        assert rel.name == "Alice is friends with Bob", rel.name
        assert rel.extra.get("entity1_name") == "Alice", rel.extra
        assert rel.extra.get("entity2_name") == "Bob", rel.extra
        ent = next(c for c in items if c.family_id == "fam_alice")
        assert ent.role == "entity"
        assert ent.name == "Alice", ent.name

    def test_role_entity_returns_only_entities(self, seeded_conn):
        conn, _ids = seeded_conn
        items = concept_dal.list_concept_families(conn, role="entity")
        roles = {c.role for c in items}
        assert roles == {"entity"}, roles
        fids = {c.family_id for c in items}
        assert fids == {"fam_alice", "fam_bob"}, fids

    def test_role_relation_returns_only_relations(self, seeded_conn):
        conn, _ids = seeded_conn
        items = concept_dal.list_concept_families(conn, role="relation")
        roles = {c.role for c in items}
        assert roles == {"relation"}, roles
        fids = {c.family_id for c in items}
        assert fids == {"relfam1"}, fids
        # list path enriches relation Concepts too: synthesized name + endpoints
        rel = next(c for c in items if c.family_id == "relfam1")
        assert rel.name == "Alice is friends with Bob", rel.name
        assert rel.extra.get("entity1_name") == "Alice", rel.extra
        assert rel.extra.get("entity2_name") == "Bob", rel.extra


# ---------------------------------------------------------------------------
# 4. role_to_owner_type maps all 4 roles correctly
# ---------------------------------------------------------------------------

class TestRoleToOwnerType:
    def test_all_four_roles_map(self):
        assert concept_dal.role_to_owner_type("entity") == "entity_obs"
        assert concept_dal.role_to_owner_type("relation") == "relation_assert"
        assert concept_dal.role_to_owner_type("episode") == "episode"
        assert concept_dal.role_to_owner_type("document") == "document_version"

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError):
            concept_dal.role_to_owner_type("observation")

    def test_emb_repo_role_to_owner_type_matches(self):
        """The embeddings repo has its own role->owner_type map; the two
        copies must agree so a concept role dispatches to the same physical
        owner_type at the DAL level and the embedding-search level."""
        for role in ("entity", "relation", "episode", "document"):
            assert emb_repo.role_to_owner_type(role) == \
                concept_dal.role_to_owner_type(role), role


# ---------------------------------------------------------------------------
# 5. search_concept_embeddings dispatches by role and ranks by similarity
# ---------------------------------------------------------------------------

class TestSearchConceptEmbeddings:
    def test_entity_role_dispatch_and_ranking(self, seeded_conn):
        """search_concept_embeddings(role='entity', qv) returns ranked entity
        results, ordered by descending cosine similarity (exact over corpus)."""
        conn, _ids = seeded_conn
        # obs_alice points AT the query; obs_bob points AWAY.
        emb_repo.insert_embedding(
            conn, "e_alice", "entity_obs", "obs_alice", "content",
            "h_alice", "test-model", 4, _vec([1.0, 0.0, 0.0, 0.0]),
            "", "2026-06-14",
        )
        emb_repo.insert_embedding(
            conn, "e_bob", "entity_obs", "obs_bob", "content",
            "h_bob", "test-model", 4, _vec([0.0, 1.0, 0.0, 0.0]),
            "", "2026-06-14",
        )
        conn.commit()

        qv = _vec([1.0, 0.0, 0.0, 0.0])
        # The canonical unified vector-search entry is search_concept_embeddings
        # (the DAL dispatches role->owner_type; embeddings repo does the search).
        results = emb_repo.search_concept_embeddings(
            conn, "entity", qv, "test-model", limit=10,
        )
        assert results, "entity role must return ranked candidates"
        # obs_alice (sim 1.0) must rank above obs_bob (sim 0.0)
        assert results[0]["owner_id"] == "obs_alice", (
            "entity vector search must rank by similarity; got "
            f"{[r['owner_id'] for r in results]}"
        )

    def test_relation_role_dispatch_and_ranking(self, seeded_conn):
        conn, _ids = seeded_conn
        emb_repo.insert_embedding(
            conn, "er1", "relation_assert", "rel1", "content",
            "hr1", "test-model", 4, _vec([1.0, 0.0, 0.0, 0.0]),
            "", "2026-06-14",
        )
        conn.commit()

        qv = _vec([1.0, 0.0, 0.0, 0.0])
        results = emb_repo.search_concept_embeddings(
            conn, "relation", qv, "test-model", limit=10,
        )
        assert results, "relation role must return ranked candidates"
        assert results[0]["relation_id"] == "rel1", results

    def test_unified_entry_matches_per_role_entry(self, seeded_conn):
        """The unified search_concept_embeddings(role=...) MUST return the same
        result set as the per-role search_*_embeddings for that role —
        proving the unified entry is pure dispatch, not a re-ranking."""
        conn, _ids = seeded_conn
        emb_repo.insert_embedding(
            conn, "e_alice", "entity_obs", "obs_alice", "content",
            "h_alice", "test-model", 4, _vec([1.0, 0.0, 0.0, 0.0]),
            "", "2026-06-14",
        )
        emb_repo.insert_embedding(
            conn, "e_bob", "entity_obs", "obs_bob", "content",
            "h_bob", "test-model", 4, _vec([0.6, 0.8, 0.0, 0.0]),
            "", "2026-06-14",
        )
        conn.commit()

        qv = _vec([1.0, 0.0, 0.0, 0.0])
        unified = emb_repo.search_concept_embeddings(
            conn, "entity", qv, "test-model", limit=10,
        )
        direct = emb_repo.search_entity_embeddings(
            conn, qv, "test-model", limit=10,
        )
        assert [r["owner_id"] for r in unified] == \
            [r["owner_id"] for r in direct]


# ---------------------------------------------------------------------------
# 6. The GET concept API endpoint surfaces a role-aware Concept DTO
# ---------------------------------------------------------------------------

class TestGetConceptAPIRoleAware:
    def test_get_concept_endpoint_returns_role_for_entity(self, seeded_conn):
        """The /api/v1/concepts/<fid> GET handler merges the unified Concept
        DTO into the response, so ``role`` is set and consistent with the
        underlying family. We exercise the storage-level DTO path it relies on
        (``storage.get_concept``) directly to keep this a deterministic unit
        test (no live :16200 server required)."""
        conn, ids = seeded_conn
        mgr = _make_manager(conn)

        # Entity concept via the unified manager entry.
        dto = mgr.get_concept(ids["alice"])
        assert dto is not None
        assert dto.role == "entity", dto
        assert dto.name == "Alice", dto

        # Relation concept via the unified manager entry.
        dto_r = mgr.get_concept(ids["rel"])
        assert dto_r is not None
        assert dto_r.role == "relation", dto_r
        assert dto_r.subject_family_id == "fam_alice", dto_r
        assert dto_r.object_family_id == "fam_bob", dto_r

    def test_get_concept_endpoint_merges_role_into_payload(self, seeded_conn):
        """The route handler (concepts.get_concept) lifts the DTO's
        role/name/content into the legacy payload dict. We replicate the exact
        merge logic the handler uses, on a real Concept DTO, to prove the
        role-aware fields survive the merge the handler performs."""
        conn, ids = seeded_conn
        mgr = _make_manager(conn)
        legacy = mgr.get_concept_by_family_id(ids["rel"])
        assert legacy is not None
        dto = mgr.get_concept(ids["rel"])
        dto_dict = dto.to_dict()

        # This mirrors the handler's merge (concepts.py get_concept):
        merged = dict(legacy)
        merged.update({k: v for k, v in dto_dict.items()
                       if k in ("role", "name", "content", "family_id",
                                "subject_family_id", "object_family_id",
                                "version_id", "status", "confidence",
                                "episode_id")})
        assert merged["role"] == "relation", merged
        assert merged.get("subject_family_id") == "fam_alice", merged
        assert merged.get("object_family_id") == "fam_bob", merged
