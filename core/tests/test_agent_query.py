from datetime import datetime, timedelta
from pathlib import Path

from core.models import Entity, Episode, Relation
from core.storage.sqlite import SQLiteGraphStorageManager


def _store(path: Path, graph_id: str = "agent_g"):
    return SQLiteGraphStorageManager(storage_path=str(path / "graphs" / graph_id), graph_id=graph_id)


def _episode(store, episode_id: str, text: str, source: str = "ThreeBody1.md"):
    doc_dir = Path(store.storage_path) / "content"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_file = doc_dir / source
    if not doc_file.exists():
        doc_file.write_text(text, encoding="utf-8")
    ep = Episode(
        absolute_id=episode_id,
        content=text,
        event_time=datetime.now(),
        processed_time=datetime.now(),
        source_document=source,
    )
    store.save_episode(ep, text=text, document_path=str(doc_file), doc_hash=episode_id)
    return ep


def _seed_agent_graph(store):
    now = datetime.now()
    _episode(store, "epver_doc1", "# 三体 1\n汪淼调查科学边界。", "三体1.md")
    _episode(store, "epver_doc2", "# 三体 2\n汪淼仍被提及，罗辑登场。", "三体2.md")

    wang_1 = Entity("conver_wang_1", "confam_wang", "汪淼", "科学家，贯穿线索人物。", now, now, "epver_doc1", "三体1.md")
    wang_2 = Entity("conver_wang_2", "confam_wang", "汪淼", "科学家，仍被提及。", now + timedelta(seconds=1), now + timedelta(seconds=1), "epver_doc2", "三体2.md")
    luoji = Entity("conver_luoji", "confam_luoji", "罗辑", "主角之一。", now, now, "epver_doc2", "三体2.md")
    store.save_entity(wang_1)
    store.save_entity(wang_2)
    store.save_entity(luoji)
    store.save_episode_mentions("epver_doc1", [wang_1.absolute_id])
    store.save_episode_mentions("epver_doc2", [wang_2.absolute_id, luoji.absolute_id])

    rel = Relation(
        "conver_rel",
        "confam_rel_mentions",
        wang_2.absolute_id,
        luoji.absolute_id,
        "汪淼与罗辑都被第二本文档提及。",
        now,
        now,
        "epver_doc2",
        "三体2.md",
    )
    store.save_relation(rel)


def test_agent_views_support_structural_queries(tmp_path):
    store = _store(tmp_path)
    try:
        _seed_agent_graph(store)

        # V1.5: latest entity observations per family
        latest = store.read_sql(
            """
            SELECT ef.entity_family_id, ef.canonical_name
            FROM entity_families ef
            WHERE EXISTS (
                SELECT 1 FROM entity_observations eo
                WHERE eo.entity_family_id = ef.entity_family_id AND eo.status = 'active'
            )
            ORDER BY ef.canonical_name
            """
        )
        assert latest["row_count"] == 2
        assert {row["entity_family_id"] for row in latest["rows"]} == {"confam_luoji", "confam_wang"}

        # V1.5: entity observations across episodes (document versions)
        docs = store.read_sql(
            """
            SELECT eo.entity_family_id, eo.name, COUNT(DISTINCT eo.episode_id) AS doc_count
            FROM entity_observations eo
            WHERE eo.status = 'active'
            GROUP BY eo.entity_family_id, eo.name
            ORDER BY doc_count DESC, eo.name
            """
        )
        assert docs["rows"][0]["entity_family_id"] == "confam_wang"
        assert docs["rows"][0]["doc_count"] == 2

        # V1.5: relation edges via relation_assertions joined with entity names
        rel_edges = store.read_sql(
            """
            SELECT ra.relation_family_id,
                   sub_eo.name AS subject_name,
                   obj_eo.name AS object_name,
                   ra.content AS relation_content
            FROM relation_assertions ra
            JOIN entity_observations sub_eo
              ON sub_eo.entity_id = ra.subject_entity_id AND sub_eo.status = 'active'
            JOIN entity_observations obj_eo
              ON obj_eo.entity_id = ra.object_entity_id AND obj_eo.status = 'active'
            WHERE ra.relation_family_id = 'confam_rel_mentions'
              AND ra.status = 'active'
            """
        )
        assert rel_edges["row_count"] == 1
        names = {rel_edges["rows"][0]["subject_name"], rel_edges["rows"][0]["object_name"]}
        assert names == {"汪淼", "罗辑"}
    finally:
        store.close()


def test_agent_read_sql_is_read_only_and_limited(tmp_path):
    store = _store(tmp_path)
    try:
        _seed_agent_graph(store)

        result = store.read_sql(
            "SELECT canonical_name FROM entity_families ORDER BY canonical_name", limit=1
        )
        assert result["row_count"] == 1
        assert result["truncated"] is True

        for sql in (
            "UPDATE entity_families SET canonical_name = 'bad'",
            "ATTACH DATABASE 'other.db' AS other",
            "WITH x AS (SELECT 1) DELETE FROM entity_families",
            "PRAGMA table_info(entity_observations)",
        ):
            try:
                store.read_sql(sql)
                assert False, f"expected rejection for {sql}"
            except ValueError:
                pass
    finally:
        store.close()


def test_agent_semantic_search_falls_back_to_name_lookup(tmp_path):
    store = _store(tmp_path)
    try:
        _seed_agent_graph(store)
        result = store.agent_semantic_search("汪淼", role="entity", top_k=5, threshold=0.0)
        assert result["total"] >= 1
        assert result["results"][0].family_id == "confam_wang"
    finally:
        store.close()
