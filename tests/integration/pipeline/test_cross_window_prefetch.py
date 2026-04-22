"""跨窗预取：实体 embedding 与关系分组辅助逻辑。"""
from __future__ import annotations

from unittest.mock import MagicMock

from processor.pipeline.entity import EntityProcessor
from processor.pipeline.relation import RelationProcessor


def test_build_relations_by_pair_matches_process_relations_prefix():
    storage = MagicMock()
    llm = MagicMock()
    rp = RelationProcessor(storage, llm)

    extracted = [
        {"entity1_name": "A", "entity2_name": "B", "content": "x"},
        {"entity1_name": "A", "entity2_name": "B", "content": "y"},
    ]
    name_to_id = {"A": "e1", "B": "e2"}

    by_pair, n_filtered = rp.build_relations_by_pair_from_inputs(extracted, name_to_id)
    assert n_filtered >= 0
    assert len(by_pair) >= 1
    pair_key = tuple(sorted(("e1", "e2")))
    assert pair_key in by_pair


def test_build_entity_candidate_table_prefetched_skips_encode():
    storage = MagicMock()
    storage.get_latest_entities_projection.return_value = [
        {
            "family_id": "p1",
            "name": "n1",
            "content": "",
            "version_count": 1,
            "embedding_array": None,
        },
    ]
    ec = MagicMock()
    ec.is_available.return_value = True
    ec.encode = MagicMock(side_effect=AssertionError("encode should not run when prefetched"))
    storage.embedding_client = ec

    llm = MagicMock()
    llm.effective_entity_snippet_length.return_value = 50

    ep = EntityProcessor(storage, llm, verbose=False)
    ents = [{"name": "n1", "content": "c1"}]
    ep._build_entity_candidate_table(
        ents,
        similarity_threshold=0.7,
        prefetched_embeddings=(None, None),
    )
    ec.encode.assert_not_called()


def test_encode_entities_for_candidate_table_returns_none_without_client():
    storage = MagicMock()
    storage.embedding_client = None
    llm = MagicMock()
    ep = EntityProcessor(storage, llm, verbose=False)
    a, b = ep.encode_entities_for_candidate_table([{"name": "a", "content": "b"}])
    assert a is None and b is None
