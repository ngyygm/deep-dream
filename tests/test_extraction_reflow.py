"""步骤2–5 重排：按关系裁剪实体等。"""
from __future__ import annotations

from datetime import datetime, timezone

from processor.models import MemoryCache
from processor.pipeline.extraction import dedupe_extracted_entities, dedupe_extracted_relations
from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor


def _make_processor(tmp_path, **overrides):
    defaults = dict(
        storage_path=str(tmp_path),
        window_size=100,
        overlap=20,
        llm_api_key=None,
        llm_model="gpt-4",
        embedding_use_local=False,
        max_llm_concurrency=2,
        entity_post_enhancement=False,
    )
    defaults.update(overrides)
    return TemporalMemoryGraphProcessor(**defaults)


def test_extract_only_prunes_entities_not_in_relations(tmp_path, monkeypatch):
    proc = _make_processor(tmp_path)

    def fake_extract_entities(*_a, **_k):
        return [
            {"name": "A", "content": "xa"},
            {"name": "B", "content": "xb"},
            {"name": "Orphan", "content": "xo"},
        ]

    def fake_extract_relations(*_a, **_k):
        return [
            {"entity1_name": "A", "entity2_name": "B", "content": "r1"},
        ]

    monkeypatch.setattr(proc.llm_client, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(proc.llm_client, "extract_relations", fake_extract_relations)

    cache = MemoryCache(
        absolute_id="t1",
        content="",
        event_time=datetime.now(timezone.utc),
        source_document="doc",
    )
    ents, rels = proc._extract_only(
        cache,
        "body",
        "doc",
        verbose=False,
        verbose_steps=False,
        progress_callback=None,
    )
    assert {e["name"] for e in ents} == {"A", "B"}
    assert len(rels) == 1


def test_extract_only_supplements_missing_relation_endpoint(tmp_path, monkeypatch):
    proc = _make_processor(tmp_path)

    def fake_extract_entities(*_a, **_k):
        return [
            {"name": "A", "content": "xa"},
        ]

    def fake_extract_relations(*_a, **_k):
        return [
            {"entity1_name": "A", "entity2_name": "NewConcept", "content": "r1"},
        ]

    def fake_extract_entities_by_names(*_a, **_k):
        return [{"name": "NewConcept", "content": "from supplement"}]

    monkeypatch.setattr(proc.llm_client, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(proc.llm_client, "extract_relations", fake_extract_relations)
    monkeypatch.setattr(proc.llm_client, "extract_entities_by_names", fake_extract_entities_by_names)

    cache = MemoryCache(
        absolute_id="t2",
        content="",
        event_time=datetime.now(timezone.utc),
        source_document="doc",
    )
    ents, rels = proc._extract_only(
        cache,
        "body",
        "doc",
        verbose=False,
        verbose_steps=False,
        progress_callback=None,
    )
    assert {e["name"] for e in ents} == {"A", "NewConcept"}
    assert len(rels) == 1


def test_extract_only_skips_prune_when_no_relations(tmp_path, monkeypatch):
    proc = _make_processor(tmp_path)

    def fake_extract_entities(*_a, **_k):
        return [
            {"name": "Only", "content": "x"},
        ]

    def fake_extract_relations(*_a, **_k):
        return []

    monkeypatch.setattr(proc.llm_client, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(proc.llm_client, "extract_relations", fake_extract_relations)

    cache = MemoryCache(
        absolute_id="t3",
        content="",
        event_time=datetime.now(timezone.utc),
        source_document="doc",
    )
    ents, _rels = proc._extract_only(
        cache,
        "body",
        "doc",
        verbose=False,
        verbose_steps=False,
        progress_callback=None,
    )
    assert len(ents) == 1
    assert ents[0]["name"] == "Only"


def test_extract_only_enhancement_only_final_entities(tmp_path, monkeypatch):
    proc = _make_processor(tmp_path, entity_post_enhancement=True)

    def fake_extract_entities(*_a, **_k):
        return [
            {"name": "A", "content": "xa"},
            {"name": "B", "content": "xb"},
            {"name": "Orphan", "content": "xo"},
        ]

    def fake_extract_relations(*_a, **_k):
        return [{"entity1_name": "A", "entity2_name": "B", "content": "r"}]

    enhanced: list[str] = []

    def fake_enhance(_memory_cache, _input_text, entity, **_k):
        enhanced.append(entity["name"])
        return entity["content"] + "+"

    monkeypatch.setattr(proc.llm_client, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(proc.llm_client, "extract_relations", fake_extract_relations)
    monkeypatch.setattr(proc.llm_client, "enhance_entity_content", fake_enhance)

    cache = MemoryCache(
        absolute_id="t4",
        content="",
        event_time=datetime.now(timezone.utc),
        source_document="doc",
    )
    proc._extract_only(
        cache,
        "body",
        "doc",
        verbose=False,
        verbose_steps=False,
        progress_callback=None,
    )
    assert set(enhanced) == {"A", "B"}


def test_dedupe_extracted_entities_prefers_longer_content():
    raw = [
        {"name": "A", "content": "short"},
        {"name": "A", "content": "a much longer second content"},
        {"name": "B", "content": "b"},
    ]
    assert dedupe_extracted_entities(raw) == [
        {"name": "A", "content": "a much longer second content"},
        {"name": "B", "content": "b"},
    ]


def test_dedupe_extracted_relations_undirected_and_content():
    raw = [
        {"entity1_name": "B", "entity2_name": "A", "content": "R"},
        {"entity1_name": "A", "entity2_name": "B", "content": "R"},
        {"entity1_name": "A", "entity2_name": "B", "content": "other"},
    ]
    out = dedupe_extracted_relations(raw)
    assert len(out) == 2
    pairs = {(r["entity1_name"], r["entity2_name"], r["content"]) for r in out}
    assert ("A", "B", "R") in pairs
    assert ("A", "B", "other") in pairs


def test_extract_only_passes_deduped_entities_to_relation_extract(tmp_path, monkeypatch):
    proc = _make_processor(tmp_path)
    captured: list = []

    def fake_extract_entities(*_a, **_k):
        return [
            {"name": "A", "content": "c1"},
            {"name": "A", "content": "c2"},
        ]

    def fake_extract_relations(_mc, _txt, *, entities, **_k):
        captured.append([dict(e) for e in entities])
        return []

    monkeypatch.setattr(proc.llm_client, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(proc.llm_client, "extract_relations", fake_extract_relations)

    cache = MemoryCache(
        absolute_id="t5",
        content="",
        event_time=datetime.now(timezone.utc),
        source_document="doc",
    )
    proc._extract_only(
        cache,
        "body",
        "doc",
        verbose=False,
        verbose_steps=False,
        progress_callback=None,
    )
    # 步骤3 在关系为空时会重试，每次重试都会再调 extract_relations；实体列表应始终为去重后 1 条
    assert captured
    assert all(len(batch) == 1 for batch in captured)
    assert captured[0][0] == {"name": "A", "content": "c1"}
