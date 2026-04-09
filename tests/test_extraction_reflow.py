"""步骤2–5 重排：按关系裁剪实体、补全缺失端点、去重。"""
from __future__ import annotations

from datetime import datetime, timezone
from processor.models import Episode
from processor.pipeline.extraction import (
    dedupe_extracted_entities,
    dedupe_extracted_relations,
)
from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor


def _make_processor(tmp_path, **overrides):
    defaults = dict(
        storage_path=str(tmp_path),
        window_size=200,
        llm_api_key=None,
        llm_model="gpt-4",
        embedding_use_local=False,
        max_llm_concurrency=1,
    )
    defaults.update(overrides)
    return TemporalMemoryGraphProcessor(**defaults)


# ---- _extract_only 端到端（mock LLM）----


def test_extract_only_prunes_entities_not_in_relations(tmp_path, monkeypatch):
    proc = _make_processor(tmp_path)

    def fake_extract_entities(*_a, **_k):
        return [
            {"name": "Alpha概念", "content": "一个重要的概念实体，具有多种含义"},
            {"name": "Beta系统", "content": "一个分布式系统架构，包含多个子系统"},
            {"name": "Orphan孤立实体", "content": "一个与其他实体无关联的独立概念"},
        ]

    def fake_extract_relations(*_a, **_k):
        return [
            {"entity1_name": "Alpha概念", "entity2_name": "Beta系统", "content": "Alpha概念是Beta系统的核心理论基础"},
        ]

    monkeypatch.setattr(proc.llm_client, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(proc.llm_client, "extract_relations", fake_extract_relations)

    cache = Episode(
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
    # 孤立实体（无关系端点）现在会被保留作为独立知识节点
    assert {e["name"] for e in ents} == {"Alpha概念", "Beta系统", "Orphan孤立实体"}
    assert len(rels) == 1


def test_extract_only_supplements_missing_relation_endpoint(tmp_path, monkeypatch):
    proc = _make_processor(tmp_path)

    def fake_extract_entities(*_a, **_k):
        return [
            {"name": "Alpha概念", "content": "一个重要的概念实体描述"},
        ]

    def fake_extract_relations(*_a, **_k):
        return [
            {"entity1_name": "Alpha概念", "entity2_name": "NewConcept新概念", "content": "一个明确的具体关系描述内容"},
        ]

    def fake_extract_entities_by_names(*_a, **_k):
        return [{"name": "NewConcept新概念", "content": "补充的实体描述信息"}]

    monkeypatch.setattr(proc.llm_client, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(proc.llm_client, "extract_relations", fake_extract_relations)
    monkeypatch.setattr(proc.llm_client, "extract_entities_by_names", fake_extract_entities_by_names)

    cache = Episode(
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
    assert {e["name"] for e in ents} == {"Alpha概念", "NewConcept新概念"}
    assert len(rels) == 1


def test_extract_only_skips_prune_when_no_relations(tmp_path, monkeypatch):
    proc = _make_processor(tmp_path)

    def fake_extract_entities(*_a, **_k):
        return [
            {"name": "OnlyEntity唯一实体", "content": "一个独立存在的概念实体描述"},
        ]

    def fake_extract_relations(*_a, **_k):
        return []

    monkeypatch.setattr(proc.llm_client, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(proc.llm_client, "extract_relations", fake_extract_relations)

    cache = Episode(
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
    assert ents[0]["name"] == "OnlyEntity唯一实体"


def test_extract_only_enhancement_only_final_entities(tmp_path):
    proc = _make_processor(tmp_path, entity_post_enhancement=True)

    def fake_extract_entities(*_a, **_k):
        return [
            {"name": "Alpha概念", "content": "一个重要的概念实体描述"},
            {"name": "Beta系统", "content": "一个分布式系统架构描述"},
        ]

    _enhanced = []

    def fake_extract_relations(*_a, **_k):
        return [
            {"entity1_name": "Alpha概念", "entity2_name": "Beta系统", "content": "一个明确的具体关系描述"},
        ]

    def fake_enhance(_mc, _txt, entity):
        _enhanced.append(entity["name"])
        return f"增强后的{entity['content']}，包含更多细节"

    proc.llm_client.extract_entities = fake_extract_entities
    proc.llm_client.extract_relations = fake_extract_relations
    proc.llm_client.enhance_entity_content = fake_enhance

    cache = Episode(
        absolute_id="t4",
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

    assert len(ents) == 2


# ---- 独立去重函数测试 ----


def test_dedupe_extracted_entities_prefers_longer_content():
    raw = [
        {"name": "Alpha概念", "content": "short content"},
        {"name": "Alpha概念", "content": "a much longer second content for this entity"},
        {"name": "Beta系统", "content": "beta system description with enough length"},
    ]
    result = dedupe_extracted_entities(raw)
    assert len(result) == 2
    alpha = next(e for e in result if e["name"] == "Alpha概念")
    assert "much longer" in alpha["content"]
    beta = next(e for e in result if e["name"] == "Beta系统")
    assert beta["content"] == "beta system description with enough length"


def test_dedupe_extracted_relations_undirected_and_content():
    raw = [
        {"entity1_name": "Beta系统", "entity2_name": "Alpha概念", "content": "明确的关系描述R内容"},
        {"entity1_name": "Alpha概念", "entity2_name": "Beta系统", "content": "明确的关系描述R内容"},
        {"entity1_name": "Alpha概念", "entity2_name": "Beta系统", "content": "另一个不同的关系描述"},
    ]
    out = dedupe_extracted_relations(raw)
    assert len(out) == 2
    pairs = {(r["entity1_name"], r["entity2_name"], r["content"]) for r in out}
    # After normalization, both endpoints should be in sorted order
    assert any("描述R内容" in r["content"] for r in out)
    assert any("不同" in r["content"] for r in out)


def test_extract_only_passes_deduped_entities_to_relation_extract(tmp_path, monkeypatch):
    proc = _make_processor(tmp_path)
    captured: list = []

    def fake_extract_entities(*_a, **_k):
        return [
            {"name": "Alpha概念", "content": "第一次描述内容信息"},
            {"name": "Alpha概念", "content": "第二次更长的描述内容信息补充"},
        ]

    def fake_extract_relations(_mc, _txt, *, entities, **_k):
        captured.append([dict(e) for e in entities])
        return []

    monkeypatch.setattr(proc.llm_client, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(proc.llm_client, "extract_relations", fake_extract_relations)

    cache = Episode(
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
