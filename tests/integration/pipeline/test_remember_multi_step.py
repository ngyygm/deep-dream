import pytest
from datetime import datetime, timezone

from processor.llm.client import LLMClient
from processor.models import Entity, Episode
from processor.pipeline._v1_legacy import dedupe_recalled_entities
from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor
from processor.pipeline.relation import RelationProcessor
from processor.storage.manager import StorageManager


def _make_processor(tmp_path, **overrides):
    defaults = dict(
        storage_path=str(tmp_path / "graph"),
        window_size=100,
        overlap=20,
        llm_api_key=None,
        llm_model="gpt-4",
        embedding_use_local=False,
        max_llm_concurrency=2,
        remember_config={"mode": "multi_step", "entity_content_batch_size": 1, "validation_retries": 1},
    )
    defaults.update(overrides)
    return TemporalMemoryGraphProcessor(**defaults)


def _make_episode(absolute_id: str = "ep_test") -> Episode:
    now = datetime.now(timezone.utc)
    episode = Episode(
        absolute_id=absolute_id,
        content="缓存摘要",
        event_time=now,
        source_document="test.txt",
    )
    episode.processed_time = now
    return episode


def _save_entity(storage: StorageManager, family_id: str, name: str) -> Entity:
    now = datetime.now(timezone.utc)
    entity = Entity(
        absolute_id=f"{family_id}_v1",
        family_id=family_id,
        name=name,
        content=f"{name}的内容描述",
        event_time=now,
        processed_time=now,
        episode_id="ep_test",
        source_document="",
    )
    storage.save_entity(entity)
    return entity


def test_dedupe_recalled_entities_keeps_abstract_and_time_concepts():
    recalled = dedupe_recalled_entities([
        {"name": "设计原则", "content": "抽象概念"},
        {"name": "设计原则（核心）", "content": "更长的抽象概念说明"},
        {"name": "1903年", "content": "时间概念"},
        {"name": "处理进度", "content": "系统泄漏"},
    ])
    names = [item["name"] for item in recalled]
    assert any(name.startswith("设计原则") for name in names)
    assert "1903年" in names
    assert "处理进度" not in names


@pytest.mark.skip(reason="V1 pipeline removed; V2/V3 have different extraction flow")
def test_extract_only_multi_step_stabilizes_concepts_before_relation_writing(tmp_path):
    processor = _make_processor(tmp_path)
    episode = _make_episode()

    processor.llm_client.extract_anchor_entities = lambda *args, **kwargs: []
    processor.llm_client.extract_named_entities = lambda *args, **kwargs: [
        {"name": "张三", "content": "人物候选"},
        {"name": "项目A", "content": "项目候选"},
    ]
    processor.llm_client.extract_entities_by_focus = lambda *args, focus, **kwargs: (
        [{"name": "设计原则", "content": "抽象原则候选"}] if focus == "abstract" else []
    )
    processor.llm_client.fill_missing_entities = lambda *args, **kwargs: [
        {"name": "2025年", "content": "时间概念候选"}
    ]

    batch_calls = []

    def _extract_by_names(_episode, _text, names, verbose=False):
        batch_calls.append(list(names))
        return [{"name": name, "content": f"{name}的详细描述"} for name in names]

    processor.llm_client.extract_entities_by_names = _extract_by_names

    def _relation_candidates(_episode, _text, entities, **kwargs):
        names = {entity["name"] for entity in entities}
        candidates = []
        if {"张三", "项目A"}.issubset(names):
            candidates.append({"entity1_name": "张三", "entity2_name": "项目A", "content": "负责开发"})
        if {"项目A", "设计原则"}.issubset(names):
            candidates.append({"entity1_name": "项目A", "entity2_name": "设计原则", "content": "遵循"})
        if {"设计原则", "2025年"}.issubset(names):
            candidates.append({"entity1_name": "设计原则", "entity2_name": "2025年", "content": "在该年提出"})
        return candidates

    processor.llm_client.extract_relation_candidates = _relation_candidates
    processor.llm_client.write_relation_contents_for_pairs = lambda _episode, _text, _entities, relation_pairs, **kwargs: [
        {
            "entity1_name": pair["entity1_name"],
            "entity2_name": pair["entity2_name"],
            "content": f"{pair['entity1_name']}与{pair['entity2_name']}存在明确关系。",
        }
        for pair in relation_pairs
    ]

    entities, relations = processor._extract_only(
        episode,
        "张三在2025年负责项目A，并且项目A遵循新的设计原则。",
        "doc.txt",
        verbose=False,
        verbose_steps=False,
    )

    names = [entity["name"] for entity in entities]
    assert names == ["张三", "项目A", "设计原则", "2025年"]
    assert batch_calls == [["张三"], ["项目A"], ["设计原则"], ["2025年"]]
    stable_name_set = set(names)
    assert relations
    assert all(relation["entity1_name"] in stable_name_set for relation in relations)
    assert all(relation["entity2_name"] in stable_name_set for relation in relations)


def test_relation_processor_preserves_distinct_relations_for_same_pair(tmp_path):
    storage = StorageManager(str(tmp_path / "storage"))
    llm_client = LLMClient(api_key=None, context_window_tokens=8000)
    processor = RelationProcessor(storage=storage, llm_client=llm_client)
    processor.preserve_distinct_relations_per_pair = True

    _save_entity(storage, "ent_zhangsan", "张三")
    _save_entity(storage, "ent_project_a", "项目A")

    processed = processor.process_relations_batch(
        [
            {"entity1_name": "张三", "entity2_name": "项目A", "content": "张三负责项目A的研发推进。"},
            {"entity1_name": "张三", "entity2_name": "项目A", "content": "项目A记录了张三在2025年的交付结果。"},
        ],
        {"张三": "ent_zhangsan", "项目A": "ent_project_a"},
        "ep_test",
        max_workers=1,
        verbose_relation=False,
    )

    family_ids = {relation.family_id for relation in processed if relation is not None}
    assert len(family_ids) == 2


@pytest.mark.skip(reason="V1 pipeline removed; V2/V3 have different extraction flow")
def test_multi_step_writes_debug_snapshots_when_distill_dir_enabled(tmp_path):
    processor = _make_processor(
        tmp_path,
        distill_data_dir=str(tmp_path / "distill"),
        remember_config={
            "mode": "multi_step",
            "entity_write_batch_size": 1,
            "pre_alignment_validation_retries": 1,
        },
    )
    processor.llm_client._distill_task_id = "task_debug"
    episode = _make_episode()

    processor.llm_client.extract_anchor_entities = lambda *args, **kwargs: [{"name": "第一章", "content": "章节锚点"}]
    processor.llm_client.extract_named_entities = lambda *args, **kwargs: [{"name": "张三", "content": "人物候选"}]
    processor.llm_client.extract_entities_by_focus = lambda *args, **kwargs: [{"name": "设计原则", "content": "抽象候选"}]
    processor.llm_client.fill_missing_entities = lambda *args, **kwargs: []
    processor.llm_client.extract_relation_candidates = lambda *args, **kwargs: [
        {"entity1_name": "张三", "entity2_name": "设计原则", "content": "存在关系线索"}
    ]
    processor.llm_client.extract_entities_by_names = lambda _episode, _text, names, **kwargs: [
        {"name": name, "content": f"{name}的详细描述"} for name in names
    ]
    processor.llm_client.write_relation_contents_for_pairs = lambda _episode, _text, _entities, relation_pairs, **kwargs: [
        {
            "entity1_name": pair["entity1_name"],
            "entity2_name": pair["entity2_name"],
            "content": f"{pair['entity1_name']}与{pair['entity2_name']}在文本中存在明确关系。",
        }
        for pair in relation_pairs
    ]

    processor._extract_only(
        episode,
        "第一章里张三提出了设计原则。",
        "doc.txt",
        verbose=False,
        verbose_steps=False,
    )

    debug_dir = tmp_path / "distill" / "remember_debug" / "task_debug"
    assert (debug_dir / "window_001_02_anchor_recall.json").exists()
    assert (debug_dir / "window_001_11_pre_alignment_validation.json").exists()
    assert (debug_dir / "window_001_12_window_summary.json").exists()
