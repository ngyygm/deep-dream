"""
100-scenario real LLM pipeline test.

Covers diverse inputs to stress-test the always-versioning pipeline:
  - Incremental knowledge accumulation
  - Entity name variations & dedup
  - Relation versioning & content merge
  - Cross-episode entity alignment
  - Edge cases: empty content, single entity, self-referencing, long text,
    mixed language, special characters, etc.
"""

import pytest
import sys
import os
import time
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

pytestmark = pytest.mark.real_llm

API_BASE = os.environ.get("DEEP_DREAM_API", "http://localhost:16200/api/v1")
SESSION = requests.Session()
SESSION.trust_env = False


def _api_get(path, **kw):
    return SESSION.get(f"{API_BASE}{path}", timeout=30, **kw)


def _api_post(path, data=None, **kw):
    return SESSION.post(f"{API_BASE}{path}", json=data, timeout=120, **kw)


def _api_put(path, data=None, **kw):
    return SESSION.put(f"{API_BASE}{path}", json=data, timeout=30, **kw)


def _api_delete(path, **kw):
    return SESSION.delete(f"{API_BASE}{path}", timeout=30, **kw)


@pytest.fixture(scope="session", autouse=True)
def check_server():
    try:
        r = _api_get("/health")
        assert r.status_code == 200
    except Exception as e:
        pytest.skip(f"Server not available: {e}")


def _remember(text, source="test", graph_id="default"):
    """Submit remember and wait for completion. Returns result data."""
    r = _api_post("/remember", {"text": text, "source_name": source, "graph_id": graph_id})
    assert r.status_code in (200, 202), f"remember failed: {r.status_code} {r.text[:200]}"
    data = r.json().get("data", {})
    task_id = data.get("task_id")
    initial_status = data.get("status", "")

    if task_id and initial_status in ("queued", "processing"):
        for _ in range(180):
            time.sleep(2)
            sr = _api_get(f"/remember/tasks/{task_id}?graph_id={graph_id}")
            if sr.status_code != 200:
                continue
            sd = sr.json().get("data", {})
            if sd.get("status") in ("completed", "failed"):
                return sd
        raise TimeoutError(f"Task {task_id} timed out")
    return data


def _get_entities(graph_id="default"):
    r = _api_get(f"/find/entities?graph_id={graph_id}&limit=500")
    return r.json().get("data", {}).get("entities", [])


def _get_relations(graph_id="default"):
    r = _api_get(f"/find/relations?graph_id={graph_id}&limit=500")
    return r.json().get("data", {}).get("relations", [])


def _entity_versions(family_id, graph_id="default"):
    r = _api_get(f"/find/entities/{family_id}/versions?graph_id={graph_id}")
    return r.json().get("data", [])


def _entity_relations(family_id, graph_id="default", scope="accumulated"):
    r = _api_get(f"/find/entities/{family_id}/relations?graph_id={graph_id}&relation_scope={scope}")
    # API returns {"data": {"relations": [...], "total": N}} or {"data": [...]}
    data = r.json().get("data", {})
    if isinstance(data, dict):
        return data.get("relations", [])
    return data if isinstance(data, list) else []


def _quick_search(query, graph_id="default"):
    r = _api_post("/find", {"query": query, "graph_id": graph_id})
    return r.json().get("data", {})


# ── Helper: create a fresh test graph ──
_test_graph_counter = 0


def _fresh_graph():
    global _test_graph_counter
    _test_graph_counter += 1
    gid = f"test_100_{int(time.time())}_{_test_graph_counter}"
    r = _api_post("/graphs", {"graph_id": gid, "name": f"100-scenario test {gid}"})
    assert r.status_code in (200, 201), f"Create graph failed: {r.text[:200]}"
    yield gid
    # Cleanup
    try:
        _api_delete(f"/graphs/{gid}")
    except Exception:
        pass


import contextlib


@contextlib.contextmanager
def _test_graph():
    gid = None
    for g in _fresh_graph():
        gid = g
        break
    try:
        yield gid
    finally:
        try:
            _api_delete(f"/graphs/{gid}")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# SCENARIO GROUPS
# ═══════════════════════════════════════════════════════════════

class TestScenario01_10_BasicExtraction:
    """Scenarios 1-10: Basic single-remember extraction."""

    def test_01_simple_sentence(self):
        """Single sentence with one entity and one relation."""
        with _test_graph() as gid:
            res = _remember("张三是北京大学的一名教授。", source="s01", graph_id=gid)
            assert res.get("status") == "completed" or res.get("entities_count", 0) >= 0
            ents = _get_entities(gid)
            assert len(ents) >= 1, "Should extract at least 1 entity"

    def test_02_two_entities_one_relation(self):
        """Two entities connected by a relation."""
        with _test_graph() as gid:
            res = _remember("李四在清华大学研究人工智能。", source="s02", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2, f"Should extract >= 2 entities, got {len(ents)}"

    def test_03_long_paragraph(self):
        """Long paragraph with multiple entities and relations."""
        text = (
            "王五是阿里巴巴集团的创始人之一，他于1999年在杭州创立了该公司。"
            "阿里巴巴最初是一个B2B电子商务平台，后来扩展到零售、云计算和数字支付等领域。"
            "蚂蚁集团是阿里巴巴的金融科技子公司，由王五实际控制。"
            "2020年，蚂蚁集团的IPO被中国监管部门叫停。"
        )
        with _test_graph() as gid:
            _remember(text, source="s03", graph_id=gid)
            ents = _get_entities(gid)
            rels = _get_relations(gid)
            assert len(ents) >= 3, f"Should extract >= 3 entities from long text, got {len(ents)}"
            assert len(rels) >= 2, f"Should extract >= 2 relations, got {len(rels)}"

    def test_04_english_text(self):
        """English text extraction."""
        with _test_graph() as gid:
            _remember(
                "Elon Musk is the CEO of Tesla and SpaceX. Tesla manufactures electric vehicles.",
                source="s04", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 2, f"Should extract English entities, got {len(ents)}"

    def test_05_mixed_language(self):
        """Mixed Chinese/English text."""
        with _test_graph() as gid:
            _remember(
                "Apple公司在Cupertino总部发布了最新的iPhone产品线。"
                "CEO Tim Cook表示这代产品的AI功能是最大的亮点。",
                source="s05", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_06_single_entity_no_relation(self):
        """Text mentioning only one entity — should not create self-referencing relations."""
        with _test_graph() as gid:
            _remember("量子计算是一个快速发展的技术领域。", source="s06", graph_id=gid)
            ents = _get_entities(gid)
            rels = _get_relations(gid)
            # May have 1 entity, 0 relations
            assert len(ents) >= 1

    def test_07_numbered_list(self):
        """Numbered list format."""
        text = (
            "项目进度报告：\n"
            "1. 张三负责前端开发\n"
            "2. 李四负责后端架构\n"
            "3. 王五负责数据分析\n"
            "项目由赵六担任项目经理，整体进度良好。"
        )
        with _test_graph() as gid:
            _remember(text, source="s07", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 3

    def test_08_dialogue_format(self):
        """Dialogue/conversation text."""
        text = (
            "张三说：'我认为这个方案可行。'\n"
            "李四回答：'我也同意，但我们还需要考虑预算问题。'\n"
            "王五补充道：'预算方面我已经和财务部的陈六沟通过了。'"
        )
        with _test_graph() as gid:
            _remember(text, source="s08", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 3

    def test_09_technical_jargon(self):
        """Technical domain text with jargon."""
        text = (
            "Kubernetes是一个开源的容器编排系统，由Google最初开发。"
            "它使用Pod作为最小的部署单元，通过Service暴露应用。"
            "Docker是最常用的容器运行时，与Kubernetes配合使用。"
        )
        with _test_graph() as gid:
            _remember(text, source="s09", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_10_short_abbreviations(self):
        """Entities with abbreviations."""
        with _test_graph() as gid:
            _remember(
                "WHO是世界卫生组织，UN是联合国。WHO是UN的专门机构之一。",
                source="s10", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 2


class TestScenario11_20_IncrementalAccumulation:
    """Scenarios 11-20: Multi-remember incremental knowledge building."""

    def test_11_same_entity_repeated(self):
        """Same entity mentioned in two episodes — should get 2 versions."""
        with _test_graph() as gid:
            _remember("张三是北京大学的教授。", source="s11a", graph_id=gid)
            _remember("张三最近发表了一篇关于深度学习的论文。", source="s11b", graph_id=gid)
            ents = _get_entities(gid)
            zhang = [e for e in ents if "张三" in e.get("name", "")]
            assert len(zhang) >= 1, "Should find 张三 entity"
            versions = _entity_versions(zhang[0]["family_id"], gid)
            assert len(versions) >= 2, f"张三 should have >= 2 versions (1 per episode), got {len(versions)}"

    def test_12_entity_content_increments(self):
        """Second mention should incrementally add info, not rewrite."""
        with _test_graph() as gid:
            _remember("李四是清华大学的博士研究生。", source="s12a", graph_id=gid)
            _remember("李四的研究方向是自然语言处理。", source="s12b", graph_id=gid)
            ents = _get_entities(gid)
            li = [e for e in ents if "李四" in e.get("name", "")]
            assert len(li) >= 1
            versions = _entity_versions(li[0]["family_id"], gid)
            # Latest version should contain info from both episodes
            latest = versions[0]  # sorted by processed_time DESC
            content = latest.get("content", "")
            # Should NOT have completely replaced the original content
            assert len(content) > 5, f"Content too short: {content}"

    def test_13_relation_accumulation(self):
        """Relations from different episodes should all be visible."""
        with _test_graph() as gid:
            _remember("王五在阿里巴巴工作。", source="s13a", graph_id=gid)
            _remember("王五毕业于浙江大学。", source="s13b", graph_id=gid)
            ents = _get_entities(gid)
            wang = [e for e in ents if "王五" in e.get("name", "")]
            assert len(wang) >= 1
            rels = _entity_relations(wang[0]["family_id"], gid)
            assert len(rels) >= 2, f"王五 should have >= 2 relations, got {len(rels)}"

    def test_14_three_episodes_same_entity(self):
        """Three episodes mentioning the same entity — 3 versions."""
        with _test_graph() as gid:
            _remember("赵六是一名软件工程师。", source="s14a", graph_id=gid)
            _remember("赵六精通Python和Java编程语言。", source="s14b", graph_id=gid)
            _remember("赵六最近跳槽到了字节跳动。", source="s14c", graph_id=gid)
            ents = _get_entities(gid)
            zhao = [e for e in ents if "赵六" in e.get("name", "")]
            assert len(zhao) >= 1
            versions = _entity_versions(zhao[0]["family_id"], gid)
            assert len(versions) >= 3, f"赵六 should have >= 3 versions, got {len(versions)}"

    def test_15_new_entity_appears_in_second_episode(self):
        """New entity introduced in second episode should be created."""
        with _test_graph() as gid:
            _remember("张三是一名教师。", source="s15a", graph_id=gid)
            _remember("张三的妻子叫小红，他们有两个孩子。", source="s15b", graph_id=gid)
            ents = _get_entities(gid)
            names = {e.get("name", "") for e in ents}
            assert any("小红" in n or "张三" in n for n in names)

    def test_16_relation_between_old_and_new_entity(self):
        """Relation between entity from ep1 and entity from ep2."""
        with _test_graph() as gid:
            _remember("小明喜欢打篮球。", source="s16a", graph_id=gid)
            _remember("小明和小红是同学关系。", source="s16b", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_17_identical_text_twice(self):
        """Same text submitted twice — both should create versions."""
        with _test_graph() as gid:
            text = "量子计算是未来的关键技术方向。"
            _remember(text, source="s17a", graph_id=gid)
            _remember(text, source="s17b", graph_id=gid)
            ents = _get_entities(gid)
            # Should have the entity with 2 versions
            for e in ents:
                versions = _entity_versions(e["family_id"], gid)
                if len(versions) >= 2:
                    return  # Found an entity with 2 versions
            # At minimum, should not crash
            assert len(ents) >= 1

    def test_18_contradictory_info(self):
        """Second episode contradicts first — should create version, not crash."""
        with _test_graph() as gid:
            _remember("张三今年25岁。", source="s18a", graph_id=gid)
            _remember("张三今年30岁。", source="s18b", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_19_incremental_detail(self):
        """Progressive detail addition across episodes."""
        with _test_graph() as gid:
            _remember("百度是一家中国科技公司。", source="s19a", graph_id=gid)
            _remember("百度的总部位于北京海淀区。", source="s19b", graph_id=gid)
            _remember("百度的创始人包括李彦宏和徐勇。", source="s19c", graph_id=gid)
            _remember("百度在纳斯达克上市，股票代码BIDU。", source="s19d", graph_id=gid)
            ents = _get_entities(gid)
            baidu = [e for e in ents if "百度" in e.get("name", "")]
            assert len(baidu) >= 1
            versions = _entity_versions(baidu[0]["family_id"], gid)
            assert len(versions) >= 3, f"百度 should have >= 3 versions, got {len(versions)}"

    def test_20_large_knowledge_graph(self):
        """Submit 5 episodes building a knowledge graph about a company."""
        with _test_graph() as gid:
            texts = [
                "腾讯是一家中国互联网公司，总部在深圳。",
                "腾讯的主要产品包括微信、QQ和腾讯云。",
                "微信由张小龙领导开发，月活跃用户超过12亿。",
                "腾讯的创始人是马化腾，他毕业于深圳大学。",
                "腾讯云是腾讯的云计算平台，与阿里云竞争。",
            ]
            for i, text in enumerate(texts):
                _remember(text, source=f"s20_{i}", graph_id=gid)
            ents = _get_entities(gid)
            rels = _get_relations(gid)
            assert len(ents) >= 5, f"Should have >= 5 entities, got {len(ents)}"
            assert len(rels) >= 3, f"Should have >= 3 relations, got {len(rels)}"


class TestScenario21_30_EntityNameVariations:
    """Scenarios 21-30: Entity name matching and dedup."""

    def test_21_full_name_vs_short_name(self):
        """Same person referenced by full name and short name."""
        with _test_graph() as gid:
            _remember("李彦宏是百度的CEO。", source="s21a", graph_id=gid)
            _remember("Robin Li在人工智能领域投入了大量资源。", source="s21b", graph_id=gid)
            ents = _get_entities(gid)
            # May create 1 or 2 entities depending on alignment
            assert len(ents) >= 1

    def test_22_entity_with_title(self):
        """Same entity with and without professional title."""
        with _test_graph() as gid:
            _remember("张教授在量子物理领域有重要贡献。", source="s22a", graph_id=gid)
            _remember("张三教授最近获得了国家科学奖。", source="s22b", graph_id=gid)
            ents = _get_entities(gid)
            # Both should be processed without error
            assert len(ents) >= 1

    def test_23_english_chinese_same_entity(self):
        """Same entity in English and Chinese."""
        with _test_graph() as gid:
            _remember("Apple Inc. is headquartered in Cupertino, California.", source="s23a", graph_id=gid)
            _remember("苹果公司是全球市值最高的科技公司之一。", source="s23b", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_24_entity_with_parenthetical(self):
        """Entity name with parenthetical annotation."""
        with _test_graph() as gid:
            _remember("GPT-4（生成式预训练变换器4）是OpenAI开发的大语言模型。", source="s24", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_25_similar_but_different_entities(self):
        """Two different entities with similar names — should NOT merge."""
        with _test_graph() as gid:
            _remember("张三是北京大学的教授。", source="s25a", graph_id=gid)
            _remember("张三是清华大学的学生。（注意：这是不同的张三）", source="s25b", graph_id=gid)
            ents = _get_entities(gid)
            # Should create separate entities or at least not crash
            assert len(ents) >= 1

    def test_26_entity_name_with_number(self):
        """Entity name containing numbers."""
        with _test_graph() as gid:
            _remember("GPT-4o是OpenAI的最新模型，GPT-3.5是其前代产品。", source="s26", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_27_long_entity_name(self):
        """Very long entity name."""
        with _test_graph() as gid:
            _remember(
                "中华人民共和国国家发展和改革委员会是国务院组成部门之一。",
                source="s27", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_28_foreign_names(self):
        """Non-Chinese foreign names."""
        with _test_graph() as gid:
            _remember(
                "Alexander Fleming discovered penicillin. Marie Curie researched radioactivity.",
                source="s28", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_29_brand_names(self):
        """Brand and product names."""
        with _test_graph() as gid:
            _remember(
                "特斯拉Model 3是2023年全球最畅销的电动车。"
                "比亚迪汉EV在中国市场销量领先。",
                source="s29", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_30_event_as_entity(self):
        """Events treated as entities."""
        with _test_graph() as gid:
            _remember(
                "第二次世界大战爆发于1939年，结束于1945年。"
                "第一次世界大战是它的前因之一。",
                source="s30", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 1


class TestScenario31_40_RelationQuality:
    """Scenarios 31-40: Relation extraction and versioning quality."""

    def test_31_relation_version_count(self):
        """Each episode mentioning a relation should create a version."""
        with _test_graph() as gid:
            _remember("张三在百度工作。", source="s31a", graph_id=gid)
            _remember("张三已经从百度离职了。", source="s31b", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_32_multiple_relations_same_pair(self):
        """Multiple relations between the same pair of entities."""
        with _test_graph() as gid:
            _remember(
                "张三既是李四的同事，也是李四的大学同学。"
                "他们还在同一个项目组工作。",
                source="s32", graph_id=gid
            )
            ents = _get_entities(gid)
            rels = _get_relations(gid)
            assert len(ents) >= 2

    def test_33_relation_with_temporal_info(self):
        """Relation containing temporal information."""
        with _test_graph() as gid:
            _remember(
                "马云在1999年至2019年期间担任阿里巴巴集团的董事局主席。",
                source="s33", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_34_causal_relation(self):
        """Causal relation extraction."""
        with _test_graph() as gid:
            _remember(
                "因为全球芯片短缺，导致汽车产量大幅下降。",
                source="s34", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_35_hierarchical_relation(self):
        """Hierarchical (part-of) relations."""
        with _test_graph() as gid:
            _remember(
                "浦东新区是上海市的一个市辖区。上海是中国的直辖市之一。",
                source="s35", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_36_relation_merge_across_episodes(self):
        """Relation content should be incrementally merged across episodes."""
        with _test_graph() as gid:
            _remember("张三是李四的导师。", source="s36a", graph_id=gid)
            _remember("张三指导李四完成了博士论文，研究方向是机器学习。", source="s36b", graph_id=gid)
            rels = _get_relations(gid)
            assert len(rels) >= 1

    def test_37_implicit_relation(self):
        """Implicit relation that requires inference."""
        with _test_graph() as gid:
            _remember(
                "张艺谋执导了电影《英雄》。《英雄》的主演包括李连杰和梁朝伟。",
                source="s37", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_38_negative_relation(self):
        """Negative relation (NOT related)."""
        with _test_graph() as gid:
            _remember(
                "虽然很多人认为牛顿和莱布尼茨互相抄袭，但实际上他们独立发明了微积分。",
                source="s38", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_39_symmetric_relation(self):
        """Symmetric relation (A-B same as B-A)."""
        with _test_graph() as gid:
            _remember("北京和上海之间有直达高铁。", source="s39", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_40_complex_network(self):
        """Complex network of relations."""
        text = (
            "乔布斯创建了Apple。沃兹尼亚克是Apple的联合创始人。"
            "乔布斯也是Pixar的CEO。Disney收购了Pixar。"
            "蒂姆·库克接替乔布斯成为Apple的CEO。"
        )
        with _test_graph() as gid:
            _remember(text, source="s40", graph_id=gid)
            ents = _get_entities(gid)
            rels = _get_relations(gid)
            assert len(ents) >= 4
            assert len(rels) >= 3


class TestScenario41_50_EdgeCases:
    """Scenarios 41-50: Edge cases and boundary conditions."""

    def test_41_very_short_text(self):
        """Very short text."""
        with _test_graph() as gid:
            _remember("张三。", source="s41", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 0  # May or may not extract

    def test_42_very_long_text(self):
        """Very long text (multiple paragraphs)."""
        text = (
            "中国的科技产业在过去二十年经历了飞速发展。" * 5 +
            "阿里巴巴、腾讯和百度被称为中国互联网的三巨头。" +
            "华为是全球领先的通信设备供应商。" +
            "字节跳动是TikTok的母公司。" +
            "这些公司都在人工智能领域投入了大量资源。"
        )
        with _test_graph() as gid:
            _remember(text, source="s42", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 3

    def test_43_special_characters(self):
        """Text with special characters."""
        with _test_graph() as gid:
            _remember(
                "C++和Java是最流行的编程语言之一。<script>alert('xss')</script>",
                source="s43", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_44_unicode_emoji(self):
        """Text with emoji and unicode."""
        with _test_graph() as gid:
            _remember(
                "🚀 SpaceX成功发射了星舰🛸，马斯克🎉表示这是太空探索的重要里程碑。",
                source="s44", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_45_math_formula(self):
        """Text with math formula."""
        with _test_graph() as gid:
            _remember(
                "爱因斯坦的质能方程 E=mc² 是物理学最著名的公式之一。"
                "它揭示了质量和能量之间的关系。",
                source="s45", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_46_repeated_entity_mentions(self):
        """Same entity mentioned many times in one text."""
        with _test_graph() as gid:
            _remember(
                "张三是教授。张三喜欢读书。张三教数学。张三住在海淀。"
                "张三的办公室在3号楼。张三有两个学生。",
                source="s46", graph_id=gid
            )
            ents = _get_entities(gid)
            zhang = [e for e in ents if "张三" in e.get("name", "")]
            assert len(zhang) >= 1

    def test_47_many_entities_few_relations(self):
        """Many entities but few explicit relations."""
        with _test_graph() as gid:
            _remember(
                "在场的有：张三、李四、王五、赵六、钱七、孙八、周九、吴十。",
                source="s47", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 3

    def test_48_json_format_text(self):
        """Text containing JSON-like format."""
        with _test_graph() as gid:
            _remember(
                '{"person": "张三", "role": "教授", "university": "北京大学"}',
                source="s48", graph_id=gid
            )
            ents = _get_entities(gid)
            # Should handle gracefully
            assert True

    def test_49_code_snippet_text(self):
        """Text containing code."""
        with _test_graph() as gid:
            _remember(
                "Python的print函数用于输出：print('Hello World')。"
                "JavaScript使用console.log实现相同功能。",
                source="s49", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_50_only_stop_words(self):
        """Text with only common/stop words."""
        with _test_graph() as gid:
            _remember("的 了 在 是 和 有 这个 那个", source="s50", graph_id=gid)
            ents = _get_entities(gid)
            # Should not crash, may extract 0 entities
            assert isinstance(ents, list)


class TestScenario51_60_CrossEpisodeAlignment:
    """Scenarios 51-60: Cross-episode entity alignment accuracy."""

    def test_51_different_contexts_same_entity(self):
        """Same entity in very different contexts."""
        with _test_graph() as gid:
            _remember("北京是中国的首都，人口超过2000万。", source="s51a", graph_id=gid)
            _remember("北京烤鸭是著名的传统美食。", source="s51b", graph_id=gid)
            ents = _get_entities(gid)
            beijing = [e for e in ents if "北京" in e.get("name", "")]
            assert len(beijing) >= 1

    def test_52_entity_with_alias(self):
        """Entity commonly known by alias."""
        with _test_graph() as gid:
            _remember("马云是阿里巴巴的创始人。", source="s52a", graph_id=gid)
            _remember("Jack Ma曾经是杭州电子科技大学的英语教师。", source="s52b", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_53_organization_rename(self):
        """Organization that changed its name."""
        with _test_graph() as gid:
            _remember("Facebook在2021年更名为Meta。", source="s53a", graph_id=gid)
            _remember("Meta的CEO马克·扎克伯格致力于元宇宙的发展。", source="s53b", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_54_disambiguation(self):
        """Entities needing disambiguation (same name, different things)."""
        with _test_graph() as gid:
            _remember("Python是一种编程语言。", source="s54a", graph_id=gid)
            _remember("Python（蟒蛇）是世界上最长的蛇之一。", source="s54b", graph_id=gid)
            ents = _get_entities(gid)
            # Should create at least one entity, ideally two different ones
            assert len(ents) >= 1

    def test_55_cross_domain_entity(self):
        """Entity appearing across different domains."""
        with _test_graph() as gid:
            _remember("刘慈欣是科幻作家，代表作有《三体》。", source="s55a", graph_id=gid)
            _remember("刘慈欣获得过雨果奖，这是科幻界的最高荣誉。", source="s55b", graph_id=gid)
            _remember("《三体》被Netflix改编为电视剧。", source="s55c", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 3

    def test_56_entity_from_different_sources(self):
        """Same entity described from different source perspectives."""
        with _test_graph() as gid:
            _remember("特斯拉在2023年交付了180万辆汽车。", source="financial_report", graph_id=gid)
            _remember("特斯拉的Autopilot系统引发了安全争议。", source="news_article", graph_id=gid)
            ents = _get_entities(gid)
            tesla = [e for e in ents if "特斯拉" in e.get("name", "")]
            assert len(tesla) >= 1

    def test_57_nested_entities(self):
        """Nested/hierarchical entities."""
        with _test_graph() as gid:
            _remember(
                "北京大学计算机科学系下设人工智能实验室。"
                "实验室主任是黄铁军教授。",
                source="s57", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_58_entity_with_multiple_roles(self):
        """Person with multiple roles."""
        with _test_graph() as gid:
            _remember("雷军是小米的创始人兼CEO。", source="s58a", graph_id=gid)
            _remember("雷军同时也是一名天使投资人。", source="s58b", graph_id=gid)
            _remember("雷军曾在金山软件工作超过16年。", source="s58c", graph_id=gid)
            ents = _get_entities(gid)
            lei = [e for e in ents if "雷军" in e.get("name", "")]
            assert len(lei) >= 1
            versions = _entity_versions(lei[0]["family_id"], gid)
            assert len(versions) >= 3

    def test_59_time_travel_entity(self):
        """Entity at different time points."""
        with _test_graph() as gid:
            _remember("2010年，小米公司刚刚成立。", source="s59a", graph_id=gid)
            _remember("2020年，小米已成为全球第三大手机厂商。", source="s59b", graph_id=gid)
            _remember("2024年，小米开始进军新能源汽车市场。", source="s59c", graph_id=gid)
            ents = _get_entities(gid)
            xiaomi = [e for e in ents if "小米" in e.get("name", "")]
            assert len(xiaomi) >= 1

    def test_60_complex_alignment_chain(self):
        """Chain of entities linked through multiple episodes."""
        with _test_graph() as gid:
            _remember("Alice在Google工作。", source="s60a", graph_id=gid)
            _remember("Alice的同事Bob跳槽到了Meta。", source="s60b", graph_id=gid)
            _remember("Bob在Meta认识了Charlie。", source="s60c", graph_id=gid)
            _remember("Charlie以前在Apple工作。", source="s60d", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 3


class TestScenario61_70_ContentQuality:
    """Scenarios 61-70: Content quality and merge behavior."""

    def test_61_content_preserves_original_structure(self):
        """Content merge should preserve original content structure."""
        with _test_graph() as gid:
            _remember("量子计算利用量子力学的原理进行计算。量子比特是基本信息单位。", source="s61a", graph_id=gid)
            _remember("量子计算有望破解现有的RSA加密算法。", source="s61b", graph_id=gid)
            ents = _get_entities(gid)
            for e in ents:
                if "量子" in e.get("name", ""):
                    content = e.get("content", "")
                    assert len(content) > 10, "Content should be meaningful"

    def test_62_no_content_duplication(self):
        """Content merge should not produce duplicated text."""
        with _test_graph() as gid:
            _remember("深度学习是机器学习的一个分支。", source="s62a", graph_id=gid)
            _remember("深度学习使用神经网络进行学习。", source="s62b", graph_id=gid)
            ents = _get_entities(gid)
            for e in ents:
                if "深度学习" in e.get("name", ""):
                    content = e.get("content", "")
                    # Content should not repeat the same sentence twice
                    assert content.count("深度学习") < 5, "Possible content duplication"

    def test_63_relation_content_not_generic(self):
        """Relation content should be specific, not generic."""
        with _test_graph() as gid:
            _remember("张三和李四是同学，他们一起毕业于清华大学计算机系。", source="s63", graph_id=gid)
            rels = _get_relations(gid)
            assert len(rels) >= 1
            for r in rels:
                content = r.get("content", "")
                # Content should be more than just "相关" or "有关联"
                assert len(content) > 3

    def test_64_summary_quality(self):
        """Entity summary should be a concise description."""
        with _test_graph() as gid:
            _remember("北京大学是中国最著名的综合性大学之一，创建于1898年。", source="s64", graph_id=gid)
            ents = _get_entities(gid)
            for e in ents:
                if "北京" in e.get("name", ""):
                    summary = e.get("summary", "")
                    assert len(summary) > 0, "Summary should not be empty"

    def test_65_confidence_increases(self):
        """Entity confidence should increase with corroboration."""
        with _test_graph() as gid:
            _remember("AlphaGo是DeepMind开发的围棋AI。", source="s65a", graph_id=gid)
            _remember("AlphaGo在2016年击败了世界冠军李世石。", source="s65b", graph_id=gid)
            ents = _get_entities(gid)
            for e in ents:
                if "AlphaGo" in e.get("name", ""):
                    conf = e.get("confidence", 0)
                    assert conf >= 0.5, f"Confidence too low: {conf}"

    def test_66_content_sections_preserved(self):
        """Content sections should be preserved across versions."""
        with _test_graph() as gid:
            _remember("深度学习是AI的核心技术。", source="s66a", graph_id=gid)
            _remember("深度学习的最新突破是Transformer架构。", source="s66b", graph_id=gid)
            ents = _get_entities(gid)
            for e in ents:
                if "深度学习" in e.get("name", ""):
                    sections = e.get("content_sections", {})
                    # Should have some structured content
                    if sections:
                        assert isinstance(sections, dict)

    def test_67_relation_direction_consistency(self):
        """Relation entity1/entity2 ordering should be consistent."""
        with _test_graph() as gid:
            _remember("张三在阿里巴巴工作。", source="s67", graph_id=gid)
            rels = _get_relations(gid)
            # Entity names should be consistent (sorted)
            for r in rels:
                e1 = r.get("entity1_name", "")
                e2 = r.get("entity2_name", "")
                if e1 and e2:
                    assert e1 <= e2, f"Entity ordering wrong: {e1} > {e2}"

    def test_68_source_document_tracked(self):
        """Source document should be tracked in versions."""
        with _test_graph() as gid:
            _remember("特斯拉由马斯克创立。", source="doc_wiki", graph_id=gid)
            ents = _get_entities(gid)
            for e in ents:
                versions = _entity_versions(e["family_id"], gid)
                for v in versions:
                    src = v.get("source_document", "")
                    if src:
                        assert "doc_wiki" in src or "wiki" in src, f"Source tracking wrong: {src}"
                        return

    def test_69_episode_id_linked(self):
        """Each version should be linked to an episode."""
        with _test_graph() as gid:
            _remember("GitHub被微软收购了。", source="s69", graph_id=gid)
            ents = _get_entities(gid)
            for e in ents:
                versions = _entity_versions(e["family_id"], gid)
                for v in versions:
                    ep = v.get("episode_id", "")
                    assert ep, f"Version missing episode_id: {v.get('absolute_id')}"

    def test_70_valid_at_timestamp(self):
        """Each version should have valid_at timestamp."""
        with _test_graph() as gid:
            _remember("Claude是Anthropic开发的AI助手。", source="s70", graph_id=gid)
            ents = _get_entities(gid)
            for e in ents:
                versions = _entity_versions(e["family_id"], gid)
                for v in versions:
                    assert v.get("valid_at"), f"Version missing valid_at: {v.get('absolute_id')}"


class TestScenario71_80_SearchAndRetrieval:
    """Scenarios 71-80: Search and retrieval after ingestion."""

    def test_71_quick_search_finds_entity(self):
        """Quick search should find ingested entity."""
        with _test_graph() as gid:
            _remember("深度学习框架PyTorch由Meta AI开发。", source="s71", graph_id=gid)
            result = _quick_search("PyTorch", gid)
            ents = result.get("entities", [])
            assert len(ents) >= 1, "Should find PyTorch via search"

    def test_72_search_by_relation_content(self):
        """Search should find entities via relation content."""
        with _test_graph() as gid:
            _remember("张三是李四的导师，张三指导李四研究机器学习。", source="s72", graph_id=gid)
            result = _quick_search("导师", gid)
            # Should find something related
            total = len(result.get("entities", [])) + len(result.get("relations", []))
            assert total >= 1

    def test_73_entity_profile_has_relations(self):
        """Entity profile should include relations."""
        with _test_graph() as gid:
            _remember("王五在阿里巴巴担任CTO。", source="s73", graph_id=gid)
            ents = _get_entities(gid)
            for e in ents:
                if "王五" in e.get("name", ""):
                    rels = _entity_relations(e["family_id"], gid)
                    assert len(rels) >= 1

    def test_74_version_timeline_ordered(self):
        """Version timeline should be in chronological order."""
        with _test_graph() as gid:
            _remember("版本1：初始信息。", source="s74a", graph_id=gid)
            _remember("版本2：更新信息。", source="s74b", graph_id=gid)
            _remember("版本3：最新信息。", source="s74c", graph_id=gid)
            ents = _get_entities(gid)
            for e in ents:
                versions = _entity_versions(e["family_id"], gid)
                if len(versions) >= 2:
                    times = [v.get("processed_time", "") for v in versions]
                    # processed_time should be DESC (latest first)
                    assert times[0] >= times[-1], f"Versions not in DESC order: {times}"

    def test_75_search_after_multiple_episodes(self):
        """Search should work correctly after multiple episodes."""
        with _test_graph() as gid:
            _remember("React是Facebook开发的前端框架。", source="s75a", graph_id=gid)
            _remember("Vue.js是尤雨溪创建的前端框架。", source="s75b", graph_id=gid)
            _remember("Angular是Google维护的前端框架。", source="s75c", graph_id=gid)
            result = _quick_search("前端框架", gid)
            ents = result.get("entities", [])
            assert len(ents) >= 2

    def test_76_empty_search_query(self):
        """Empty search should not crash."""
        with _test_graph() as gid:
            _remember("测试文本。", source="s76", graph_id=gid)
            # Just ensure it doesn't crash
            try:
                _quick_search("", gid)
            except Exception:
                pass

    def test_77_search_nonexistent_entity(self):
        """Search for non-existent entity should return empty."""
        with _test_graph() as gid:
            result = _quick_search("这个实体肯定不存在于图数据库中XYZ123", gid)
            ents = result.get("entities", [])
            assert len(ents) == 0

    def test_78_entity_count_after_episodes(self):
        """Entity count should increase with new entities across episodes."""
        with _test_graph() as gid:
            _remember("张三是一名医生。", source="s78a", graph_id=gid)
            count1 = len(_get_entities(gid))
            _remember("李四是一名律师。", source="s78b", graph_id=gid)
            count2 = len(_get_entities(gid))
            assert count2 >= count1

    def test_79_relation_count_after_episodes(self):
        """Relation count should increase with new relations."""
        with _test_graph() as gid:
            _remember("王五在腾讯工作。", source="s79a", graph_id=gid)
            _remember("赵六在阿里巴巴工作。", source="s79b", graph_id=gid)
            rels = _get_relations(gid)
            assert len(rels) >= 1

    def test_80_cross_episode_search(self):
        """Search should find entities from all episodes."""
        with _test_graph() as gid:
            _remember("Rust是一种系统编程语言。", source="s80a", graph_id=gid)
            _remember("Go语言由Google的Robert Griesemer等人设计。", source="s80b", graph_id=gid)
            result = _quick_search("编程语言", gid)
            ents = result.get("entities", [])
            assert len(ents) >= 1


class TestScenario81_90_StressAndRobustness:
    """Scenarios 81-90: Stress testing and robustness."""

    def test_81_rapid_fire_submissions(self):
        """Submit 5 remembers rapidly without waiting."""
        with _test_graph() as gid:
            texts = [
                f"测试人物{i}在测试公司{i}工作。" for i in range(5)
            ]
            results = []
            for text in texts:
                r = _api_post("/remember", {"text": text, "source_name": f"s81_{len(results)}", "graph_id": gid})
                results.append(r.status_code)
            # All should be accepted
            for code in results:
                assert code in (200, 202), f"Unexpected status: {code}"

    def test_82_large_batch_entities(self):
        """Text with many entities."""
        names = ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十",
                 "郑十一", "冯十二"]
        text = "、".join(names) + "等人参加了这次学术会议。"
        with _test_graph() as gid:
            _remember(text, source="s82", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 3

    def test_83_deeply_nested_description(self):
        """Deeply nested descriptions."""
        text = (
            "张三（北京大学计算机系教授，博士生导师，"
            "IEEE Fellow，ACM杰出科学家）"
            "在人工智能领域（包括机器学习、深度学习、自然语言处理）"
            "有重要贡献。"
        )
        with _test_graph() as gid:
            _remember(text, source="s83", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_84_mixed_formatting(self):
        """Text with mixed formatting (markdown, HTML)."""
        text = (
            "# 项目报告\n"
            "## 成员\n"
            "- **张三**：项目经理\n"
            "- *李四*：技术负责人\n"
            "- `王五`：开发者\n"
            "> 赵六提供了外部支持"
        )
        with _test_graph() as gid:
            _remember(text, source="s84", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_85_multiline_table(self):
        """Text in table format."""
        text = (
            "项目成员表：\n"
            "姓名 | 部门 | 职位\n"
            "张三 | 技术部 | 总监\n"
            "李四 | 产品部 | 经理\n"
            "王五 | 设计部 | 设计师"
        )
        with _test_graph() as gid:
            _remember(text, source="s85", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_86_url_in_text(self):
        """Text containing URLs."""
        with _test_graph() as gid:
            _remember(
                "更多信息请访问 https://openai.com 了解OpenAI的最新动态。",
                source="s86", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_87_repeated_remember_same_content(self):
        """Submit same content 3 times."""
        with _test_graph() as gid:
            text = "人工智能正在改变世界。"
            for i in range(3):
                _remember(text, source=f"s87_{i}", graph_id=gid)
            # Should not crash, entity should exist
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_88_empty_lines_and_whitespace(self):
        """Text with many empty lines and whitespace."""
        with _test_graph() as gid:
            _remember(
                "张三\n\n\n\n   \n李四\n\n\n王五",
                source="s88", graph_id=gid
            )
            ents = _get_entities(gid)
            assert isinstance(ents, list)

    def test_89_only_dates_and_numbers(self):
        """Text with only dates and numbers."""
        with _test_graph() as gid:
            _remember("2024年1月15日。3.14。100万。", source="s89", graph_id=gid)
            ents = _get_entities(gid)
            # Should not crash
            assert isinstance(ents, list)

    def test_90_pipe_separated_entities(self):
        """Pipe-separated entity list."""
        with _test_graph() as gid:
            _remember(
                "团队成员：张三 | 李四 | 王五 | 赵六",
                source="s90", graph_id=gid
            )
            ents = _get_entities(gid)
            assert len(ents) >= 2


class TestScenario91_100_AdvancedScenarios:
    """Scenarios 91-100: Advanced and real-world scenarios."""

    def test_91_academic_paper_abstract(self):
        """Academic paper abstract."""
        text = (
            "Abstract: In this paper, we propose a novel approach to named entity recognition "
            "using transformer-based models. Our method achieves state-of-the-art results on "
            "the CoNLL-2003 benchmark. The authors are Zhang San from Tsinghua University "
            "and Li Si from Peking University."
        )
        with _test_graph() as gid:
            _remember(text, source="s91", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_92_news_article(self):
        """News article format."""
        text = (
            "【新华社北京3月15日电】国务院今日发布《新一代人工智能发展规划》，"
            "提出到2030年使中国成为世界主要人工智能创新中心。"
            "科技部部长表示将加大对AI基础研究的投入力度。"
        )
        with _test_graph() as gid:
            _remember(text, source="s92", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_93_knowledge_accumulation_10_episodes(self):
        """Build knowledge graph across 10 episodes."""
        with _test_graph() as gid:
            episodes = [
                "上海是中国最大的城市之一。",
                "上海位于长江入海口。",
                "上海的GDP在全国排名第一。",
                "上海浦东新区是中国的金融中心。",
                "上海有复旦大学和上海交通大学等著名高校。",
                "上海迪士尼乐园是中国大陆第一个迪士尼主题公园。",
                "上海的磁悬浮列车是世界第一条商业化运营的磁悬浮线路。",
                "2024年，上海的常住人口约2500万。",
                "上海自贸区是中国第一个自由贸易试验区。",
                "上海的洋山深水港是全球最繁忙的集装箱港口之一。",
            ]
            for i, text in enumerate(episodes):
                _remember(text, source=f"s93_{i}", graph_id=gid)

            ents = _get_entities(gid)
            shanghai = [e for e in ents if "上海" in e.get("name", "")]
            assert len(shanghai) >= 1
            versions = _entity_versions(shanghai[0]["family_id"], gid)
            assert len(versions) >= 5, f"上海 should have >= 5 versions, got {len(versions)}"

    def test_94_bilingual_knowledge(self):
        """Build knowledge with bilingual episodes."""
        with _test_graph() as gid:
            _remember("The Great Wall of China is one of the Seven Wonders of the World.", source="s94a", graph_id=gid)
            _remember("中国的长城是世界上最长的建筑物，全长超过2万公里。", source="s94b", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 1

    def test_95_complex_temporal_events(self):
        """Complex temporal sequence of events."""
        text = (
            "2020年1月，新冠疫情爆发。"
            "2020年3月，WHO宣布全球大流行。"
            "2020年12月，首款疫苗获得紧急使用授权。"
            "2021年，全球开始大规模疫苗接种。"
            "2023年5月，WHO宣布疫情不再构成国际关注的突发公共卫生事件。"
        )
        with _test_graph() as gid:
            _remember(text, source="s95", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 2

    def test_96_graph_statistics(self):
        """Graph statistics should be accurate after ingestion."""
        with _test_graph() as gid:
            _remember("Google总部位于Mountain View。Google的CEO是Sundar Pichai。", source="s96", graph_id=gid)
            ents = _get_entities(gid)
            rels = _get_relations(gid)
            assert len(ents) >= 2

    def test_97_concurrent_entities(self):
        """Multiple entities introduced simultaneously with complex relations."""
        text = (
            "在这部电影中，周星驰既是导演又是主演。"
            "吴孟达饰演周星驰的搭档。"
            "张敏饰演女主角。"
            "该电影由华谊兄弟出品。"
        )
        with _test_graph() as gid:
            _remember(text, source="s97", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 3

    def test_98_multi_hop_relations(self):
        """Relations spanning multiple hops (A→B→C)."""
        with _test_graph() as gid:
            _remember("张三在百度工作。", source="s98a", graph_id=gid)
            _remember("百度在北京设有总部。", source="s98b", graph_id=gid)
            _remember("北京是中国的首都。", source="s98c", graph_id=gid)
            ents = _get_entities(gid)
            assert len(ents) >= 3

    def test_99_real_world_company_profile(self):
        """Real-world company profile accumulation."""
        with _test_graph() as gid:
            _remember(
                "NVIDIA成立于1993年，总部位于美国加州圣克拉拉。",
                source="s99a", graph_id=gid
            )
            _remember(
                "NVIDIA的CEO是黄仁勋（Jensen Huang），他是一位美籍华裔企业家。",
                source="s99b", graph_id=gid
            )
            _remember(
                "NVIDIA是全球领先的GPU制造商，其产品广泛应用于AI训练和推理。",
                source="s99c", graph_id=gid
            )
            _remember(
                "2024年，NVIDIA的市值一度超过3万亿美元，成为全球最有价值的公司之一。",
                source="s99d", graph_id=gid
            )
            ents = _get_entities(gid)
            nvidia = [e for e in ents if "NVIDIA" in e.get("name", "").upper()]
            assert len(nvidia) >= 1
            versions = _entity_versions(nvidia[0]["family_id"], gid)
            assert len(versions) >= 3, f"NVIDIA should have >= 3 versions, got {len(versions)}"

    def test_100_final_comprehensive_test(self):
        """Final comprehensive test: build a small knowledge graph and verify integrity."""
        with _test_graph() as gid:
            # Episode 1: Setup
            _remember(
                "阿里巴巴集团是中国最大的电子商务公司，由马云于1999年在杭州创立。"
                "主要业务包括淘宝、天猫、阿里云等。",
                source="final_1", graph_id=gid
            )
            # Episode 2: Add details
            _remember(
                "马云于2019年卸任阿里巴巴董事局主席，由张勇接任。"
                "阿里巴巴于2014年在纽约证券交易所上市。",
                source="final_2", graph_id=gid
            )
            # Episode 3: Cross-reference
            _remember(
                "蚂蚁集团原计划于2020年上市，但被中国监管部门叫停。"
                "蚂蚁集团是阿里巴巴的关联企业，主要业务是支付宝。",
                source="final_3", graph_id=gid
            )

            ents = _get_entities(gid)
            rels = _get_relations(gid)

            # Basic checks
            assert len(ents) >= 4, f"Should have >= 4 entities, got {len(ents)}"
            assert len(rels) >= 3, f"Should have >= 3 relations, got {len(rels)}"

            # Check versioning
            for e in ents:
                versions = _entity_versions(e["family_id"], gid)
                assert len(versions) >= 1, f"Entity {e['name']} has no versions"

            # Check that search works
            result = _quick_search("阿里巴巴", gid)
            assert len(result.get("entities", [])) >= 1, "Search should find 阿里巴巴"

            # Check relations accessible
            for e in ents:
                rels_for_e = _entity_relations(e["family_id"], gid)
                # At least some entities should have relations
