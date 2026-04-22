"""
Test V2 entity alignment judgment with real LLM.

Tests the P5 prompt: judge_entity_same_v2()
Runs only when USE_REAL_LLM=1 is set.

Quality metrics:
- Same-entity pairs should be judged as same (true)
- Different-entity pairs should be judged as different (false)
- Borderline cases: related but distinct entities should be different
"""

import pytest

pytestmark = pytest.mark.real_llm


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SAME_ENTITY_CASES = [
    {
        "id": "same_person_bilingual",
        "name_a": "Alexander Fleming",
        "content_a": "英国细菌学家，1928年发现了青霉素。",
        "name_b": "Fleming",
        "content_b": "在圣玛丽医院实验室中偶然发现了青霉菌的抑菌作用。",
        "expected_same": True,
    },
    {
        "id": "same_person_fullname",
        "name_a": "袁隆平",
        "content_a": "中国农业科学家，被誉为杂交水稻之父。",
        "name_b": "袁隆平",
        "content_b": "研发了杂交水稻技术，获得国家最高科学技术奖。",
        "expected_same": True,
    },
    {
        "id": "same_concept_bilingual",
        "name_a": "量子霸权",
        "content_a": "2019年Google使用Sycamore处理器实现的量子计算里程碑。",
        "name_b": "Quantum Supremacy",
        "content_b": "指量子计算机在特定任务上超越经典计算机的能力。",
        "expected_same": True,
    },
    {
        "id": "same_place_abbrev",
        "name_a": "北京",
        "content_a": "中国的首都，位于华北平原。",
        "name_b": "北京市",
        "content_b": "中华人民共和国直辖市，政治文化中心。",
        "expected_same": True,
    },
]

DIFFERENT_ENTITY_CASES = [
    {
        "id": "person_vs_work",
        "name_a": "曹雪芹",
        "content_a": "清代作家，创作了中国古典四大名著之一的红楼梦。",
        "name_b": "红楼梦",
        "content_b": "中国古典四大名著之一，以贾宝玉和林黛玉的爱情悲剧为主线。",
        "expected_same": False,
    },
    {
        "id": "person_vs_event",
        "name_a": "曹操",
        "content_a": "东汉末年政治家、军事家，字孟德。",
        "name_b": "官渡之战",
        "content_b": "曹操在此战中击败了袁绍，统一了北方。",
        "expected_same": False,
    },
    {
        "id": "city_vs_country",
        "name_a": "长沙",
        "content_a": "湖南省省会城市，袁隆平逝世于此。",
        "name_b": "湖南",
        "content_b": "中国中南部省份，省会为长沙。",
        "expected_same": False,
    },
    {
        "id": "concept_vs_instance",
        "name_a": "量子计算",
        "content_a": "利用量子力学原理进行计算的技术领域。",
        "name_b": "Sycamore",
        "content_b": "Google研发的量子处理器。",
        "expected_same": False,
    },
    {
        "id": "similar_different",
        "name_a": "Howard Florey",
        "content_a": "1940年成功提纯了青霉素的科学家之一。",
        "name_b": "Ernst Chain",
        "content_b": "与Florey一起提纯青霉素的科学家。",
        "expected_same": False,
    },
    {
        "id": "org_vs_person",
        "name_a": "Google",
        "content_a": "美国科技公司，开发了Kubernetes和量子计算技术。",
        "name_b": "Guido van Rossum",
        "content_b": "Python编程语言的创建者。",
        "expected_same": False,
    },
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _create_llm_client(real_llm_config, shared_embedding_client):
    """Create an LLMClient with real config for testing."""
    from processor.llm.client import LLMClient
    kwargs = {
        "api_key": real_llm_config.get("llm_api_key"),
        "model_name": real_llm_config.get("llm_model"),
        "base_url": real_llm_config.get("llm_base_url"),
        "think_mode": real_llm_config.get("llm_think_mode", False),
        "max_tokens": real_llm_config.get("llm_max_tokens"),
        "context_window_tokens": real_llm_config.get("llm_context_window_tokens"),
        "max_llm_concurrency": 2,
        "embedding_client": shared_embedding_client,
        "content_snippet_length": 100,
        "relation_content_snippet_length": 0,
    }
    return LLMClient(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEntityAlignment:
    """Test P5 prompt: judge_entity_same_v2() with real LLM."""

    @pytest.fixture(scope="class")
    def llm_client(self, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        return _create_llm_client(real_llm_config, shared_embedding_client)

    @pytest.mark.parametrize("case", SAME_ENTITY_CASES, ids=lambda c: c["id"])
    def test_same_entities_detected(self, llm_client, case):
        """Same entities should be judged as same (True)."""
        result = llm_client.judge_entity_same_v2(
            case["name_a"], case["content_a"],
            case["name_b"], case["content_b"],
        )
        assert result is True, (
            f"[{case['id']}] Expected True for same entities: "
            f"'{case['name_a']}' vs '{case['name_b']}'"
        )

    @pytest.mark.parametrize("case", DIFFERENT_ENTITY_CASES, ids=lambda c: c["id"])
    def test_different_entities_rejected(self, llm_client, case):
        """Different entities should be judged as different (False)."""
        result = llm_client.judge_entity_same_v2(
            case["name_a"], case["content_a"],
            case["name_b"], case["content_b"],
        )
        assert result is False, (
            f"[{case['id']}] Expected False for different entities: "
            f"'{case['name_a']}' vs '{case['name_b']}'"
        )
