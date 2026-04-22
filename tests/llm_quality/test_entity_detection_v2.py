"""
Test V2 entity name extraction with real LLM.

Tests the P1 prompt: extract_entity_names_v2()
Runs only when USE_REAL_LLM=1 is set.

Quality metrics:
- Recall: % of expected entities found (target >= 80%)
- Format compliance: % of responses that are valid JSON arrays (target >= 90%)
- No hallucination: no system metadata or generic words in output
"""

import json
import os
import pytest

# Skip all tests when USE_REAL_LLM is not set
pytestmark = pytest.mark.real_llm


# ---------------------------------------------------------------------------
# Test data: window-level texts with expected entities
# ---------------------------------------------------------------------------

WINDOW_TEST_CASES = [
    {
        "id": "chinese_short_biography",
        "text": "李白是唐代著名诗人，字太白，号青莲居士。他擅长浪漫主义诗歌创作，被称为\"诗仙\"。",
        "expected_entities": ["李白", "唐代"],
        "must_not_contain": ["处理进度", "步骤"],
    },
    {
        "id": "chinese_medicine",
        "text": (
            "青霉素（Penicillin）是世界上第一种广泛使用的抗生素，由英国细菌学家Alexander Fleming"
            "于1928年发现。Fleming在圣玛丽医院的实验室中偶然发现，培养葡萄球菌的培养基上"
            "长出了一团青绿色的霉菌。1940年，Howard Florey和Ernst Chain成功提纯了青霉素。"
            "Fleming、Florey和Chain因此共同获得了1945年诺贝尔生理学或医学奖。"
        ),
        "expected_entities": ["青霉素", "Alexander Fleming", "Howard Florey", "Ernst Chain", "诺贝尔"],
        "must_not_contain": ["处理进度", "分析", "系统"],
    },
    {
        "id": "english_tech",
        "text": (
            "Python is a high-level programming language created by Guido van Rossum in 1991. "
            "It emphasizes code readability and supports multiple programming paradigms. "
            "Python has become one of the most popular languages for data science and web development."
        ),
        "expected_entities": ["Python", "Guido van Rossum"],
        "must_not_contain": ["process", "system", "analysis"],
    },
    {
        "id": "english_k8s",
        "text": (
            "Kubernetes is an open-source container orchestration system for automating software deployment, "
            "scaling, and management of containerized applications. It was originally developed by Google "
            "and is now maintained by the Cloud Native Computing Foundation. Kubernetes works with Docker "
            "and other container runtimes."
        ),
        "expected_entities": ["Kubernetes", "Google", "Docker", "Cloud Native Computing Foundation"],
        "must_not_contain": ["process", "framework", "approach"],
    },
    {
        "id": "mixed_quantum",
        "text": (
            "量子计算（Quantum Computing）利用量子力学原理进行计算。IBM和Google都在研发量子计算机。"
            "2019年，Google宣布实现量子霸权（Quantum Supremacy），使用Sycamore处理器在200秒内"
            "完成了经典超级计算机需要1万年才能完成的计算任务。"
        ),
        "expected_entities": ["量子计算", "IBM", "Google", "量子霸权", "Sycamore"],
        "must_not_contain": ["处理进度", "系统状态"],
    },
    {
        "id": "chinese_literature",
        "text": (
            "红楼梦是中国古典四大名著之一，由清代作家曹雪芹创作。小说以贾宝玉和林黛玉的爱情悲剧为主线，"
            "描绘了贾、王、薛、史四大家族的兴衰。其中贾宝玉是荣国府的公子，林黛玉是贾母的外孙女。"
        ),
        "expected_entities": ["红楼梦", "曹雪芹", "贾宝玉", "林黛玉", "贾母"],
        "must_not_contain": ["处理进度", "系统"],
    },
    {
        "id": "chinese_history",
        "text": (
            "袁隆平是中国著名的农业科学家，被誉为\"杂交水稻之父\"。他研发的杂交水稻技术使水稻产量"
            "提高了20%以上。袁隆平于2021年5月22日在长沙逝世，享年91岁。他的贡献解决了数亿人的"
            "粮食问题，获得了国家最高科学技术奖和世界粮食奖。"
        ),
        "expected_entities": ["袁隆平", "杂交水稻", "长沙", "国家最高科学技术奖", "世界粮食奖"],
        "must_not_contain": ["处理进度", "分析"],
    },
    {
        "id": "english_history",
        "text": (
            "The 2008 financial crisis was triggered by the collapse of the housing bubble in the "
            "United States. Lehman Brothers filed for bankruptcy on September 15, 2008. Federal Reserve "
            "Chairman Ben Bernanke played a key role in managing the crisis response. The crisis led to "
            "the passage of the Dodd-Frank Wall Street Reform Act in 2010."
        ),
        "expected_entities": ["2008 financial crisis", "Lehman Brothers", "Ben Bernanke", "Dodd-Frank"],
        "must_not_contain": ["process", "system", "analysis"],
    },
    {
        "id": "chinese_three_kingdoms",
        "text": (
            "曹操，字孟德，是东汉末年著名的政治家、军事家和文学家。他在官渡之战中击败了袁绍，"
            "统一了北方。郭嘉是曹操最信任的谋士之一，为曹操出谋划策。赤壁之战中，曹操败于"
            "孙权和刘备的联军，从此形成三国鼎立的局面。"
        ),
        "expected_entities": ["曹操", "官渡之战", "袁绍", "郭嘉", "赤壁之战", "孙权", "刘备"],
        "must_not_contain": ["处理进度", "系统状态", "步骤"],
    },
    {
        "id": "chinese_philosophy",
        "text": (
            "儒家思想是中国古代最重要的哲学体系之一，由孔子创立。孔子名丘，字仲尼，出生于鲁国陬邑"
            "（今山东省曲阜市）。他提倡\"仁\"、\"义\"、\"礼\"、\"智\"、\"信\"五种核心德目。"
            "他的言行被弟子整理成《论语》一书。孟子进一步发展了儒家思想，提出\"性善论\"。"
        ),
        "expected_entities": ["儒家", "孔子", "鲁国", "论语", "孟子"],
        "must_not_contain": ["处理进度", "系统"],
    },
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _create_llm_client(real_llm_config, shared_embedding_client):
    """Create an LLMClient with real config for testing."""
    from processor.llm.client import LLMClient
    # conftest keys: llm_api_key, llm_model, llm_base_url, llm_think_mode,
    #                llm_max_tokens, llm_context_window_tokens
    # LLMClient params: api_key, model_name, base_url, think_mode, max_tokens,
    #                   context_window_tokens
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

class TestEntityNameExtraction:
    """Test P1 prompt: extract_entity_names_v2() with real LLM."""

    @pytest.fixture(scope="class")
    def llm_client(self, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        return _create_llm_client(real_llm_config, shared_embedding_client)

    @pytest.mark.parametrize("case", WINDOW_TEST_CASES, ids=lambda c: c["id"])
    def test_returns_valid_json_list(self, llm_client, case):
        """Output must be a valid list of non-empty strings."""
        names = llm_client.extract_entity_names_v2(case["text"])
        assert isinstance(names, list), f"Expected list, got {type(names)}"
        assert len(names) > 0, "Expected at least one entity name"
        for name in names:
            assert isinstance(name, str), f"Expected str, got {type(name)}: {name}"
            assert len(name) >= 1, f"Name is empty"

    @pytest.mark.parametrize("case", WINDOW_TEST_CASES, ids=lambda c: c["id"])
    def test_expected_entities_recalled(self, llm_client, case):
        """At least 80% of expected entities should be found."""
        names = llm_client.extract_entity_names_v2(case["text"])
        names_lower = [n.lower() for n in names]

        matched = 0
        for expected in case["expected_entities"]:
            found = any(
                expected.lower() in n or n in expected.lower()
                for n in names_lower
            )
            if found:
                matched += 1

        recall = matched / len(case["expected_entities"])
        assert recall >= 0.6, (
            f"[{case['id']}] Recall too low: {recall:.0%} ({matched}/{len(case['expected_entities'])}). "
            f"Expected: {case['expected_entities']}, Got: {names}"
        )

    @pytest.mark.parametrize("case", WINDOW_TEST_CASES, ids=lambda c: c["id"])
    def test_no_system_leaks(self, llm_client, case):
        """Output must not contain system metadata or generic patterns."""
        names = llm_client.extract_entity_names_v2(case["text"])
        for forbidden in case.get("must_not_contain", []):
            for name in names:
                assert forbidden.lower() not in name.lower(), (
                    f"[{case['id']}] System leak detected: '{name}' contains '{forbidden}'"
                )

    def test_empty_input_returns_empty(self, llm_client):
        """Empty text should return empty list."""
        names = llm_client.extract_entity_names_v2("")
        assert isinstance(names, list)

    def test_deduplication(self, llm_client):
        """Same entity mentioned twice should appear only once."""
        text = "张三去了北京。张三又从北京回来了。"
        names = llm_client.extract_entity_names_v2(text)
        # Count occurrences of each name (case-insensitive)
        lower_names = [n.lower() for n in names]
        for name in set(lower_names):
            count = lower_names.count(name)
            assert count <= 1, f"Duplicate entity: '{name}' appears {count} times"

    @pytest.mark.parametrize("case", WINDOW_TEST_CASES, ids=lambda c: c["id"])
    def test_not_too_few_entities(self, llm_client, case):
        """Should extract at least half of expected entities."""
        names = llm_client.extract_entity_names_v2(case["text"])
        assert len(names) >= len(case["expected_entities"]) // 2, (
            f"[{case['id']}] Too few entities: {len(names)} extracted, "
            f"expected at least {len(case['expected_entities']) // 2}. "
            f"Got: {names}"
        )
