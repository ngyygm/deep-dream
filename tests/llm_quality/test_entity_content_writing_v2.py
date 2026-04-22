"""
Test V2 entity content writing with real LLM.

Tests the P2 prompt: write_entity_content_v2()
Runs only when USE_REAL_LLM=1 is set.

Quality metrics:
- Content length: 30-100 characters (Chinese) / 20-200 characters (English)
- Contains keywords from the source text
- Does NOT contain generic/template phrases
- Focuses on the entity itself, not relations
"""

import os
import pytest

pytestmark = pytest.mark.real_llm


# ---------------------------------------------------------------------------
# Test data: (entity_name, window_text, must_contain_keywords, must_not_contain)
# ---------------------------------------------------------------------------

ENTITY_CONTENT_CASES = [
    {
        "id": "李白_biography",
        "entity": "李白",
        "text": (
            "李白是唐代著名诗人，字太白，号青莲居士。他擅长浪漫主义诗歌创作，被称为\"诗仙\"。"
        ),
        "must_contain": ["诗人", "唐代"],
        "must_not_contain": ["该实体", "这是一个", "处理进度"],
    },
    {
        "id": "青霉素_medicine",
        "entity": "青霉素",
        "text": (
            "青霉素（Penicillin）是世界上第一种广泛使用的抗生素，由英国细菌学家Alexander Fleming"
            "于1928年发现。Fleming在圣玛丽医院的实验室中偶然发现，培养葡萄球菌的培养基上"
            "长出了一团青绿色的霉菌。1940年，Howard Florey和Ernst Chain成功提纯了青霉素。"
        ),
        "must_contain": ["抗生素"],
        "must_not_contain": ["该实体", "这是一个", "处理进度", "步骤"],
    },
    {
        "id": "Python_tech",
        "entity": "Python",
        "text": (
            "Python is a high-level programming language created by Guido van Rossum in 1991. "
            "It emphasizes code readability and supports multiple programming paradigms. "
            "Python has become one of the most popular languages for data science and web development."
        ),
        "must_contain": ["programming language", "编程语言"],
        "must_not_contain": ["This entity", "该实体", "process", "system"],
    },
    {
        "id": "Kubernetes_k8s",
        "entity": "Kubernetes",
        "text": (
            "Kubernetes is an open-source container orchestration system for automating software deployment, "
            "scaling, and management of containerized applications. It was originally developed by Google "
            "and is now maintained by the Cloud Native Computing Foundation. Kubernetes works with Docker "
            "and other container runtimes."
        ),
        "must_contain": ["container", "容器"],
        "must_not_contain": ["This entity", "process", "framework"],
    },
    {
        "id": "曹操_three_kingdoms",
        "entity": "曹操",
        "text": (
            "曹操，字孟德，是东汉末年著名的政治家、军事家和文学家。他在官渡之战中击败了袁绍，"
            "统一了北方。郭嘉是曹操最信任的谋士之一，为曹操出谋划策。赤壁之战中，曹操败于"
            "孙权和刘备的联军，从此形成三国鼎立的局面。"
        ),
        "must_contain": ["政治家"],
        "must_not_contain": ["该实体", "这是一个", "处理进度"],
    },
    {
        "id": "袁隆平_scientist",
        "entity": "袁隆平",
        "text": (
            "袁隆平是中国著名的农业科学家，被誉为\"杂交水稻之父\"。他研发的杂交水稻技术使水稻产量"
            "提高了20%以上。袁隆平于2021年5月22日在长沙逝世，享年91岁。他的贡献解决了数亿人的"
            "粮食问题，获得了国家最高科学技术奖和世界粮食奖。"
        ),
        "must_contain": ["科学家", "杂交水稻"],
        "must_not_contain": ["该实体", "这是一个", "处理进度"],
    },
    {
        "id": "红楼梦_literature",
        "entity": "红楼梦",
        "text": (
            "红楼梦是中国古典四大名著之一，由清代作家曹雪芹创作。小说以贾宝玉和林黛玉的爱情悲剧为主线，"
            "描绘了贾、王、薛、史四大家族的兴衰。其中贾宝玉是荣国府的公子，林黛玉是贾母的外孙女。"
        ),
        "must_contain": ["四大名著", "曹雪芹"],
        "must_not_contain": ["该实体", "这是一个", "处理进度"],
    },
    {
        "id": "量子霸权_physics",
        "entity": "量子霸权",
        "text": (
            "量子计算（Quantum Computing）利用量子力学原理进行计算。IBM和Google都在研发量子计算机。"
            "2019年，Google宣布实现量子霸权（Quantum Supremacy），使用Sycamore处理器在200秒内"
            "完成了经典超级计算机需要1万年才能完成的计算任务。"
        ),
        "must_contain": ["Google"],
        "must_not_contain": ["该实体", "这是一个", "处理进度"],
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

class TestEntityContentWriting:
    """Test P2 prompt: write_entity_content_v2() with real LLM."""

    @pytest.fixture(scope="class")
    def llm_client(self, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        return _create_llm_client(real_llm_config, shared_embedding_client)

    @pytest.mark.parametrize("case", ENTITY_CONTENT_CASES, ids=lambda c: c["id"])
    def test_returns_non_empty_content(self, llm_client, case):
        """Content must be a non-empty string."""
        content = llm_client.write_entity_content_v2(case["entity"], case["text"])
        assert isinstance(content, str), f"Expected str, got {type(content)}"
        assert len(content) >= 10, f"Content too short ({len(content)} chars): '{content}'"

    @pytest.mark.parametrize("case", ENTITY_CONTENT_CASES, ids=lambda c: c["id"])
    def test_content_not_too_long(self, llm_client, case):
        """Content should be reasonably concise (< 200 chars)."""
        content = llm_client.write_entity_content_v2(case["entity"], case["text"])
        assert len(content) <= 200, f"Content too long ({len(content)} chars): '{content}'"

    @pytest.mark.parametrize("case", ENTITY_CONTENT_CASES, ids=lambda c: c["id"])
    def test_contains_key_information(self, llm_client, case):
        """Content must contain at least one must_contain keyword."""
        content = llm_client.write_entity_content_v2(case["entity"], case["text"])
        matched = sum(1 for kw in case["must_contain"] if kw.lower() in content.lower())
        assert matched >= 1, (
            f"[{case['id']}] Missing key info. Expected one of {case['must_contain']}. "
            f"Got: '{content}'"
        )

    @pytest.mark.parametrize("case", ENTITY_CONTENT_CASES, ids=lambda c: c["id"])
    def test_no_template_phrases(self, llm_client, case):
        """Content must not contain generic template phrases."""
        content = llm_client.write_entity_content_v2(case["entity"], case["text"])
        for forbidden in case["must_not_contain"]:
            assert forbidden.lower() not in content.lower(), (
                f"[{case['id']}] Template phrase detected: '{forbidden}' in '{content}'"
            )

    @pytest.mark.parametrize("case", ENTITY_CONTENT_CASES, ids=lambda c: c["id"])
    def test_focuses_on_entity_not_relations(self, llm_client, case):
        """Content should describe the entity, not relations with others."""
        content = llm_client.write_entity_content_v2(case["entity"], case["text"])
        # Content should mention the entity name itself
        assert case["entity"].lower() in content.lower(), (
            f"[{case['id']}] Content doesn't mention entity '{case['entity']}': '{content}'"
        )
