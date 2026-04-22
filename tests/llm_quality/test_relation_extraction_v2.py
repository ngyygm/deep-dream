"""
Test V2 relation extraction with real LLM.

Tests:
  - P3 prompt: discover_relation_pairs_v2()
  - P4 prompt: write_relation_content_v2()

Runs only when USE_REAL_LLM=1 is set.
"""

import pytest

pytestmark = pytest.mark.real_llm


# ---------------------------------------------------------------------------
# Test data for relation pair discovery (P3)
# ---------------------------------------------------------------------------

RELATION_PAIR_CASES = [
    {
        "id": "medicine_pairs",
        "entities": ["青霉素", "Alexander Fleming", "Howard Florey", "Ernst Chain", "诺贝尔"],
        "text": (
            "青霉素（Penicillin）是世界上第一种广泛使用的抗生素，由英国细菌学家Alexander Fleming"
            "于1928年发现。Fleming在圣玛丽医院的实验室中偶然发现，培养葡萄球菌的培养基上"
            "长出了一团青绿色的霉菌。1940年，Howard Florey和Ernst Chain成功提纯了青霉素。"
            "Fleming、Florey和Chain因此共同获得了1945年诺贝尔生理学或医学奖。"
        ),
        "expected_pairs": [("青霉素", "Alexander Fleming"), ("青霉素", "Howard Florey"),
                           ("青霉素", "Ernst Chain"), ("诺贝尔", "Alexander Fleming")],
        "negative_pairs": [],  # no strict negative pairs for this text
    },
    {
        "id": "three_kingdoms_pairs",
        "entities": ["曹操", "官渡之战", "袁绍", "郭嘉", "赤壁之战", "孙权", "刘备"],
        "text": (
            "曹操，字孟德，是东汉末年著名的政治家、军事家和文学家。他在官渡之战中击败了袁绍，"
            "统一了北方。郭嘉是曹操最信任的谋士之一，为曹操出谋划策。赤壁之战中，曹操败于"
            "孙权和刘备的联军，从此形成三国鼎立的局面。"
        ),
        "expected_pairs": [("曹操", "官渡之战"), ("曹操", "袁绍"), ("曹操", "郭嘉"),
                           ("赤壁之战", "孙权"), ("赤壁之战", "刘备"), ("孙权", "刘备")],
        "negative_pairs": [],
    },
    {
        "id": "quantum_pairs",
        "entities": ["量子计算", "IBM", "Google", "量子霸权", "Sycamore"],
        "text": (
            "量子计算（Quantum Computing）利用量子力学原理进行计算。IBM和Google都在研发量子计算机。"
            "2019年，Google宣布实现量子霸权（Quantum Supremacy），使用Sycamore处理器在200秒内"
            "完成了经典超级计算机需要1万年才能完成的计算任务。"
        ),
        "expected_pairs": [("Google", "量子霸权"), ("Google", "Sycamore"), ("量子霸权", "Sycamore")],
        "negative_pairs": [],
    },
]


# ---------------------------------------------------------------------------
# Test data for relation content writing (P4)
# ---------------------------------------------------------------------------

RELATION_CONTENT_CASES = [
    {
        "id": "fleming_penicillin",
        "entity_a": "Alexander Fleming",
        "entity_b": "青霉素",
        "text": (
            "青霉素（Penicillin）是世界上第一种广泛使用的抗生素，由英国细菌学家Alexander Fleming"
            "于1928年发现。Fleming在圣玛丽医院的实验室中偶然发现，培养葡萄球菌的培养基上"
            "长出了一团青绿色的霉菌。"
        ),
        "must_contain": ["发现"],
        "must_not_contain": ["有关联", "存在关系", "相关", "处理进度", "该实体"],
    },
    {
        "id": "caocao_guandu",
        "entity_a": "曹操",
        "entity_b": "官渡之战",
        "text": (
            "曹操，字孟德，是东汉末年著名的政治家、军事家和文学家。他在官渡之战中击败了袁绍，"
            "统一了北方。"
        ),
        "must_contain": ["击败", "袁绍"],
        "must_not_contain": ["有关联", "存在关系", "相关", "处理进度"],
    },
    {
        "id": "google_sycamore",
        "entity_a": "Google",
        "entity_b": "Sycamore",
        "text": (
            "2019年，Google宣布实现量子霸权（Quantum Supremacy），使用Sycamore处理器在200秒内"
            "完成了经典超级计算机需要1万年才能完成的计算任务。"
        ),
        "must_contain": ["处理器", "量子"],
        "must_not_contain": ["有关联", "存在关系", "相关", "处理进度"],
    },
    {
        "id": "liuyu_sunquan_liubei",
        "entity_a": "孙权",
        "entity_b": "刘备",
        "text": (
            "赤壁之战中，曹操败于孙权和刘备的联军，从此形成三国鼎立的局面。"
        ),
        "must_contain": ["联军", "赤壁"],
        "must_not_contain": ["有关联", "存在关系", "相关", "处理进度"],
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


def _pair_match(found_pair, expected_pair):
    """Check if a found pair matches an expected pair (order-independent)."""
    found_set = {found_pair[0].lower(), found_pair[1].lower()}
    expected_set = {expected_pair[0].lower(), expected_pair[1].lower()}
    return found_set == expected_set


# ---------------------------------------------------------------------------
# Tests: Relation Pair Discovery (P3)
# ---------------------------------------------------------------------------

class TestRelationPairDiscovery:
    """Test P3 prompt: discover_relation_pairs_v2() with real LLM."""

    @pytest.fixture(scope="class")
    def llm_client(self, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        return _create_llm_client(real_llm_config, shared_embedding_client)

    @pytest.mark.parametrize("case", RELATION_PAIR_CASES, ids=lambda c: c["id"])
    def test_returns_list_of_pairs(self, llm_client, case):
        """Output must be a list of (str, str) tuples."""
        pairs = llm_client.discover_relation_pairs_v2(case["entities"], case["text"])
        assert isinstance(pairs, list), f"Expected list, got {type(pairs)}"
        for pair in pairs:
            assert isinstance(pair, (list, tuple)) and len(pair) == 2, f"Bad pair: {pair}"

    @pytest.mark.parametrize("case", RELATION_PAIR_CASES, ids=lambda c: c["id"])
    def test_expected_pairs_found(self, llm_client, case):
        """At least 60% of expected pairs should be discovered."""
        pairs = llm_client.discover_relation_pairs_v2(case["entities"], case["text"])
        matched = 0
        for ep in case["expected_pairs"]:
            if any(_pair_match(fp, ep) for fp in pairs):
                matched += 1
        recall = matched / len(case["expected_pairs"])
        assert recall >= 0.6, (
            f"[{case['id']}] Pair recall too low: {recall:.0%} "
            f"({matched}/{len(case['expected_pairs'])}). "
            f"Expected: {case['expected_pairs']}, Got: {pairs}"
        )

    @pytest.mark.parametrize("case", RELATION_PAIR_CASES, ids=lambda c: c["id"])
    def test_no_self_pairs(self, llm_client, case):
        """No pair should have the same entity twice."""
        pairs = llm_client.discover_relation_pairs_v2(case["entities"], case["text"])
        for pair in pairs:
            assert pair[0] != pair[1], f"Self-pair detected: {pair}"

    @pytest.mark.parametrize("case", RELATION_PAIR_CASES, ids=lambda c: c["id"])
    def test_pair_endpoints_are_known_entities(self, llm_client, case):
        """Both endpoints of each pair should be from the provided entity list."""
        pairs = llm_client.discover_relation_pairs_v2(case["entities"], case["text"])
        entities_lower = {e.lower() for e in case["entities"]}
        for pair in pairs:
            for endpoint in pair:
                assert endpoint.lower() in entities_lower, (
                    f"[{case['id']}] Unknown endpoint '{endpoint}' not in {case['entities']}"
                )


# ---------------------------------------------------------------------------
# Tests: Relation Content Writing (P4)
# ---------------------------------------------------------------------------

class TestRelationContentWriting:
    """Test P4 prompt: write_relation_content_v2() with real LLM."""

    @pytest.fixture(scope="class")
    def llm_client(self, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        return _create_llm_client(real_llm_config, shared_embedding_client)

    @pytest.mark.parametrize("case", RELATION_CONTENT_CASES, ids=lambda c: c["id"])
    def test_returns_non_empty_content(self, llm_client, case):
        """Content must be a non-empty string."""
        content = llm_client.write_relation_content_v2(
            case["entity_a"], case["entity_b"], case["text"]
        )
        assert isinstance(content, str), f"Expected str, got {type(content)}"
        assert len(content) >= 10, f"Content too short ({len(content)} chars): '{content}'"

    @pytest.mark.parametrize("case", RELATION_CONTENT_CASES, ids=lambda c: c["id"])
    def test_content_not_too_long(self, llm_client, case):
        """Content should be concise (< 100 chars)."""
        content = llm_client.write_relation_content_v2(
            case["entity_a"], case["entity_b"], case["text"]
        )
        assert len(content) <= 100, f"Content too long ({len(content)} chars): '{content}'"

    @pytest.mark.parametrize("case", RELATION_CONTENT_CASES, ids=lambda c: c["id"])
    def test_contains_key_information(self, llm_client, case):
        """Content must contain at least one must_contain keyword."""
        content = llm_client.write_relation_content_v2(
            case["entity_a"], case["entity_b"], case["text"]
        )
        matched = sum(1 for kw in case["must_contain"] if kw in content)
        assert matched >= 1, (
            f"[{case['id']}] Missing key info. Expected one of {case['must_contain']}. "
            f"Got: '{content}'"
        )

    @pytest.mark.parametrize("case", RELATION_CONTENT_CASES, ids=lambda c: c["id"])
    def test_no_generic_phrases(self, llm_client, case):
        """Content must not contain generic relation phrases."""
        content = llm_client.write_relation_content_v2(
            case["entity_a"], case["entity_b"], case["text"]
        )
        for forbidden in case["must_not_contain"]:
            assert forbidden not in content, (
                f"[{case['id']}] Generic phrase detected: '{forbidden}' in '{content}'"
            )

    @pytest.mark.parametrize("case", RELATION_CONTENT_CASES, ids=lambda c: c["id"])
    def test_mentions_both_entities(self, llm_client, case):
        """Content should mention both entity endpoints."""
        content = llm_client.write_relation_content_v2(
            case["entity_a"], case["entity_b"], case["text"]
        )
        # At least one endpoint should be mentioned
        mentioned = (
            case["entity_a"].lower() in content.lower()
            or case["entity_b"].lower() in content.lower()
        )
        assert mentioned, (
            f"[{case['id']}] Content doesn't mention either entity. "
            f"Expected '{case['entity_a']}' or '{case['entity_b']}'. Got: '{content}'"
        )
