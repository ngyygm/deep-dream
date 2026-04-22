"""
Tests for entity name validation and relation entity handling in LLMClient:
_build_relation_entity_catalog, _normalize_entity_pair,
_normalize_and_filter_relations_by_entities, _parse_relations_response.
"""

import json

import pytest

from processor.llm.client import LLMClient

# 与 service_config.llm.context_window_tokens / server.config.DEFAULTS 对齐
_TEST_CONTEXT_WINDOW_TOKENS = 8000


def _llm_client(**kwargs):
    kwargs.setdefault("context_window_tokens", _TEST_CONTEXT_WINDOW_TOKENS)
    return LLMClient(**kwargs)


class TestRelationEntityCatalog:
    """_build_relation_entity_catalog 与 relation_content_snippet_length 配置。"""

    def test_name_only_when_relation_snippet_length_zero(self):
        client = _llm_client(relation_content_snippet_length=0)
        s, valid, order = client._build_relation_entity_catalog(
            [
                {"name": "A", "content": "should_not_appear_in_catalog"},
                {"name": "B", "content": ""},
            ],
        )
        assert "should_not_appear" not in s
        assert "|" not in s
        assert s.strip() == "- A\n- B"
        assert valid == {"A", "B"}
        assert order == ["A", "B"]


class TestNormalizeEntityPair:
    """Tests for _normalize_entity_pair static method."""

    def test_ordering_alphabetical(self):
        result = LLMClient._normalize_entity_pair("Alice", "Bob")
        assert result == ("Alice", "Bob")

    def test_ordering_reversed(self):
        result = LLMClient._normalize_entity_pair("Bob", "Alice")
        assert result == ("Alice", "Bob")

    def test_same_entity(self):
        result = LLMClient._normalize_entity_pair("Charlie", "Charlie")
        assert result == ("Charlie", "Charlie")

    def test_strips_whitespace(self):
        result = LLMClient._normalize_entity_pair("  Alice  ", "  Bob  ")
        assert result == ("Alice", "Bob")

    def test_chinese_entity_names(self):
        result = LLMClient._normalize_entity_pair("张三", "李四")
        # Chinese characters sort by Unicode code point; 张 < 李 is determined
        # by their code points. The important thing is consistency.
        e1, e2 = result
        assert e1 <= e2

    def test_consistent_regardless_of_input_order(self):
        pair1 = LLMClient._normalize_entity_pair("X", "Y")
        pair2 = LLMClient._normalize_entity_pair("Y", "X")
        assert pair1 == pair2


class TestRelationEntityNameFiltering:
    def setup_method(self):
        self.client = _llm_client()

    def test_keeps_unknown_endpoints_for_step4_supplement(self):
        relations = [
            {
                "entity1_name": "报复",
                "entity2_name": "贾瑞（小说人物）",
                "content": "报复与贾瑞有关",
            },
            {
                "entity1_name": "贾瑞（小说人物）",
                "entity2_name": "王熙凤（小说人物）",
                "content": "二人存在互动",
            },
        ]

        filtered, normalized_count, filtered_count = self.client._normalize_and_filter_relations_by_entities(
            relations,
            {"贾瑞（小说人物）", "王熙凤（小说人物）"},
        )

        assert normalized_count == 0
        assert filtered_count == 0
        assert len(filtered) == 2
        revenge_edges = [
            r for r in filtered
            if "报复" in (r["entity1_name"], r["entity2_name"])
        ]
        assert len(revenge_edges) == 1
        assert "贾瑞（小说人物）" in (
            revenge_edges[0]["entity1_name"],
            revenge_edges[0]["entity2_name"],
        )

    def test_normalizes_unique_base_name(self):
        relations = [
            {
                "entity1_name": "贾瑞",
                "entity2_name": "王熙凤（小说人物）",
                "content": "二人存在互动",
            }
        ]

        filtered, normalized_count, filtered_count = self.client._normalize_and_filter_relations_by_entities(
            relations,
            {"贾瑞（小说人物）", "王熙凤（小说人物）"},
        )

        assert normalized_count == 1
        assert filtered_count == 0
        assert len(filtered) == 1
        assert filtered[0]["entity1_name"] == "贾瑞（小说人物）"
        assert filtered[0]["entity2_name"] == "王熙凤（小说人物）"

    def test_ambiguous_base_resolves_to_latest_bracketed_in_catalog_order(self):
        """多個「贾瑞（…）」去括号后 Jaccard 并列时，按 catalog 顺序取较后者。"""
        relations = [
            {
                "entity1_name": "贾瑞",
                "entity2_name": "王熙凤（小说人物）",
                "content": "二人存在互动",
            }
        ]

        catalog = {"贾瑞（小说人物）", "贾瑞（另一版本）", "王熙凤（小说人物）"}
        order = ["贾瑞（小说人物）", "王熙凤（小说人物）", "贾瑞（另一版本）"]
        filtered, normalized_count, filtered_count = self.client._normalize_and_filter_relations_by_entities(
            relations,
            catalog,
            catalog_name_order=order,
        )

        assert normalized_count == 1
        assert filtered_count == 0
        assert len(filtered) == 1
        assert filtered[0]["entity1_name"] == "贾瑞（另一版本）"
        assert filtered[0]["entity2_name"] == "王熙凤（小说人物）"


class TestParseRelationsResponse:
    def setup_method(self):
        self.client = _llm_client()

    def test_accepts_name_fields_maps_to_catalog(self):
        raw = json.dumps([
            {
                "entity1_name": "贾宝玉（小说人物）",
                "entity2_name": "林黛玉（小说人物）",
                "content": "二人有明确互动",
            }
        ], ensure_ascii=False)

        parsed = self.client._parse_relations_response(
            raw,
            {"贾宝玉（小说人物）", "林黛玉（小说人物）"},
        )

        assert len(parsed) == 1
        assert {parsed[0]["entity1_name"], parsed[0]["entity2_name"]} == {"贾宝玉（小说人物）", "林黛玉（小说人物）"}

    def test_accepts_unknown_second_endpoint_for_supplement(self):
        raw = json.dumps([
            {
                "entity1_name": "贾宝玉（小说人物）",
                "entity2_name": "薛宝钗（小说人物）",
                "content": "二人有明确互动",
            }
        ], ensure_ascii=False)

        parsed = self.client._parse_relations_response(
            raw,
            {"贾宝玉（小说人物）", "林黛玉（小说人物）"},
        )

        assert len(parsed) == 1
        assert {parsed[0]["entity1_name"], parsed[0]["entity2_name"]} == {
            "贾宝玉（小说人物）",
            "薛宝钗（小说人物）",
        }

    def test_ambiguous_partial_name_keeps_raw_below_jaccard_threshold(self):
        """Jaccard 未达阈值时不强行映射；保留简称供步骤4 补全，不再整条丢弃。"""
        raw = json.dumps([
            {
                "entity1_name": "真人",
                "entity2_name": "太尉",
                "content": "二者有明确互动",
            }
        ], ensure_ascii=False)

        parsed = self.client._parse_relations_response(
            raw,
            {"罗真人", "张真人", "洪太尉"},
        )

        assert len(parsed) == 1
        assert {parsed[0]["entity1_name"], parsed[0]["entity2_name"]} == {"太尉", "真人"}

    def test_accepts_name_only_relation_and_maps_to_closest_entity(self):
        raw = json.dumps([
            {
                "entity1_name": "真人",
                "entity2_name": "太尉",
                "content": "二者有明确互动",
            }
        ], ensure_ascii=False)

        # 默认阈值 0.9 时「真人/太尉」与全名 Jaccard 约 2/3；降低阈值以断言映射到目录全名
        client = _llm_client(relation_endpoint_jaccard_threshold=0.65)
        parsed = client._parse_relations_response(
            raw,
            {"罗真人", "洪太尉"},
        )

        assert len(parsed) == 1
        assert {parsed[0]["entity1_name"], parsed[0]["entity2_name"]} == {"罗真人", "洪太尉"}

    def test_jaccard_uses_stripped_names_maps_to_bracketed_catalog(self):
        """比较时去掉括号；目录仅有带说明全名时，简称应对齐到该全名。"""
        raw = json.dumps([
            {
                "entity1_name": "贾宝玉",
                "entity2_name": "林黛玉",
                "content": "二人有明确互动",
            }
        ], ensure_ascii=False)

        catalog = {"贾宝玉（小说人物）", "林黛玉（小说人物）"}
        order = ["贾宝玉（小说人物）", "林黛玉（小说人物）"]
        parsed = self.client._parse_relations_response(
            raw, catalog, catalog_name_order=order
        )

        assert len(parsed) == 1
        assert {parsed[0]["entity1_name"], parsed[0]["entity2_name"]} == catalog

    def test_tie_break_bracketed_prefers_later_catalog_name(self):
        """同分且均带括号时，取 catalog_name_order 中较后者（新版本）。"""
        raw = json.dumps([
            {
                "entity1_name": "贾宝玉",
                "entity2_name": "薛宝钗",
                "content": "互动",
            }
        ], ensure_ascii=False)

        catalog = {"贾宝玉（旧）", "贾宝玉（新）", "薛宝钗"}
        order = ["薛宝钗", "贾宝玉（旧）", "贾宝玉（新）"]
        parsed = self.client._parse_relations_response(
            raw, catalog, catalog_name_order=order
        )

        assert len(parsed) == 1
        names = {parsed[0]["entity1_name"], parsed[0]["entity2_name"]}
        assert "贾宝玉（新）" in names
        assert "薛宝钗" in names
