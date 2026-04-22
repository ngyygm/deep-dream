"""
V2 remember pipeline integration tests with real LLM.

Tests the complete remember_text() flow with remember_mode="v2":
  Steps 1-8: Extraction
  Steps 9-10: Alignment
  Storage: Entities and relations persisted correctly

Quality metrics:
- Entities stored match extraction results
- No false positive merges for clearly different entities
- Multi-remember dedup: second remember of related text merges same entities
- Entity content is meaningful (not generic template text)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# Allow `from conftest import ...` to find tests/llm_quality/conftest.py
sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.real_llm

from tests.fixtures.extraction_gold import EXTRACTION_GOLD
from conftest import _make_v2_processor


class TestV2RememberIntegration:
    """Full V2 pipeline integration tests."""

    @pytest.fixture(scope="class")
    def processor(self, tmp_path_factory, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        tmp = tmp_path_factory.mktemp("v2_integration")
        return _make_v2_processor(tmp, real_llm_config, shared_embedding_client)

    def test_single_remember_stores_entities(self, processor):
        """Single remember_text call should store entities in the graph."""
        text = EXTRACTION_GOLD["biography_mayun"]["text"]
        processor.remember_text(text, doc_name="test_bio", verbose=False, verbose_steps=False)

        entities = processor.storage.get_all_entities()
        assert len(entities) >= 3, f"Expected >= 3 entities, got {len(entities)}"

        # Check that key entities are present
        entity_names = {e.name for e in entities}
        assert any("马云" in n for n in entity_names), f"Missing 马云. Got: {entity_names}"
        assert any("阿里巴巴" in n for n in entity_names), f"Missing 阿里巴巴. Got: {entity_names}"

    def test_single_remember_stores_relations(self, processor):
        """Single remember_text should store relations."""
        text = EXTRACTION_GOLD["biography_mayun"]["text"]
        processor.remember_text(text, doc_name="test_bio_rel", verbose=False, verbose_steps=False)

        relations = processor.storage.get_all_relations()
        assert len(relations) >= 1, f"Expected >= 1 relation, got {len(relations)}"

    def test_entity_content_quality(self, processor):
        """Stored entity content should be meaningful (not generic template)."""
        text = EXTRACTION_GOLD["history_tang"]["text"]
        processor.remember_text(text, doc_name="test_tang_quality", verbose=False, verbose_steps=False)

        entities = processor.storage.get_all_entities()
        generic_patterns = ["该实体", "这是一个", "这是一"]
        for e in entities:
            content = e.content or ""
            assert len(content) >= 15, (
                f"Entity '{e.name}' content too short ({len(content)} chars)"
            )
            for pattern in generic_patterns:
                assert pattern not in content, (
                    f"Entity '{e.name}' has generic content: '{content[:80]}'"
                )

    def test_not_all_entities_orphaned(self, processor):
        """Most entities should be connected by at least one relation."""
        text = EXTRACTION_GOLD["business_tesla"]["text"]
        processor.remember_text(text, doc_name="test_no_orphans", verbose=False, verbose_steps=False)

        entities = processor.storage.get_all_entities()
        relations = processor.storage.get_all_relations()

        if len(entities) <= 1:
            pytest.skip("Not enough entities to check connectivity")

        # Get entity IDs that appear in at least one relation
        connected_ids = set()
        for r in relations:
            connected_ids.add(r.entity1_absolute_id)
            connected_ids.add(r.entity2_absolute_id)

        # Count orphans
        orphan_count = sum(1 for e in entities if e.absolute_id not in connected_ids)
        # Allow some orphans but not all
        assert orphan_count < len(entities), (
            f"All {len(entities)} entities are orphans (no relations)"
        )


class TestV2MultiRemember:
    """Test cross-remember deduplication."""

    def test_second_remember_merges_same_entities(self, tmp_path, real_llm_config, shared_embedding_client):
        """Two remember calls with overlapping entities should merge, not duplicate."""
        if not real_llm_config:
            pytest.skip("No LLM config")
        proc = _make_v2_processor(tmp_path, real_llm_config, shared_embedding_client)

        # First remember
        text1 = "马云创立了阿里巴巴，是中国最大的电子商务公司。"
        proc.remember_text(text1, doc_name="remember_1", verbose=False, verbose_steps=False)

        # Second remember with overlapping entity
        text2 = "阿里巴巴后来推出了淘宝网和支付宝，深刻改变了中国人的购物和支付方式。"
        proc.remember_text(text2, doc_name="remember_2", verbose=False, verbose_steps=False)

        entities = proc.storage.get_all_entities()

        # "阿里巴巴" should not have many distinct family_ids
        alibaba_fids = set(e.family_id for e in entities if "阿里巴巴" in (e.name or ""))
        assert len(alibaba_fids) <= 2, (
            f"'阿里巴巴' has {len(alibaba_fids)} distinct family_ids — merge failed. "
            f"Entities: {[(e.name, e.family_id) for e in entities if '阿里巴巴' in (e.name or '')]}"
        )

    def test_different_entities_not_merged(self, tmp_path, real_llm_config, shared_embedding_client):
        """Two remember calls with different entities should not merge them."""
        if not real_llm_config:
            pytest.skip("No LLM config")
        proc = _make_v2_processor(tmp_path, real_llm_config, shared_embedding_client)

        text1 = "Python is a programming language created by Guido van Rossum."
        proc.remember_text(text1, doc_name="diff_1", verbose=False, verbose_steps=False)

        text2 = "Java is a programming language created by James Gosling at Sun Microsystems."
        proc.remember_text(text2, doc_name="diff_2", verbose=False, verbose_steps=False)

        entities = proc.storage.get_all_entities()
        python_fids = set(e.family_id for e in entities if "python" in (e.name or "").lower())
        java_fids = set(e.family_id for e in entities if "java" in (e.name or "").lower())

        # Python and Java should NOT share a family_id
        overlap = python_fids & java_fids
        assert len(overlap) == 0, (
            f"Python and Java were incorrectly merged: shared family_ids={overlap}"
        )

    def test_multi_remember_accumulates_knowledge(self, tmp_path, real_llm_config, shared_embedding_client):
        """Multiple remember calls should accumulate entities across calls."""
        if not real_llm_config:
            pytest.skip("No LLM config")
        proc = _make_v2_processor(tmp_path, real_llm_config, shared_embedding_client)

        texts = [
            "唐朝由李渊建立，定都长安。",
            "唐太宗李世民开创了贞观之治。",
            "李白是唐代最著名的诗人，被称为诗仙。",
        ]
        for i, text in enumerate(texts):
            proc.remember_text(text, doc_name=f"tang_{i}", verbose=False, verbose_steps=False)

        entities = proc.storage.get_all_entities()
        entity_names = {e.name for e in entities}

        # Should have accumulated entities from all three calls
        assert len(entities) >= 4, (
            f"Expected >= 4 accumulated entities, got {len(entities)}: {entity_names}"
        )
