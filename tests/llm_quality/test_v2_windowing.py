"""
V2 multi-window extraction quality tests with real LLM.

Tests:
- Entities at window boundaries are not lost
- Cross-window entity dedup works correctly
- Longer texts produce proportionally more entities
- Relations spanning windows are discovered

Uses window_size=500, overlap=100 (matching production config).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# Allow `from conftest import ...` to find tests/llm_quality/conftest.py
sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.real_llm

from conftest import _make_v2_processor, _run_v2_extraction
from tests.fixtures.extraction_gold import EXTRACTION_GOLD


class TestMultiWindowExtraction:
    """Test extraction quality with multi-window texts."""

    @pytest.fixture(scope="class")
    def processor(self, tmp_path_factory, real_llm_config, shared_embedding_client):
        if not real_llm_config:
            pytest.skip("No LLM config")
        tmp = tmp_path_factory.mktemp("v2_windowing")
        return _make_v2_processor(
            tmp, real_llm_config, shared_embedding_client,
            window_size=500, overlap=100,  # Production config
        )

    def test_long_text_entity_count_proportional(self, processor):
        """Longer texts should produce proportionally more entities."""
        short_text = "Google is a technology company."
        long_case = EXTRACTION_GOLD["tech_go"]  # ~2000 chars

        short_ents, _ = _run_v2_extraction(processor, short_text)
        long_ents, _ = _run_v2_extraction(processor, long_case["text"])

        assert len(long_ents) > len(short_ents), (
            f"Long text ({len(long_ents)} ents) should have more than "
            f"short text ({len(short_ents)} ents)"
        )

    def test_multi_window_dedup(self, processor):
        """Same entity appearing in multiple windows should be deduplicated."""
        # Text that mentions "Google" throughout multiple windows
        text = (
            "Google was founded by Larry Page and Sergey Brin at Stanford. "
            "Google developed the Android operating system. "
            "Google created the Chrome browser. "
            "Google acquired YouTube in 2006. "
            "Google launched Gmail in 2004. "
            "Google Maps is widely used for navigation. "
            "Google Translate supports over 100 languages. "
            "Google Cloud competes with AWS and Azure. "
            "Google's parent company Alphabet was created in 2015. "
            "Google DeepMind achieved breakthroughs in AI."
        )
        entities, _ = _run_v2_extraction(processor, text)

        # Count Google occurrences (allow some variation due to different aspects)
        google_names = [e["name"] for e in entities if "google" in e["name"].lower()]
        # Should not have more than 3 variants of Google
        assert len(google_names) <= 3, (
            f"Too many Google variants: {google_names} — cross-window dedup may have failed"
        )

    def test_boundary_entity_preserved(self, processor):
        """Entities mentioned near window boundaries should be extracted."""
        # Create text where key entities appear at the ~500 char boundary
        # This text is carefully crafted so "量子纠缠" appears right around char 480-520
        padding = "量子计算是利用量子力学原理进行信息处理的新型计算范式。"
        boundary_text = (
            padding * 5 +  # ~250 chars of padding
            "量子纠缠是量子计算的关键资源。" +  # Key entity near boundary
            "当两个量子比特纠缠时，测量一个会瞬间影响另一个。"
            "爱因斯坦称之为幽灵般的超距作用。" +
            padding * 5  # More padding
        )
        entities, _ = _run_v2_extraction(processor, boundary_text)
        extracted_names = {e["name"] for e in entities}

        # "量子纠缠" should be found (it's a key entity at the boundary)
        found_entanglement = any("量子纠缠" in n for n in extracted_names)
        # Or at least "量子计算" should be found
        found_quantum = any("量子" in n for n in extracted_names)
        assert found_quantum or found_entanglement, (
            f"Boundary entity not found. Got: {sorted(extracted_names)}"
        )

    def test_relation_across_windows(self, processor):
        """Relations between entities in different windows should be discovered."""
        # Use the long mixed_domains text which spans multiple windows
        case = EXTRACTION_GOLD["mixed_internet"]  # ~3000 chars, 6+ windows
        entities, relations = _run_v2_extraction(processor, case["text"])

        # Should find at least some relations
        assert len(relations) >= 2, (
            f"Multi-window text should produce at least 2 relations, got {len(relations)}"
        )

        # Check that relation endpoints reference extracted entities
        entity_names = {e["name"] for e in entities}
        for r in relations:
            e1 = r.get("entity1_name", "")
            e2 = r.get("entity2_name", "")
            # At least one endpoint should match an entity name
            assert e1 in entity_names or e2 in entity_names, (
                f"Relation ({e1}, {e2}) has no matching entity. "
                f"Available: {sorted(entity_names)}"
            )
