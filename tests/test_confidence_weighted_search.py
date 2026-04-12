"""
Confidence-weighted search ranking — 32 tests across 4 dimensions.

P0 blueprint gap fix: The confidence engine (blueprint line 86) requires
"影响默认召回排序与展示" — confidence must affect Find ranking.

The HybridSearcher.confidence_rerank() method existed but was dead code.
Now all hybrid search endpoints call it before serializing results.

D1: Endpoint integration — confidence_rerank called in each search path (8 tests)
D2: Ranking correctness — high confidence ranks above low confidence (8 tests)
D3: Edge cases — missing confidence, empty results, alpha boundary (8 tests)
D4: End-to-end search quality with mixed data (8 tests)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_storage(tmp_path):
    from processor.storage.manager import StorageManager
    sm = StorageManager(str(tmp_path / "graph"))
    yield sm
    if hasattr(sm, '_vector_store') and sm._vector_store:
        sm._vector_store.close()


def _make_entity(family_id, name, content, confidence=0.7, source_document="test"):
    from processor.models import Entity
    now = datetime.now(timezone.utc)
    return Entity(
        absolute_id=str(uuid.uuid4()),
        family_id=family_id,
        name=name,
        content=content,
        event_time=now,
        processed_time=now,
        episode_id="ep_test",
        source_document=source_document,
        confidence=confidence,
    )


def _make_relation(family_id, e1_abs, e2_abs, content, confidence=0.7):
    from processor.models import Relation
    now = datetime.now(timezone.utc)
    return Relation(
        absolute_id=str(uuid.uuid4()),
        family_id=family_id,
        entity1_absolute_id=e1_abs,
        entity2_absolute_id=e2_abs,
        content=content,
        event_time=now,
        processed_time=now,
        episode_id="ep_test",
        source_document="test",
        confidence=confidence,
    )


def _seed_entities(storage):
    """Create entities with varying confidence levels."""
    e_high = _make_entity("ent_high", "Alice", "Engineer at Google", confidence=0.95)
    e_med = _make_entity("ent_med", "Bob", "Designer at Apple", confidence=0.5)
    e_low = _make_entity("ent_low", "Charlie", "Manager at Meta", confidence=0.1)
    e_none = _make_entity("ent_none", "Diana", "Analyst at Amazon", confidence=None)
    storage.bulk_save_entities([e_high, e_med, e_low, e_none])
    return e_high, e_med, e_low, e_none


def _seed_relations(storage, e1, e2, e3):
    """Create relations with varying confidence levels."""
    r_high = _make_relation("rel_high", e1.absolute_id, e2.absolute_id,
                            "Alice mentors Bob", confidence=0.9)
    r_low = _make_relation("rel_low", e2.absolute_id, e3.absolute_id,
                           "Bob reports to Charlie", confidence=0.2)
    storage.bulk_save_relations([r_high, r_low])
    return r_high, r_low


def _make_registry(storage_path):
    from server.registry import GraphRegistry
    config = {
        "default_embedding_provider": "none",
        "default_llm_provider": "none",
        "remember_workers": 0,
        "remember_max_retries": 0,
        "remember_retry_delay_seconds": 0,
    }
    return GraphRegistry(base_storage_path=storage_path, config=config)


# ══════════════════════════════════════════════════════════════════════════
# D1: Endpoint integration — confidence_rerank called in each search path
# ══════════════════════════════════════════════════════════════════════════


class TestEndpointIntegration:
    """D1: Verify confidence_rerank is called in all hybrid search paths."""

    @pytest.fixture
    def app(self, tmp_path):
        from server.api import create_app
        registry = _make_registry(str(tmp_path / "graphs"))
        app = create_app(registry=registry, config={"rate_limit_per_minute": 600})
        app.config['TESTING'] = True
        return app

    @pytest.fixture
    def client(self, app):
        return app.test_client()

    def test_find_entities_search_hybrid_calls_rerank(self, client):
        """D1.1: find_entities_search hybrid mode calls confidence_rerank."""
        with patch('server.blueprints.entities.HybridSearcher') as MockSearcher:
            mock_searcher = MagicMock()
            e1 = _make_entity("ent_d1a", "Test", "Content", confidence=0.9)
            mock_searcher.search_entities.return_value = [(e1, 0.8)]
            mock_searcher.confidence_rerank.side_effect = lambda items, alpha=0.2: items
            MockSearcher.return_value = mock_searcher

            with patch('server.blueprints.entities._get_processor') as mock_proc:
                proc = MagicMock()
                mock_proc.return_value = proc

                client.post('/api/v1/find/entities/search',
                            json={"query_name": "test", "search_mode": "hybrid"})

                mock_searcher.confidence_rerank.assert_called_once()
                call_args = mock_searcher.confidence_rerank.call_args
                assert call_args[1].get('alpha', call_args[0][1] if len(call_args[0]) > 1 else None) == 0.2 or \
                       call_args[0][1] == 0.2

    def test_find_relations_search_hybrid_calls_rerank(self, client):
        """D1.2: find_relations_search hybrid mode calls confidence_rerank."""
        with patch('server.blueprints.relations.HybridSearcher') as MockSearcher:
            mock_searcher = MagicMock()
            r1 = _make_relation("rel_d2", "a", "b", "Test", confidence=0.9)
            mock_searcher.search_relations.return_value = [(r1, 0.7)]
            mock_searcher.confidence_rerank.side_effect = lambda items, alpha=0.2: items
            MockSearcher.return_value = mock_searcher

            with patch('server.blueprints.relations._get_processor') as mock_proc:
                proc = MagicMock()
                mock_proc.return_value = proc

                client.post('/api/v1/find/relations/search',
                            json={"query_text": "test", "search_mode": "hybrid"})

                mock_searcher.confidence_rerank.assert_called_once()

    def test_find_unified_hybrid_calls_rerank(self, client):
        """D1.3: find_unified hybrid mode calls confidence_rerank."""
        with patch('server.blueprints.relations.HybridSearcher') as MockSearcher:
            mock_searcher = MagicMock()
            e1 = _make_entity("ent_d3", "Test", "Content", confidence=0.9)
            r1 = _make_relation("rel_d3", "a", "b", "Test", confidence=0.9)
            mock_searcher.search_entities.return_value = [(e1, 0.8)]
            mock_searcher.search_relations.return_value = [(r1, 0.7)]
            mock_searcher.confidence_rerank.side_effect = lambda items, alpha=0.2: items
            MockSearcher.return_value = mock_searcher

            with patch('server.blueprints.relations._get_processor') as mock_proc:
                proc = MagicMock()
                proc.storage.batch_get_entity_degrees.return_value = {}
                mock_proc.return_value = proc

                client.post('/api/v1/find',
                            json={"query": "test", "search_mode": "hybrid", "expand": False})

                # Should be called twice: once for entities, once for relations
                assert mock_searcher.confidence_rerank.call_count == 2

    def test_quick_search_calls_rerank(self, client):
        """D1.4: quick_search calls confidence_rerank for both phases."""
        with patch('server.blueprints.relations.HybridSearcher') as MockSearcher:
            mock_searcher = MagicMock()
            e1 = _make_entity("ent_d4", "Test", "Content", confidence=0.9)
            r1 = _make_relation("rel_d4", "a", "b", "Test", confidence=0.9)
            mock_searcher.search_entities.return_value = [(e1, 0.8)]
            mock_searcher.search_relations.return_value = [(r1, 0.7)]
            mock_searcher.confidence_rerank.side_effect = lambda items, alpha=0.2: items
            MockSearcher.return_value = mock_searcher

            with patch('server.blueprints.relations._get_processor') as mock_proc:
                proc = MagicMock()
                proc.storage.get_family_ids_by_names.return_value = {}
                mock_proc.return_value = proc

                client.post('/api/v1/find/quick-search',
                            json={"query": "test"})

                assert mock_searcher.confidence_rerank.call_count == 2

    def test_find_unified_nonhybrid_skips_rerank(self, client):
        """D1.5: Non-hybrid search does NOT call confidence_rerank."""
        with patch('server.blueprints.relations.HybridSearcher') as MockSearcher:
            mock_searcher = MagicMock()
            MockSearcher.return_value = mock_searcher

            with patch('server.blueprints.relations._get_processor') as mock_proc:
                proc = MagicMock()
                proc.storage.search_entities_by_similarity.return_value = []
                proc.storage.search_relations_by_similarity.return_value = []
                mock_proc.return_value = proc

                client.post('/api/v1/find',
                            json={"query": "test", "search_mode": "semantic", "expand": False})

                mock_searcher.confidence_rerank.assert_not_called()

    def test_find_entities_bm25_skips_rerank(self, client):
        """D1.6: BM25 search does NOT call confidence_rerank."""
        with patch('server.blueprints.entities.HybridSearcher') as MockSearcher:
            mock_searcher = MagicMock()
            MockSearcher.return_value = mock_searcher

            with patch('server.blueprints.entities._get_processor') as mock_proc:
                proc = MagicMock()
                proc.storage.search_entities_by_bm25.return_value = []
                mock_proc.return_value = proc

                client.post('/api/v1/find/entities/search',
                            json={"query_name": "test", "search_mode": "bm25"})

                mock_searcher.confidence_rerank.assert_not_called()

    def test_find_unified_node_degree_skips_confidence_rerank(self, client):
        """D1.7: node_degree reranker skips confidence rerank."""
        with patch('server.blueprints.relations.HybridSearcher') as MockSearcher:
            mock_searcher = MagicMock()
            e1 = _make_entity("ent_d7", "Test", "Content", confidence=0.9)
            mock_searcher.search_entities.return_value = [(e1, 0.8)]
            mock_searcher.search_relations.return_value = []
            mock_searcher.node_degree_rerank.side_effect = lambda items, dm: items
            mock_searcher.confidence_rerank.side_effect = lambda items, alpha=0.2: items
            MockSearcher.return_value = mock_searcher

            with patch('server.blueprints.relations._get_processor') as mock_proc:
                proc = MagicMock()
                proc.storage.batch_get_entity_degrees.return_value = {}
                mock_proc.return_value = proc

                client.post('/api/v1/find',
                            json={"query": "test", "search_mode": "hybrid",
                                  "reranker": "node_degree", "expand": False})

                # confidence_rerank should NOT be called when node_degree reranker is active
                mock_searcher.confidence_rerank.assert_not_called()

    def test_rerank_alpha_is_0_2(self, client):
        """D1.8: All calls use alpha=0.2 (20% confidence weight)."""
        with patch('server.blueprints.entities.HybridSearcher') as MockSearcher:
            mock_searcher = MagicMock()
            e1 = _make_entity("ent_d8", "Test", "Content", confidence=0.9)
            mock_searcher.search_entities.return_value = [(e1, 0.8)]
            mock_searcher.confidence_rerank.side_effect = lambda items, alpha=0.2: items
            MockSearcher.return_value = mock_searcher

            with patch('server.blueprints.entities._get_processor') as mock_proc:
                proc = MagicMock()
                mock_proc.return_value = proc

                client.post('/api/v1/find/entities/search',
                            json={"query_name": "test", "search_mode": "hybrid"})

                # Verify alpha=0.2 was passed
                _, kwargs = mock_searcher.confidence_rerank.call_args
                assert kwargs.get('alpha', 0.2) == 0.2


# ══════════════════════════════════════════════════════════════════════════
# D2: Ranking correctness — high confidence ranks above low confidence
# ══════════════════════════════════════════════════════════════════════════


class TestRankingCorrectness:
    """D2: Verify ranking is affected by confidence values."""

    def test_high_confidence_entity_ranks_first(self, tmp_storage):
        """D2.1: Entity with high confidence ranks above low confidence with same RRF score."""
        from processor.search.hybrid import HybridSearcher
        e_high = _make_entity("ent_r1h", "HighConf", "Content", confidence=0.9)
        e_low = _make_entity("ent_r1l", "LowConf", "Content", confidence=0.1)
        tmp_storage.bulk_save_entities([e_high, e_low])

        searcher = HybridSearcher(tmp_storage)
        items = [(e_low, 0.5), (e_high, 0.5)]
        result = searcher.confidence_rerank(items, alpha=0.2)

        assert result[0][0].family_id == "ent_r1h"

    def test_high_confidence_relation_ranks_first(self, tmp_storage):
        """D2.2: Relation with high confidence ranks above low confidence."""
        from processor.search.hybrid import HybridSearcher
        r_high = _make_relation("rel_r2h", "a", "b", "Strong link", confidence=0.9)
        r_low = _make_relation("rel_r2l", "c", "d", "Weak link", confidence=0.1)
        tmp_storage.bulk_save_relations([r_high, r_low])

        searcher = HybridSearcher(tmp_storage)
        items = [(r_low, 0.5), (r_high, 0.5)]
        result = searcher.confidence_rerank(items, alpha=0.2)

        assert result[0][0].family_id == "rel_r2h"

    def test_low_rrf_high_confidence_can_overtake(self, tmp_storage):
        """D2.3: Lower RRF score + high confidence can overtake higher RRF + low confidence."""
        from processor.search.hybrid import HybridSearcher
        e_high = _make_entity("ent_r3h", "HighConf", "Content", confidence=0.95)
        e_low = _make_entity("ent_r3l", "LowConf", "Content", confidence=0.1)
        tmp_storage.bulk_save_entities([e_high, e_low])

        searcher = HybridSearcher(tmp_storage)
        # High conf has lower RRF score
        items = [(e_low, 0.5), (e_high, 0.3)]
        result = searcher.confidence_rerank(items, alpha=0.5)

        # With alpha=0.5: low = 0.5*(1-0.5+0.5*0.1)=0.275, high = 0.3*(1-0.5+0.5*0.95)=0.3925
        assert result[0][0].family_id == "ent_r3h"

    def test_many_entities_mixed_confidence(self, tmp_storage):
        """D2.4: Multiple entities with mixed confidence get properly sorted."""
        from processor.search.hybrid import HybridSearcher
        entities = [
            _make_entity(f"ent_r4_{i}", f"Entity{i}", f"Content {i}",
                         confidence=round(0.1 + i * 0.1, 1))
            for i in range(8)
        ]
        tmp_storage.bulk_save_entities(entities)

        searcher = HybridSearcher(tmp_storage)
        # All same RRF score
        items = [(e, 0.5) for e in entities]
        # Shuffle: put lowest confidence first
        items.reverse()
        result = searcher.confidence_rerank(items, alpha=0.3)

        # Should be sorted by confidence (highest first)
        confidences = [getattr(r[0], 'confidence', 0.5) for r in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_mixed_entities_and_relations(self, tmp_storage):
        """D2.5: Confidence rerank works for mixed entity/relation items."""
        from processor.search.hybrid import HybridSearcher
        e = _make_entity("ent_r5", "Entity", "Content", confidence=0.9)
        r = _make_relation("rel_r5", "a", "b", "Link", confidence=0.2)

        searcher = HybridSearcher(tmp_storage)
        items = [(r, 0.5), (e, 0.5)]
        result = searcher.confidence_rerank(items, alpha=0.3)

        assert result[0][0].family_id == "ent_r5"

    def test_equal_confidence_preserves_rrf_order(self, tmp_storage):
        """D2.6: Equal confidence preserves original RRF ranking."""
        from processor.search.hybrid import HybridSearcher
        e1 = _make_entity("ent_r6a", "E1", "Content", confidence=0.7)
        e2 = _make_entity("ent_r6b", "E2", "Content", confidence=0.7)
        e3 = _make_entity("ent_r6c", "E3", "Content", confidence=0.7)
        tmp_storage.bulk_save_entities([e1, e2, e3])

        searcher = HybridSearcher(tmp_storage)
        items = [(e1, 0.9), (e2, 0.6), (e3, 0.3)]
        result = searcher.confidence_rerank(items, alpha=0.3)

        assert [r[0].family_id for r in result] == ["ent_r6a", "ent_r6b", "ent_r6c"]

    def test_extreme_confidence_difference(self, tmp_storage):
        """D2.7: Very high vs very low confidence creates clear ranking separation."""
        from processor.search.hybrid import HybridSearcher
        e_high = _make_entity("ent_r7h", "HighConf", "Content", confidence=1.0)
        e_low = _make_entity("ent_r7l", "LowConf", "Content", confidence=0.01)
        tmp_storage.bulk_save_entities([e_high, e_low])

        searcher = HybridSearcher(tmp_storage)
        items = [(e_low, 0.5), (e_high, 0.5)]
        result = searcher.confidence_rerank(items, alpha=0.3)

        # High should be clearly first; check score gap
        assert result[0][1] - result[1][1] > 0.05

    def test_very_low_confidence_ranks_last(self, tmp_storage):
        """D2.8: Entity with very low confidence (0.01) ranks last among equal scores."""
        from processor.search.hybrid import HybridSearcher
        e_low = _make_entity("ent_r8z", "LowConf", "Content", confidence=0.01)
        e_high = _make_entity("ent_r8h", "HighConf", "Content", confidence=0.8)
        e_med = _make_entity("ent_r8m", "MedConf", "Content", confidence=0.4)
        tmp_storage.bulk_save_entities([e_low, e_high, e_med])

        searcher = HybridSearcher(tmp_storage)
        items = [(e_low, 0.5), (e_high, 0.5), (e_med, 0.5)]
        result = searcher.confidence_rerank(items, alpha=0.3)

        assert result[-1][0].family_id == "ent_r8z"


# ══════════════════════════════════════════════════════════════════════════
# D3: Edge cases — missing confidence, empty results, alpha boundary
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """D3: Edge cases for confidence-weighted reranking."""

    def test_none_confidence_treated_as_0_5(self, tmp_storage):
        """D3.1: Entity with confidence=None treated as 0.5."""
        from processor.search.hybrid import HybridSearcher
        e_none = _make_entity("ent_e1", "NoConf", "Content", confidence=None)
        e_05 = _make_entity("ent_e1b", "HalfConf", "Content", confidence=0.5)
        tmp_storage.bulk_save_entities([e_none, e_05])

        searcher = HybridSearcher(tmp_storage)
        items = [(e_none, 0.5), (e_05, 0.5)]
        result = searcher.confidence_rerank(items, alpha=0.3)

        # Both should have same score
        assert abs(result[0][1] - result[1][1]) < 0.001

    def test_empty_items_returns_empty(self, tmp_storage):
        """D3.2: Empty input returns empty output."""
        from processor.search.hybrid import HybridSearcher
        searcher = HybridSearcher(tmp_storage)
        result = searcher.confidence_rerank([], alpha=0.3)
        assert result == []

    def test_single_item_returns_same(self, tmp_storage):
        """D3.3: Single item passes through with adjusted score."""
        from processor.search.hybrid import HybridSearcher
        e = _make_entity("ent_e3", "Single", "Content", confidence=0.8)
        tmp_storage.save_entity(e)

        searcher = HybridSearcher(tmp_storage)
        items = [(e, 0.5)]
        result = searcher.confidence_rerank(items, alpha=0.3)

        assert len(result) == 1
        assert result[0][0].family_id == "ent_e3"
        # score * (1 - alpha + alpha * confidence) = 0.5 * (0.7 + 0.3*0.8) = 0.47
        expected = 0.5 * (1 - 0.3 + 0.3 * 0.8)
        assert abs(result[0][1] - round(expected, 6)) < 0.001

    def test_alpha_zero_preserves_original_scores(self, tmp_storage):
        """D3.4: alpha=0 means confidence has no effect on scores."""
        from processor.search.hybrid import HybridSearcher
        e1 = _make_entity("ent_e4a", "A", "Content", confidence=0.1)
        e2 = _make_entity("ent_e4b", "B", "Content", confidence=0.9)
        tmp_storage.bulk_save_entities([e1, e2])

        searcher = HybridSearcher(tmp_storage)
        items = [(e1, 0.8), (e2, 0.3)]
        result = searcher.confidence_rerank(items, alpha=0.0)

        # Scores should be unchanged
        assert result[0][0].family_id == "ent_e4a"
        assert result[0][1] == 0.8

    def test_alpha_one_pure_confidence_score(self, tmp_storage):
        """D3.5: alpha=1.0 means adjusted = score * confidence."""
        from processor.search.hybrid import HybridSearcher
        e1 = _make_entity("ent_e5a", "A", "Content", confidence=0.2)
        e2 = _make_entity("ent_e5b", "B", "Content", confidence=0.8)
        tmp_storage.bulk_save_entities([e1, e2])

        searcher = HybridSearcher(tmp_storage)
        items = [(e1, 0.5), (e2, 0.5)]
        result = searcher.confidence_rerank(items, alpha=1.0)

        # adjusted = score * confidence: e1=0.1, e2=0.4
        assert result[0][0].family_id == "ent_e5b"
        assert abs(result[0][1] - 0.4) < 0.001

    def test_negative_confidence_handled(self, tmp_storage):
        """D3.6: Negative confidence (data error) doesn't crash."""
        from processor.search.hybrid import HybridSearcher
        e = _make_entity("ent_e6", "Negative", "Content", confidence=-0.1)
        tmp_storage.save_entity(e)

        searcher = HybridSearcher(tmp_storage)
        items = [(e, 0.5)]
        # Should not crash
        result = searcher.confidence_rerank(items, alpha=0.3)
        assert len(result) == 1

    def test_confidence_above_1_handled(self, tmp_storage):
        """D3.7: Confidence > 1.0 doesn't crash (data error)."""
        from processor.search.hybrid import HybridSearcher
        e = _make_entity("ent_e7", "OverConfident", "Content", confidence=1.5)
        tmp_storage.save_entity(e)

        searcher = HybridSearcher(tmp_storage)
        items = [(e, 0.5)]
        # Should not crash
        result = searcher.confidence_rerank(items, alpha=0.3)
        assert len(result) == 1

    def test_large_item_set(self, tmp_storage):
        """D3.8: Reranking works with 100+ items without performance issues."""
        from processor.search.hybrid import HybridSearcher
        entities = [
            _make_entity(f"ent_e8_{i}", f"E{i}", f"Content {i}",
                         confidence=round((i % 10) / 10, 1))
            for i in range(100)
        ]
        tmp_storage.bulk_save_entities(entities)

        searcher = HybridSearcher(tmp_storage)
        items = [(e, 0.5) for e in entities]
        result = searcher.confidence_rerank(items, alpha=0.2)

        assert len(result) == 100
        # All scores should be positive
        assert all(score > 0 for _, score in result)


# ══════════════════════════════════════════════════════════════════════════
# D4: End-to-end search quality with mixed data
# ══════════════════════════════════════════════════════════════════════════


class TestEndToEndSearchQuality:
    """D4: End-to-end tests with real storage and search flows."""

    def test_entity_search_ranks_higher_confidence_first(self, tmp_storage):
        """D4.1: Two entities with same content but different confidence."""
        from processor.search.hybrid import HybridSearcher
        e_high = _make_entity("ent_q1h", "Python", "Programming language", confidence=0.95)
        e_low = _make_entity("ent_q1l", "Python", "Programming language", confidence=0.1)
        tmp_storage.bulk_save_entities([e_high, e_low])

        searcher = HybridSearcher(tmp_storage)
        # Simulate what the endpoint does
        hybrid_ents = [(e_low, 0.5), (e_high, 0.5)]
        reranked = searcher.confidence_rerank(hybrid_ents, alpha=0.2)

        assert reranked[0][0].family_id == "ent_q1h"

    def test_relation_search_ranks_higher_confidence_first(self, tmp_storage):
        """D4.2: Two relations with same content but different confidence."""
        from processor.search.hybrid import HybridSearcher
        e1 = _make_entity("ent_q2a", "A", "Content")
        e2 = _make_entity("ent_q2b", "B", "Content")
        tmp_storage.bulk_save_entities([e1, e2])

        r_high = _make_relation("rel_q2h", e1.absolute_id, e2.absolute_id,
                                "A collaborates with B", confidence=0.9)
        r_low = _make_relation("rel_q2l", e1.absolute_id, e2.absolute_id,
                               "A collaborates with B", confidence=0.2)
        tmp_storage.bulk_save_relations([r_high, r_low])

        searcher = HybridSearcher(tmp_storage)
        hybrid_rels = [(r_low, 0.5), (r_high, 0.5)]
        reranked = searcher.confidence_rerank(hybrid_rels, alpha=0.2)

        assert reranked[0][0].family_id == "rel_q2h"

    def test_score_reflected_in_serialized_output(self, tmp_storage):
        """D4.3: Adjusted score is reflected in entity dict serialization."""
        from server.blueprints.helpers import entity_to_dict
        from processor.search.hybrid import HybridSearcher
        e_high = _make_entity("ent_q3h", "HighConf", "Content", confidence=0.9)
        e_low = _make_entity("ent_q3l", "LowConf", "Content", confidence=0.1)
        tmp_storage.bulk_save_entities([e_high, e_low])

        searcher = HybridSearcher(tmp_storage)
        items = [(e_low, 0.5), (e_high, 0.5)]
        reranked = searcher.confidence_rerank(items, alpha=0.2)

        # Serialize with adjusted scores
        dicts = [entity_to_dict(e, _score=score) for e, score in reranked]
        assert dicts[0]["_score"] > dicts[1]["_score"]

    def test_three_tier_confidence_ranking(self, tmp_storage):
        """D4.4: Three tiers of confidence (high/med/low) maintain proper order."""
        from processor.search.hybrid import HybridSearcher
        e_high = _make_entity("ent_q4h", "Verified", "Confirmed info", confidence=0.95)
        e_med = _make_entity("ent_q4m", "Probable", "Likely info", confidence=0.5)
        e_low = _make_entity("ent_q4l", "Speculative", "Rumor", confidence=0.1)
        tmp_storage.bulk_save_entities([e_high, e_med, e_low])

        searcher = HybridSearcher(tmp_storage)
        # All same RRF score, shuffle order
        items = [(e_med, 0.5), (e_low, 0.5), (e_high, 0.5)]
        result = searcher.confidence_rerank(items, alpha=0.2)

        fids = [r[0].family_id for r in result]
        assert fids == ["ent_q4h", "ent_q4m", "ent_q4l"]

    def test_pipeline_corroboration_boosts_search_ranking(self, tmp_storage):
        """D4.5: Entity that was corroborated (high conf) ranks above uncorroborated."""
        from processor.search.hybrid import HybridSearcher
        # Simulate corroborated entity (high confidence from multiple sources)
        e_corroborated = _make_entity("ent_q5c", "Earth", "Planet in solar system",
                                      confidence=0.95)
        e_uncorroborated = _make_entity("ent_q5u", "Mars", "Red planet",
                                        confidence=0.3)
        tmp_storage.bulk_save_entities([e_corroborated, e_uncorroborated])

        searcher = HybridSearcher(tmp_storage)
        items = [(e_uncorroborated, 0.5), (e_corroborated, 0.5)]
        result = searcher.confidence_rerank(items, alpha=0.2)

        assert result[0][0].name == "Earth"

    def test_contradicted_entity_demoted(self, tmp_storage):
        """D4.6: Entity with lowered confidence from contradiction ranks lower."""
        from processor.search.hybrid import HybridSearcher
        e_good = _make_entity("ent_q6g", "Reliable Source", "Verified data",
                              confidence=0.85)
        e_bad = _make_entity("ent_q6b", "Contradicted", "Conflicting data",
                             confidence=0.15)
        tmp_storage.bulk_save_entities([e_good, e_bad])

        searcher = HybridSearcher(tmp_storage)
        # Even if bad entity had higher RRF score, good overtakes with higher alpha
        items = [(e_bad, 0.6), (e_good, 0.4)]
        result = searcher.confidence_rerank(items, alpha=0.5)

        # Good entity should overtake due to confidence (alpha=0.5 gives more weight)
        # bad:  0.6 * (0.5 + 0.5*0.15) = 0.6 * 0.575 = 0.345
        # good: 0.4 * (0.5 + 0.5*0.85) = 0.4 * 0.925 = 0.37
        assert result[0][0].name == "Reliable Source"

    def test_dream_discovered_relations_ranked_by_confidence(self, tmp_storage):
        """D4.7: Dream-discovered relations with low confidence rank lower."""
        from processor.search.hybrid import HybridSearcher
        e1 = _make_entity("ent_q7a", "X", "Content")
        e2 = _make_entity("ent_q7b", "Y", "Content")
        e3 = _make_entity("ent_q7c", "Z", "Content")
        tmp_storage.bulk_save_entities([e1, e2, e3])

        r_dream = _make_relation("rel_q7d", e1.absolute_id, e2.absolute_id,
                                 "Dream link", confidence=0.3)
        r_real = _make_relation("rel_q7r", e2.absolute_id, e3.absolute_id,
                                "Explicit link", confidence=0.9)
        tmp_storage.bulk_save_relations([r_dream, r_real])

        searcher = HybridSearcher(tmp_storage)
        items = [(r_dream, 0.5), (r_real, 0.5)]
        result = searcher.confidence_rerank(items, alpha=0.2)

        assert result[0][0].family_id == "rel_q7r"

    def test_mixed_entities_and_relations_combined_ranking(self, tmp_storage):
        """D4.8: Entities and relations mixed together get correct ranking."""
        from processor.search.hybrid import HybridSearcher
        e_high = _make_entity("ent_q8h", "High", "Content", confidence=0.9)
        e_low = _make_entity("ent_q8l", "Low", "Content", confidence=0.2)
        r_med = _make_relation("rel_q8m", e_high.absolute_id, e_low.absolute_id,
                               "Link", confidence=0.5)
        tmp_storage.bulk_save_entities([e_high, e_low])
        tmp_storage.bulk_save_relations([r_med])

        searcher = HybridSearcher(tmp_storage)
        items = [(e_low, 0.5), (r_med, 0.5), (e_high, 0.5)]
        result = searcher.confidence_rerank(items, alpha=0.2)

        # Order should be: high(0.9) > med(0.5) > low(0.2)
        confidences = [getattr(r[0], 'confidence', 0.5) for r in result]
        assert confidences == sorted(confidences, reverse=True)
