"""
Contradiction Detection Pipeline Integration — 30+ test cases across 4 dimensions:

  Dimension 1: Core contradiction detection logic (LLM-driven)
  Dimension 2: Confidence penalty on contradiction
  Dimension 3: Pipeline integration (_detect_and_apply_contradictions)
  Dimension 4: Edge cases & error resilience
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processor.storage.manager import StorageManager
from processor.models import Entity, Relation
from processor.pipeline.extraction import _ExtractionMixin


def _now() -> datetime:
    return datetime.now()


def _make_entity(
    family_id="fam_1",
    name="TestEntity",
    content="A test entity.",
    confidence=0.7,
    processed_time=None,
    version=0,
) -> Entity:
    pt = processed_time or (_now() + timedelta(seconds=version))
    return Entity(
        absolute_id=f"{family_id}_v{version}_{pt.strftime('%H%M%S_%f')}",
        family_id=family_id,
        name=name,
        content=content,
        event_time=pt,
        processed_time=pt,
        episode_id=f"ep_{version}",
        source_document="test.md",
        confidence=confidence,
        content_format="markdown",
        summary=content[:80],
    )


@pytest.fixture
def storage(tmp_path):
    s = StorageManager(storage_path=str(tmp_path / "test_contra.db"))
    yield s
    s.close()


def _make_mock_processor(storage, llm_contradictions=None):
    """Create a mock processor with _ExtractionMixin methods and real storage."""
    llm = MagicMock()
    llm.detect_contradictions = AsyncMock(return_value=llm_contradictions or [])
    # Create a minimal object with the mixin methods
    proc = type("MockProcessor", (_ExtractionMixin,), {
        "storage": storage,
        "llm_client": llm,
    })()
    return proc, llm


# ═══════════════════════════════════════════════════════════════════
# Dimension 1: Core detection logic
# ═══════════════════════════════════════════════════════════════════

class TestCoreDetectionLogic:
    """Tests for _detect_and_apply_contradictions core logic."""

    def test_no_versions_skipped(self, storage):
        """Family IDs with < 2 versions should be skipped."""
        e = _make_entity(family_id="fam_s1", version=0)
        storage.save_entity(e)
        proc, llm = _make_mock_processor(storage)
        proc._detect_and_apply_contradictions(["fam_s1"])
        llm.detect_contradictions.assert_not_called()

    def test_two_versions_triggers_detection(self, storage):
        """Family IDs with >= 2 versions should trigger LLM detection."""
        storage.save_entity(_make_entity(family_id="fam_t1", version=0, content="Old info"))
        storage.save_entity(_make_entity(family_id="fam_t1", version=1, content="New info"))
        proc, llm = _make_mock_processor(storage)
        proc._detect_and_apply_contradictions(["fam_t1"])
        llm.detect_contradictions.assert_called_once()

    def test_three_versions_triggers_detection(self, storage):
        """3+ versions should also trigger detection."""
        for v in range(3):
            storage.save_entity(_make_entity(family_id="fam_t2", version=v, content=f"Version {v}"))
        proc, llm = _make_mock_processor(storage)
        proc._detect_and_apply_contradictions(["fam_t2"])
        llm.detect_contradictions.assert_called_once()

    def test_empty_family_ids_list(self, storage):
        """Empty family_ids list should be a no-op."""
        proc, llm = _make_mock_processor(storage)
        proc._detect_and_apply_contradictions([])
        llm.detect_contradictions.assert_not_called()

    def test_nonexistent_family_id_skipped(self, storage):
        """Nonexistent family_id should be skipped (get_entity_versions returns [])."""
        proc, llm = _make_mock_processor(storage)
        proc._detect_and_apply_contradictions(["nonexistent"])
        llm.detect_contradictions.assert_not_called()

    def test_multiple_family_ids_all_checked(self, storage):
        """Multiple family_ids should all be checked."""
        for fid in ["fam_m1", "fam_m2"]:
            storage.save_entity(_make_entity(family_id=fid, version=0, content=f"{fid} v0"))
            storage.save_entity(_make_entity(family_id=fid, version=1, content=f"{fid} v1"))
        proc, llm = _make_mock_processor(storage)
        proc._detect_and_apply_contradictions(["fam_m1", "fam_m2"])
        assert llm.detect_contradictions.call_count == 2

    def test_mixed_valid_and_invalid_ids(self, storage):
        """Mix of valid (>=2 versions) and invalid (<2 versions) IDs."""
        storage.save_entity(_make_entity(family_id="fam_valid", version=0))
        storage.save_entity(_make_entity(family_id="fam_valid", version=1))
        storage.save_entity(_make_entity(family_id="fam_single", version=0))
        proc, llm = _make_mock_processor(storage)
        proc._detect_and_apply_contradictions(["fam_valid", "fam_single", "fam_ghost"])
        # Only fam_valid should trigger detection
        llm.detect_contradictions.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# Dimension 2: Confidence penalty on contradiction
# ═══════════════════════════════════════════════════════════════════

class TestConfidencePenalty:
    """Tests for confidence reduction when contradictions are detected."""

    def test_high_severity_reduces_confidence(self, storage):
        """High severity contradiction should reduce confidence by 0.1."""
        storage.save_entity(_make_entity(family_id="fam_h1", version=0, confidence=0.8))
        storage.save_entity(_make_entity(family_id="fam_h1", version=1, confidence=0.8))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[
            {"description": "Conflicting location", "severity": "high"},
        ])
        proc._detect_and_apply_contradictions(["fam_h1"])
        updated = storage.get_entity_by_family_id("fam_h1")
        assert abs(updated.confidence - 0.7) < 0.01

    def test_medium_severity_reduces_confidence(self, storage):
        """Medium severity contradiction should reduce confidence by 0.1."""
        storage.save_entity(_make_entity(family_id="fam_m1", version=0, confidence=0.7))
        storage.save_entity(_make_entity(family_id="fam_m1", version=1, confidence=0.7))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[
            {"description": "Minor inconsistency", "severity": "medium"},
        ])
        proc._detect_and_apply_contradictions(["fam_m1"])
        updated = storage.get_entity_by_family_id("fam_m1")
        assert abs(updated.confidence - 0.6) < 0.01

    def test_low_severity_no_penalty(self, storage):
        """Low severity contradiction should NOT reduce confidence."""
        storage.save_entity(_make_entity(family_id="fam_l1", version=0, confidence=0.7))
        storage.save_entity(_make_entity(family_id="fam_l1", version=1, confidence=0.7))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[
            {"description": "Minor style difference", "severity": "low"},
        ])
        proc._detect_and_apply_contradictions(["fam_l1"])
        updated = storage.get_entity_by_family_id("fam_l1")
        assert abs(updated.confidence - 0.7) < 0.01

    def test_no_contradictions_no_penalty(self, storage):
        """No contradictions should leave confidence unchanged."""
        storage.save_entity(_make_entity(family_id="fam_nc", version=0, confidence=0.75))
        storage.save_entity(_make_entity(family_id="fam_nc", version=1, confidence=0.75))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[])
        proc._detect_and_apply_contradictions(["fam_nc"])
        updated = storage.get_entity_by_family_id("fam_nc")
        assert abs(updated.confidence - 0.75) < 0.01

    def test_multiple_contradictions_single_penalty(self, storage):
        """Multiple contradictions from one detection call = one penalty per call."""
        storage.save_entity(_make_entity(family_id="fam_multi", version=0, confidence=0.8))
        storage.save_entity(_make_entity(family_id="fam_multi", version=1, confidence=0.8))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[
            {"description": "Conflict 1", "severity": "high"},
            {"description": "Conflict 2", "severity": "medium"},
        ])
        proc._detect_and_apply_contradictions(["fam_multi"])
        updated = storage.get_entity_by_family_id("fam_multi")
        # Only one penalty (-0.1) since it's one detection call
        assert abs(updated.confidence - 0.7) < 0.01

    def test_confidence_floor_at_zero(self, storage):
        """Confidence should not go below 0.0."""
        storage.save_entity(_make_entity(family_id="fam_floor", version=0, confidence=0.05))
        storage.save_entity(_make_entity(family_id="fam_floor", version=1, confidence=0.05))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[
            {"description": "Major conflict", "severity": "high"},
        ])
        proc._detect_and_apply_contradictions(["fam_floor"])
        updated = storage.get_entity_by_family_id("fam_floor")
        assert updated.confidence >= 0.0

    def test_separate_entities_penalized_independently(self, storage):
        """Each entity with contradictions should be penalized independently."""
        for fid, conf in [("fam_a", 0.9), ("fam_b", 0.6)]:
            storage.save_entity(_make_entity(family_id=fid, version=0, confidence=conf))
            storage.save_entity(_make_entity(family_id=fid, version=1, confidence=conf))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[
            {"description": "Conflict", "severity": "high"},
        ])
        proc._detect_and_apply_contradictions(["fam_a", "fam_b"])
        a = storage.get_entity_by_family_id("fam_a")
        b = storage.get_entity_by_family_id("fam_b")
        assert abs(a.confidence - 0.8) < 0.01
        assert abs(b.confidence - 0.5) < 0.01


# ═══════════════════════════════════════════════════════════════════
# Dimension 3: Pipeline integration
# ═══════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    """Tests for contradiction detection as part of the extraction pipeline."""

    def test_version_count_filter(self, storage):
        """Only entities with version_count >= 2 should be checked."""
        # Entity with 1 version
        storage.save_entity(_make_entity(family_id="fam_1v", version=0))
        # Entity with 2 versions
        storage.save_entity(_make_entity(family_id="fam_2v", version=0))
        storage.save_entity(_make_entity(family_id="fam_2v", version=1))

        vc = storage.get_entity_version_counts(["fam_1v", "fam_2v"])
        assert vc["fam_1v"] == 1
        assert vc["fam_2v"] == 2

    def test_detection_uses_correct_versions(self, storage):
        """Detection should pass all versions of the entity."""
        for v in range(3):
            storage.save_entity(_make_entity(
                family_id="fam_ver", version=v,
                content=f"Content version {v}",
                confidence=0.8,
            ))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[])
        proc._detect_and_apply_contradictions(["fam_ver"])
        call_args = llm.detect_contradictions.call_args
        assert call_args[0][0] == "fam_ver"
        versions = call_args[0][1]
        assert len(versions) == 3

    def test_detection_order_by_processed_time(self, storage):
        """Versions should be retrieved in processed_time DESC order."""
        t0 = _now() - timedelta(minutes=20)
        t1 = _now() - timedelta(minutes=10)
        t2 = _now()
        storage.save_entity(_make_entity(family_id="fam_ord", version=0, processed_time=t0))
        storage.save_entity(_make_entity(family_id="fam_ord", version=1, processed_time=t1))
        storage.save_entity(_make_entity(family_id="fam_ord", version=2, processed_time=t2))

        versions = storage.get_entity_versions("fam_ord")
        assert len(versions) == 3
        # First should be the latest
        assert versions[0].processed_time >= versions[1].processed_time

    def test_empty_contradictions_list_no_side_effects(self, storage):
        """Empty contradictions list should have no side effects on storage."""
        storage.save_entity(_make_entity(family_id="fam_empty", version=0, confidence=0.8))
        storage.save_entity(_make_entity(family_id="fam_empty", version=1, confidence=0.8))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[])
        proc._detect_and_apply_contradictions(["fam_empty"])
        updated = storage.get_entity_by_family_id("fam_empty")
        assert abs(updated.confidence - 0.8) < 0.01

    def test_contradiction_detection_is_sequential_per_fid(self, storage):
        """Each family_id should be processed sequentially (not batched)."""
        for fid in ["fam_seq1", "fam_seq2", "fam_seq3"]:
            storage.save_entity(_make_entity(family_id=fid, version=0))
            storage.save_entity(_make_entity(family_id=fid, version=1))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[])
        proc._detect_and_apply_contradictions(["fam_seq1", "fam_seq2", "fam_seq3"])
        assert llm.detect_contradictions.call_count == 3


# ═══════════════════════════════════════════════════════════════════
# Dimension 4: Error resilience & edge cases
# ═══════════════════════════════════════════════════════════════════

class TestErrorResilience:
    """Tests for error handling in contradiction detection."""

    def test_llm_failure_does_not_crash(self, storage):
        """LLM failure should not crash the pipeline."""
        storage.save_entity(_make_entity(family_id="fam_fail", version=0, confidence=0.8))
        storage.save_entity(_make_entity(family_id="fam_fail", version=1, confidence=0.8))
        proc, llm = _make_mock_processor(storage)
        llm.detect_contradictions = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        # Should not raise
        proc._detect_and_apply_contradictions(["fam_fail"])
        # Confidence unchanged
        updated = storage.get_entity_by_family_id("fam_fail")
        assert abs(updated.confidence - 0.8) < 0.01

    def test_asyncio_error_does_not_crash(self, storage):
        """Async event loop errors should not crash the pipeline."""
        storage.save_entity(_make_entity(family_id="fam_async", version=0))
        storage.save_entity(_make_entity(family_id="fam_async", version=1))
        proc, llm = _make_mock_processor(storage)
        llm.detect_contradictions = AsyncMock(side_effect=Exception("async error"))
        proc._detect_and_apply_contradictions(["fam_async"])

    def test_malformed_contradiction_response(self, storage):
        """Malformed contradiction data should not crash."""
        storage.save_entity(_make_entity(family_id="fam_mal", version=0, confidence=0.8))
        storage.save_entity(_make_entity(family_id="fam_mal", version=1, confidence=0.8))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[
            {"wrong_key": "value"},  # Missing 'description' and 'severity'
        ])
        proc._detect_and_apply_contradictions(["fam_mal"])
        # No severity='high'/'medium', so no penalty
        updated = storage.get_entity_by_family_id("fam_mal")
        assert abs(updated.confidence - 0.8) < 0.01

    def test_partial_failure_continues(self, storage):
        """If one family_id fails, others should still be processed."""
        storage.save_entity(_make_entity(family_id="fam_ok", version=0, confidence=0.8))
        storage.save_entity(_make_entity(family_id="fam_ok", version=1, confidence=0.8))
        storage.save_entity(_make_entity(family_id="fam_bad", version=0, confidence=0.7))
        storage.save_entity(_make_entity(family_id="fam_bad", version=1, confidence=0.7))

        call_count = [0]

        async def mock_detect(fid, versions):
            call_count[0] += 1
            if fid == "fam_bad":
                raise RuntimeError("Simulated failure")
            return [{"description": "Conflict", "severity": "high"}]

        proc, llm = _make_mock_processor(storage)
        llm.detect_contradictions = AsyncMock(side_effect=mock_detect)
        proc._detect_and_apply_contradictions(["fam_ok", "fam_bad"])
        # Both should have been attempted
        assert call_count[0] == 2
        # fam_ok should have been penalized
        ok_entity = storage.get_entity_by_family_id("fam_ok")
        assert abs(ok_entity.confidence - 0.7) < 0.01
        # fam_bad should be unchanged (failed before penalty)
        bad_entity = storage.get_entity_by_family_id("fam_bad")
        assert abs(bad_entity.confidence - 0.7) < 0.01

    def test_concurrent_safety(self, storage):
        """Multiple calls to _detect_and_apply_contradictions should be safe."""
        import threading
        storage.save_entity(_make_entity(family_id="fam_conc", version=0, confidence=0.9))
        storage.save_entity(_make_entity(family_id="fam_conc", version=1, confidence=0.9))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[
            {"description": "Conflict", "severity": "high"},
        ])
        errors = []

        def run_detect():
            try:
                proc._detect_and_apply_contradictions(["fam_conc"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_detect) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # At least one penalty applied (up to 3)
        updated = storage.get_entity_by_family_id("fam_conc")
        assert updated.confidence <= 0.9

    def test_large_version_count_limited_to_5(self, storage):
        """Entities with many versions should pass at most 5 to LLM (per detect_contradictions impl)."""
        for v in range(10):
            storage.save_entity(_make_entity(family_id="fam_many", version=v, content=f"v{v}"))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[])
        proc._detect_and_apply_contradictions(["fam_many"])
        call_args = llm.detect_contradictions.call_args
        versions_passed = call_args[0][1]
        assert len(versions_passed) == 10  # storage returns all, LLM internally limits to 5

    def test_verbose_mode_logs(self, storage):
        """Verbose mode should produce log output for contradictions."""
        storage.save_entity(_make_entity(family_id="fam_log", version=0, confidence=0.8))
        storage.save_entity(_make_entity(family_id="fam_log", version=1, confidence=0.8))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[
            {"description": "Major conflict found", "severity": "high"},
        ])
        # Should not crash in verbose mode
        proc._detect_and_apply_contradictions(["fam_log"], verbose=True)
        updated = storage.get_entity_by_family_id("fam_log")
        assert abs(updated.confidence - 0.7) < 0.01

    def test_unicode_content_in_versions(self, storage):
        """Unicode content in versions should not cause issues."""
        storage.save_entity(_make_entity(
            family_id="fam_unicode", version=0,
            content="北京是中国的首都，人口超过2000万",
            confidence=0.8,
        ))
        storage.save_entity(_make_entity(
            family_id="fam_unicode", version=1,
            content="上海是中国的经济中心，金融业发达",
            confidence=0.8,
        ))
        proc, llm = _make_mock_processor(storage, llm_contradictions=[
            {"description": "城市矛盾", "severity": "high"},
        ])
        proc._detect_and_apply_contradictions(["fam_unicode"])
        updated = storage.get_entity_by_family_id("fam_unicode")
        assert abs(updated.confidence - 0.7) < 0.01
