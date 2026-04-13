"""
Tests for automatic summary evolution in the remember pipeline.

Verifies that entities with sufficient version history get their summaries
auto-evolved via LLM, while new entities and low-version entities are skipped.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processor.storage.manager import StorageManager
from processor.models import Entity
from processor.pipeline.extraction import _ExtractionMixin


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_entity(
    family_id="fam_1",
    name="TestEntity",
    content="A test entity.",
    confidence=0.7,
    processed_time=None,
    version=0,
    summary=None,
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
        summary=summary or content[:80],
    )


@pytest.fixture
def storage(tmp_path):
    s = StorageManager(storage_path=str(tmp_path / "test_summary.db"))
    yield s
    s.close()


def _make_mock_processor(storage, evolve_summary_result=None):
    """Create a mock processor with _ExtractionMixin methods and real storage."""
    llm = MagicMock()

    # Make evolve_entity_summary return a coroutine
    async def _evolve(entity, old_version=None):
        return evolve_summary_result or f"Evolved summary for {entity.name}"

    llm.evolve_entity_summary = MagicMock(side_effect=_evolve)

    proc = type("MockProcessor", (_ExtractionMixin,), {
        "storage": storage,
        "llm_client": llm,
    })()
    return proc, llm


# ═══════════════════════════════════════════════════════════════════
# Dimension 1: Threshold logic
# ═══════════════════════════════════════════════════════════════════

class TestSummaryEvolutionThreshold:
    """Tests for the version count threshold that triggers summary evolution."""

    def test_single_version_skipped(self, storage):
        """Entities with < 3 versions should NOT trigger summary evolution."""
        storage.save_entity(_make_entity(family_id="fam_1v", version=0))
        proc, llm = _make_mock_processor(storage)
        proc._auto_evolve_summaries(["fam_1v"])
        llm.evolve_entity_summary.assert_not_called()

    def test_two_versions_skipped(self, storage):
        """Entities with 2 versions should NOT trigger summary evolution."""
        storage.save_entity(_make_entity(family_id="fam_2v", version=0))
        storage.save_entity(_make_entity(family_id="fam_2v", version=1))
        proc, llm = _make_mock_processor(storage)
        proc._auto_evolve_summaries(["fam_2v"])
        llm.evolve_entity_summary.assert_not_called()

    def test_three_versions_triggers(self, storage):
        """Entities with 3 versions SHOULD trigger summary evolution."""
        for v in range(3):
            storage.save_entity(_make_entity(
                family_id="fam_3v", version=v, content=f"Version {v} content"
            ))
        proc, llm = _make_mock_processor(storage)
        proc._auto_evolve_summaries(["fam_3v"])
        llm.evolve_entity_summary.assert_called_once()

    def test_five_versions_triggers(self, storage):
        """Entities with 5 versions should also trigger."""
        for v in range(5):
            storage.save_entity(_make_entity(
                family_id="fam_5v", version=v, content=f"Content v{v}"
            ))
        proc, llm = _make_mock_processor(storage)
        proc._auto_evolve_summaries(["fam_5v"])
        llm.evolve_entity_summary.assert_called_once()

    def test_empty_family_ids(self, storage):
        """Empty list should be a no-op."""
        proc, llm = _make_mock_processor(storage)
        proc._auto_evolve_summaries([])
        llm.evolve_entity_summary.assert_not_called()

    def test_nonexistent_family_id_skipped(self, storage):
        """Nonexistent family_id should be skipped gracefully."""
        proc, llm = _make_mock_processor(storage)
        proc._auto_evolve_summaries(["nonexistent"])
        llm.evolve_entity_summary.assert_not_called()

    def test_mixed_valid_and_invalid_ids(self, storage):
        """Mix of valid (>=3) and invalid (<3) IDs — only valid should trigger."""
        # Valid: 3 versions
        for v in range(3):
            storage.save_entity(_make_entity(
                family_id="fam_valid", version=v, content=f"v{v}"
            ))
        # Invalid: 1 version
        storage.save_entity(_make_entity(family_id="fam_short", version=0))

        proc, llm = _make_mock_processor(storage)
        proc._auto_evolve_summaries(["fam_valid", "fam_short", "fam_ghost"])
        llm.evolve_entity_summary.assert_called_once()
        call_args = llm.evolve_entity_summary.call_args
        assert call_args[0][0].family_id == "fam_valid"


# ═══════════════════════════════════════════════════════════════════
# Dimension 2: Summary update correctness
# ═══════════════════════════════════════════════════════════════════

class TestSummaryEvolutionUpdate:
    """Tests for the summary being correctly updated in storage."""

    def test_summary_updated_in_storage(self, storage):
        """Summary should be updated after evolution."""
        for v in range(3):
            storage.save_entity(_make_entity(
                family_id="fam_update", version=v,
                content=f"Version {v} content about quantum computing",
                summary=f"Short summary v{v}",
            ))
        evolved = "量子计算是利用量子力学原理进行信息处理的计算范式"
        proc, llm = _make_mock_processor(storage, evolve_summary_result=evolved)
        proc._auto_evolve_summaries(["fam_update"])

        updated = storage.get_entity_by_family_id("fam_update")
        assert updated.summary == evolved

    def test_evolve_called_with_correct_args(self, storage):
        """evolve_entity_summary should be called with current and old version."""
        for v in range(3):
            storage.save_entity(_make_entity(
                family_id="fam_args", version=v, content=f"Content v{v}"
            ))
        proc, llm = _make_mock_processor(storage)
        proc._auto_evolve_summaries(["fam_args"])

        call_args = llm.evolve_entity_summary.call_args
        current = call_args[0][0]
        old = call_args[0][1]
        assert current.family_id == "fam_args"
        # Most recent version should be first
        versions = storage.get_entity_versions("fam_args")
        assert current.absolute_id == versions[0].absolute_id
        assert old.absolute_id == versions[1].absolute_id


# ═══════════════════════════════════════════════════════════════════
# Dimension 3: Skip logic
# ═══════════════════════════════════════════════════════════════════

class TestSummaryEvolutionSkip:
    """Tests for when summary evolution should be skipped."""

    def test_skip_when_existing_summary_is_long(self, storage):
        """Entities with existing summary > 50 chars and unchanged content should be skipped."""
        long_summary = "这是一个非常详细的综合性摘要，全面描述了该实体的所有重要特征、历史演变过程和与其他概念的深层关联关系（共51字以上）"
        for v in range(3):
            storage.save_entity(_make_entity(
                family_id="fam_longsum", version=v,
                content="Same content",  # Same content across versions
                summary=long_summary if v == 2 else f"Summary v{v}",
            ))
        proc, llm = _make_mock_processor(storage)
        proc._auto_evolve_summaries(["fam_longsum"])
        # Content unchanged and existing summary is long → skip
        llm.evolve_entity_summary.assert_not_called()

    def test_trigger_when_existing_summary_is_short(self, storage):
        """Entities with short summary should still trigger evolution even with 3+ versions."""
        for v in range(3):
            storage.save_entity(_make_entity(
                family_id="fam_shortsum", version=v,
                content=f"Content about topic v{v}",
                summary=f"v{v}",  # Very short summary
            ))
        proc, llm = _make_mock_processor(storage)
        proc._auto_evolve_summaries(["fam_shortsum"])
        llm.evolve_entity_summary.assert_called_once()

    def test_trigger_when_content_changed_even_with_long_summary(self, storage):
        """Content changed significantly should trigger evolution even with long existing summary."""
        long_summary = "这是一个非常详细的综合性摘要，全面描述了该实体的所有重要特征、历史演变和深层关联关系（共51字以上）"
        for v in range(3):
            storage.save_entity(_make_entity(
                family_id="fam_changed", version=v,
                content=f"Different content for version {v}",  # Different content
                summary=long_summary if v == 2 else f"Summary v{v}",
            ))
        proc, llm = _make_mock_processor(storage)
        proc._auto_evolve_summaries(["fam_changed"])
        llm.evolve_entity_summary.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# Dimension 4: Error resilience
# ═══════════════════════════════════════════════════════════════════

class TestSummaryEvolutionErrors:
    """Tests for error handling in summary evolution."""

    def test_llm_failure_does_not_crash(self, storage):
        """LLM failure should not crash the pipeline."""
        for v in range(3):
            storage.save_entity(_make_entity(
                family_id="fam_fail", version=v, content=f"Content v{v}"
            ))
        proc, llm = _make_mock_processor(storage)

        async def _fail(entity, old_version=None):
            raise RuntimeError("LLM timeout")

        llm.evolve_entity_summary = MagicMock(side_effect=_fail)
        # Should not raise
        proc._auto_evolve_summaries(["fam_fail"])

    def test_partial_failure_continues(self, storage):
        """If one entity fails, others should still be processed."""
        for fid in ["fam_ok", "fam_bad"]:
            for v in range(3):
                storage.save_entity(_make_entity(
                    family_id=fid, version=v, content=f"{fid} content v{v}"
                ))

        call_count = [0]

        async def _conditional(entity, old_version=None):
            call_count[0] += 1
            if entity.family_id == "fam_bad":
                raise RuntimeError("Simulated failure")
            return f"Summary for {entity.name}"

        proc, llm = _make_mock_processor(storage)
        llm.evolve_entity_summary = MagicMock(side_effect=_conditional)
        proc._auto_evolve_summaries(["fam_ok", "fam_bad"])

        # Both should have been attempted
        assert call_count[0] == 2
        # fam_ok should have been updated
        ok_entity = storage.get_entity_by_family_id("fam_ok")
        assert "Summary" in ok_entity.summary

    def test_empty_llm_response_uses_fallback(self, storage):
        """Empty LLM response should not overwrite the summary."""
        for v in range(3):
            storage.save_entity(_make_entity(
                family_id="fam_empty", version=v,
                content=f"Content v{v}",
                summary="Original summary",
            ))

        async def _empty(entity, old_version=None):
            return ""  # Empty response

        proc, llm = _make_mock_processor(storage)
        llm.evolve_entity_summary = MagicMock(side_effect=_empty)
        proc._auto_evolve_summaries(["fam_empty"])

        # Summary should NOT be overwritten with empty string
        updated = storage.get_entity_by_family_id("fam_empty")
        assert updated.summary == "Original summary"
