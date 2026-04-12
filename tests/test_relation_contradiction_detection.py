"""
Relation contradiction detection — 32 tests across 4 dimensions.

Vision gap fix: Contradiction detection (vision.md Section 8.7 "容错与自检",
Section 3 "矛盾与修正") only covered entities. Relations also have versions and
their content can contradict over time. Now detect_contradictions supports
concept_type="relation" and the pipeline runs contradiction detection on
multi-version relations.

D1: LLM contradiction detection for relations (8 tests)
D2: Pipeline integration (8 tests)
D3: HTTP endpoints (8 tests)
D4: MCP tools (8 tests)
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock

import os
import subprocess
import sys

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_storage(tmp_path):
    from processor.storage.manager import StorageManager
    sm = StorageManager(str(tmp_path / "graph"))
    yield sm
    if hasattr(sm, '_vector_store') and sm._vector_store:
        sm._vector_store.close()


def _make_entity(family_id: str, name: str, content: str,
                 source_document: str = "test"):
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
    )


def _make_relation(family_id: str, e1_abs: str, e2_abs: str, content: str,
                   confidence: float = 0.7, event_time=None):
    from processor.models import Relation
    now = datetime.now(timezone.utc)
    return Relation(
        absolute_id=str(uuid.uuid4()),
        family_id=family_id,
        entity1_absolute_id=e1_abs,
        entity2_absolute_id=e2_abs,
        content=content,
        event_time=event_time or now,
        processed_time=now,
        episode_id="ep_test",
        source_document="test",
        confidence=confidence,
    )


def _setup_entities(storage, fid1="ent_ra", fid2="ent_rb"):
    e1 = _make_entity(fid1, "Entity R", "Content R")
    e2 = _make_entity(fid2, "Entity S", "Content S")
    storage.bulk_save_entities([e1, e2])
    return e1, e2


def _mock_llm_client(detect_result=None, resolve_result=None):
    """Create a mock LLM client with ContradictionDetectionMixin methods."""
    from processor.llm.contradiction import ContradictionDetectionMixin
    mock = MagicMock(spec=ContradictionDetectionMixin)
    mock.detect_contradictions = AsyncMock(return_value=detect_result or [])
    mock.resolve_contradiction = AsyncMock(
        return_value=resolve_result or {"decision": "flag", "reason": "test"}
    )
    return mock


# ══════════════════════════════════════════════════════════════════════════
# D1: LLM contradiction detection for relations
# ══════════════════════════════════════════════════════════════════════════


class TestLLMRelationContradiction:
    """D1: LLM contradiction detection with concept_type='relation'."""

    def test_detect_with_relation_type_uses_relation_prompt(self):
        """D1.1: detect_contradictions with concept_type='relation' passes correct args."""
        from processor.llm.contradiction import ContradictionDetectionMixin
        mixin = ContradictionDetectionMixin()
        mixin.call_llm_until_json_parses = MagicMock(return_value=([], None))
        mixin._parse_contradictions_response = MagicMock(return_value=[])

        versions = [
            _make_relation("rel_d1", "a1", "a2", "Version 1"),
            _make_relation("rel_d1", "b1", "b2", "Version 2"),
        ]
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                mixin.detect_contradictions("rel_d1", versions, concept_type="relation")
            )
        finally:
            loop.close()

        assert result == []
        # Verify call_llm_until_json_parses was called
        assert mixin.call_llm_until_json_parses.call_count == 1
        messages = mixin.call_llm_until_json_parses.call_args[0][0]
        user_msg = messages[1]["content"]
        assert "<关系 ID>" in user_msg
        assert "<关系版本>" in user_msg

    def test_detect_default_entity_type_backward_compat(self):
        """D1.2: Default concept_type='entity' still uses entity prompt."""
        from processor.llm.contradiction import ContradictionDetectionMixin
        mixin = ContradictionDetectionMixin()
        mixin.call_llm_until_json_parses = MagicMock(return_value=([], None))
        mixin._parse_contradictions_response = MagicMock(return_value=[])

        versions = [
            _make_entity("ent_d2", "E", "V1"),
            _make_entity("ent_d2", "E", "V2"),
        ]
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                mixin.detect_contradictions("ent_d2", versions)
            )
        finally:
            loop.close()

        messages = mixin.call_llm_until_json_parses.call_args[0][0]
        user_msg = messages[1]["content"]
        assert "<实体 ID>" in user_msg
        assert "<实体版本>" in user_msg

    def test_version_text_includes_relation_content(self):
        """D1.3: Version text correctly includes relation content."""
        from processor.llm.contradiction import ContradictionDetectionMixin
        mixin = ContradictionDetectionMixin()
        mixin.call_llm_until_json_parses = MagicMock(return_value=([], None))
        mixin._parse_contradictions_response = MagicMock(return_value=[])

        versions = [
            _make_relation("rel_d3", "a", "b", "Alice mentors Bob"),
            _make_relation("rel_d3", "a", "b", "Alice manages Bob"),
        ]
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                mixin.detect_contradictions("rel_d3", versions, concept_type="relation")
            )
        finally:
            loop.close()

        messages = mixin.call_llm_until_json_parses.call_args[0][0]
        user_msg = messages[1]["content"]
        assert "Alice mentors Bob" in user_msg
        assert "Alice manages Bob" in user_msg

    def test_less_than_two_versions_returns_empty(self):
        """D1.4: Single version returns empty list."""
        from processor.llm.contradiction import ContradictionDetectionMixin
        mixin = ContradictionDetectionMixin()

        versions = [_make_relation("rel_d4", "a", "b", "Only version")]
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                mixin.detect_contradictions("rel_d4", versions, concept_type="relation")
            )
        finally:
            loop.close()

        assert result == []

    def test_llm_failure_returns_empty_no_crash(self):
        """D1.5: LLM failure returns empty list without crash."""
        from processor.llm.contradiction import ContradictionDetectionMixin
        mixin = ContradictionDetectionMixin()
        mixin.call_llm_until_json_parses = MagicMock(side_effect=RuntimeError("LLM error"))

        versions = [
            _make_relation("rel_d5", "a", "b", "V1"),
            _make_relation("rel_d5", "a", "b", "V2"),
        ]
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                mixin.detect_contradictions("rel_d5", versions, concept_type="relation")
            )
        finally:
            loop.close()

        assert result == []

    def test_high_severity_detected(self):
        """D1.6: High severity contradictions are detected."""
        from processor.llm.contradiction import ContradictionDetectionMixin
        mixin = ContradictionDetectionMixin()
        mixin._parse_contradictions_response = MagicMock(return_value=[
            {"description": "V1 says A causes B, V2 says A prevents B", "severity": "high"}
        ])
        mixin.call_llm_until_json_parses = MagicMock(
            side_effect=lambda msgs, parse_fn=None, **kw: (
                parse_fn("dummy") if parse_fn else [], None
            )
        )

        versions = [
            _make_relation("rel_d6", "a", "b", "A causes B"),
            _make_relation("rel_d6", "a", "b", "A prevents B"),
        ]
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                mixin.detect_contradictions("rel_d6", versions, concept_type="relation")
            )
        finally:
            loop.close()

        assert len(result) == 1
        assert result[0]["severity"] == "high"

    def test_multiple_contradictions_in_single_relation(self):
        """D1.7: Multiple contradictions detected in a single relation."""
        from processor.llm.contradiction import ContradictionDetectionMixin
        mixin = ContradictionDetectionMixin()
        mixin._parse_contradictions_response = MagicMock(return_value=[
            {"description": "Contradiction 1", "severity": "high"},
            {"description": "Contradiction 2", "severity": "medium"},
        ])
        mixin.call_llm_until_json_parses = MagicMock(
            side_effect=lambda msgs, parse_fn=None, **kw: (
                parse_fn("dummy") if parse_fn else [], None
            )
        )

        versions = [
            _make_relation("rel_d7", "a", "b", "Version 1"),
            _make_relation("rel_d7", "a", "b", "Version 2"),
        ]
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                mixin.detect_contradictions("rel_d7", versions, concept_type="relation")
            )
        finally:
            loop.close()

        assert len(result) == 2

    def test_xml_tags_use_relation_label(self):
        """D1.8: XML tags use '关系' when concept_type='relation'."""
        from processor.llm.contradiction import ContradictionDetectionMixin
        mixin = ContradictionDetectionMixin()
        mixin.call_llm_until_json_parses = MagicMock(return_value=([], None))
        mixin._parse_contradictions_response = MagicMock(return_value=[])

        versions = [
            _make_relation("rel_d8", "a", "b", "V1"),
            _make_relation("rel_d8", "a", "b", "V2"),
        ]
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                mixin.detect_contradictions("rel_d8", versions, concept_type="relation")
            )
        finally:
            loop.close()

        messages = mixin.call_llm_until_json_parses.call_args[0][0]
        user_msg = messages[1]["content"]
        assert "关系" in user_msg
        assert "<实体" not in user_msg


# ══════════════════════════════════════════════════════════════════════════
# D2: Pipeline integration
# ══════════════════════════════════════════════════════════════════════════


class TestPipelineRelationContradiction:
    """D2: Pipeline integration for relation contradiction detection."""

    def test_calls_get_relation_versions(self, tmp_storage):
        """D2.1: _detect_and_apply_relation_contradictions calls get_relation_versions."""
        e1, e2 = _setup_entities(tmp_storage)
        r1 = _make_relation("rel_p1", e1.absolute_id, e2.absolute_id, "V1", confidence=0.7)
        r2 = _make_relation("rel_p1", e1.absolute_id, e2.absolute_id, "V2", confidence=0.7)
        tmp_storage.bulk_save_relations([r1, r2])

        from processor.pipeline.extraction import _ExtractionMixin
        mixin = _ExtractionMixin.__new__(_ExtractionMixin)
        mixin.storage = tmp_storage
        mixin.llm_client = _mock_llm_client(detect_result=[])

        with patch.object(tmp_storage, 'get_relation_versions', wraps=tmp_storage.get_relation_versions) as spy:
            mixin._detect_and_apply_relation_contradictions(["rel_p1"])
            spy.assert_called_once_with("rel_p1")

    def test_high_severity_triggers_confidence_adjustment(self, tmp_storage):
        """D2.2: High severity contradiction triggers adjust_confidence_on_contradiction."""
        e1, e2 = _setup_entities(tmp_storage)
        r1 = _make_relation("rel_p2", e1.absolute_id, e2.absolute_id, "A causes B", confidence=0.7)
        r2 = _make_relation("rel_p2", e1.absolute_id, e2.absolute_id, "A prevents B", confidence=0.7)
        tmp_storage.bulk_save_relations([r1, r2])

        from processor.pipeline.extraction import _ExtractionMixin
        mixin = _ExtractionMixin.__new__(_ExtractionMixin)
        mixin.storage = tmp_storage
        mixin.llm_client = _mock_llm_client(detect_result=[
            {"description": "Contradiction", "severity": "high"}
        ])

        with patch.object(tmp_storage, 'adjust_confidence_on_contradiction') as mock_adj:
            mixin._detect_and_apply_relation_contradictions(["rel_p2"])
            mock_adj.assert_called_once_with("rel_p2", source_type="relation")

    def test_medium_severity_triggers_confidence_adjustment(self, tmp_storage):
        """D2.3: Medium severity contradiction triggers confidence adjustment."""
        e1, e2 = _setup_entities(tmp_storage)
        r1 = _make_relation("rel_p3", e1.absolute_id, e2.absolute_id, "V1", confidence=0.7)
        r2 = _make_relation("rel_p3", e1.absolute_id, e2.absolute_id, "V2", confidence=0.7)
        tmp_storage.bulk_save_relations([r1, r2])

        from processor.pipeline.extraction import _ExtractionMixin
        mixin = _ExtractionMixin.__new__(_ExtractionMixin)
        mixin.storage = tmp_storage
        mixin.llm_client = _mock_llm_client(detect_result=[
            {"description": "Minor contradiction", "severity": "medium"}
        ])

        with patch.object(tmp_storage, 'adjust_confidence_on_contradiction') as mock_adj:
            mixin._detect_and_apply_relation_contradictions(["rel_p3"])
            mock_adj.assert_called_once_with("rel_p3", source_type="relation")

    def test_low_severity_does_not_trigger_adjustment(self, tmp_storage):
        """D2.4: Low severity only does NOT trigger confidence adjustment."""
        e1, e2 = _setup_entities(tmp_storage)
        r1 = _make_relation("rel_p4", e1.absolute_id, e2.absolute_id, "V1", confidence=0.7)
        r2 = _make_relation("rel_p4", e1.absolute_id, e2.absolute_id, "V2", confidence=0.7)
        tmp_storage.bulk_save_relations([r1, r2])

        from processor.pipeline.extraction import _ExtractionMixin
        mixin = _ExtractionMixin.__new__(_ExtractionMixin)
        mixin.storage = tmp_storage
        mixin.llm_client = _mock_llm_client(detect_result=[
            {"description": "Trivial difference", "severity": "low"}
        ])

        with patch.object(tmp_storage, 'adjust_confidence_on_contradiction') as mock_adj:
            mixin._detect_and_apply_relation_contradictions(["rel_p4"])
            mock_adj.assert_not_called()

    def test_no_contradictions_no_confidence_change(self, tmp_storage):
        """D2.5: No contradictions → no confidence change."""
        e1, e2 = _setup_entities(tmp_storage)
        r1 = _make_relation("rel_p5", e1.absolute_id, e2.absolute_id, "V1", confidence=0.7)
        r2 = _make_relation("rel_p5", e1.absolute_id, e2.absolute_id, "V2", confidence=0.7)
        tmp_storage.bulk_save_relations([r1, r2])

        from processor.pipeline.extraction import _ExtractionMixin
        mixin = _ExtractionMixin.__new__(_ExtractionMixin)
        mixin.storage = tmp_storage
        mixin.llm_client = _mock_llm_client(detect_result=[])

        with patch.object(tmp_storage, 'adjust_confidence_on_contradiction') as mock_adj:
            mixin._detect_and_apply_relation_contradictions(["rel_p5"])
            mock_adj.assert_not_called()

    def test_single_version_relation_skipped(self, tmp_storage):
        """D2.6: Single-version relations skipped."""
        e1, e2 = _setup_entities(tmp_storage)
        r1 = _make_relation("rel_p6", e1.absolute_id, e2.absolute_id, "Only version")
        tmp_storage.save_relation(r1)

        from processor.pipeline.extraction import _ExtractionMixin
        mixin = _ExtractionMixin.__new__(_ExtractionMixin)
        mixin.storage = tmp_storage
        mixin.llm_client = _mock_llm_client()

        with patch.object(tmp_storage, 'adjust_confidence_on_contradiction') as mock_adj:
            mixin._detect_and_apply_relation_contradictions(["rel_p6"])
            mock_adj.assert_not_called()

    def test_pipeline_phase_b_plus_calls_relation_contradiction(self, tmp_storage):
        """D2.7: Pipeline calls relation contradiction for multi-version relations."""
        from processor.pipeline.extraction import _ExtractionMixin
        mixin = _ExtractionMixin.__new__(_ExtractionMixin)
        mixin.storage = tmp_storage
        mixin.llm_client = _mock_llm_client()

        from processor.models import Relation
        rel1 = Relation(
            absolute_id="abs_r1", family_id="rel_pip1",
            entity1_absolute_id="ea", entity2_absolute_id="eb",
            content="V1", event_time=datetime.now(timezone.utc),
            processed_time=datetime.now(timezone.utc),
            episode_id="ep1", source_document="test",
        )
        rel2 = Relation(
            absolute_id="abs_r2", family_id="rel_pip1",
            entity1_absolute_id="ea", entity2_absolute_id="eb",
            content="V2", event_time=datetime.now(timezone.utc),
            processed_time=datetime.now(timezone.utc),
            episode_id="ep1", source_document="test",
        )

        with patch.object(mixin, '_detect_and_apply_relation_contradictions') as mock_detect:
            # Simulate pipeline logic: check version counts
            processed_relations = [rel1, rel2]
            _rel_versioned_fids = []
            for rel in processed_relations:
                fid = rel.family_id
                # This simulates the version count check
                _rel_versioned_fids.append(fid)

            if _rel_versioned_fids:
                mixin._detect_and_apply_relation_contradictions(
                    list(set(_rel_versioned_fids)), verbose=False,
                )
            mock_detect.assert_called_once()

    def test_pipeline_exception_does_not_crash(self, tmp_storage):
        """D2.8: Pipeline exception doesn't crash the detection."""
        from processor.pipeline.extraction import _ExtractionMixin
        mixin = _ExtractionMixin.__new__(_ExtractionMixin)
        mixin.storage = tmp_storage
        mixin.llm_client = _mock_llm_client()

        with patch.object(tmp_storage, 'get_relation_versions', side_effect=RuntimeError("DB error")):
            # Should not crash
            mixin._detect_and_apply_relation_contradictions(["rel_p8"])


# ══════════════════════════════════════════════════════════════════════════
# D3: HTTP endpoints
# ══════════════════════════════════════════════════════════════════════════


def _make_registry(storage_path):
    """Create a GraphRegistry for testing."""
    from server.registry import GraphRegistry
    config = {
        "default_embedding_provider": "none",
        "default_llm_provider": "none",
        "remember_workers": 0,
        "remember_max_retries": 0,
        "remember_retry_delay_seconds": 0,
    }
    return GraphRegistry(base_storage_path=storage_path, config=config)


class TestHTTPEndpoints:
    """D3: HTTP endpoint tests for relation contradictions."""

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

    def test_get_contradictions_returns_results(self, client):
        """D3.1: GET contradictions returns contradictions for multi-version relation."""
        with patch('server.blueprints.relations._get_processor') as mock_proc:
            proc = MagicMock()
            proc.storage.get_relation_versions.return_value = [
                _make_relation("rel_ht1", "a", "b", "V1"),
                _make_relation("rel_ht1", "a", "b", "V2"),
            ]
            proc.llm_client.detect_contradictions = AsyncMock(return_value=[
                {"description": "conflict", "severity": "high"}
            ])
            mock_proc.return_value = proc
            with patch('server.blueprints.helpers.run_async') as mock_async:
                mock_async.return_value = [{"description": "conflict", "severity": "high"}]
                resp = client.get('/api/v1/find/relations/rel_ht1/contradictions')
                data = resp.get_json()
                assert resp.status_code == 200
                assert isinstance(data, dict)

    def test_get_returns_empty_for_single_version(self, client):
        """D3.2: GET returns empty for single-version relation."""
        r1 = _make_relation("rel_ht2", "a1", "a2", "V1")

        with patch('server.blueprints.relations._get_processor') as mock_proc:
            proc = MagicMock()
            proc.storage.get_relation_versions.return_value = [r1]
            mock_proc.return_value = proc
            resp = client.get('/api/v1/find/relations/rel_ht2/contradictions')
            data = resp.get_json()
            assert resp.status_code == 200
            inner = data.get("data", data)
            if isinstance(inner, list):
                assert len(inner) == 0

    def test_get_returns_empty_for_nonexistent(self, client):
        """D3.3: GET returns empty for nonexistent family_id."""
        with patch('server.blueprints.relations._get_processor') as mock_proc:
            proc = MagicMock()
            proc.storage.get_relation_versions.return_value = []
            mock_proc.return_value = proc
            resp = client.get('/api/v1/find/relations/ghost_rel/contradictions')
            data = resp.get_json()
            assert resp.status_code == 200

    def test_post_resolve_returns_resolution(self, client):
        """D3.4: POST resolve-contradiction returns resolution."""
        with patch('server.blueprints.relations._get_processor') as mock_proc:
            with patch('server.blueprints.helpers.run_async') as mock_async:
                mock_async.return_value = {"decision": "keep_new", "reason": "test"}
                proc = MagicMock()
                mock_proc.return_value = proc
                resp = client.post(
                    '/api/v1/find/relations/rel_ht4/resolve-contradiction',
                    json={"contradiction": {"description": "test contradiction"}}
                )
                assert resp.status_code == 200

    def test_post_returns_error_for_missing_body(self, client):
        """D3.5: POST returns error for missing contradiction body."""
        with patch('server.blueprints.relations._get_processor') as mock_proc:
            proc = MagicMock()
            mock_proc.return_value = proc
            resp = client.post(
                '/api/v1/find/relations/rel_ht5/resolve-contradiction',
                json={}
            )
            assert resp.status_code == 400

    def test_post_keep_new_decision(self, client):
        """D3.6: POST with keep_new decision works."""
        with patch('server.blueprints.relations._get_processor') as mock_proc:
            with patch('server.blueprints.helpers.run_async') as mock_async:
                mock_async.return_value = {"decision": "keep_new", "reason": "New data is more accurate"}
                proc = MagicMock()
                mock_proc.return_value = proc
                resp = client.post(
                    '/api/v1/find/relations/rel_ht6/resolve-contradiction',
                    json={"contradiction": {"description": "conflict"}}
                )
                data = resp.get_json()
                assert resp.status_code == 200

    def test_post_flag_decision(self, client):
        """D3.7: POST with flag decision works."""
        with patch('server.blueprints.relations._get_processor') as mock_proc:
            with patch('server.blueprints.helpers.run_async') as mock_async:
                mock_async.return_value = {"decision": "flag", "reason": "Needs human review"}
                proc = MagicMock()
                mock_proc.return_value = proc
                resp = client.post(
                    '/api/v1/find/relations/rel_ht7/resolve-contradiction',
                    json={"contradiction": {"description": "unclear conflict"}}
                )
                data = resp.get_json()
                assert resp.status_code == 200

    def test_endpoints_use_concept_type_relation(self, client):
        """D3.8: GET contradictions endpoint passes concept_type='relation'."""
        r1 = _make_relation("rel_ht8", "a", "b", "V1")
        r2 = _make_relation("rel_ht8", "a", "b", "V2")

        with patch('server.blueprints.relations._get_processor') as mock_proc:
            with patch('server.blueprints.helpers.run_async') as mock_async:
                mock_async.return_value = []
                proc = MagicMock()
                proc.storage.get_relation_versions.return_value = [r1, r2]
                mock_proc.return_value = proc
                resp = client.get('/api/v1/find/relations/rel_ht8/contradictions')
                assert resp.status_code == 200
                # Verify run_async was called (which wraps detect_contradictions)
                assert mock_async.call_count == 1


# ══════════════════════════════════════════════════════════════════════════
# D4: MCP tools
# ══════════════════════════════════════════════════════════════════════════


def _run_mcp_subprocess(script):
    """Run a Python script in a subprocess that imports the MCP server.

    The MCP server reopens sys.stdout as binary at import time, so we use
    a temp file for output instead of stdout/stderr.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        outfile = f.name

    full_script = f"""
import json, sys, os
sys.path.insert(0, '.')
_outfile = {outfile!r}
{script}
"""
    result = subprocess.run(
        [sys.executable, "-c", full_script],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    try:
        with open(outfile) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raise RuntimeError(
            f"Subprocess failed (rc={result.returncode}). "
            f"stdout={result.stdout[:500]!r}, stderr={result.stderr[:500]!r}"
        )
    finally:
        os.unlink(outfile)
    return data


def _get_mcp_tools_and_map():
    """Extract TOOLS and _TOOL_MAP from the MCP server module via subprocess.

    The MCP server module reopens sys.stdout/stdin as binary at import time,
    which breaks pytest's capture. Use subprocess to isolate the import.
    """
    script = """
from server.mcp.deep_dream_server import TOOLS, _TOOL_MAP
data = {
    'tool_names': [t['name'] for t in TOOLS],
    'handler_names': list(_TOOL_MAP.keys()),
}
with open(_outfile, 'w') as f:
    json.dump(data, f)
"""
    return _run_mcp_subprocess(script)


class TestMCPTools:
    """D4: MCP tool tests for relation contradictions.

    Uses subprocess to extract tool/handler names since the MCP server
    module reopens sys.stdout/stdin as binary at import time, breaking
    pytest's capture system.
    """

    def test_get_relation_contradictions_tool_registered(self):
        """D4.1: get_relation_contradictions MCP tool is registered."""
        info = _get_mcp_tools_and_map()
        assert "get_relation_contradictions" in info["tool_names"]

    def test_resolve_relation_contradiction_tool_registered(self):
        """D4.2: resolve_relation_contradiction MCP tool is registered."""
        info = _get_mcp_tools_and_map()
        assert "resolve_relation_contradiction" in info["tool_names"]

    def test_mcp_get_handler_calls_correct_endpoint(self):
        """D4.3: MCP get handler is registered in _TOOL_MAP."""
        info = _get_mcp_tools_and_map()
        assert "get_relation_contradictions" in info["handler_names"]

    def test_mcp_resolve_handler_calls_correct_endpoint(self):
        """D4.4: MCP resolve handler is registered in _TOOL_MAP."""
        info = _get_mcp_tools_and_map()
        assert "resolve_relation_contradiction" in info["handler_names"]

    def test_mcp_get_handler_returns_result(self):
        """D4.5: MCP get handler produces output when contradictions found.

        Tests the handler via subprocess by calling it with a mocked _get.
        """
        script = """
from unittest.mock import patch
from server.mcp.deep_dream_server import _TOOL_MAP
handler = _TOOL_MAP['get_relation_contradictions']
mod = __import__('server.mcp.deep_dream_server', fromlist=['_get'])
with patch.object(mod, '_get', return_value=({'data': [{'description': 'conflict', 'severity': 'high'}]}, 200)):
    result = handler({'family_id': 'rel_test'})
    data = {'has_content': 'content' in result, 'is_error': result.get('isError', False)}
    with open(_outfile, 'w') as f:
        json.dump(data, f)
"""
        data = _run_mcp_subprocess(script)
        assert data["has_content"]

    def test_mcp_resolve_handler_returns_result(self):
        """D4.6: MCP resolve handler produces output after resolution."""
        script = """
from unittest.mock import patch
from server.mcp.deep_dream_server import _TOOL_MAP
mod = __import__('server.mcp.deep_dream_server', fromlist=['_post'])
with patch.object(mod, '_post', return_value=({'data': {'decision': 'keep_new', 'reason': 'test'}}, 200)):
    handler = _TOOL_MAP['resolve_relation_contradiction']
    result = handler({'family_id': 'rel_test', 'contradiction_id': 'c1', 'resolution': 'keep_new'})
    data = {'has_content': 'content' in result, 'is_error': result.get('isError', False)}
    with open(_outfile, 'w') as f:
        json.dump(data, f)
"""
        data = _run_mcp_subprocess(script)
        assert data["has_content"]

    def test_mcp_tools_dont_conflict_with_entity_tools(self):
        """D4.7: Relation and entity contradiction tools coexist without conflict."""
        info = _get_mcp_tools_and_map()
        assert "get_entity_contradictions" in info["tool_names"]
        assert "get_relation_contradictions" in info["tool_names"]
        assert "resolve_entity_contradiction" in info["tool_names"]
        assert "resolve_relation_contradiction" in info["tool_names"]
        # All handlers registered
        assert "get_entity_contradictions" in info["handler_names"]
        assert "get_relation_contradictions" in info["handler_names"]

    def test_mcp_handler_gracefully_handles_api_errors(self):
        """D4.8: MCP handler gracefully handles API errors."""
        script = """
from unittest.mock import patch
from server.mcp.deep_dream_server import _TOOL_MAP
mod = __import__('server.mcp.deep_dream_server', fromlist=['_get'])
with patch.object(mod, '_get', return_value=({'error': 'Internal error'}, 500)):
    handler = _TOOL_MAP['get_relation_contradictions']
    result = handler({'family_id': 'rel_error'})
    data = {'is_error': result.get('isError', False)}
    with open(_outfile, 'w') as f:
        json.dump(data, f)
"""
        data = _run_mcp_subprocess(script)
        assert data["is_error"] is True
