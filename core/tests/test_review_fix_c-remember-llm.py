"""code review 修复回归（组 c-remember-llm）。

- f7 批量置信度门：entity_alignment 窗口批量 merge_into 判决低于
  batch_resolution_confidence_threshold 时不执行（降级 create_new）；
  batch_resolution_enabled kill-switch 关闭时走完整 sequential fallback。
- f8 候选截断：resolve_entities_window_batch 的 prompt 不再把候选截到前 6，
  第 7-10 名候选（max_similar_entities 默认 10）必须出现在 prompt 里。
- f9 暂停完整性：pause/cancel epilogue 与成功 epilogue 共用
  _recompute_document_publish_windows——从持久化 episodes 重算，targeted
  修复被暂停时已完整入库的文档保持 active 而非整体降级 incomplete。
"""
import json
import sqlite3
from datetime import datetime

import pytest

from core.llm.consolidation import _ConsolidationMixin
from core.models import Entity
from core.remember.entity_alignment import _EntityBatchMixin
from core.remember.orchestrator import RememberControlFlow
from core.remember.orchestrator_pipeline import _PipelineMixin
from core.storage.sqlite.repositories import documents as doc_repo
from core.storage.sqlite.repositories import episodes as ep_repo
from core.storage.sqlite.schema_v15 import init_schema_v15

NOW = "2026-08-24T00:00:00Z"


# ---------------------------------------------------------------------------
# f7：批量实体判决置信度门
# ---------------------------------------------------------------------------

def _ent(fid, name, content="旧内容"):
    return Entity(
        absolute_id=f"{fid}_v1", family_id=fid, name=name, content=content,
        event_time=datetime.now(), processed_time=datetime.now(),
        episode_id="ep1", source_document="doc.md",
    )


class _StubStorage:
    def resolve_family_id(self, fid):
        return None

    def get_entity_by_family_id(self, fid):
        return None

    def register_entity_redirect(self, src, dst):
        pass


class _BatchHost(_EntityBatchMixin):
    """最小宿主：记录关键写路径调用次数，验证门是否拦截了合并。"""

    def __init__(self, threshold=0.75, batch_enabled=True):
        self.storage = _StubStorage()
        self.llm_client = None
        self.batch_resolution_confidence_threshold = threshold
        self.batch_resolution_enabled = batch_enabled
        self.calls = {"merge": 0, "version": 0, "gate_create": 0, "fallback": 0}

    def _entity_tree_log(self):
        return False

    def _gate_create_entity(self, name, content, episode_id, source_document="",
                            base_time=None, confidence=None, judged_candidate_names=None):
        self.calls["gate_create"] += 1
        return _ent("NEW_fam", name, content)

    def _build_entity_version(self, family_id, name, content, episode_id,
                              source_document="", base_time=None,
                              old_content="", old_content_format="plain"):
        self.calls["version"] += 1
        return _ent(family_id, name, content)

    def _merge_two_contents(self, old_entity, entity_name, entity_content,
                            source_document, episode_id, base_time=None):
        self.calls["merge"] += 1
        return old_entity.content or ""

    def _mark_versioned(self, family_id, already_versioned_family_ids=None,
                        _version_lock=None):
        pass

    def _alignment_guard(self, *args, **kwargs):
        return None

    def _try_context_alias_merge(self, **kwargs):
        return None

    def _process_entity_sequential_fallback(self, extracted_entity, *args, **kwargs):
        self.calls["fallback"] += 1
        return _ent("FB_fam", extracted_entity["name"], extracted_entity["content"]), [], {}


def _cands():
    # 首候选与实体不同名（避开精确同名快路径）、score≥0.25（避开低相似快路径）
    return [{
        "family_id": "fam_zgl", "name": "诸葛亮", "combined_score": 0.6,
        "merge_safe": False, "name_match_type": "none",
        "entity": _ent("fam_zgl", "诸葛亮", "蜀汉丞相"),
    }]


def _run_batch(host, verdict):
    return host._process_entity_with_batch_candidates(
        extracted_entity={"name": "孔明", "content": "蜀汉军师，字孔明"},
        candidates=_cands(),
        episode_id="ep1",
        similarity_threshold=0.7,
        precomputed_verdict=verdict,
    )


class TestBatchConfidenceGate:

    def test_low_confidence_merge_degrades_to_create_new(self):
        """conservative 档（阈值 0.9）下 conf=0.3 的 merge_into 不得执行。"""
        host = _BatchHost(threshold=0.9)
        entity, relations, mapping, to_persist = _run_batch(host, {
            "match_existing_id": "fam_zgl", "update_mode": "merge_into_latest",
            "merged_name": "诸葛亮", "confidence": 0.3, "relations_to_create": [],
        })
        # 不合并、不在既有 family 上建版本，而是新建
        assert entity.family_id == "NEW_fam"
        assert host.calls["merge"] == 0
        assert host.calls["version"] == 0
        assert host.calls["gate_create"] == 1
        assert "孔明" in mapping

    def test_high_confidence_merge_still_executes(self):
        """conf 超过阈值时合并照常执行（门不误伤正常路径）。"""
        host = _BatchHost(threshold=0.9)
        entity, relations, mapping, to_persist = _run_batch(host, {
            "match_existing_id": "fam_zgl", "update_mode": "merge_into_latest",
            "merged_name": "诸葛亮", "confidence": 0.95, "relations_to_create": [],
        })
        assert entity.family_id == "fam_zgl"
        assert host.calls["merge"] == 1
        assert host.calls["version"] == 1

    def test_kill_switch_disables_batch_resolution(self):
        """batch_resolution_enabled=False → 完整 sequential fallback。"""
        host = _BatchHost(threshold=0.0, batch_enabled=False)
        entity, relations, mapping, to_persist = _run_batch(host, {
            "match_existing_id": "fam_zgl", "update_mode": "merge_into_latest",
            "merged_name": "诸葛亮", "confidence": 0.99, "relations_to_create": [],
        })
        assert host.calls["fallback"] == 1
        assert host.calls["merge"] == 0
        assert entity.family_id == "FB_fam"
        assert to_persist is None

    def test_create_new_verdict_ignores_stale_match_id(self):
        """update_mode=create_new 但 LLM 顺手填了 match_existing_id → 仍按新建。"""
        host = _BatchHost(threshold=0.9)
        entity, relations, mapping, to_persist = _run_batch(host, {
            "match_existing_id": "fam_zgl", "update_mode": "create_new",
            "merged_name": "", "confidence": 0.99, "relations_to_create": [],
        })
        assert entity.family_id == "NEW_fam"
        assert host.calls["gate_create"] == 1
        assert host.calls["merge"] == 0
        assert host.calls["version"] == 0


# ---------------------------------------------------------------------------
# f8：窗口批量判决 prompt 的候选截断
# ---------------------------------------------------------------------------

class _PromptCaptureClient(_ConsolidationMixin):
    def __init__(self):
        self.prompts = []

    @staticmethod
    def _parse_json_response(text):
        return {}

    def call_llm_until_json_parses(self, messages, parse_fn=None, json_parse_retries=1):
        self.prompts.append(messages[0]["content"])
        return {"results": []}, ""


class TestWindowBatchCandidateTruncation:

    def test_all_ten_candidates_appear_in_prompt(self):
        """候选默认 10 个（max_similar_entities）：第 7-10 名必须进 prompt。"""
        client = _PromptCaptureClient()
        cands = [
            {"family_id": f"fam_{i}", "name": f"候选实体{i}", "content": "描述"}
            for i in range(1, 11)
        ]
        client.resolve_entities_window_batch(
            [{"name": "孔明", "content": "蜀汉军师"}],
            {"孔明": cands},
            context_text="孔明出山辅佐刘备。",
        )
        assert len(client.prompts) == 1
        prompt = client.prompts[0]
        for idx in range(1, 11):
            assert f"候选{idx}:" in prompt, f"候选{idx} 被截断出 prompt"
        assert "fam_10" in prompt and "候选实体10" in prompt


# ---------------------------------------------------------------------------
# f9：pause/cancel epilogue 的完整性口径
# ---------------------------------------------------------------------------

class _StorageShim:
    def __init__(self, conn):
        self._conn_obj = conn

    def _conn(self):
        return self._conn_obj


class _PipelineHost(_PipelineMixin):
    pass


@pytest.fixture
def v15_conn(tmp_path):
    db = tmp_path / "f9_helper.db"
    conn = sqlite3.connect(str(db))
    init_schema_v15(conn)
    yield conn
    conn.close()


def _seed_doc(conn, n_windows, doc_id="doc1", ver_id="ver1"):
    doc_repo.insert_document(conn, doc_id, title="F9", managed_path="f9.md",
                             created_at=NOW, updated_at=NOW)
    doc_repo.insert_document_version(conn, ver_id, doc_id, "hash1", processed_at=NOW)
    doc_repo.update_current_version(conn, doc_id, ver_id, updated_at=NOW)
    for i in range(n_windows):
        ep_repo.insert_episode(
            conn, f"ep{i}", f"epfam{i}", doc_id, ver_id,
            source_text=f"窗口{i}原文", memory_text=f"窗口{i}摘要",
            chunk_index=i, chunk_hash=f"ch{i}", episode_type="chunk",
            activity_type="processing", processed_at=NOW, run_id="run_seed",
        )
    conn.commit()


class TestRecomputeDocumentPublishWindows:

    def test_complete_doc_partial_repair_stays_active(self):
        """6 窗全有 active episode；修复 run 只完成窗口 2 就暂停 → 仍完整。"""
        conn = sqlite3.connect(":memory:")
        init_schema_v15(conn)
        _seed_doc(conn, 6)
        host = _PipelineHost()
        host.storage = _StorageShim(conn)
        doc_id, active = host._recompute_document_publish_windows(
            last_episode_id="ep5", override_doc_id="", total_chunks=6,
            failed_window_indices=[], successful_window_indices=[2],
            default_document_id="doc_atomic",
        )
        assert doc_id == "doc1"
        assert active == {0, 1, 2, 3, 4, 5}

    def test_genuinely_missing_windows_stay_incomplete(self):
        """只有 3/6 窗有 active episode：missing 仍按持久化事实计算。"""
        conn = sqlite3.connect(":memory:")
        init_schema_v15(conn)
        _seed_doc(conn, 3)
        host = _PipelineHost()
        host.storage = _StorageShim(conn)
        _, active = host._recompute_document_publish_windows(
            last_episode_id="ep2", override_doc_id="", total_chunks=6,
            failed_window_indices=[], successful_window_indices=[],
            default_document_id="doc_atomic",
        )
        assert active == {0, 1, 2}

    def test_override_doc_id_resolution_and_failed_window_excluded(self):
        """last_episode_id 缺失时经 documents.current_version_id 解析；
        本次 run 失败的窗口从 active 集合剔除（与成功 epilogue 一致）。"""
        conn = sqlite3.connect(":memory:")
        init_schema_v15(conn)
        _seed_doc(conn, 6)
        host = _PipelineHost()
        host.storage = _StorageShim(conn)
        doc_id, active = host._recompute_document_publish_windows(
            last_episode_id="", override_doc_id="doc1", total_chunks=6,
            failed_window_indices=[3], successful_window_indices=[2],
            default_document_id="doc_atomic",
        )
        assert doc_id == "doc1"
        assert active == {0, 1, 2, 4, 5}

    def test_publish_final_state_noop_without_setter(self, v15_conn):
        host = _PipelineHost()
        host.storage = _StorageShim(v15_conn)
        # set_publish_state 为 None（存储无该能力）时不抛错
        host._publish_final_ingestion_state(
            set_publish_state=None, last_episode_id="", override_doc_id="",
            total_chunks=3, failed_window_indices=[], successful_window_indices=[],
            default_document_id="doc_x",
        )


# 多窗口文本：window_size_chars=500 时切出 ≥3 窗
F9_TEXT = "\n\n".join([
    "Alice met Bob at the cafe one sunny morning in spring.",
    "Alice and Bob discussed quantum physics for hours over coffee.",
    "Carol joined later and told Alice about the new telescope at the observatory.",
    "Bob said the telescope could help their research on dark matter.",
    "Alice wrote careful notes about dark matter while Carol watched quietly.",
    "Later that week Carol visited the observatory with Bob and Alice together.",
    "Alice and Carol talked about the telescope again during the visit.",
    "Bob mentioned that dark matter research needed more telescope time.",
] * 2)


def _make_processor(tmp_path):
    from core.remember.orchestrator import TemporalMemoryGraphProcessor
    from core.storage.embedding import EmbeddingClient
    emb = EmbeddingClient(model_path="/nonexistent/mock-model", use_local=True)
    emb.model = None
    return TemporalMemoryGraphProcessor(
        storage_path=str(tmp_path / "lib_f9_pause"),
        embedding_client=emb,
        remember_config={"profile": "strong-v1", "window_size_chars": 500,
                         "overlap_chars": 50},
    )


class TestPauseEpilogueKeepsCompleteDocActive:

    def test_pause_during_targeted_repair_keeps_doc_active(self, tmp_path):
        """已完整入库的多窗口文档，targeted 修复 run 中途被暂停：
        完整性必须从持久化 episodes 重算（active），而不是按本次 run
        未跑完把全文档降级成 incomplete（旧 bug：missing=all）。"""
        from core.models import Episode
        from core.text_chunking import apply_document_metadata_prefix
        from core.utils import compute_doc_hash

        proc = _make_processor(tmp_path)
        doc_file = tmp_path / "f9_pause.md"
        doc_name = "f9_pause.md"
        # 与 pipeline 相同口径计算每窗口 chunk 哈希（window-0 加元数据前缀）
        chunks = proc.document_processor.chunk_text(F9_TEXT)
        n_win = len(chunks)
        assert n_win >= 3
        hashes = [
            compute_doc_hash(apply_document_metadata_prefix(doc_name, c[0], idx))
            for idx, c in enumerate(chunks)
        ]
        conn = proc.storage._conn()
        # 播种：每窗口一条 active episode + 抽取缓存（模拟此前一次全量成功的 run）
        for idx, h in enumerate(hashes):
            proc.storage.save_episode(
                Episode(absolute_id=f"ep_seed_{idx}",
                        content=f"窗口{idx}的记忆摘要",
                        event_time=datetime.now(), processed_time=datetime.now(),
                        source_document=doc_name),
                text=chunks[idx][0], document_path=str(doc_file),
                doc_hash=h, run_id="run_seed", chunk_index=idx,
            )
            assert proc.storage.save_extraction_result(
                h, [], [], document_path=str(doc_file))
        doc_row = conn.execute(
            "SELECT document_id, current_version_id FROM documents LIMIT 1"
        ).fetchone()
        doc_id, ver_id = doc_row
        assert ver_id  # save_episode 应已挂 current version
        n_active = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE document_version_id = ? AND status = 'active'",
            (ver_id,),
        ).fetchone()[0]
        assert n_active == n_win

        # targeted 修复（与 task_worker 修复路径同参：override_doc_id + 目标窗口）。
        # 抽取缓存命中让 step1 免 LLM；前两次 poll 放行，之后暂停——
        # 暂停点落在目标窗口已交给 step9/10 之后、后续窗口之前。
        _poll_count = {"n": 0}

        def _control():
            _poll_count["n"] += 1
            return "pause" if _poll_count["n"] > 2 else None

        with pytest.raises(RememberControlFlow):
            proc.remember_text(
                F9_TEXT, doc_name=doc_name, document_path=str(doc_file),
                target_window_indices=[1, 2], override_doc_id=doc_id,
                control_callback=_control,
            )

        row = conn.execute(
            "SELECT state, total_windows, complete_windows, missing_windows "
            "FROM document_ingestion_state WHERE document_id = ?", (doc_id,)
        ).fetchone()
        assert row is not None
        state, total, complete, missing = row
        assert state == "active"
        assert total == n_win
        assert complete == n_win
        assert json.loads(missing) == []
