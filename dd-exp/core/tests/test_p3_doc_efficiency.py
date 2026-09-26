"""P3.2 / P3.6 文档级效率回归测试。

- P3.2：save_episode 的文档级工作（全文件读、content hash、跨文档去重查询、
  current 重写、版本快照）在同一 run 内只做一次，后续窗口复用首个窗口的
  解析结果；run_id 为空（直连调用）保持原有逐次完整解析语义。
- P3.6：队列侧窗口哈希优先复用 remember run 产出（task.result.window_hashes），
  只有无完整产出时才回退现场重算；入队估算走算术公式不再跑 chunk_text；
  window-0 元数据前缀统一来自 core.text_chunking.apply_document_metadata_prefix
  （不变式 a：orchestrator 与 task_queue 字节一致）。

全部用 tmp_path 落盘，不依赖真实 LLM / 网络。
"""
from datetime import datetime
from pathlib import Path

import pytest

from core.models import Episode
from core.remember.document import DocumentProcessor
from core.server.task_journal import RememberTask
from core.server.task_progress import estimate_chunk_count
from core.server.task_queue import RememberTaskQueue
from core.storage.sqlite import SQLiteGraphStorageManager
from core.storage.sqlite import content_fs
from core.text_chunking import apply_document_metadata_prefix
from core.utils import compute_doc_hash

DOC_NAME = "p3_doc.md"

# 6 个 H2 小节 × 每节约 1000 字：window_size=2000 时切出 6 个窗口
DOC_TEXT = "".join(
    ["# 系统设计文档\n\n"]
    + [
        f"## 第{i + 1}节 设计要点\n\n"
        + "爱丽丝介绍存储方案，鲍勃询问并发写入，卡罗尔补充元数据设计。\n\n" * 30
        for i in range(6)
    ]
)


# ---------------------------------------------------------------------------
# 工具：计数处理器 / 假 processor / 播种
# ---------------------------------------------------------------------------

class CountingDocumentProcessor(DocumentProcessor):
    """统计 chunk_text 调用次数（P3.6 的核心指标）。"""

    def __init__(self, counter, window_size, overlap):
        super().__init__(window_size=window_size, overlap=overlap)
        self.chunk_calls = counter

    def chunk_text(self, content):
        self.chunk_calls["chunk_text"] += 1
        return super().chunk_text(content)


class FakeProcessor:
    """queue 测试用的最小 processor 外壳（不真正跑 remember）。"""

    def __init__(self, storage, counter, window_size=2000, overlap=200):
        self.storage = storage
        self.load_cache_memory = False
        self.document_processor = CountingDocumentProcessor(
            counter, window_size, overlap)


def _pipeline_window_hashes(processor, text, doc_name):
    """按 orchestrator 的方式计算窗口哈希：chunk + window-0 前缀 + hash。"""
    hashes = []
    for idx, item in enumerate(processor.document_processor.chunk_text(text)):
        chunk = item[0] if isinstance(item, (list, tuple)) and item else str(item)
        chunk = apply_document_metadata_prefix(doc_name, chunk, idx)
        hashes.append(compute_doc_hash(chunk))
    return hashes


def _episode(idx, tag):
    return Episode(
        absolute_id=f"ep_{tag}_{idx}",
        content=f"窗口{idx}的记忆摘要",
        event_time=datetime.now(),
        processed_time=datetime.now(),
        source_document=DOC_NAME,
    )


def _seed_complete_document(env, doc_file):
    """模拟一次完整 remember run 的入库（每窗口 episode + 抽取缓存）。

    返回按 orchestrator 语义计算的窗口哈希列表（消耗 1 次 chunk_text）。
    """
    hashes = _pipeline_window_hashes(env["proc"], DOC_TEXT, DOC_NAME)
    for idx, h in enumerate(hashes):
        env["storage"].save_episode(
            _episode(idx, "seed"), text=f"窗口{idx}原文片段",
            document_path=str(doc_file), doc_hash=h, run_id="run_seed",
        )
        env["storage"].save_extraction_result(h, entities=[], relations=[])
    return hashes


def _active_version_id(storage):
    return storage._conn().execute(
        "SELECT document_version_id FROM document_versions "
        "WHERE status = 'active' LIMIT 1"
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# 不变式 a：window-0 元数据前缀
# ---------------------------------------------------------------------------

class TestApplyDocumentMetadataPrefix:

    def test_window0_normal_doc_gets_prefix(self):
        out = apply_document_metadata_prefix("notes.md", "正文", 0)
        assert out == "[文档元数据] 文档名：notes.md [/文档元数据]\n\n正文"

    def test_auto_and_api_names_skip_prefix(self):
        # 自动生成名 / api 直传名不注入前缀（与 orchestrator 行为一致）
        for name in ("auto_20240101_120000", "api://abc-def"):
            assert apply_document_metadata_prefix(name, "正文", 0) == "正文"

    def test_nonzero_window_and_empty_name_skip_prefix(self):
        assert apply_document_metadata_prefix("notes.md", "正文", 3) == "正文"
        assert apply_document_metadata_prefix("", "正文", 0) == "正文"

    def test_bytes_equal_to_legacy_inline_fstring(self):
        # 不变式 a：与旧的内联 f-string 字节一致，防止 helper 漂移
        doc_name, chunk = "设计文档.md", "窗口正文内容"
        legacy = f"[文档元数据] 文档名：{doc_name} [/文档元数据]\n\n{chunk}"
        assert (apply_document_metadata_prefix(doc_name, chunk, 0).encode("utf-8")
                == legacy.encode("utf-8"))


# ---------------------------------------------------------------------------
# P3.2：save_episode 文档级工作每 run 一次
# ---------------------------------------------------------------------------

class TestSaveEpisodeDocumentWorkOncePerRun:

    def _make_storage(self, tmp_path):
        return SQLiteGraphStorageManager(
            storage_path=str(tmp_path / "library"), vector_dim=1024,
            graph_id="test")

    def _install_counters(self, monkeypatch):
        """统计文档级工作：全文件读 / current 重写 / 版本快照 / 去重查询。"""
        counters = {"doc_read": 0, "current_write": 0, "snapshot": 0, "dedup_q": 0}

        orig_read_text = Path.read_text

        def counting_read_text(self, *args, **kwargs):
            # 只统计对被测文档文件的读取（排除测试框架自身的文件读取）
            if self.name == DOC_NAME:
                counters["doc_read"] += 1
            return orig_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", counting_read_text)
        monkeypatch.setattr(
            content_fs, "write_current_file",
            _wrap(counters, "current_write", content_fs.write_current_file))
        monkeypatch.setattr(
            content_fs, "write_version_snapshot",
            _wrap(counters, "snapshot", content_fs.write_version_snapshot))
        return counters

    def _install_dedup_counter(self, store):
        """save_episode 的跨文档去重 SELECT 每次文档级解析恰好执行一次。"""
        seen = []

        def trace(statement):
            if "SELECT dv.document_id FROM document_versions dv" in statement:
                seen.append(statement)

        store._conn().set_trace_callback(trace)
        return seen

    def test_document_work_once_within_run(self, tmp_path, monkeypatch):
        store = self._make_storage(tmp_path)
        doc_file = tmp_path / DOC_NAME
        doc_file.write_text(DOC_TEXT, encoding="utf-8")
        counters = self._install_counters(monkeypatch)
        dedup = self._install_dedup_counter(store)

        n_windows = 4
        for i in range(n_windows):
            store.save_episode(
                _episode(i, "a"), text=f"窗口{i}原文片段",
                document_path=str(doc_file), doc_hash=f"hash_a_{i}",
                run_id="run_A")

        # 文档级工作每 run 恰好一次；episode 行仍每窗口一次
        assert counters["doc_read"] == 1
        assert counters["current_write"] == 1
        assert counters["snapshot"] == 1
        assert len(dedup) == 1

        conn = store._conn()
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM document_versions WHERE status='active'"
        ).fetchone()[0] == 1
        rows = conn.execute(
            "SELECT chunk_index, document_version_id FROM episodes "
            "WHERE status='active' ORDER BY chunk_index"
        ).fetchall()
        assert [r[0] for r in rows] == list(range(n_windows))
        assert len({r[1] for r in rows}) == 1  # 同一版本
        store.close()

    def test_new_run_redoes_document_work(self, tmp_path, monkeypatch):
        # 新 run 重新做一次文档级解析（同内容 → 版本复用，不新增版本）
        store = self._make_storage(tmp_path)
        doc_file = tmp_path / DOC_NAME
        doc_file.write_text(DOC_TEXT, encoding="utf-8")
        counters = self._install_counters(monkeypatch)

        for i in range(2):
            store.save_episode(_episode(i, "a"), text=f"窗口{i}",
                               document_path=str(doc_file),
                               doc_hash=f"hash_a_{i}", run_id="run_A")
        for i in range(2, 4):
            store.save_episode(_episode(i, "b"), text=f"窗口{i}",
                               document_path=str(doc_file),
                               doc_hash=f"hash_b_{i}", run_id="run_B")

        assert counters["doc_read"] == 2
        assert counters["current_write"] == 2
        assert counters["snapshot"] == 1  # content_hash 相同 → 版本复用
        conn = store._conn()
        assert conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE status='active'"
        ).fetchone()[0] == 4
        assert conn.execute(
            "SELECT COUNT(*) FROM document_versions WHERE status='active'"
        ).fetchone()[0] == 1
        store.close()

    def test_without_run_id_keeps_legacy_behavior(self, tmp_path, monkeypatch):
        # run_id 为空（直连调用/旧测试路径）不启用 per-run 缓存：
        # 每窗口都做完整文档级解析
        store = self._make_storage(tmp_path)
        doc_file = tmp_path / DOC_NAME
        doc_file.write_text(DOC_TEXT, encoding="utf-8")
        counters = self._install_counters(monkeypatch)

        n_windows = 4
        for i in range(n_windows):
            store.save_episode(
                _episode(i, "legacy"), text=f"窗口{i}原文片段",
                document_path=str(doc_file), doc_hash=f"hash_l_{i}", run_id="")

        assert counters["doc_read"] == n_windows
        assert counters["current_write"] == n_windows
        conn = store._conn()
        assert conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE status='active'"
        ).fetchone()[0] == n_windows
        store.close()

    def test_concurrent_same_run_document_init_exactly_once(
            self, tmp_path, monkeypatch):
        # 并发竞态：多个窗口同 run 同时抵达 save_episode，锁 + double-check
        # 保证文档级解析恰发生一次（生产路径 step1 在 _cache_lock 下串行，
        # 这里直接压测存储层的兜底语义）。文档先用种子 run 落库提交，
        # 避免 per-thread 连接下未提交版本行的 FK 可见性问题。
        #
        # 测试接缝：chunk_index 由"先 COUNT 再 INSERT"得出，并发直连调用
        # 本就可能撞 UNIQUE(document_version_id, chunk_index)（与 P3.2 无关
        # 的既有行为，管线内不会并发）。本测试关注文档级初始化，故对
        # insert_episode 覆写 chunk_index 为全局唯一值以隔离该既有竞态。
        import itertools
        import threading

        from core.storage.sqlite.repositories import episodes as ep_repo_mod

        store = self._make_storage(tmp_path)
        doc_file = tmp_path / DOC_NAME
        doc_file.write_text(DOC_TEXT, encoding="utf-8")
        # 种子 run：创建并提交文档 + 版本（计数器装好前完成）
        store.save_episode(
            _episode(99, "seed"), text="种子窗口",
            document_path=str(doc_file), doc_hash="hash_seed", run_id="run_seed")
        counters = self._install_counters(monkeypatch)

        _real_insert = ep_repo_mod.insert_episode
        _insert_lock = threading.Lock()
        _uniq = itertools.count(1000)

        def _unique_chunk_insert(conn, ep_id, *args, **kwargs):
            with _insert_lock:
                kwargs["chunk_index"] = next(_uniq)
                return _real_insert(conn, ep_id, *args, **kwargs)

        monkeypatch.setattr(ep_repo_mod, "insert_episode", _unique_chunk_insert)

        n_threads = 4
        barrier = threading.Barrier(n_threads)
        errors = []

        def _worker(idx):
            import sqlite3 as _sqlite3

            ep = _episode(idx, "race")
            for attempt in range(5):
                try:
                    if attempt == 0:
                        barrier.wait(timeout=5)
                    store.save_episode(
                        ep, text=f"窗口{idx}原文片段",
                        document_path=str(doc_file), doc_hash=f"hash_r_{idx}",
                        run_id="run_race")
                    return
                except _sqlite3.OperationalError as exc:
                    # WAL 读快照→写升级冲突（并发直连 save_episode 的既有
                    # 形态，与 P3.2 无关）：回滚后重试，重试走 run 缓存命中，
                    # 不影响"文档级初始化恰一次"的被测语义
                    if "locked" not in str(exc):
                        errors.append(exc)
                        return
                    try:
                        store._conn().rollback()
                    except Exception:
                        pass
                except Exception as exc:  # noqa: BLE001 — 线程内异常统一收集
                    errors.append(exc)
                    return
            errors.append(RuntimeError("重试耗尽"))

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "并发 save_episode 线程卡死"

        assert errors == []
        # 文档级解析恰一次：文件读 / current 重写 / 版本快照各 1 次
        assert counters["doc_read"] == 1
        assert counters["current_write"] == 1
        assert counters["snapshot"] == 0  # 同内容 → 版本复用，无新快照
        conn = store._conn()
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM document_versions WHERE status='active'"
        ).fetchone()[0] == 1
        rows = conn.execute(
            "SELECT document_version_id FROM episodes WHERE status='active'"
        ).fetchall()
        assert len(rows) == n_threads + 1
        assert len({r[0] for r in rows}) == 1  # 全部挂在同一版本
        store.close()


def _wrap(counters, key, original):
    def wrapper(*args, **kwargs):
        counters[key] += 1
        return original(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# P3.6 端到端：remember run 产出的 window_hashes 与落库 chunk_hash 一致
# ---------------------------------------------------------------------------

class TestRememberRunProducesWindowHashes:

    def test_window_hashes_match_episodes_chunk_hash(self, tmp_path):
        # 全 mock LLM/embedding 的真实 remember run：result.window_hashes
        # 必须与 episodes.chunk_hash 逐窗一致（task_queue 复用时才不会误报
        # 缺失窗口），且 run 内每窗口哈希只计算一次（缓存查找复用同值）。
        from core.remember.orchestrator import TemporalMemoryGraphProcessor
        from core.storage.embedding import EmbeddingClient

        emb = EmbeddingClient(model_path="/nonexistent/mock-model", use_local=True)
        emb.model = None
        proc = TemporalMemoryGraphProcessor(
            storage_path=str(tmp_path / "lib_e2e"), embedding_client=emb,
            remember_config={"profile": "strong-v1", "window_size_chars": 500,
                             "overlap_chars": 50},
        )
        try:
            text = "\n\n".join([
                "Alice met Bob at the cafe one sunny morning in spring.",
                "Alice and Bob discussed quantum physics for hours over coffee.",
                "Carol joined later and told Alice about the new telescope.",
                "Bob said the telescope could help their research on dark matter.",
                "Alice wrote careful notes about dark matter while Carol watched.",
                "Later that week Carol visited the observatory with Bob and Alice.",
                "Carol and Bob argued about funding for the telescope project.",
                "Alice reminded everyone that the observatory visit was fruitful.",
                "The dark matter notes convinced Bob to schedule another meeting.",
                "Carol finally agreed the telescope time was well spent that year.",
            ])
            assert len(proc.document_processor.chunk_text(text)) >= 2  # 多窗口
            result = proc.remember_text(text, doc_name="p3_hashes.md", verbose=False)
            hashes = result.get("window_hashes")
            assert isinstance(hashes, list)
            assert result.get("total_chunks") == len(hashes) >= 2
            assert all(isinstance(h, str) and h for h in hashes)
            conn = proc.storage._conn()
            stored = [r[0] for r in conn.execute(
                "SELECT chunk_hash FROM episodes "
                "WHERE status='active' AND episode_type != 'retrieval_slice' "
                "ORDER BY chunk_index").fetchall()]
            assert hashes == stored  # 与入库哈希逐窗一致（含 window-0 前缀）
        finally:
            proc.close()


# ---------------------------------------------------------------------------
# P3.6：队列侧 chunk 只算一次
# ---------------------------------------------------------------------------

@pytest.fixture
def queue_env(tmp_path, monkeypatch):
    counter = {"chunk_text": 0}
    storage = SQLiteGraphStorageManager(
        storage_path=str(tmp_path / "library"), vector_dim=1024, graph_id="test")
    proc = FakeProcessor(storage, counter)
    queue = RememberTaskQueue(
        proc, Path(storage.storage_path),
        processor_factory=lambda: proc, max_workers=1)
    # 测试不真正执行任务：吞掉入队分发，防止 worker 线程消费
    monkeypatch.setattr(queue._queue, "put", lambda *a, **kw: None)
    doc_file = tmp_path / DOC_NAME
    doc_file.write_text(DOC_TEXT, encoding="utf-8")
    yield {"storage": storage, "proc": proc, "queue": queue,
           "counter": counter, "doc_file": doc_file}
    storage.close()


def _make_task(task_id="t_p3"):
    return RememberTask(
        task_id=task_id, text=DOC_TEXT, source_name=DOC_NAME,
        load_cache=False, control_action=None, event_time=None,
        original_path="")


class TestQueueChunkComputedOnce:

    def test_submit_uses_formula_without_chunking(self, queue_env):
        # 入队估算只走算术公式（显式兜底），不再为一条提示语切全文
        task = _make_task("t_est")
        queue_env["queue"].submit(task)
        assert queue_env["counter"]["chunk_text"] == 0
        assert task.total_chunks == estimate_chunk_count(
            len(DOC_TEXT), queue_env["queue"]._window_size,
            queue_env["queue"]._overlap)
        assert task.total_chunks >= 1

    def test_detect_repair_windows_reuses_run_output(self, queue_env):
        # run 已产出（崩溃恢复/续跑）：task.result 带 window_hashes 时
        # 修复检测零重算
        hashes = _seed_complete_document(queue_env, queue_env["doc_file"])
        task = _make_task("t_resume")
        task.result = {"window_hashes": list(hashes), "total_chunks": len(hashes)}
        before = queue_env["counter"]["chunk_text"]
        missing = queue_env["queue"].detect_repair_windows(task)
        assert queue_env["counter"]["chunk_text"] == before
        assert missing == []

    def test_detect_repair_windows_fallback_matches_pipeline_bytes(self, queue_env):
        # 无 run 产出时回退现场重算：与入库哈希字节一致 → 不误报缺失
        hashes = _seed_complete_document(queue_env, queue_env["doc_file"])
        assert len(hashes) >= 2
        task = _make_task("t_fallback")
        task.result = None
        before = queue_env["counter"]["chunk_text"]
        missing = queue_env["queue"].detect_repair_windows(task)
        assert queue_env["counter"]["chunk_text"] == before + 1  # 兜底恰好一次
        assert missing == []

    def test_partial_run_output_falls_back(self, queue_env):
        # window_hashes 含空洞（run 中途崩溃）→ 视为不完整，回退重算
        hashes = _seed_complete_document(queue_env, queue_env["doc_file"])
        task = _make_task("t_partial")
        task.result = {"window_hashes": hashes[:1] + [None] + hashes[2:],
                       "total_chunks": len(hashes)}
        before = queue_env["counter"]["chunk_text"]
        queue_env["queue"].detect_repair_windows(task)
        assert queue_env["counter"]["chunk_text"] == before + 1

    def test_assess_document_integrity_uses_shared_prefix(self, queue_env):
        # 独立完整性检查（无 task 上下文）与 orchestrator 入库哈希字节一致
        hashes = _seed_complete_document(queue_env, queue_env["doc_file"])
        ver_id = _active_version_id(queue_env["storage"])
        before = queue_env["counter"]["chunk_text"]
        report = queue_env["queue"].assess_document_integrity(ver_id)
        assert queue_env["counter"]["chunk_text"] == before + 1
        assert report["complete"] is True
        assert report["missing_windows"] == 0
        assert report["total_windows"] == len(hashes)
