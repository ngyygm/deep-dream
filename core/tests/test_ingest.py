"""core/ingest.py 测试：统一文件入口 + log profile 零 LLM 快速通道。

log profile 全程不构造任何 LLM/embedding client——测试只建临时
LibraryManager（无 embedding client），并断言 embeddings 表保持为空。
"""
import pytest

from core.ingest import (
    ingest_file,
    ingest_log,
    ingest_text,
    _parse_line_timestamp,
    _template_line,
)
from core.storage.sqlite.library_manager import LibraryManager

# 报告 dict 必须携带的字段（任务契约）
_REQUIRED_REPORT_KEYS = {
    "profile", "file", "lines", "windows", "skipped_duplicate_windows",
    "documents_created", "episodes_created", "patterns_distilled",
    "duration_ms",
}

# 带时间戳的合成日志：10:00 附近 3 行 + 20 分钟间隔后 3 行 → 2 个时间窗
TS_LOG = "\n".join([
    "2026-08-23T10:00:00Z INFO worker heartbeat seq=1 status=ok",
    "2026-08-23T10:00:05Z INFO worker heartbeat seq=2 status=ok",
    "2026-08-23T10:00:10Z WARN disk usage at 85 percent on /data",
    "2026-08-23T10:20:00Z ERROR db connection timeout after 30s host=10.0.0.5",
    "2026-08-23T10:20:07Z ERROR db connection timeout after 30s host=10.0.0.6",
    "2026-08-23T10:20:09Z INFO retry attempt 3 of 5",
]) + "\n"

# 不带时间戳的合成日志：block_b 与 block_a 完全相同（重复窗口），
# line_window=5 → 3 个行数窗，其中 1 个重复被跳过
_NT_LINE = "GET request id={i} path=/api/v1/items duration_ms={d}"
NT_LOG = "\n".join(
    [_NT_LINE.format(i=i, d=i * 7) for i in range(1, 6)]
    + [_NT_LINE.format(i=i, d=i * 7) for i in range(1, 6)]
    + [f"POST request id={i} path=/api/v1/items duration_ms={i * 9}"
       for i in range(6, 9)]
) + "\n"


@pytest.fixture
def lib(tmp_path):
    """临时 LibraryManager（无 embedding client → 任何 embedding 写入都会炸）。"""
    mgr = LibraryManager(str(tmp_path / "lib"))
    yield mgr
    mgr.close()


# ── 时间戳识别 / 行模板化（纯函数）──────────────────────────

def test_parse_line_timestamp_variants():
    assert _parse_line_timestamp(
        "2026-08-23T10:00:00Z INFO x").second == 0
    assert _parse_line_timestamp(
        "2026-08-23 10:00:01,123 [main] INFO y").second == 1
    assert _parse_line_timestamp(
        "[2026-08-23 10:00:01] INFO bracketed").hour == 10
    assert _parse_line_timestamp(
        "Aug 23 10:00:01 host syslogd[1]: ok").month == 8
    assert _parse_line_timestamp("no timestamp here at all") is None
    assert _parse_line_timestamp("") is None


def test_template_line_normalization():
    assert _template_line(
        "req id=3f2b8c1a deadbeef99 host=10.0.0.5 uuid="
        "123e4567-e89b-12d3-a456-426614174000 msg=\"done\""
    ) == ("req id=⟨H⟩ ⟨H⟩ host=⟨IP⟩ uuid=⟨H⟩ msg=⟨S⟩")
    assert _template_line("count = 42 ratio=0.5") == "count = ⟨N⟩ ratio=⟨N⟩"
    assert _template_line("   ") == ""


# ── log profile：带时间戳（时间窗模式）──────────────────────

def test_ingest_text_log_timestamp_windows(lib):
    report = ingest_text(TS_LOG, "app.log", profile="log", storage=lib)

    assert _REQUIRED_REPORT_KEYS <= set(report)
    assert report["profile"] == "log"
    assert report["window_mode"] == "time"
    assert report["lines"] == 7  # 尾随换行 split 出的空行计入
    assert report["windows"] == 2
    assert report["skipped_duplicate_windows"] == 0
    assert report["documents_created"] == 2      # 原文 + 蒸馏
    assert report["episodes_created"] == 3       # 2 窗口 + 1 蒸馏
    assert report["patterns_distilled"] >= 1
    assert report["duration_ms"] >= 0

    conn = lib._conn()
    docs = conn.execute(
        "SELECT document_id, title FROM documents "
        "WHERE status='active' ORDER BY title").fetchall()
    titles = {d[1] for d in docs}
    assert titles == {"app.log", "app.log 日志模式蒸馏"}

    # 原文文档：2 个 log_window episode，偏移/行号/事件时间正确
    raw_doc_id = report["document_ids"][0]
    eps = conn.execute(
        "SELECT name, start_offset, end_offset, line_start, line_end, "
        "chunk_index, episode_type, event_time FROM episodes "
        "WHERE document_id=? AND status='active' ORDER BY chunk_index",
        (raw_doc_id,)).fetchall()
    assert len(eps) == 2
    assert all(e[6] == "log_window" for e in eps)
    assert eps[0][1] == 0  # start_offset（第一窗从文件头开始）
    expected_w2_start = len(TS_LOG.split("\n")[0]) + 1 \
        + len(TS_LOG.split("\n")[1]) + 1 + len(TS_LOG.split("\n")[2]) + 1
    assert eps[1][1] == expected_w2_start  # 第二窗字节偏移
    assert (eps[0][3], eps[0][4]) == (1, 3)   # line_start/line_end（1 基）
    assert (eps[1][3], eps[1][4]) == (4, 6)
    assert eps[0][0].startswith("W001 2026-08-23T10:00:00Z")
    assert eps[0][7] == "2026-08-23T10:00:00Z"  # 窗口起始时间作 event_time

    # ingestion_state=active（搜索视图按它过滤非 active 文档）
    states = {
        r[0]: r[1] for r in conn.execute(
            "SELECT document_id, state FROM document_ingestion_state").fetchall()
    }
    for doc_id in report["document_ids"]:
        assert states.get(doc_id) == "active"

    # FTS（bm25）能搜到日志内容——走 library_manager 现有 search 方法
    for word in ("heartbeat", "timeout"):
        hits = lib.search_concepts_by_bm25(word, limit=10)
        assert hits, f"bm25 应能搜到 {word!r}"
        assert any(word in (h.get("content") or "") for h in hits)
        assert all(h.get("role") == "episode" for h in hits)

    # 零 LLM：没有任何 embedding / 实体 / 关系产生
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM entity_observations").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM relation_assertions").fetchone()[0] == 0


# ── log profile：无时间戳（行数窗模式）+ 重复窗口去重 ────────

def test_ingest_text_log_line_mode_and_duplicate_windows(lib):
    report = ingest_text(NT_LOG, "access.log", profile="log", storage=lib,
                         line_window=5)

    assert report["window_mode"] == "line"
    assert report["lines"] == 14  # 13 行内容 + 尾随换行 split 出的空行
    assert report["windows"] == 3
    assert report["skipped_duplicate_windows"] == 1
    assert report["documents_created"] == 2
    assert report["episodes_created"] == 3  # 去重后 2 窗口 + 1 蒸馏
    assert report["patterns_distilled"] >= 1

    conn = lib._conn()
    raw_eps = conn.execute(
        "SELECT source_text, chunk_index FROM episodes "
        "WHERE document_id=? AND status='active' AND episode_type='log_window' "
        "ORDER BY chunk_index", (report["document_ids"][0],)).fetchall()
    assert len(raw_eps) == 2  # 重复窗口没有落库

    # 蒸馏文档：单 episode，含 top 模板（数字已模板化）
    distill_ep = conn.execute(
        "SELECT source_text FROM episodes WHERE document_id=? "
        "AND status='active' AND episode_type='log_distillation'",
        (report["distill_document_id"],)).fetchone()
    assert distill_ep is not None
    md = distill_ep[0]
    assert md.startswith("# access.log 日志模式蒸馏")
    assert "⟨N⟩" in md
    assert "GET request id=⟨N⟩" in md  # 出现 10 次的 top 模板在蒸馏文档里
    assert report["patterns_distilled"] == md.count("## 模式 ")

    # 蒸馏文档同样能被 bm25 搜到（FTS 已同步）
    hits = lib.search_concepts_by_bm25("日志模式蒸馏", limit=10)
    assert any(h.get("family_id") == report["distill_document_id"]
               or "日志模式蒸馏" in (h.get("content") or "") for h in hits)


def test_ingest_log_reingest_same_content_is_idempotent(lib):
    """同内容重复入库：复用版本、替换窗口 episode，不产生重复行。"""
    ingest_text(TS_LOG, "app.log", profile="log", storage=lib)
    report2 = ingest_text(TS_LOG, "app.log", profile="log", storage=lib)

    conn = lib._conn()
    n_docs = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE status='active'").fetchone()[0]
    n_eps = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE status='active'").fetchone()[0]
    n_versions = conn.execute(
        "SELECT COUNT(*) FROM document_versions WHERE status='active'").fetchone()[0]
    assert n_docs == 2
    assert n_versions == 2
    assert n_eps == 3  # 2 窗口 + 1 蒸馏，与首次一致
    assert report2["documents_created"] == 2
    # 重复入库后 FTS 仍可搜且无重复行
    hits = lib.search_concepts_by_bm25("heartbeat", limit=20)
    ep_ids = [h["family_id"] for h in hits]
    assert len(ep_ids) == len(set(ep_ids))


def test_ingest_log_content_fallback_a_b_a(lib):
    """A→B→A 内容回退：第三次入库复用第一次的历史版本，不撞 UNIQUE。

    ver_id 由 content_hash 派生（确定性 ID），直接 insert 会违反
    UNIQUE(document_id, document_version_id) / UNIQUE(document_id,
    content_hash) 把整批回滚；必须复活旧 superseded 版本并指回 current。
    """
    content_a = "aaa=1\naaa=2\n"
    content_b = "bbb=1\nbbb=2\n"
    ingest_text(content_a, "app.log", profile="log", storage=lib, distill=False)
    ingest_text(content_b, "app.log", profile="log", storage=lib, distill=False)
    report3 = ingest_text(content_a, "app.log", profile="log", storage=lib,
                          distill=False)

    conn = lib._conn()
    doc_id = report3["document_ids"][0]
    # 同一文档只有 A/B 两个版本，且回退后 A 是唯一 active、current 指回 A
    vers = conn.execute(
        "SELECT document_version_id, status FROM document_versions "
        "WHERE document_id=? ORDER BY document_version_id",
        (doc_id,)).fetchall()
    assert len(vers) == 2
    active = [v for v in vers if v[1] == "active"]
    assert len(active) == 1
    current = conn.execute(
        "SELECT current_version_id FROM documents WHERE document_id=?",
        (doc_id,)).fetchone()[0]
    assert current == active[0][0]

    # 回退版本的 episodes 已替换为 A 的窗口，FTS 可搜
    eps = conn.execute(
        "SELECT source_text FROM episodes "
        "WHERE document_id=? AND status='active'", (doc_id,)).fetchall()
    assert len(eps) == 1 and "aaa=1" in eps[0][0]
    hits = lib.search_concepts_by_bm25("aaa", limit=5)
    assert any("aaa=1" in (h.get("content") or "") for h in hits)


# ── log profile：偏移约定（字符，非字节）+ 前导行 ────────────

# 带时间戳的中文日志：2 行 10:00 窗 + 1 行 10:20 窗 → 2 个时间窗，
# 且中文使字符数 ≠ UTF-8 字节数（保证偏移断言真正区分两种口径）
CJK_TS_LOG = "\n".join([
    "2026-08-23T10:00:00Z INFO 服务启动完成 耗时三秒",
    "2026-08-23T10:00:05Z INFO 心跳检查正常 状态良好",
    "2026-08-23T10:20:00Z WARN 数据库连接超时 重试第三次",
]) + "\n"


def test_ingest_log_char_offsets_with_cjk(lib):
    """episode 偏移按 Python 字符计（与全库约定一致），不是 UTF-8 字节。"""
    report = ingest_text(CJK_TS_LOG, "cjk.log", profile="log", storage=lib)

    conn = lib._conn()
    eps = conn.execute(
        "SELECT start_offset, end_offset FROM episodes "
        "WHERE document_id=? AND status='active' AND episode_type='log_window' "
        "ORDER BY chunk_index", (report["document_ids"][0],)).fetchall()
    assert len(eps) == 2
    lines = CJK_TS_LOG.split("\n")
    expected_w2_start = len(lines[0]) + 1 + len(lines[1]) + 1
    # 断言本身有效：中文行字符数 ≠ 字节数（否则测试拦不住字节口径）
    assert expected_w2_start != (
        len(lines[0].encode()) + 1 + len(lines[1].encode()) + 1)
    assert eps[0][0] == 0
    assert eps[0][1] == len(lines[0]) + 1 + len(lines[1])  # 不含末尾换行
    assert eps[1][0] == expected_w2_start
    assert eps[1][1] == expected_w2_start + len(lines[2])

    # 蒸馏文档的单 episode 同样按字符：end_offset == len(source_text)
    distill_ep = conn.execute(
        "SELECT start_offset, end_offset, source_text FROM episodes "
        "WHERE document_id=? AND status='active' "
        "AND episode_type='log_distillation'",
        (report["distill_document_id"],)).fetchone()
    assert distill_ep is not None
    assert distill_ep[0] == 0
    assert distill_ep[1] == len(distill_ep[2])


def test_ingest_log_leading_lines_before_first_timestamp(lib):
    """首个时间戳之前的前导行不丢弃：单独成窗并落库。"""
    log = "\n".join([
        "service bootstrap preamble without any timestamp",
        "2026-08-23T10:00:00Z INFO heartbeat seq=1",
        "2026-08-23T10:00:05Z INFO heartbeat seq=2",
    ]) + "\n"
    report = ingest_text(log, "lead.log", profile="log", storage=lib,
                         distill=False)

    assert report["window_mode"] == "time"
    assert report["windows"] == 2

    conn = lib._conn()
    eps = conn.execute(
        "SELECT name, source_text, line_start, line_end, start_offset "
        "FROM episodes WHERE document_id=? AND status='active' "
        "ORDER BY chunk_index", (report["document_ids"][0],)).fetchall()
    assert len(eps) == 2
    # 前导行自成首窗：内容完整保留、行号从 1 开始、偏移从 0 起
    assert "preamble" in eps[0][1]
    assert (eps[0][2], eps[0][3]) == (1, 1)
    assert eps[0][4] == 0
    assert eps[0][0] == "W001"  # 无时间戳 → 名字只有窗序号
    assert (eps[1][2], eps[1][3]) == (2, 3)
    assert eps[1][1].startswith("2026-08-23T10:00:00Z")


# ── ingest_file 文件直传 ─────────────────────────────────────

def test_ingest_file_log_profile(tmp_path, lib):
    log_path = tmp_path / "train-run.log"
    log_path.write_text(NT_LOG, encoding="utf-8")
    report = ingest_file(str(log_path), profile="log", storage=lib,
                         line_window=5)

    assert report["profile"] == "log"
    assert report["file"] == str(log_path.resolve())
    assert report["skipped_duplicate_windows"] == 1
    assert report["documents_created"] == 2

    conn = lib._conn()
    row = conn.execute(
        "SELECT title, absolute_path, source_mode FROM documents "
        "WHERE document_id=?", (report["document_ids"][0],)).fetchone()
    assert row[0] == "train-run.log"      # title = 文件名
    assert row[1] == str(log_path.resolve())
    assert row[2] == "managed"


def test_ingest_file_unknown_profile_raises(tmp_path, lib):
    p = tmp_path / "x.txt"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        ingest_file(str(p), profile="yaml", storage=lib)


# ── prose profile：委托 remember 管线（stub processor，零 LLM）──

class _StubProcessor:
    """记录调用参数的最小 processor 桩。"""

    def __init__(self):
        self.calls = []
        self.storage = None

    def remember_text(self, text, doc_name="", **kw):
        self.calls.append((text, doc_name, kw))
        return {
            "episode_id": "ep_stub_1",
            "document_version_id": "docver_stub",
            "chunks_processed": 2,
            "total_chunks": 2,
            "entities": 3,
            "relations": 1,
            "storage_path": "/tmp/stub",
        }


def test_ingest_prose_delegates_to_processor():
    stub = _StubProcessor()
    report = ingest_text("hello knowledge graph", "note.md",
                         profile="prose", processor=stub)

    assert len(stub.calls) == 1
    text, doc_name, kw = stub.calls[0]
    assert text == "hello knowledge graph"
    assert doc_name == "note.md"
    assert kw["source_document"] == "note.md"

    assert _REQUIRED_REPORT_KEYS <= set(report)
    assert report["profile"] == "prose"
    assert report["windows"] == 2
    assert report["episodes_created"] == 2
    assert report["documents_created"] == 1
    assert report["patterns_distilled"] == 0
    assert report["run"]["episode_id"] == "ep_stub_1"


def test_ingest_prose_via_registry_uses_given_processor():
    """传了显式 processor 时不再向 registry 要 processor。"""
    stub = _StubProcessor()

    class _BoomRegistry:
        def get_processor(self, graph_id):
            raise AssertionError("显式 processor 传入时不应触碰 registry")

    report = ingest_text("t", "t.md", profile="prose", processor=stub,
                         registry=_BoomRegistry())
    assert report["run"]["episode_id"] == "ep_stub_1"


# ── ingest_log 底层入口直接可用 ──────────────────────────────

def test_ingest_log_direct_entry(lib):
    report = ingest_log("alpha=1\nalpha=2\n", "tiny.log", storage=lib,
                        distill=False)
    assert report["documents_created"] == 1  # 关闭蒸馏时只有原文文档
    assert report["patterns_distilled"] == 0
    assert report["distill_document_id"] == ""
    assert lib.search_concepts_by_bm25("alpha", limit=5)
