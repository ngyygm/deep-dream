"""统一文件/文本入库入口：``ingest_file`` / ``ingest_text``。

两种 profile：

- ``prose``（自然语言文稿）：读入文本后委托现有 remember 管线
  （``registry.get_processor(graph_id).remember_text(text, doc_name)``），
  本模块不做切块/抽取。
- ``log``（日志/运行采集数据，**零 LLM**）：按行读入，识别行首时间戳后
  按时间窗（默认 300s）切块（无时间戳则按行数窗，默认 400 行），
  直接沿管线同款的库方法写入 documents/document_versions/episodes
  并同步 episodes_fts，置 document_ingestion_state=active；随后对全文件
  做行模板化模式蒸馏（数字→⟨N⟩、十六进制/uuid→⟨H⟩、引号串→⟨S⟩、
  IP→⟨IP⟩），生成一份 markdown 蒸馏文档作为第二个零 LLM 文档入库。

本模块不 import 任何 LLM 客户端；log profile 全程不构造 LLM/embedding。
"""
from __future__ import annotations

import hashlib
import re
import time as _time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.storage.sqlite import content_fs
from core.storage.sqlite.helpers import now_utc_str
from core.storage.sqlite.repositories import documents as doc_repo
from core.storage.sqlite.repositories import episodes as ep_repo

__all__ = ["ingest_file", "ingest_text", "ingest_log"]

PROFILE_PROSE = "prose"
PROFILE_LOG = "log"

# ── 行首时间戳识别（三种常见格式）───────────────────────────────
# 1) ISO8601 / "YYYY-MM-DD HH:MM:SS[.ms][Z|+HH:MM]"（[T ] 分隔）
_TS_ISO_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
    r"(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)"
)
# 2) 括号包裹 "[YYYY-MM-DD HH:MM:SS]"（logging.Formatter 常见）
_TS_BRACKET_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\]"
)
# 3) syslog 风格 "Aug 23 10:00:01"
_TS_SYSLOG_RE = re.compile(r"^([A-Z][a-z]{2})\s{1,2}(\d{1,2})\s+(\d{2}:\d{2}:\d{2})\b")
_SYSLOG_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# ── 模式蒸馏：行模板化正则（先后顺序有语义：uuid→hex→IP→引号→数字）──
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{8,}\b")
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_QUOTED_RE = re.compile(r'"[^"\n]*"|\'[^\'\n]*\'')
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

_TPL_NUM = "⟨N⟩"    # 数字
_TPL_HEX = "⟨H⟩"    # 十六进制 / uuid
_TPL_STR = "⟨S⟩"    # 引号串
_TPL_IP = "⟨IP⟩"    # 点分地址


# ═══════════════════════════════════════════════════════════════
# 顶层入口
# ═══════════════════════════════════════════════════════════════

def ingest_file(path, profile: str = PROFILE_PROSE, **kw) -> Dict[str, Any]:
    """读文件并按 profile 入库。

    Parameters
    ----------
    path: 文件路径（str/Path）。
    profile: ``'prose'`` 走 remember 管线；``'log'`` 走零 LLM 快速通道。
    kw: 其余参数同 :func:`ingest_text`（encoding/processor/registry/
        storage/time_window_s/line_window/top_patterns/distill 等）。
    """
    fp = Path(path)
    encoding = kw.pop("encoding", "utf-8")
    text = fp.read_text(encoding=encoding)
    return ingest_text(
        text, fp.name, profile=profile,
        absolute_path=str(fp.resolve()), **kw,
    )


def ingest_text(text: str, name: str, profile: str = PROFILE_PROSE, *,
                absolute_path: str = "",
                processor=None, registry=None, storage=None,
                graph_id: Optional[str] = None,
                config: Optional[dict] = None,
                storage_path: str = "./library",
                time_window_s: float = 300.0,
                line_window: int = 400,
                top_patterns: int = 30,
                distill: bool = True,
                **kw) -> Dict[str, Any]:
    """统一文本入库入口。

    Parameters
    ----------
    text: 原始文本内容。
    name: 来源名（文件名/标签），作为文档标题。
    profile: ``'prose'`` 委托 remember 管线（需要 processor/registry）；
        ``'log'`` 零 LLM 快速通道（只需要 storage）。
    absolute_path: 原文件绝对路径（可空；ingest_file 自动填充）。
    processor: prose profile 显式传入的 processor（优先级最高）。
    registry: GraphRegistry（prose 未传 processor 时惰性使用）。
    storage: log profile 显式传入的存储 facade（LibraryManager）；
        未传时从 processor/registry 惰性获取。
    graph_id: 图 ID（单库模式下归一化为 library）。
    config / storage_path: 惰性构造 GraphRegistry 时的配置（参考 CLI 的
        CliContext 模式，但本模块不 import core.cli）。
    time_window_s / line_window: log profile 的切块窗（时间秒数/行数）。
    top_patterns: 模式蒸馏保留的 top N 模板数。
    distill: 是否生成并入库模式蒸馏文档。
    """
    if profile == PROFILE_LOG:
        store = storage if storage is not None else _resolve_storage(
            processor=processor, registry=registry, graph_id=graph_id,
            config=config, storage_path=storage_path,
        )
        return ingest_log(
            text, name, storage=store, absolute_path=absolute_path,
            time_window_s=time_window_s, line_window=line_window,
            top_patterns=top_patterns, distill=distill,
        )
    if profile == PROFILE_PROSE:
        return _ingest_prose(text, name, absolute_path=absolute_path,
                             processor=processor, registry=registry,
                             graph_id=graph_id, config=config,
                             storage_path=storage_path, **kw)
    raise ValueError(f"未知 profile: {profile!r}（可选: prose/log）")


# ═══════════════════════════════════════════════════════════════
# prose profile：委托 remember 管线
# ═══════════════════════════════════════════════════════════════

def _lazy_registry(config: Optional[dict], storage_path: str):
    """惰性构造 GraphRegistry（默认 ./library，与 CLI 语义一致）。"""
    from core.server.registry import GraphRegistry
    if config is None:
        try:
            from core.server.config import DEFAULTS
            import copy
            config = copy.deepcopy(DEFAULTS)
        except Exception:
            config = {}
    return GraphRegistry(storage_path, config)


def _resolve_processor(processor=None, registry=None, graph_id=None,
                       config=None, storage_path="./library"):
    """prose profile 的 processor 解析：显式传入 > registry > 惰性建 registry。"""
    if processor is not None:
        return processor
    if registry is None:
        registry = _lazy_registry(config, storage_path)
    from core.server.registry import GraphRegistry
    return registry.get_processor(GraphRegistry.normalize_graph_id(graph_id))


def _resolve_storage(processor=None, registry=None, graph_id=None,
                     config=None, storage_path="./library"):
    """log profile 的存储解析：显式 storage > processor.storage > registry。"""
    proc = _resolve_processor(
        processor=processor, registry=registry, graph_id=graph_id,
        config=config, storage_path=storage_path)
    return proc.storage


def _ingest_prose(text: str, name: str, *, absolute_path: str = "",
                  processor=None, registry=None, graph_id=None,
                  config=None, storage_path="./library", **kw) -> Dict[str, Any]:
    """委托 ``remember_text`` 管线并整理为统一报告 dict。"""
    proc = _resolve_processor(
        processor=processor, registry=registry, graph_id=graph_id,
        config=config, storage_path=storage_path)
    t0 = _time.perf_counter()
    result = proc.remember_text(
        text, doc_name=name,
        verbose=bool(kw.pop("verbose", False)),
        source_document=absolute_path or kw.pop("source_document", None) or name,
    )
    duration_ms = round((_time.perf_counter() - t0) * 1000, 1)
    return {
        "profile": PROFILE_PROSE,
        "file": name,
        "lines": len((text or "").splitlines()),
        "windows": result.get("total_chunks", result.get("chunks_processed", 0)),
        "skipped_duplicate_windows": 0,
        "documents_created": 1 if result.get("episode_id") else 0,
        "episodes_created": result.get("chunks_processed", 0),
        "patterns_distilled": 0,
        "duration_ms": duration_ms,
        "run": result,
    }


# ═══════════════════════════════════════════════════════════════
# log profile：零 LLM 快速通道
# ═══════════════════════════════════════════════════════════════

def ingest_log(text: str, name: str, *, storage,
               absolute_path: str = "",
               time_window_s: float = 300.0,
               line_window: int = 400,
               top_patterns: int = 30,
               distill: bool = True) -> Dict[str, Any]:
    """日志/采集数据入库（零 LLM）。

    步骤：按行切窗（时间戳→时间窗，否则行数窗）→ 内容哈希去重窗口
    → documents/document_versions/episodes/episodes_fts 直写（与管线
    同款库方法、同款顺序）→ ingestion_state=active → 模式蒸馏文档。
    """
    t0 = _time.perf_counter()
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    display = absolute_path or name

    if not text.strip():
        return {
            "profile": PROFILE_LOG, "file": display, "lines": 0,
            "windows": 0, "skipped_duplicate_windows": 0,
            "documents_created": 0, "episodes_created": 0,
            "patterns_distilled": 0,
            "duration_ms": round((_time.perf_counter() - t0) * 1000, 1),
            "warning": "text is empty or whitespace-only",
        }

    char_starts = _line_char_offsets(lines)
    windows, mode = _split_windows(
        lines, time_window_s=time_window_s, line_window=line_window)

    # 内容哈希去重：同哈希窗口跳过（计 skipped_duplicate_windows）
    seen_hashes = set()
    kept: List[dict] = []
    skipped = 0
    for w in windows:
        h = hashlib.sha256(w["text"].encode("utf-8")).hexdigest()
        if h in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(h)
        kept.append(w)

    episode_rows = [
        _window_episode_row(w, i, char_starts)
        for i, w in enumerate(kept)
    ]
    doc = _write_zero_llm_document(
        storage, identity=f"ingest-log\0{absolute_path or name}", title=name,
        text=text, episode_rows=episode_rows, absolute_path=absolute_path,
        total_windows=len(kept))

    documents_created = 1
    episodes_created = doc["episodes_created"]
    document_ids = [doc["document_id"]]
    distill_doc_id = ""

    patterns: List[dict] = []
    if distill:
        patterns = _distill_patterns(lines, top_n=top_patterns)
    if patterns:
        md = _render_distill_markdown(name, lines, patterns)
        ddoc = _write_zero_llm_document(
            storage, identity=f"ingest-log\0{absolute_path or name}\0patterns",
            title=f"{name} 日志模式蒸馏", text=md,
            episode_rows=[{
                "name": "pattern_distillation",
                "source_text": md,
                "start_offset": 0, "end_offset": len(md),  # 字符偏移，同窗口 episode
                "line_start": 1, "line_end": len(md.split("\n")),
                "episode_type": "log_distillation",
                "activity_type": "日志模式蒸馏",
                "event_time": now_utc_str(),
            }],
            absolute_path=absolute_path, total_windows=1)
        documents_created += 1
        episodes_created += ddoc["episodes_created"]
        document_ids.append(ddoc["document_id"])
        distill_doc_id = ddoc["document_id"]

    return {
        "profile": PROFILE_LOG,
        "file": display,
        "lines": len(lines),
        "windows": len(windows),
        "skipped_duplicate_windows": skipped,
        "documents_created": documents_created,
        "episodes_created": episodes_created,
        "patterns_distilled": len(patterns),
        "duration_ms": round((_time.perf_counter() - t0) * 1000, 1),
        "window_mode": mode,
        "document_ids": document_ids,
        "distill_document_id": distill_doc_id,
    }


# ── 切窗 ─────────────────────────────────────────────────────

def _line_char_offsets(lines: List[str]) -> List[int]:
    """每行在整份文本中的起始字符偏移（换行符计 1 字符）。

    与全库约定一致：episode 偏移一律按 Python 字符计数（参考
    ``library_manager._write_retrieval_slices``），不是 UTF-8 字节偏移。
    """
    offsets: List[int] = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line) + 1  # +1 换行符
    return offsets


def _parse_line_timestamp(line: str) -> Optional[datetime]:
    """识别行首时间戳；识别不了返回 None。"""
    m = _TS_ISO_RE.match(line) or _TS_BRACKET_RE.match(line)
    if m:
        return _parse_iso_datetime(m.group(1))
    m = _TS_SYSLOG_RE.match(line)
    if m:
        month = _SYSLOG_MONTHS.get(m.group(1))
        if not month:
            return None
        try:
            hh, mm, ss = (int(x) for x in m.group(3).split(":"))
            # 年份取固定值：窗口切分只看间隔，不依赖真实年份
            return datetime(2000, month, int(m.group(2)), hh, mm, ss,
                            tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _parse_iso_datetime(raw: str) -> Optional[datetime]:
    """把 ISO8601 变体归一成 aware datetime（naive 视作 UTC）。"""
    s = raw.strip().replace(",", ".")
    if len(s) >= 11 and s[10] == " ":
        s = s[:10] + "T" + s[11:]
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    tz_fix = re.search(r"([+-]\d{2})(\d{2})$", s)
    if tz_fix:
        s = s[:tz_fix.start()] + tz_fix.group(1) + ":" + tz_fix.group(2)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _split_windows(lines: List[str], *, time_window_s: float,
                   line_window: int) -> tuple:
    """切块：有时间戳→按时间窗（默认 300s）；无→按行数窗（默认 400 行）。
    首个时间戳之前的前导行单独成窗（无时间戳 → 行数窗语义）。

    Returns:
        (windows, mode)，window 为 dict：
        {first, last, text, start_ts, end_ts}（行号为 0 基）。
    """
    stamps = [_parse_line_timestamp(l) for l in lines]
    windows: List[dict] = []

    def _flush(first: int, last: int):
        # 窗口尾部连续空行不计入（尾随换行 split 出的末尾空行同理）
        while last > first and not lines[last].strip():
            last -= 1
        chunk_lines = lines[first:last + 1]
        if not any(l.strip() for l in chunk_lines):
            return
        ts_in_window = [t for t in stamps[first:last + 1] if t is not None]
        windows.append({
            "first": first, "last": last,
            "text": "\n".join(chunk_lines),
            "start_ts": ts_in_window[0] if ts_in_window else None,
            "end_ts": ts_in_window[-1] if ts_in_window else None,
        })

    if any(t is not None for t in stamps):
        mode = "time"
        base_ts = None
        first = 0
        for i, ts in enumerate(stamps):
            if ts is not None and (base_ts is None or
                                   (ts - base_ts).total_seconds() > time_window_s):
                # 首个时间戳之前的前导行（first < i）单独成窗，不能丢弃
                if first < i:
                    _flush(first, i - 1)
                first = i
                base_ts = ts
        _flush(first, len(lines) - 1)
    else:
        mode = "line"
        step = max(1, int(line_window))
        for first in range(0, len(lines), step):
            _flush(first, min(first + step, len(lines)) - 1)
    return windows, mode


def _window_episode_row(w: dict, index: int, char_starts: List[int]) -> dict:
    """窗口 → episode 行（name=窗序号或时间范围，offset 按整文件字符算）。"""
    seq = f"W{index + 1:03d}"
    if w["start_ts"] is not None and w["end_ts"] is not None:
        name = f"{seq} {_fmt_ts(w['start_ts'])}~{_fmt_ts(w['end_ts'])}"
    else:
        name = seq
    start_off = char_starts[w["first"]]
    end_off = char_starts[w["last"]] + len(w["text"].split("\n")[-1] or "")
    return {
        "name": name,
        "source_text": w["text"],
        "start_offset": start_off,
        "end_offset": end_off,
        "line_start": w["first"] + 1,   # 1 基行号，与 vault_indexer 一致
        "line_end": w["last"] + 1,
        "episode_type": "log_window",
        "activity_type": "日志采集",
        "event_time": _fmt_ts(w["start_ts"]) if w["start_ts"] else now_utc_str(),
    }


def _fmt_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


# ── 模式蒸馏 ─────────────────────────────────────────────────

def _template_line(line: str) -> str:
    """行模板化：uuid/长十六进制→⟨H⟩、IP→⟨IP⟩、引号串→⟨S⟩、数字→⟨N⟩。"""
    s = _UUID_RE.sub(_TPL_HEX, line)
    s = _HEX_RE.sub(_TPL_HEX, s)
    s = _IP_RE.sub(_TPL_IP, s)
    s = _QUOTED_RE.sub(_TPL_STR, s)
    s = _NUM_RE.sub(_TPL_NUM, s)
    return s.strip()


def _distill_patterns(lines: List[str], top_n: int = 30) -> List[dict]:
    """统计各模板出现次数/首末行号/原始样例，取 top_n。

    Returns:
        [{template, count, first_line, last_line, sample}]，按次数降序。
    """
    stats: Dict[str, dict] = {}
    for i, line in enumerate(lines):
        tpl = _template_line(line)
        if not tpl:
            continue
        rec = stats.get(tpl)
        if rec is None:
            stats[tpl] = {
                "template": tpl, "count": 1,
                "first_line": i + 1, "last_line": i + 1,
                "sample": line.strip(),
            }
        else:
            rec["count"] += 1
            rec["last_line"] = i + 1
    ordered = sorted(stats.values(),
                     key=lambda r: (-r["count"], r["first_line"]))
    return ordered[:max(0, int(top_n))]


def _render_distill_markdown(name: str, lines: List[str],
                             patterns: List[dict]) -> str:
    """生成 markdown 蒸馏文档："# {文件名} 日志模式蒸馏" + 每模式一节。"""
    out = [f"# {name} 日志模式蒸馏", ""]
    out.append(f"- 来源: {name}")
    out.append(f"- 总行数: {len(lines)}")
    out.append(f"- 蒸馏模式数: {len(patterns)}")
    out.append(f"- 生成时间: {now_utc_str()}")
    out.append("")
    for rank, p in enumerate(patterns, 1):
        out.append(f"## 模式 {rank}（出现 {p['count']} 次）")
        out.append("")
        out.append(f"- 模板: `{p['template']}`")
        out.append(f"- 行号: {p['first_line']}–{p['last_line']}")
        out.append(f"- 样例: `{p['sample']}`")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


# ── 零 LLM 文档写入（与管线同款库方法/顺序）─────────────────────

def _write_zero_llm_document(storage, *, identity: str, title: str, text: str,
                             episode_rows: List[dict],
                             absolute_path: str = "",
                             total_windows: Optional[int] = None) -> dict:
    """沿管线的写入顺序直建 document/version/episodes + FTS，置 active。

    顺序与 ``_resolve_episode_document`` / ``vault_indexer`` 一致：
    insert_document → (同哈希版本复用并清空其 episodes——含 A→B→A 内容
    回退时复活历史 superseded 版本 / 否则级联快照旧版本 + 新版本)
    → update_current_version → 逐 episode insert + fts_sync →
    set_document_ingestion_state('active')。整个写入在 ``_write_batch``
    临界段内完成，出错整体回滚。
    """
    doc_id = "doc_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    content_hash = content_fs.compute_content_hash(text)
    now = now_utc_str()
    lib_path = str(storage.library_path)

    with storage._write_batch():
        conn = storage._conn()
        managed_path = content_fs.write_current_file(
            lib_path, title or doc_id, text, doc_id=doc_id)
        doc_repo.insert_document(
            conn, doc_id, title, managed_path=managed_path,
            source_mode="managed", absolute_path=absolute_path,
            created_at=now, updated_at=now,
        )

        old_ver = doc_repo.get_active_version(conn, doc_id)
        # 同哈希既有版本：当前 active 命中（重复入库）或历史 superseded 命中
        # （A→B→A 内容回退）都复用版本行——ver_id 由 content_hash 派生，
        # 直接 insert 会撞 UNIQUE(document_id, document_version_id) /
        # UNIQUE(document_id, content_hash)，IntegrityError 把整批回滚。
        reuse_ver = (old_ver if old_ver is not None
                     and old_ver.get("content_hash") == content_hash
                     else doc_repo.get_version_by_hash(conn, doc_id, content_hash))
        if reuse_ver is not None:
            # 复用版本，替换该版本下的 active episodes。
            # episodes 有表级 UNIQUE(document_version_id, chunk_index,
            # chunk_hash)，软 supersede 不释放槽位，必须物理删除后才可
            # 重插同 (版本, 窗口号, 哈希)。仅删无下游引用的行——零 LLM
            # 文档的 episode 永远没有 mentions/observations/assertions。
            ver_id = reuse_ver["document_version_id"]
            if reuse_ver.get("status") != "active":
                # A→B→A 回退：先级联 supersede 当前版本（idx_docver_one_active
                # 每文档只允许一个 active），再复活旧版本并指回 current。
                doc_repo.supersede_active_version_cascade(conn, doc_id)
                doc_repo.reactivate_version(conn, ver_id)
            old_ep_ids = ep_repo.supersede_episodes_by_version(conn, ver_id)
            ep_repo.fts_delete_episodes(conn, old_ep_ids)
            conn.execute(
                """DELETE FROM episodes WHERE document_version_id = ?
                   AND NOT EXISTS (SELECT 1 FROM entity_mentions em
                                    WHERE em.episode_id = episodes.episode_id)
                   AND NOT EXISTS (SELECT 1 FROM entity_observations eo
                                    WHERE eo.episode_id = episodes.episode_id)
                   AND NOT EXISTS (SELECT 1 FROM relation_assertions ra
                                    WHERE ra.episode_id = episodes.episode_id)
                   AND NOT EXISTS (SELECT 1 FROM relation_mentions rm
                                    WHERE rm.episode_id = episodes.episode_id)""",
                (ver_id,),
            )
            doc_repo.update_current_version(conn, doc_id, ver_id, updated_at=now)
        else:
            if old_ver:
                doc_repo.supersede_active_version_cascade(conn, doc_id)
            ver_id = f"docver_{doc_id}_{content_hash[:16]}"
            content_fs.write_version_snapshot(lib_path, doc_id, content_hash, text)
            doc_repo.insert_document_version(
                conn, ver_id, doc_id, content_hash,
                version_content_path=f"content/versions/{doc_id}/{content_hash}.md",
                title=title, char_count=len(text),
                line_count=len(text.splitlines()),
                byte_size=len(text.encode("utf-8")), processed_at=now,
            )
            doc_repo.update_current_version(conn, doc_id, ver_id, updated_at=now)

        written = 0
        for idx, row in enumerate(episode_rows):
            ep_id = f"ep_{uuid.uuid4().hex[:16]}"
            ep_repo.insert_episode(
                conn, ep_id, f"epfam_{doc_id}_{idx}", doc_id, ver_id,
                source_text=row["source_text"], memory_text="",
                start_offset=int(row["start_offset"]),
                end_offset=int(row["end_offset"]),
                line_start=int(row["line_start"]), line_end=int(row["line_end"]),
                chunk_index=idx,
                chunk_hash=hashlib.sha256(
                    row["source_text"].encode("utf-8")).hexdigest()[:16],
                name=row["name"],
                episode_type=row.get("episode_type", "log_window"),
                activity_type=row.get("activity_type", ""),
                event_time=row.get("event_time") or now,
                processed_at=now,
            )
            ep_repo.fts_sync_episode(
                conn, ep_id, doc_id, ver_id,
                name=row["name"], source_text=row["source_text"])
            written += 1

        storage.set_document_ingestion_state(
            doc_id, "active",
            total_windows=total_windows if total_windows is not None else written,
            complete_windows=written,
        )
    return {"document_id": doc_id, "version_id": ver_id,
            "episodes_created": written, "title": title}
