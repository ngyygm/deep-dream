"""
Temporal_Memory_Graph 自然语言记忆图 API

一个以自然语言为核心的统一记忆图服务。系统只有两个核心职责：
  - Remember：接收自然语言文本或文档，自动构建概念实体/关系图。
  - Find：通过语义检索从总图中唤醒相关的局部记忆区域。

所有记忆写入同一张总图（统一大脑），不区分记忆库。
系统不负责 select，外部智能体根据 find 结果自行决策。
"""
from __future__ import annotations

import argparse
import base64
import errno
import json
import os
import queue
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from flask import Flask, jsonify, make_response, request

from config_loader import load_config, resolve_embedding_model
from processor import TemporalMemoryGraphProcessor
from processor.models import Entity, MemoryCache, Relation


# ---------------------------------------------------------------------------
# 文件读取工具
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {".txt", ".md", ".text", ".log", ".csv", ".json", ".xml",
                   ".yaml", ".yml", ".ini", ".conf", ".cfg", ".rst", ".html"}


def _read_file_content(path: str) -> str:
    """读取文件内容为纯文本。支持 txt/md 等纯文本，以及 pdf/docx（需对应依赖）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    ext = p.suffix.lower()

    if ext in _TEXT_EXTENSIONS or ext == "":
        return p.read_text(encoding="utf-8")

    if ext == ".pdf":
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("读取 PDF 需要安装 PyPDF2: pip install PyPDF2")
        text_parts: list[str] = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)

    if ext in (".docx", ".doc"):
        try:
            import docx
        except ImportError:
            raise ImportError("读取 DOCX 需要安装 python-docx: pip install python-docx")
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text)

    return p.read_text(encoding="utf-8")


def entity_to_dict(e: Entity) -> Dict[str, Any]:
    return {
        "id": e.id,
        "entity_id": e.entity_id,
        "name": e.name,
        "content": e.content,
        "physical_time": e.physical_time.isoformat() if e.physical_time else None,
        "memory_cache_id": e.memory_cache_id,
        "doc_name": getattr(e, "doc_name", "") or "",
    }


def relation_to_dict(r: Relation) -> Dict[str, Any]:
    return {
        "id": r.id,
        "relation_id": r.relation_id,
        "entity1_absolute_id": r.entity1_absolute_id,
        "entity2_absolute_id": r.entity2_absolute_id,
        "content": r.content,
        "physical_time": r.physical_time.isoformat() if r.physical_time else None,
        "memory_cache_id": r.memory_cache_id,
        "doc_name": getattr(r, "doc_name", "") or "",
    }


def memory_cache_to_dict(c: MemoryCache) -> Dict[str, Any]:
    return {
        "id": c.id,
        "content": c.content,
        "physical_time": c.physical_time.isoformat() if c.physical_time else None,
        "doc_name": getattr(c, "doc_name", "") or "",
        "activity_type": getattr(c, "activity_type", None),
    }


def ok(data: Any) -> tuple:
    out: Dict[str, Any] = {"success": True, "data": data}
    try:
        if hasattr(request, "start_time"):
            out["elapsed_ms"] = round((time.time() - request.start_time) * 1000, 2)
    except RuntimeError:
        pass
    return jsonify(out), 200


def err(message: str, status: int = 400) -> tuple:
    out: Dict[str, Any] = {"success": False, "error": message}
    try:
        if hasattr(request, "start_time"):
            out["elapsed_ms"] = round((time.time() - request.start_time) * 1000, 2)
    except RuntimeError:
        pass
    return jsonify(out), status


@dataclass
class RememberTask:
    task_id: str
    text: str
    source_name: str
    load_cache: Optional[bool]
    event_time: Optional[datetime]
    original_path: str
    status: str = "queued"          # queued | running | completed | failed
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class RememberJournal:
    """将 remember 任务落盘到 storage_path/remember_journal，进程崩溃重启后可恢复未完成任务。"""

    def __init__(self, storage_root: Path):
        self.dir = Path(storage_root) / "remember_journal"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.json"

    def write(self, task: RememberTask) -> None:
        d: Dict[str, Any] = {
            "task_id": task.task_id,
            "source_name": task.source_name,
            "original_path": task.original_path,
            "status": task.status,
            "event_time": task.event_time.isoformat() if task.event_time else None,
            "load_cache": task.load_cache,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "error": task.error,
            "result": task.result,
        }
        p = self._path(task.task_id)
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        tmp.replace(p)

    def read_record(self, task_id: str) -> Optional[Dict[str, Any]]:
        p = self._path(task_id)
        if not p.exists():
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def iter_records(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        corrupt_dir = self.dir / "corrupt"
        for p in sorted(self.dir.glob("*.json")):
            if p.name.endswith(".tmp"):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception as exc:
                try:
                    corrupt_dir.mkdir(parents=True, exist_ok=True)
                    dest = corrupt_dir / f"{p.stem}.bad.json"
                    if dest.exists():
                        dest = corrupt_dir / f"{p.stem}_{int(time.time())}.bad.json"
                    p.rename(dest)
                    print(
                        f"[remember_journal] 已隔离无法解析的文件: {p.name} -> {dest.name} ({exc})",
                        file=sys.stderr,
                    )
                except Exception:
                    pass
                continue
        return out


def _remember_task_from_record(rec: Dict[str, Any], text: str) -> RememberTask:
    et_raw = rec.get("event_time")
    event_time: Optional[datetime] = None
    if et_raw:
        try:
            event_time = datetime.fromisoformat(str(et_raw).replace("Z", "+00:00"))
        except ValueError:
            event_time = None
    return RememberTask(
        task_id=str(rec["task_id"]),
        text=text,
        source_name=str(rec.get("source_name") or "api_input"),
        load_cache=rec.get("load_cache"),
        event_time=event_time,
        original_path=str(rec.get("original_path") or ""),
        status=str(rec.get("status") or "queued"),
        result=rec.get("result"),
        error=rec.get("error"),
        created_at=float(rec.get("created_at") or time.time()),
        started_at=rec.get("started_at"),
        finished_at=rec.get("finished_at"),
    )


class RememberTaskQueue:
    """异步记忆写入任务队列（两阶段线程模型）。
    每个任务：phase1 生成文档整体记忆，phase2 跑滑窗链。
    A 的 phase1 完成后即可启动 B 的 phase1（B 以 A 的整体记忆为初始）；phase2 串行执行以保持 cache 链一致。
    并行度由 remember_workers 控制（同时进行 phase1 的线程数）。
    任务状态写入 remember_journal，异常退出后重启会重新入队未完成任务（从 originals 原文重跑完整流水线）。"""

    def __init__(
        self,
        processor,
        storage_path: Path,
        max_workers: int = 1,
        max_history: int = 200,
        max_retries: int = 2,
        retry_delay_seconds: float = 2,
    ):
        self._processor = processor
        self._journal = RememberJournal(storage_path)
        self._queue: "queue.Queue[RememberTask]" = queue.Queue()
        self._phase2_queue: "queue.Queue[Tuple[RememberTask, Any]]" = queue.Queue()
        self._tasks: Dict[str, RememberTask] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._max_history = max_history
        self._max_retries = max(0, max_retries)
        self._retry_delay = max(0.0, retry_delay_seconds)
        self._shared_last_overall = None
        self._phase1_done_count = 0
        self._phase2_lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._recover_from_disk()
        for i in range(max(1, max_workers)):
            t = threading.Thread(target=self._worker, name=f"remember-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def _persist(self, task: RememberTask) -> None:
        try:
            self._journal.write(task)
        except Exception as e:
            print(f"[Remember] 警告：journal 写入失败 task_id={task.task_id[:8]}…: {e}")

    def _recover_from_disk(self) -> None:
        n_resume = 0
        for rec in self._journal.iter_records():
            tid = rec.get("task_id")
            if not tid:
                continue
            st = rec.get("status")
            if st in ("completed", "failed"):
                continue
            if st in ("queued", "running"):
                op = rec.get("original_path")
                if not op or not Path(op).exists():
                    rec2 = dict(rec)
                    rec2["status"] = "failed"
                    rec2["error"] = "重启恢复失败：originals 中原文文件不存在"
                    rec2["finished_at"] = time.time()
                    try:
                        tdead = _remember_task_from_record(rec2, text="")
                        self._journal.write(tdead)
                    except Exception:
                        pass
                    print(f"[Remember] 恢复跳过 task_id={str(tid)[:8]}…：原文缺失")
                    continue
                try:
                    text = Path(op).read_text(encoding="utf-8")
                except Exception as e:
                    rec2 = dict(rec)
                    rec2["status"] = "failed"
                    rec2["error"] = f"重启恢复失败：无法读取原文: {e}"
                    rec2["finished_at"] = time.time()
                    try:
                        tdead = _remember_task_from_record(rec2, text="")
                        self._journal.write(tdead)
                    except Exception:
                        pass
                    continue
                task = _remember_task_from_record(rec, text=text)
                task.status = "queued"
                task.started_at = None
                task.finished_at = None
                task.error = None
                task.result = None
                with self._lock:
                    self._tasks[tid] = task
                self._queue.put(task)
                self._persist(task)
                n_resume += 1
                print(f"[Remember] 恢复未完成任务并入队: task_id={tid[:8]}…, source_name={task.source_name!r}")
        if n_resume:
            print(f"[Remember] 启动恢复：重新入队 {n_resume} 个未完成任务（已完成/失败仅保留在 journal，按需通过 status 查询）")

    def submit(self, task: RememberTask) -> str:
        with self._lock:
            self._tasks[task.task_id] = task
            self._trim_history()
        self._persist(task)
        self._queue.put(task)
        print(f"[Remember] 任务入队: task_id={task.task_id[:8]}…, source_name={task.source_name!r}")
        return task.task_id

    def get_status(self, task_id: str) -> Optional[RememberTask]:
        with self._lock:
            t = self._tasks.get(task_id)
        if t is not None:
            return t
        rec = self._journal.read_record(task_id)
        if rec is None:
            return None
        text = ""
        op = rec.get("original_path")
        if op and Path(op).exists():
            try:
                text = Path(op).read_text(encoding="utf-8")
            except Exception:
                pass
        return _remember_task_from_record(rec, text=text)

    def list_tasks(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            items = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
        out = []
        for t in items[:limit]:
            out.append({
                "task_id": t.task_id,
                "source_name": t.source_name,
                "status": t.status,
                "created_at": t.created_at,
                "started_at": t.started_at,
                "finished_at": t.finished_at,
                "error": t.error,
            })
        return out

    def _worker(self):
        """两阶段：phase1 生成整体记忆（可多线程并行），phase2 滑窗串行（单锁）。"""
        while True:
            task = self._queue.get()
            try:
                with self._lock:
                    task.status = "running"
                    task.started_at = time.time()
                self._persist(task)
                print(f"[Remember] 开始处理: task_id={task.task_id[:8]}…, source_name={task.source_name!r}, 文本长度={len(task.text)} 字符")

                # 若非首个任务，等待上一任务 phase1 完成（拿到 previous_overall）
                with self._cond:
                    while self._phase1_done_count > 0 and self._shared_last_overall is None:
                        self._cond.wait()
                previous_overall = self._shared_last_overall

                last_exc = None
                for attempt in range(self._max_retries + 1):
                    try:
                        # Phase1: 仅生成文档整体记忆
                        overall = self._processor.remember_phase1_overall(
                            text=task.text,
                            doc_name=task.source_name,
                            event_time=task.event_time,
                            document_path=task.original_path,
                            previous_overall_cache=previous_overall,
                            verbose=True,
                        )
                        with self._cond:
                            self._shared_last_overall = overall
                            self._phase1_done_count += 1
                            self._cond.notify_all()

                        self._phase2_queue.put((task, overall))

                        # Phase2: 串行执行（processor 的 current_memory_cache 单链，不能多任务并行写）
                        with self._phase2_lock:
                            task2, overall2 = self._phase2_queue.get()
                            result = self._processor.remember_phase2_windows(
                                text=task2.text,
                                doc_name=task2.source_name,
                                verbose=True,
                                event_time=task2.event_time,
                                document_path=task2.original_path,
                                overall_cache=overall2,
                            )
                        result["original_path"] = task2.original_path
                        with self._lock:
                            task2.status = "completed"
                            task2.result = result
                            task2.finished_at = time.time()
                        self._persist(task2)
                        elapsed = (task2.finished_at or 0) - (task2.started_at or 0)
                        print(f"[Remember] 完成: task_id={task2.task_id[:8]}…, chunks_processed={result.get('chunks_processed')}, 耗时={elapsed:.1f}s")
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        if attempt < self._max_retries:
                            delay = self._retry_delay
                            print(f"[Remember] 失败将重试: task_id={task.task_id[:8]}…, attempt={attempt + 1}, error={exc!r}, {delay}s 后重试")
                            time.sleep(delay)
                        else:
                            with self._lock:
                                task.status = "failed"
                                task.error = str(exc)
                                task.finished_at = time.time()
                            self._persist(task)
                            print(f"[Remember] 失败: task_id={task.task_id[:8]}…, error={exc!r}")
            except Exception as exc:
                with self._lock:
                    task.status = "failed"
                    task.error = str(exc)
                    task.finished_at = time.time()
                self._persist(task)
                print(f"[Remember] 失败: task_id={task.task_id[:8]}…, error={exc!r}")
            finally:
                self._queue.task_done()

    def _trim_history(self):
        if len(self._tasks) <= self._max_history:
            return
        items = sorted(self._tasks.values(), key=lambda t: t.created_at)
        to_remove = len(self._tasks) - self._max_history
        removed = 0
        for t in items:
            if t.status in ("completed", "failed") and removed < to_remove:
                del self._tasks[t.task_id]
                removed += 1


def _extract_candidate_ids(
    storage: Any,
    body: Dict[str, Any],
    parse_time_point: Any,
) -> Tuple[Set[str], Set[str]]:
    """按 query_text / 时间等条件从主图抽取实体与关系的 absolute id 集合（供 query-one 等接口使用）。"""
    entity_absolute_ids: Set[str] = set()
    relation_absolute_ids: Set[str] = set()
    time_before = body.get("time_before")
    time_after = body.get("time_after")
    max_entities = body.get("max_entities")
    if max_entities is None:
        max_entities = 100
    max_relations = body.get("max_relations")
    if max_relations is None:
        max_relations = 500
    time_before_dt = parse_time_point(time_before) if time_before else None
    time_after_dt = parse_time_point(time_after) if time_after else None

    entity_name = (body.get("entity_name") or body.get("query_text") or "").strip()
    if entity_name:
        entities = storage.search_entities_by_similarity(
            query_name=entity_name,
            query_content=body.get("query_text") or entity_name,
            threshold=float(body.get("similarity_threshold", 0.7)),
            max_results=int(max_entities),
            text_mode=body.get("text_mode") or "name_and_content",
            similarity_method=body.get("similarity_method") or "embedding",
        )
        for e in entities:
            entity_absolute_ids.add(e.id)
    elif time_before_dt:
        entities = storage.get_all_entities_before_time(time_before_dt, limit=max_entities)
        for e in entities:
            entity_absolute_ids.add(e.id)
    else:
        entities = storage.get_all_entities(limit=max_entities)
        for e in entities:
            entity_absolute_ids.add(e.id)

    if not entity_absolute_ids:
        return entity_absolute_ids, relation_absolute_ids

    relations = storage.get_relations_by_entity_absolute_ids(
        list(entity_absolute_ids), limit=max_relations
    )
    for r in relations:
        if time_before_dt and r.physical_time and r.physical_time > time_before_dt:
            continue
        if time_after_dt and r.physical_time and r.physical_time < time_after_dt:
            continue
        relation_absolute_ids.add(r.id)
    drop_entities = set()
    for eid in entity_absolute_ids:
        e = storage.get_entity_by_absolute_id(eid)
        if e and e.physical_time:
            if time_before_dt and e.physical_time > time_before_dt:
                drop_entities.add(eid)
            elif time_after_dt and e.physical_time < time_after_dt:
                drop_entities.add(eid)
    entity_absolute_ids -= drop_entities
    return entity_absolute_ids, relation_absolute_ids


def create_app(processor: TemporalMemoryGraphProcessor, config: Optional[Dict[str, Any]] = None) -> Flask:
    app = Flask(__name__)

    # 允许测试页（不同端口）跨域调用 API，避免浏览器报 Failed to fetch
    @app.after_request
    def _cors_headers(response):
        origin = request.environ.get("HTTP_ORIGIN")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
        else:
            response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.before_request
    def _cors_preflight():
        if request.method == "OPTIONS":
            return make_response("", 204)

    @app.before_request
    def _record_start():
        request.start_time = time.time()

    config = config or {}
    remember_queue = RememberTaskQueue(
        processor,
        Path(processor.storage.storage_path),
        max_workers=config.get("remember_workers", 1),
        max_retries=config.get("remember_max_retries", 2),
        retry_delay_seconds=config.get("remember_retry_delay_seconds", 2),
    )

    def parse_time_point(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("time_point 需为 ISO 格式")

    @app.route("/health")
    def health():
        try:
            embedding_available = (
                processor.embedding_client is not None
                and processor.embedding_client.is_available()
            )
            return ok({
                "storage_path": str(processor.storage.storage_path),
                "embedding_available": embedding_available,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/health/llm")
    def health_llm():
        """检查大模型是否可访问：做一次极简调用，成功才返回 200。"""
        try:
            # 极简提示、短超时、单次重试，仅用于连通性检查
            response = processor.llm_client._call_llm(
                "请只回复一个词：OK",
                max_retries=1,
                timeout=60,
                allow_mock_fallback=False,
            )
            ok_result = (
                response is not None
                and isinstance(response, str)
                and len(response.strip()) > 0
            )
            if ok_result:
                return ok({"llm_available": True, "message": "大模型访问正常"})
            return err("大模型未返回有效结果", 503)
        except Exception as e:
            return err(f"大模型不可用: {e}", 503)

    def _parse_bool_query(name: str) -> Optional[bool]:
        v = request.args.get(name)
        if v is None or v == "":
            return None
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        return None

    @app.route("/api/remember", methods=["GET"])
    def remember():
        """记忆写入（仅 GET）：查询参数入队，立即返回 task_id。原文写入 originals/ 与 remember_journal/，进程崩溃重启后会自动恢复未完成任务并重跑。

        查询参数：
          - text 或 text_b64（二选一必填）：正文；长文本建议用标准 Base64 的 text_b64，避免 URL 长度限制
          - source_name / doc_name（可选）：来源，默认 api_input
          - load_cache_memory（可选）：1/true/0/false
          - event_time（可选）：ISO 8601 事件时间

        返回：HTTP 202。查询进度：GET /api/remember/status/<task_id>
        """
        try:
            text = (request.args.get("text") or "").strip()
            b64 = (request.args.get("text_b64") or "").strip()
            if b64:
                try:
                    pad = (-len(b64)) % 4
                    if pad:
                        b64 += "=" * pad
                    text = base64.b64decode(b64).decode("utf-8")
                except Exception:
                    return err("text_b64 不是有效的 UTF-8 Base64 内容", 400)
            if not text:
                return err("缺少 text 或 text_b64（必填其一）", 400)

            source_name = (
                (request.args.get("source_name") or request.args.get("doc_name") or "").strip() or "api_input"
            )
            load_cache = _parse_bool_query("load_cache_memory")

            event_time: Optional[datetime] = None
            et_str = (request.args.get("event_time") or "").strip() or None
            if et_str:
                try:
                    event_time = datetime.fromisoformat(et_str.replace("Z", "+00:00"))
                except ValueError:
                    return err("event_time 需为 ISO 8601 格式", 400)

            originals_dir = Path(processor.storage.storage_path) / "originals"
            originals_dir.mkdir(parents=True, exist_ok=True)
            safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in source_name)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_id = uuid.uuid4().hex[:8]
            original_filename = f"{safe_name}_{ts}_{short_id}.txt"
            original_path = str((originals_dir / original_filename).resolve())
            Path(original_path).write_text(text, encoding="utf-8")

            preview = (text[:80] + "…") if len(text) > 80 else text
            print(f"[Remember] 收到(GET): source_name={source_name!r}, 文本长度={len(text)} 字符, event_time={et_str or '未指定'}")
            print(f"[Remember] 内容预览: {preview!r}")

            task_id = uuid.uuid4().hex
            task = RememberTask(
                task_id=task_id,
                text=text,
                source_name=source_name,
                load_cache=load_cache,
                event_time=event_time,
                original_path=original_path,
            )

            remember_queue.submit(task)
            return make_response(jsonify({
                "success": True,
                "data": {
                    "task_id": task_id,
                    "status": "queued",
                    "message": "已加入队列；Find 与 Remember 可并发。崩溃重启后未完成任务会从 journal 恢复。GET /api/remember/status/<task_id> 查询进度",
                    "original_path": original_path,
                },
            }), 202)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/remember/status/<task_id>")
    def remember_status(task_id: str):
        """查询异步记忆写入任务的状态。"""
        try:
            t = remember_queue.get_status(task_id)
            if t is None:
                return err("任务不存在", 404)
            data: Dict[str, Any] = {
                "task_id": t.task_id,
                "status": t.status,
                "source_name": t.source_name,
                "original_path": t.original_path,
                "created_at": t.created_at,
                "started_at": t.started_at,
                "finished_at": t.finished_at,
            }
            if t.status == "completed" and t.result:
                data["result"] = t.result
            if t.status == "failed" and t.error:
                data["error"] = t.error
            return ok(data)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/remember/queue")
    def remember_queue_list():
        """查看记忆写入任务队列。"""
        try:
            limit = request.args.get("limit", 50, type=int)
            tasks = remember_queue.list_tasks(limit=limit)
            return ok({"tasks": tasks, "count": len(tasks)})
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 统计
    # =========================================================
    @app.route("/api/find/stats")
    def find_stats():
        try:
            total_entities = len(processor.storage.get_all_entities(limit=None))
            total_relations = len(processor.storage.get_all_relations())

            cache_json_dir = processor.storage.cache_json_dir
            cache_dir = processor.storage.cache_dir
            json_files = list(cache_json_dir.glob("*.json"))
            if json_files:
                total_memory_caches = len(json_files)
            else:
                total_memory_caches = len(list(cache_dir.glob("*.json")))

            return ok({
                "total_entities": total_entities,
                "total_relations": total_relations,
                "total_memory_caches": total_memory_caches,
            })
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 统一语义检索入口（推荐）
    # =========================================================
    @app.route("/api/find", methods=["POST"])
    def find_unified():
        """统一语义检索：用自然语言从总记忆图中唤醒相关的局部区域。

        请求体:
            query (str, 必填): 自然语言查询
            similarity_threshold (float): 语义相似度阈值，默认 0.5
            max_entities (int): 最大返回实体数，默认 20
            max_relations (int): 最大返回关系数，默认 50
            expand (bool): 是否从命中实体向外扩展邻域，默认 true
            time_before (str, ISO): 只返回此时间之前的记忆
            time_after (str, ISO): 只返回此时间之后的记忆

        返回:
            entities: 命中的概念实体列表
            relations: 命中的概念关系列表
        """
        try:
            body = request.get_json(silent=True) or {}
            query = (body.get("query") or "").strip()
            if not query:
                return err("query 为必填字段", 400)

            similarity_threshold = float(body.get("similarity_threshold", 0.5))
            max_entities = int(body.get("max_entities", 20))
            max_relations = int(body.get("max_relations", 50))
            expand = body.get("expand", True)
            time_before = body.get("time_before")
            time_after = body.get("time_after")

            try:
                time_before_dt = parse_time_point(time_before) if time_before else None
                time_after_dt = parse_time_point(time_after) if time_after else None
            except ValueError as ve:
                return err(str(ve), 400)

            storage = processor.storage

            # --- 第一步：语义召回实体 ---
            matched_entities = storage.search_entities_by_similarity(
                query_name=query,
                query_content=query,
                threshold=similarity_threshold,
                max_results=max_entities,
                text_mode="name_and_content",
                similarity_method="embedding",
            )

            # --- 第二步：语义召回关系 ---
            matched_relations = storage.search_relations_by_similarity(
                query_text=query,
                threshold=similarity_threshold,
                max_results=max_relations,
            )

            entity_abs_ids: Set[str] = {e.id for e in matched_entities}
            relation_abs_ids: Set[str] = {r.id for r in matched_relations}
            entities_by_abs: Dict[str, Entity] = {e.id: e for e in matched_entities}

            # --- 第三步：从语义命中的关系中补充关联实体 ---
            for r in list(matched_relations):
                for abs_id in (r.entity1_absolute_id, r.entity2_absolute_id):
                    if abs_id not in entity_abs_ids:
                        e = storage.get_entity_by_absolute_id(abs_id)
                        if e:
                            entities_by_abs[e.id] = e
                            entity_abs_ids.add(e.id)

            # --- 第四步：图谱邻域扩展 ---
            if expand and entity_abs_ids:
                expanded_rels = storage.get_relations_by_entity_absolute_ids(
                    list(entity_abs_ids), limit=max_relations
                )
                for r in expanded_rels:
                    if r.id not in relation_abs_ids:
                        relation_abs_ids.add(r.id)
                        matched_relations.append(r)
                    for abs_id in (r.entity1_absolute_id, r.entity2_absolute_id):
                        if abs_id not in entity_abs_ids:
                            e = storage.get_entity_by_absolute_id(abs_id)
                            if e:
                                entities_by_abs[e.id] = e
                                entity_abs_ids.add(e.id)

            # --- 第五步：时间过滤 ---
            final_entities: List[Entity] = []
            for e in entities_by_abs.values():
                if time_before_dt and e.physical_time and e.physical_time > time_before_dt:
                    continue
                if time_after_dt and e.physical_time and e.physical_time < time_after_dt:
                    continue
                final_entities.append(e)

            final_relations: List[Relation] = []
            seen_rel_ids: Set[str] = set()
            for r in matched_relations:
                if r.id in seen_rel_ids:
                    continue
                if time_before_dt and r.physical_time and r.physical_time > time_before_dt:
                    continue
                if time_after_dt and r.physical_time and r.physical_time < time_after_dt:
                    continue
                seen_rel_ids.add(r.id)
                final_relations.append(r)

            result: Dict[str, Any] = {
                "query": query,
                "entities": [entity_to_dict(e) for e in final_entities],
                "relations": [relation_to_dict(r) for r in final_relations],
                "entity_count": len(final_entities),
                "relation_count": len(final_relations),
            }

            return ok(result)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/query-one", methods=["POST"])
    def find_query_one():
        """按请求体中的 query_text / similarity_threshold 等条件从主图抽取一批实体与关系（一次性返回，无额外状态）。"""
        try:
            body = request.get_json(silent=True) or {}
            include_entities = body.get("include_entities", True)
            include_relations = body.get("include_relations", True)
            try:
                entity_ids, relation_ids = _extract_candidate_ids(
                    processor.storage, body, parse_time_point
                )
            except ValueError as ve:
                return err(str(ve), 400)
            storage = processor.storage
            entities_data: List[Dict[str, Any]] = []
            relations_data: List[Dict[str, Any]] = []
            if include_entities:
                for eid in entity_ids:
                    e = storage.get_entity_by_absolute_id(eid)
                    if e:
                        entities_data.append(entity_to_dict(e))
            if include_relations:
                for rid in relation_ids:
                    r = storage.get_relation_by_absolute_id(rid)
                    if r:
                        relations_data.append(relation_to_dict(r))
            return ok({"entities": entities_data, "relations": relations_data})
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 实体原子接口
    # =========================================================
    @app.route("/api/find/entities/all")
    def find_entities_all():
        try:
            limit = request.args.get("limit", type=int)
            entities = processor.storage.get_all_entities(limit=limit)
            return ok([entity_to_dict(e) for e in entities])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/entities/all-before-time")
    def find_entities_all_before_time():
        try:
            time_point_str = request.args.get("time_point")
            if not time_point_str:
                return err("time_point 为必填参数（ISO 格式）", 400)
            try:
                time_point = parse_time_point(time_point_str)
            except ValueError as ve:
                return err(str(ve), 400)
            limit = request.args.get("limit", type=int)
            entities = processor.storage.get_all_entities_before_time(time_point, limit=limit)
            return ok([entity_to_dict(e) for e in entities])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/entities/version-counts", methods=["POST"])
    def find_entity_version_counts():
        try:
            body = request.get_json(silent=True) or {}
            entity_ids = body.get("entity_ids")
            if not isinstance(entity_ids, list) or not all(isinstance(x, str) for x in entity_ids):
                return err("请求体需包含 entity_ids 字符串数组", 400)
            counts = processor.storage.get_entity_version_counts(entity_ids)
            return ok(counts)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/entities/by-absolute-id/<absolute_id>/embedding-preview")
    def find_entity_embedding_preview(absolute_id: str):
        try:
            num_values = request.args.get("num_values", type=int, default=5)
            preview = processor.storage.get_entity_embedding_preview(absolute_id, num_values=num_values)
            if preview is None:
                return err(f"未找到实体 embedding 或实体不存在: {absolute_id}", 404)
            return ok({"absolute_id": absolute_id, "values": preview})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/entities/by-absolute-id/<absolute_id>")
    def find_entity_by_absolute_id(absolute_id: str):
        try:
            entity = processor.storage.get_entity_by_absolute_id(absolute_id)
            if entity is None:
                return err(f"未找到实体版本: {absolute_id}", 404)
            return ok(entity_to_dict(entity))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/entities/search")
    def find_entities_search():
        try:
            query_name = (request.args.get("query_name") or "").strip()
            if not query_name:
                return err("query_name 为必填参数", 400)
            query_content = request.args.get("query_content") or None
            threshold = request.args.get("threshold", type=float, default=0.7)
            max_results = request.args.get("max_results", type=int, default=10)
            text_mode = request.args.get("text_mode") or "name_and_content"
            if text_mode not in ("name_only", "content_only", "name_and_content"):
                text_mode = "name_and_content"
            similarity_method = request.args.get("similarity_method") or "embedding"
            if similarity_method not in ("embedding", "text", "jaccard", "bleu"):
                similarity_method = "embedding"
            content_snippet_length = request.args.get("content_snippet_length", type=int, default=50)

            entities = processor.storage.search_entities_by_similarity(
                query_name=query_name,
                query_content=query_content,
                threshold=threshold,
                max_results=max_results,
                content_snippet_length=content_snippet_length,
                text_mode=text_mode,
                similarity_method=similarity_method,
            )
            return ok([entity_to_dict(e) for e in entities])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/entities/<entity_id>/versions")
    def find_entity_versions(entity_id: str):
        try:
            versions = processor.storage.get_entity_versions(entity_id)
            return ok([entity_to_dict(e) for e in versions])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/entities/<entity_id>/at-time")
    def find_entity_at_time(entity_id: str):
        try:
            time_point_str = request.args.get("time_point")
            if not time_point_str:
                return err("time_point 为必填参数（ISO 格式）", 400)
            try:
                time_point = parse_time_point(time_point_str)
            except ValueError as ve:
                return err(str(ve), 400)
            entity = processor.storage.get_entity_version_at_time(entity_id, time_point)
            if entity is None:
                return err(f"未找到该时间点版本: {entity_id}", 404)
            return ok(entity_to_dict(entity))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/entities/<entity_id>/version-count")
    def find_entity_version_count(entity_id: str):
        try:
            count = processor.storage.get_entity_version_count(entity_id)
            if count <= 0:
                return err(f"未找到实体: {entity_id}", 404)
            return ok({"entity_id": entity_id, "version_count": count})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/entities/<entity_id>")
    def find_entity_by_id(entity_id: str):
        try:
            entity = processor.storage.get_entity_by_id(entity_id)
            if entity is None:
                return err(f"未找到实体: {entity_id}", 404)
            return ok(entity_to_dict(entity))
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 关系原子接口
    # =========================================================
    @app.route("/api/find/relations/all")
    def find_relations_all():
        try:
            relations = processor.storage.get_all_relations()
            return ok([relation_to_dict(r) for r in relations])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/relations/search")
    def find_relations_search():
        try:
            query_text = (request.args.get("query_text") or "").strip()
            if not query_text:
                return err("query_text 为必填参数", 400)
            threshold = request.args.get("threshold", type=float, default=0.3)
            max_results = request.args.get("max_results", type=int, default=10)
            relations = processor.storage.search_relations_by_similarity(
                query_text=query_text,
                threshold=threshold,
                max_results=max_results,
            )
            return ok([relation_to_dict(r) for r in relations])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/relations/between")
    def find_relations_between():
        try:
            from_entity_id = (request.args.get("from_entity_id") or "").strip()
            to_entity_id = (request.args.get("to_entity_id") or "").strip()
            if not from_entity_id or not to_entity_id:
                return err("from_entity_id 与 to_entity_id 为必填参数", 400)
            relations = processor.storage.get_relations_by_entities(from_entity_id, to_entity_id)
            return ok([relation_to_dict(r) for r in relations])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/relations/by-absolute-id/<absolute_id>/embedding-preview")
    def find_relation_embedding_preview(absolute_id: str):
        try:
            num_values = request.args.get("num_values", type=int, default=5)
            preview = processor.storage.get_relation_embedding_preview(absolute_id, num_values=num_values)
            if preview is None:
                return err(f"未找到关系 embedding 或关系不存在: {absolute_id}", 404)
            return ok({"absolute_id": absolute_id, "values": preview})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/relations/by-absolute-id/<entity_absolute_id>")
    def find_relations_by_entity_absolute_id(entity_absolute_id: str):
        try:
            limit = request.args.get("limit", type=int)
            time_point_str = request.args.get("time_point")
            try:
                time_point = parse_time_point(time_point_str)
            except ValueError as ve:
                return err(str(ve), 400)
            relations = processor.storage.get_entity_relations(
                entity_absolute_id=entity_absolute_id,
                limit=limit,
                time_point=time_point,
            )
            return ok([relation_to_dict(r) for r in relations])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/relations/<relation_id>/versions")
    def find_relation_versions(relation_id: str):
        try:
            versions = processor.storage.get_relation_versions(relation_id)
            return ok([relation_to_dict(r) for r in versions])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/relations/by-entity/<entity_id>")
    def find_relations_by_entity(entity_id: str):
        try:
            limit = request.args.get("limit", type=int)
            time_point_str = request.args.get("time_point")
            try:
                time_point = parse_time_point(time_point_str)
            except ValueError as ve:
                return err(str(ve), 400)
            max_version_absolute_id = (request.args.get("max_version_absolute_id") or "").strip() or None
            relations = processor.storage.get_entity_relations_by_entity_id(
                entity_id=entity_id,
                limit=limit,
                time_point=time_point,
                max_version_absolute_id=max_version_absolute_id,
            )
            return ok([relation_to_dict(r) for r in relations])
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 记忆缓存原子接口
    # =========================================================
    @app.route("/api/find/memory-cache/latest/metadata")
    def find_latest_memory_cache_metadata():
        try:
            activity_type = request.args.get("activity_type")
            metadata = processor.storage.get_latest_memory_cache_metadata(activity_type=activity_type)
            if metadata is None:
                return err("未找到记忆缓存元数据", 404)
            return ok(metadata)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/memory-cache/latest")
    def find_latest_memory_cache():
        try:
            activity_type = request.args.get("activity_type")
            cache = processor.storage.get_latest_memory_cache(activity_type=activity_type)
            if cache is None:
                return err("未找到记忆缓存", 404)
            return ok(memory_cache_to_dict(cache))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/memory-cache/<cache_id>/text")
    def find_memory_cache_text(cache_id: str):
        try:
            text = processor.storage.get_memory_cache_text(cache_id)
            if text is None:
                return err(f"未找到记忆缓存或原文: {cache_id}", 404)
            return ok({"cache_id": cache_id, "text": text})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/memory-cache/<cache_id>")
    def find_memory_cache(cache_id: str):
        try:
            cache = processor.storage.load_memory_cache(cache_id)
            if cache is None:
                return err(f"未找到记忆缓存: {cache_id}", 404)
            return ok(memory_cache_to_dict(cache))
        except Exception as e:
            return err(str(e), 500)

    return app


def _apply_thread_cap(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    按 max_total_worker_threads 限制总线程数，避免线程爆炸。
    三层线程（优先级从高到低，超出时优先缩小低优先级）：
      1. remember_workers — 队列 worker 数（最高优先级，保证接活能力）
      2. pipeline.max_concurrent_windows — 单任务内并行滑窗数
      3. pipeline.llm_threads — 单窗口内实体/关系并行数（最低优先级，最先被缩小）
    峰值估算：remember_workers + max_concurrent_windows + max_concurrent_windows * llm_threads
    """
    cap = config.get("max_total_worker_threads")
    if cap is None or cap < 1:
        return config
    pipeline = config.get("pipeline") or {}
    rw = max(1, config.get("remember_workers", 1))
    mcw = max(1, pipeline.get("max_concurrent_windows", 1))
    lt = max(1, pipeline.get("llm_threads", 1))
    rw_orig, mcw_orig, lt_orig = rw, mcw, lt

    def peak():
        return rw + mcw + mcw * lt

    while peak() > cap:
        if lt > 1:
            lt -= 1
        elif mcw > 1:
            mcw -= 1
        elif rw > 1:
            rw -= 1
        else:
            break

    config = dict(config)
    config["remember_workers"] = rw
    if "pipeline" not in config or config["pipeline"] is None:
        config["pipeline"] = {}
    config["pipeline"] = dict(config["pipeline"])
    config["pipeline"]["max_concurrent_windows"] = mcw
    config["pipeline"]["llm_threads"] = lt
    if (rw, mcw, lt) != (rw_orig, mcw_orig, lt_orig):
        print(f"[线程上限] max_total_worker_threads={cap} 已收紧: remember_workers {rw_orig}→{rw}, max_concurrent_windows {mcw_orig}→{mcw}, llm_threads {lt_orig}→{lt} (峰值≈{peak()})")
    return config


def build_processor(config: Dict[str, Any]) -> TemporalMemoryGraphProcessor:
    storage_path = config.get("storage_path", "./graph/tmg_storage")
    chunking = config.get("chunking") or {}
    window_size = chunking.get("window_size", 1000)
    overlap = chunking.get("overlap", 200)
    llm = config.get("llm") or {}
    embedding = config.get("embedding") or {}
    pipeline = config.get("pipeline") or {}
    # llm_threads 支持写在顶层或 pipeline 下，优先 pipeline
    llm_threads = pipeline.get("llm_threads", config.get("llm_threads", 1))
    model_path, model_name, use_local = resolve_embedding_model(embedding)
    kwargs: Dict[str, Any] = {
        "storage_path": storage_path,
        "window_size": window_size,
        "overlap": overlap,
        "llm_api_key": llm.get("api_key"),
        "llm_model": llm.get("model", "gpt-4"),
        "llm_base_url": llm.get("base_url"),
        "llm_think_mode": bool(llm.get("think", llm.get("think_mode", False))),
        "embedding_model_path": model_path,
        "embedding_model_name": model_name,
        "embedding_device": embedding.get("device", "cpu"),
        "embedding_use_local": use_local,
        "llm_threads": llm_threads,
    }
    # pipeline 下其余参数（仅传入有写的键，避免覆盖为 None）
    for key in (
        "similarity_threshold", "max_similar_entities", "content_snippet_length",
        "relation_content_snippet_length", "entity_extraction_max_iterations",
        "entity_extraction_iterative", "entity_post_enhancement",
        "relation_extraction_max_iterations", "relation_extraction_absolute_max_iterations",
        "relation_extraction_iterative", "load_cache_memory",
        "jaccard_search_threshold", "embedding_name_search_threshold", "embedding_full_search_threshold",
        "max_concurrent_windows",
    ):
        if key in pipeline:
            kwargs[key] = pipeline[key]
    return TemporalMemoryGraphProcessor(**kwargs)


def _check_llm_available(processor) -> tuple[bool, str | None]:
    """启动前握手：检查配置的 LLM 是否可用。返回 (成功, 错误信息)，失败时错误信息非空。"""
    try:
        response = processor.llm_client._call_llm(
            "请只回复一个词：OK",
            max_retries=1,
            timeout=60,
            allow_mock_fallback=False,
        )
        ok_result = (
            response is not None
            and isinstance(response, str)
            and len(response.strip()) > 0
        )
        if ok_result:
            return True, None
        return False, "大模型未返回有效结果（可能超时或网络不可达）"
    except Exception as e:
        return False, f"大模型不可用: {e}"


def _tcp_bind_probe(host: str, port: int) -> Tuple[bool, Optional[str]]:
    """尝试在 host:port 上独占 bind，用于启动前检测端口是否可用。"""
    bind_addr = host if host not in ("", "0.0.0.0") else "0.0.0.0"
    sock: Optional[socket.socket] = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_addr, int(port)))
        return True, None
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            return False, "端口已被占用 (EADDRINUSE)"
        return False, str(e)
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def _resolve_listen_port(
    host: str,
    preferred_port: int,
    auto_fallback: bool,
    max_extra: int = 10,
) -> Tuple[int, bool]:
    """
    若 preferred_port 可 bind 则用之；否则在 auto_fallback 时尝试 preferred_port+1 … +max_extra。
    返回 (实际端口, 是否发生了端口切换)。
    """
    ok, _ = _tcp_bind_probe(host, preferred_port)
    if ok:
        return preferred_port, False
    if not auto_fallback:
        return preferred_port, False
    for delta in range(1, max_extra + 1):
        p = preferred_port + delta
        ok2, _ = _tcp_bind_probe(host, p)
        if ok2:
            return p, True
    return preferred_port, False


def _check_storage_writable(storage_root: Path) -> Optional[str]:
    """在 storage_path 下尝试创建/删除测试文件，不可写则返回错误说明。"""
    probe = storage_root / ".tmg_write_probe"
    try:
        storage_root.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return None
    except OSError as e:
        return f"存储路径不可写或无法创建: {storage_root} ({e})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Temporal_Memory_Graph 自然语言记忆图 API（Remember + Find）")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径（如 service_config.json）")
    parser.add_argument("--host", type=str, default=None, help="覆盖配置中的 host")
    parser.add_argument("--port", type=int, default=None, help="覆盖配置中的 port")
    parser.add_argument(
        "--skip-llm-check",
        action="store_true",
        help="跳过启动前 LLM 握手（仅适合调试 Find；Remember 仍可能在运行时失败）",
    )
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="若配置端口被占用，自动尝试后续连续端口（最多 +10）",
    )
    parser.add_argument("--debug", action="store_true", help="开启 Flask 调试模式")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        print(f"错误：配置文件不存在: {config_path}")
        return 1

    config = load_config(config_path)
    config = _apply_thread_cap(config)
    host = args.host if args.host is not None else config.get("host", "0.0.0.0")
    port = args.port if args.port is not None else config.get("port", 5001)
    storage_path = config.get("storage_path", "./graph/tmg_storage")
    storage_root = Path(storage_path)
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    wr_err = _check_storage_writable(storage_root)
    if wr_err:
        print(f"错误：{wr_err}")
        return 1

    processor = build_processor(config)

    # 启动前对配置的 LLM 做握手，不可用则报错退出（可用 --skip-llm-check 跳过）
    if args.skip_llm_check:
        print("已跳过 LLM 握手（--skip-llm-check）。Remember 与 /health/llm 可能在运行时失败。")
    else:
        print("正在检查配置的 LLM 是否可用…")
        ok_llm, err_msg = _check_llm_available(processor)
        if not ok_llm:
            print(f"错误：{err_msg}")
            print("请检查 service_config 中 llm.api_key / llm.base_url / llm.model 及网络。")
            print("若仅需测试 Find，可加参数: --skip-llm-check")
            return 1
        print("LLM 握手成功，模型可用。")

    app = create_app(processor, config)

    # 启动时加载当前大脑记忆库统计并输出
    try:
        total_entities = len(processor.storage.get_all_entities(limit=None))
        total_relations = len(processor.storage.get_all_relations())
        cache_json_dir = processor.storage.cache_json_dir
        cache_dir = processor.storage.cache_dir
        json_files = list(cache_json_dir.glob("*.json"))
        total_memory_caches = len(json_files) if json_files else len(list(cache_dir.glob("*.json")))
    except Exception:
        total_entities = total_relations = total_memory_caches = 0

    auto_fb = bool(args.auto_port or config.get("auto_port_fallback", False))
    listen_port, port_switched = _resolve_listen_port(host, port, auto_fb)
    ok_bind, bind_err = _tcp_bind_probe(host, listen_port)
    if not ok_bind:
        print(f"错误：无法在 {host}:{listen_port} 上绑定: {bind_err}")
        print(f"  配置的端口为 {port}。")
        if not auto_fb:
            print("  解决：结束占用该端口的进程，或改用 --port <其他端口>，或在配置中设置 auto_port_fallback: true 并加 --auto-port。")
            try:
                print(f"  排查示例: ss -tlnp | grep ':{port} ' 或 lsof -i :{port}")
            except Exception:
                pass
        else:
            print("  已尝试自动换端口但仍失败，请检查系统权限或防火墙设置。")
        return 1
    if port_switched:
        print(f"注意：端口 {port} 已被占用，已自动改用 {listen_port}。")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║     Temporal_Memory_Graph — 自然语言记忆图 API           ║
╚══════════════════════════════════════════════════════════╝

  当前大脑记忆库:
    实体: {total_entities}  关系: {total_relations}  记忆缓存: {total_memory_caches}

  服务地址: http://{host}:{listen_port}
  健康检查: GET  http://{host}:{listen_port}/health
  记忆写入: GET  http://{host}:{listen_port}/api/remember?text=...（或 text_b64=...）
  语义检索: POST http://{host}:{listen_port}/api/find
  原子查询: GET  http://{host}:{listen_port}/api/find/...

  存储路径: {storage_path}
  HTTP 多线程: 处理中 Find 与 Remember 可并行（Flask threaded）

  按 Ctrl+C 停止服务
""")
    threaded = bool(config.get("flask_threaded", True))
    try:
        app.run(host=host, port=listen_port, debug=args.debug, threaded=threaded)
    except OSError as e:
        print(f"错误：HTTP 服务启动失败: {e}", file=sys.stderr)
        if e.errno == errno.EADDRINUSE:
            print("  端口在探测后仍被占用（竞态），请重试或更换端口。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
