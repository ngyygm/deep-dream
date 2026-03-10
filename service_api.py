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
import os
import sys
import tempfile
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
class SubgraphEntry:
    entity_absolute_ids: Set[str] = field(default_factory=set)
    relation_absolute_ids: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    ttl_seconds: int = 3600

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    def touch(self) -> None:
        self.last_accessed_at = time.time()


class SubgraphStore:
    def __init__(self, max_count: int = 100, default_ttl_seconds: int = 3600):
        self._store: Dict[str, SubgraphEntry] = {}
        self._max_count = max_count
        self._default_ttl = default_ttl_seconds
        self._lock = threading.Lock()

    def create(self, entity_ids: Set[str], relation_ids: Set[str], ttl_seconds: Optional[int] = None) -> str:
        with self._lock:
            self._evict_if_needed()
            sid = str(uuid.uuid4())
            self._store[sid] = SubgraphEntry(
                entity_absolute_ids=set(entity_ids),
                relation_absolute_ids=set(relation_ids),
                ttl_seconds=ttl_seconds or self._default_ttl,
            )
            return sid

    def get(self, subgraph_id: str) -> Optional[SubgraphEntry]:
        with self._lock:
            entry = self._store.get(subgraph_id)
            if entry is None or entry.is_expired():
                if entry and entry.is_expired():
                    del self._store[subgraph_id]
                return None
            entry.touch()
            return entry

    def delete(self, subgraph_id: str) -> bool:
        with self._lock:
            if subgraph_id in self._store:
                del self._store[subgraph_id]
                return True
            return False

    def _evict_if_needed(self) -> None:
        while len(self._store) >= self._max_count:
            expired = [k for k, v in self._store.items() if v.is_expired()]
            for k in expired:
                del self._store[k]
            if len(self._store) >= self._max_count:
                oldest = min(self._store.items(), key=lambda x: x[1].last_accessed_at)
                del self._store[oldest[0]]


def _extract_subgraph(
    storage: Any,
    body: Dict[str, Any],
    parse_time_point: Any,
) -> Tuple[Set[str], Set[str]]:
    """从主图按条件抽取实体与关系的 absolute id 集合。"""
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


def _apply_filter(
    storage: Any,
    entry: SubgraphEntry,
    body: Dict[str, Any],
    parse_time_point: Any,
) -> None:
    """在子图 entry 上按 time_before/time_after/entity_ids 缩小集合（原地更新）。"""
    time_before = body.get("time_before")
    time_after = body.get("time_after")
    entity_ids_filter = body.get("entity_ids")
    if isinstance(entity_ids_filter, list):
        keep_entity_ids = set(entity_ids_filter)
    else:
        keep_entity_ids = None

    if time_before or time_after:
        time_before_dt = parse_time_point(time_before) if time_before else None
        time_after_dt = parse_time_point(time_after) if time_after else None
        to_drop_entity = set()
        for eid in entry.entity_absolute_ids:
            e = storage.get_entity_by_absolute_id(eid)
            if not e or not e.physical_time:
                continue
            if time_before_dt and e.physical_time > time_before_dt:
                to_drop_entity.add(eid)
            elif time_after_dt and e.physical_time < time_after_dt:
                to_drop_entity.add(eid)
        entry.entity_absolute_ids -= to_drop_entity
        to_drop_relation = set()
        for rid in entry.relation_absolute_ids:
            r = storage.get_relation_by_absolute_id(rid)
            if not r or not r.physical_time:
                continue
            if time_before_dt and r.physical_time > time_before_dt:
                to_drop_relation.add(rid)
            elif time_after_dt and r.physical_time < time_after_dt:
                to_drop_relation.add(rid)
        entry.relation_absolute_ids -= to_drop_relation

    if keep_entity_ids is not None:
        keep_absolute = set()
        for eid in entry.entity_absolute_ids:
            e = storage.get_entity_by_absolute_id(eid)
            if e and e.entity_id in keep_entity_ids:
                keep_absolute.add(eid)
        entry.entity_absolute_ids = keep_absolute
        keep_relations = set()
        for rid in entry.relation_absolute_ids:
            r = storage.get_relation_by_absolute_id(rid)
            if r and r.entity1_absolute_id in keep_absolute and r.entity2_absolute_id in keep_absolute:
                keep_relations.add(rid)
        entry.relation_absolute_ids = keep_relations


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

    remember_lock = threading.Lock()
    config = config or {}
    subgraph_store = SubgraphStore(
        max_count=config.get("subgraph_max_count", 100),
        default_ttl_seconds=config.get("subgraph_ttl_seconds", 3600),
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

    @app.route("/api/remember", methods=["POST"])
    def remember():
        """记忆写入：接收自然语言文本或文档，自动构建记忆图。

        三种输入方式：
        1. JSON {"text": "..."} — 直接传文本
        2. JSON {"file_path": "/path/to/file"} — 服务端本地文件
        3. multipart/form-data 上传文件（字段名 file）

        可选字段：source_name / doc_name, load_cache_memory
        """
        try:
            content_type = request.content_type or ""
            text: Optional[str] = None
            source_name: str = "api_input"

            if "multipart/form-data" in content_type:
                uploaded = request.files.get("file")
                if not uploaded or not uploaded.filename:
                    return err("multipart 请求需包含 file 字段", 400)
                source_name = request.form.get("source_name") or uploaded.filename or "upload"
                tmp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmp_dir, uploaded.filename)
                try:
                    uploaded.save(tmp_path)
                    text = _read_file_content(tmp_path)
                finally:
                    try:
                        os.remove(tmp_path)
                        os.rmdir(tmp_dir)
                    except OSError:
                        pass
            else:
                body = request.get_json(silent=True) or {}
                text = (body.get("text") or "").strip() or None
                file_path = (body.get("file_path") or "").strip() or None
                source_name = (
                    (body.get("source_name") or body.get("doc_name") or "").strip()
                    or (Path(file_path).name if file_path else "api_input")
                )

                if not text and file_path:
                    text = _read_file_content(file_path)
                elif not text:
                    return err("请提供 text、file_path 或上传文件", 400)

            load_cache = None
            if "multipart/form-data" not in content_type:
                lc = (request.get_json(silent=True) or {}).get("load_cache_memory")
                if lc is not None:
                    load_cache = bool(lc)

            with remember_lock:
                result = processor.remember_text(
                    text=text,
                    doc_name=source_name,
                    verbose=False,
                    load_cache_memory=load_cache,
                )
            return ok(result)
        except FileNotFoundError as e:
            return err(str(e), 404)
        except ImportError as e:
            return err(str(e), 501)
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
            create_subgraph (bool): 是否创建持久化子图以便后续操作，默认 false

        返回:
            entities: 命中的概念实体列表
            relations: 命中的概念关系列表
            subgraph_id: 若 create_subgraph=true，返回子图 ID
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
            do_create_subgraph = body.get("create_subgraph", False)
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
            final_entity_abs_ids = {e.id for e in final_entities}

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

            if do_create_subgraph:
                sid = subgraph_store.create(
                    final_entity_abs_ids,
                    seen_rel_ids,
                )
                result["subgraph_id"] = sid

            return ok(result)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 状态化子图（高级，用于多步检索）
    # =========================================================
    @app.route("/api/find/subgraph", methods=["POST"])
    def find_subgraph_create():
        try:
            body = request.get_json(silent=True) or {}
            try:
                entity_ids, relation_ids = _extract_subgraph(
                    processor.storage, body, parse_time_point
                )
            except ValueError as ve:
                return err(str(ve), 400)
            ttl = body.get("ttl_seconds")
            ttl = int(ttl) if ttl is not None else None
            sid = subgraph_store.create(entity_ids, relation_ids, ttl_seconds=ttl)
            return ok({
                "subgraph_id": sid,
                "entity_count": len(entity_ids),
                "relation_count": len(relation_ids),
            }), 201
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/subgraph/<subgraph_id>")
    def find_subgraph_get(subgraph_id: str):
        try:
            entry = subgraph_store.get(subgraph_id)
            if entry is None:
                return err("子图不存在或已过期", 404)
            return ok({
                "subgraph_id": subgraph_id,
                "entity_count": len(entry.entity_absolute_ids),
                "relation_count": len(entry.relation_absolute_ids),
                "created_at": entry.created_at,
                "ttl_seconds": entry.ttl_seconds,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/subgraph/<subgraph_id>/entities")
    def find_subgraph_entities(subgraph_id: str):
        try:
            entry = subgraph_store.get(subgraph_id)
            if entry is None:
                return err("子图不存在或已过期", 404)
            entities = []
            for eid in entry.entity_absolute_ids:
                e = processor.storage.get_entity_by_absolute_id(eid)
                if e:
                    entities.append(entity_to_dict(e))
            return ok(entities)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/subgraph/<subgraph_id>/relations")
    def find_subgraph_relations(subgraph_id: str):
        try:
            entry = subgraph_store.get(subgraph_id)
            if entry is None:
                return err("子图不存在或已过期", 404)
            relations = []
            for rid in entry.relation_absolute_ids:
                r = processor.storage.get_relation_by_absolute_id(rid)
                if r:
                    relations.append(relation_to_dict(r))
            return ok(relations)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/subgraph/<subgraph_id>/expand", methods=["POST"])
    def find_subgraph_expand(subgraph_id: str):
        try:
            entry = subgraph_store.get(subgraph_id)
            if entry is None:
                return err("子图不存在或已过期", 404)
            body = request.get_json(silent=True) or {}
            try:
                new_entity_ids, new_relation_ids = _extract_subgraph(
                    processor.storage, body, parse_time_point
                )
            except ValueError as ve:
                return err(str(ve), 400)
            entry.entity_absolute_ids |= new_entity_ids
            entry.relation_absolute_ids |= new_relation_ids
            entry.touch()
            return ok({
                "subgraph_id": subgraph_id,
                "entity_count": len(entry.entity_absolute_ids),
                "relation_count": len(entry.relation_absolute_ids),
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/subgraph/<subgraph_id>/filter", methods=["POST"])
    def find_subgraph_filter(subgraph_id: str):
        try:
            entry = subgraph_store.get(subgraph_id)
            if entry is None:
                return err("子图不存在或已过期", 404)
            body = request.get_json(silent=True) or {}
            try:
                _apply_filter(processor.storage, entry, body, parse_time_point)
            except ValueError as ve:
                return err(str(ve), 400)
            entry.touch()
            return ok({
                "subgraph_id": subgraph_id,
                "entity_count": len(entry.entity_absolute_ids),
                "relation_count": len(entry.relation_absolute_ids),
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/subgraph/<subgraph_id>", methods=["DELETE"])
    def find_subgraph_release(subgraph_id: str):
        try:
            if subgraph_store.delete(subgraph_id):
                return "", 204
            return err("子图不存在或已过期", 404)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/find/query-one", methods=["POST"])
    def find_query_one():
        try:
            body = request.get_json(silent=True) or {}
            include_entities = body.get("include_entities", True)
            include_relations = body.get("include_relations", True)
            try:
                entity_ids, relation_ids = _extract_subgraph(
                    processor.storage, body, parse_time_point
                )
            except ValueError as ve:
                return err(str(ve), 400)
            sid = subgraph_store.create(entity_ids, relation_ids)
            try:
                entities_data: List[Dict[str, Any]] = []
                relations_data: List[Dict[str, Any]] = []
                if include_entities:
                    entry = subgraph_store.get(sid)
                    if entry:
                        for eid in entry.entity_absolute_ids:
                            e = processor.storage.get_entity_by_absolute_id(eid)
                            if e:
                                entities_data.append(entity_to_dict(e))
                if include_relations:
                    entry = subgraph_store.get(sid)
                    if entry:
                        for rid in entry.relation_absolute_ids:
                            r = processor.storage.get_relation_by_absolute_id(rid)
                            if r:
                                relations_data.append(relation_to_dict(r))
                return ok({"entities": entities_data, "relations": relations_data})
            finally:
                subgraph_store.delete(sid)
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


def build_processor(config: Dict[str, Any]) -> TemporalMemoryGraphProcessor:
    storage_path = config.get("storage_path", "./graph/tmg_storage")
    chunking = config.get("chunking") or {}
    window_size = chunking.get("window_size", 1000)
    overlap = chunking.get("overlap", 200)
    llm = config.get("llm") or {}
    embedding = config.get("embedding") or {}
    model_path, model_name, use_local = resolve_embedding_model(embedding)
    return TemporalMemoryGraphProcessor(
        storage_path=storage_path,
        window_size=window_size,
        overlap=overlap,
        llm_api_key=llm.get("api_key"),
        llm_model=llm.get("model", "gpt-4"),
        llm_base_url=llm.get("base_url"),
        llm_think_mode=bool(llm.get("think", llm.get("think_mode", False))),
        embedding_model_path=model_path,
        embedding_model_name=model_name,
        embedding_device=embedding.get("device", "cpu"),
        embedding_use_local=use_local,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Temporal_Memory_Graph 自然语言记忆图 API（Remember + Find）")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径（如 service_config.json）")
    parser.add_argument("--host", type=str, default=None, help="覆盖配置中的 host")
    parser.add_argument("--port", type=int, default=None, help="覆盖配置中的 port")
    parser.add_argument("--debug", action="store_true", help="开启 Flask 调试模式")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        print(f"错误：配置文件不存在: {config_path}")
        return 1

    config = load_config(config_path)
    host = args.host if args.host is not None else config.get("host", "0.0.0.0")
    port = args.port if args.port is not None else config.get("port", 5001)
    storage_path = config.get("storage_path", "./graph/tmg_storage")
    Path(storage_path).mkdir(parents=True, exist_ok=True)

    processor = build_processor(config)
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

    print(f"""
╔══════════════════════════════════════════════════════════╗
║     Temporal_Memory_Graph — 自然语言记忆图 API           ║
╚══════════════════════════════════════════════════════════╝

  当前大脑记忆库:
    实体: {total_entities}  关系: {total_relations}  记忆缓存: {total_memory_caches}

  服务地址: http://{host}:{port}
  健康检查: GET  http://{host}:{port}/health
  记忆写入: POST http://{host}:{port}/api/remember
  语义检索: POST http://{host}:{port}/api/find
  原子查询: GET  http://{host}:{port}/api/find/...

  存储路径: {storage_path}

  按 Ctrl+C 停止服务
""")
    app.run(host=host, port=port, debug=args.debug)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
