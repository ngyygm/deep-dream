"""Scope 沙箱编排：检索 + 图回溯圈定一个有界的文档范围。

目标：给定一个查询，圈出一个有界的文档沙箱供 agent 在范围内用 bash
精读。全程无 LLM、无网络：

  1. 种子概念：调用 ``concept_search`` 的对应模式函数（bm25/semantic/
     hybrid）拿 top 概念（family_id/name/role/_score 的 dict 列表）。
  2. 图回溯：entity family 走 entity_mentions + entity_observations、
     relation family 走 relation_assertions——按 family_id 批量 IN 查询
     （``LibraryManager.concept_source_documents``）锚定 episode（含偏移
     信息），再 join episodes→documents。可见性过滤与搜索视图
     （repositories/search.py::search_fts）保持一致：episodes/documents/
     document_versions 均 active，且
     ``COALESCE(document_ingestion_state.state, 'active') = 'active'``。
  3. role='episode' 的兜底检索结果（BM25 无概念命中时的 FTS 证据行）
     直接把对应 episode 纳入范围（``include_episode_rank=False`` 关闭）。
  4. 文档排序分 = Σ(命中概念 score) + 0.05 × distinct 命中概念数；同分按
     document_id 稳定排序，截到 ``max_docs``。

产物：
  - ``build_document_scope`` → 可 JSON 序列化的 scope dict；
  - ``materialize_scope`` → ``sandbox_root/<scope_id>/`` 下的 symlink 目录
    + manifest.json（幂等），manifest 记录每个文件的命中原因 / 分数 /
    episode 偏移与目标文件 sha256。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .concept_search import (
    bm25_concept_search,
    hybrid_concept_search,
    semantic_concept_search,
)

logger = logging.getLogger(__name__)

# 物化文件名里的非法字符（保留 unicode 字母/数字/下划线/连字符/点/空格）
_UNSAFE_FS_CHARS = re.compile(r"[^\w\-. ]+")
# 物化目录里属于上一次物化的编号文件（幂等重建时先清掉）
_NUMBERED_FILE = re.compile(r"^\d{2,}-")
# scope_id 白名单：仅字母/数字/下划线/连字符（防 "../" 之类路径逃逸，
# 同款先例见 library_manager._library_file 的库内路径校验）
_SCOPE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# episode 片段长度（source_text 前缀）
_SNIPPET_CHARS = 120
# 文档排序分里每个 distinct 命中概念的加成
_MATCH_COUNT_BONUS = 0.05
# vault / external 文档优先用绝对路径；managed 用相对库根的 managed_path
_ABSOLUTE_MODES = ("vault", "external")
# 语义种子腿的相似度阈值（bm25/hybrid 腿取 0——沙箱要召回不要精排）
_SEMANTIC_SEED_THRESHOLD = 0.3


# ── 种子概念 ────────────────────────────────────────────────

def _seed_concepts(storage, query: str, mode: str, max_concepts: int) -> List[dict]:
    """按模式调用 concept_search 的统一执行函数拿种子概念。

    threshold 选择：bm25/hybrid 两腿传 0（沙箱构建要召回，阈值过滤交给
    检索函数内部的 CJK 降阈逻辑）；semantic 腿传 0.3（余弦分数的常规下限，
    无 embedding client 时 LIKE 兜底分也据此缩放）。
    """
    if mode == "bm25":
        results, _meta = bm25_concept_search(
            storage, query, None, max_concepts, 0.0)
    elif mode == "semantic":
        results, _meta = semantic_concept_search(
            storage, query, None, max_concepts, _SEMANTIC_SEED_THRESHOLD)
    elif mode == "hybrid":
        results, _meta = hybrid_concept_search(
            storage, query, None, max_concepts, 0.0)
    else:
        raise ValueError(f"unknown scope mode: {mode!r}（可选：bm25|semantic|hybrid）")
    return results or []


# ── 回溯聚合 ────────────────────────────────────────────────

def _absorb_row(docs: Dict[str, dict], row: dict, matched: set) -> None:
    """把一条回溯行合并进 docs 聚合桶（同 episode 多来源合并 matched 集合）。"""
    doc_id = row.get("document_id") or ""
    if not doc_id:
        return
    bucket = docs.get(doc_id)
    if bucket is None:
        bucket = {
            "document_id": doc_id,
            "title": row.get("title") or "",
            "source_mode": row.get("source_mode") or "managed",
            "managed_path": row.get("managed_path") or "",
            "absolute_path": row.get("absolute_path") or "",
            "matched": set(),
            "episodes": {},
        }
        docs[doc_id] = bucket
    bucket["matched"].update(matched)
    ep_id = row.get("episode_id") or ""
    if not ep_id:
        return
    ep = bucket["episodes"].get(ep_id)
    if ep is None:
        bucket["episodes"][ep_id] = {
            "episode_id": ep_id,
            "name": row.get("episode_name") or "",
            "start_offset": row.get("episode_start_offset") or 0,
            "end_offset": row.get("episode_end_offset") or 0,
            "matched": set(matched),
            "snippet": str(row.get("source_text") or "")[:_SNIPPET_CHARS],
        }
    else:
        ep["matched"].update(matched)


def _aggregate_backtrace(storage, concepts: List[dict],
                         scores: Dict[str, float]) -> Dict[str, dict]:
    """概念 → episode → document 批量回溯聚合（document_id → 桶）。"""
    fam_ids = [c["family_id"] for c in concepts]
    rows = (storage.concept_source_documents(fam_ids, include_offsets=True)
            if fam_ids else [])
    docs: Dict[str, dict] = {}
    for row in rows:
        fid = row.get("family_id") or ""
        if fid not in scores:
            continue  # 防御：回溯行不在种子集（正常不会发生）
        _absorb_row(docs, row, {fid})
    return docs


def _active_episode(storage, episode_id: str) -> Optional[dict]:
    """读取 episode 行，非 active / 不存在返回 None。"""
    try:
        ep = storage.get_episode(episode_id)
    except Exception as exc:
        logger.warning("scope: 读取 episode %s 失败: %s", episode_id, exc)
        return None
    if not ep or (ep.get("status") or "active") != "active":
        return None
    return ep


def _active_document(storage, document_id: str) -> Optional[dict]:
    """读取 document 行，非 active / 不存在返回 None。"""
    if not document_id:
        return None
    try:
        doc = storage.get_document(document_id)
    except Exception as exc:
        logger.warning("scope: 读取 document %s 失败: %s", document_id, exc)
        return None
    if not doc or doc.get("status") != "active":
        return None
    return doc


def _ingestion_active(storage, document_id: str) -> bool:
    """与搜索视图一致：document_ingestion_state 无行视为 active。"""
    try:
        result = storage.read_sql(
            "SELECT state FROM document_ingestion_state WHERE document_id = ?",
            [document_id])
    except Exception as exc:
        logger.debug("scope: 查询摄取态失败（按 active 处理）%s: %s",
                     document_id, exc)
        return True
    rows = (result or {}).get("rows") or []
    if not rows:
        return True
    first = rows[0]
    state = first.get("state") if isinstance(first, dict) else None
    return (state or "active") == "active"


def _absorb_episode_seeds(storage, docs: Dict[str, dict],
                          episode_seeds: List[dict]) -> None:
    """role='episode' 兜底结果直接纳入：episode 与文档均 active 才进沙箱。"""
    for seed in episode_seeds:
        ep_row = _active_episode(storage, seed["episode_id"])
        if not ep_row:
            continue
        doc_id = ep_row.get("document_id") or ""
        doc = _active_document(storage, doc_id)
        if not doc or not _ingestion_active(storage, doc_id):
            continue
        _absorb_row(docs, {
            "document_id": doc_id,
            "title": doc.get("title") or "",
            "source_mode": doc.get("source_mode") or "managed",
            "managed_path": doc.get("managed_path") or "",
            "absolute_path": doc.get("absolute_path") or "",
            "episode_id": ep_row.get("episode_id") or seed["episode_id"],
            "episode_name": ep_row.get("name") or "",
            "episode_start_offset": ep_row.get("start_offset") or 0,
            "episode_end_offset": ep_row.get("end_offset") or 0,
            "source_text": ep_row.get("source_text") or "",
        }, {seed["family_id"]})


# ── 路径解析 ────────────────────────────────────────────────

def _presentation_path(bucket: dict) -> str:
    """展示路径：managed → 相对库根 managed_path；vault/external → absolute_path。

    优先值缺失时回退另一个，两者皆缺返回空串（stats 里计数）。
    """
    if bucket["source_mode"] in _ABSOLUTE_MODES:
        return bucket["absolute_path"] or bucket["managed_path"] or ""
    return bucket["managed_path"] or bucket["absolute_path"] or ""


def _resolve_file_path(storage, bucket: dict) -> str:
    """物化用磁盘绝对路径：vault/external 直接用 absolute_path；managed 相对
    ``storage.library_path`` 库根解析。两者皆缺返回空串。"""
    if bucket["source_mode"] in _ABSOLUTE_MODES and bucket["absolute_path"]:
        return bucket["absolute_path"]
    root = getattr(storage, "library_path", None)
    if bucket["managed_path"] and root is not None:
        return str(Path(root) / bucket["managed_path"])
    return bucket["absolute_path"] or bucket["managed_path"] or ""


# ── 主入口 1：圈定范围 ─────────────────────────────────────

def build_document_scope(storage, query: str, *, mode: str = "hybrid",
                         max_concepts: int = 20, max_docs: int = 30,
                         include_episode_rank: bool = True) -> dict:
    """用"检索 + 图回溯"圈出一个有界的文档沙箱（无 LLM、无网络）。

    参数：
      storage：LibraryManager（或实现 concept_source_documents /
        get_episode / get_document / read_sql 的同类检索面）。
      query：检索查询。
      mode：种子概念检索模式，``bm25`` / ``semantic`` / ``hybrid``。
      max_concepts：种子概念上限。
      max_docs：返回文档数上限（按排序分截断）。
      include_episode_rank：True 时把检索返回的 role='episode' 兜底结果
        （BM25 无概念命中时的 FTS 证据行）直接纳入范围。

    返回结构::

        {
          "query": str, "mode": str,
          "concepts": [{"family_id", "name", "role", "score"}],
          "documents": [{
              "document_id", "title", "path", "file_path", "score",
              "matched_concepts": [family_id],
              "episodes": [{"episode_id", "name", "start_offset",
                            "end_offset", "matched": [family_id],
                            "snippet": source_text 前 120 字符}],
          }],
          "stats": {"seed_concepts", "episode_seeds", "episodes_found",
                    "documents_total", "documents_returned",
                    "documents_missing_path"},
        }

    ``documents[].path``：managed 文档给相对库根的 managed_path，
    vault/external 给 absolute_path，两者皆缺为空串并计入
    ``stats["documents_missing_path"]``。``documents[].file_path`` 是物化
    用的磁盘绝对路径（materialize_scope 的 symlink 目标）。
    """
    raw_results = _seed_concepts(storage, query, mode, max_concepts)

    concepts: List[dict] = []
    scores: Dict[str, float] = {}
    episode_seeds: List[dict] = []
    seen_fids = set()
    for item in raw_results:
        fid = str(item.get("family_id") or item.get("id") or "").strip()
        if not fid or fid in seen_fids:
            continue
        seen_fids.add(fid)
        role = str(item.get("role") or "")
        score = float(item.get("_score") or 0.0)
        if role == "episode":
            # BM25 兜底行：family_id 即 episode_id，不进 concepts 列表
            if include_episode_rank:
                scores[fid] = score
                episode_seeds.append({
                    "family_id": fid,
                    "episode_id": str(item.get("episode_id") or fid),
                })
            continue
        scores[fid] = score
        concepts.append({
            "family_id": fid,
            "name": item.get("name") or "",
            "role": role,
            "score": round(score, 6),
        })

    docs = _aggregate_backtrace(storage, concepts, scores)
    if episode_seeds:
        _absorb_episode_seeds(storage, docs, episode_seeds)

    episodes_found = sum(len(b["episodes"]) for b in docs.values())
    documents_total = len(docs)

    def _bucket_score(bucket: dict) -> float:
        return (sum(scores.get(f, 0.0) for f in bucket["matched"])
                + _MATCH_COUNT_BONUS * len(bucket["matched"]))

    ranked = sorted(docs.values(),
                    key=lambda b: (-_bucket_score(b), b["document_id"]))

    documents: List[dict] = []
    missing_path = 0
    for bucket in ranked[:max(int(max_docs or 0), 0)]:
        path = _presentation_path(bucket)
        if not path:
            missing_path += 1
        documents.append({
            "document_id": bucket["document_id"],
            "title": bucket["title"],
            "path": path,
            "file_path": _resolve_file_path(storage, bucket),
            "score": round(_bucket_score(bucket), 6),
            "matched_concepts": sorted(bucket["matched"]),
            "episodes": [
                {**ep, "matched": sorted(ep["matched"])}
                for ep in sorted(bucket["episodes"].values(),
                                 key=lambda e: (e["start_offset"] or 0,
                                                e["episode_id"]))
            ],
        })

    return {
        "query": query,
        "mode": mode,
        "concepts": concepts,
        "documents": documents,
        "stats": {
            "seed_concepts": len(concepts),
            "episode_seeds": len(episode_seeds),
            "episodes_found": episodes_found,
            "documents_total": documents_total,
            "documents_returned": len(documents),
            "documents_missing_path": missing_path,
        },
    }


# ── 主入口 2：物化沙箱 ─────────────────────────────────────

def _safe_filename_component(text: str, fallback: str, max_len: int = 60) -> str:
    """把标题安全化成文件名组件：替换非法字符、压空白、截长度。"""
    cleaned = _UNSAFE_FS_CHARS.sub("_", str(text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    cleaned = cleaned[:max_len].strip(" ._")
    return cleaned or str(fallback or "untitled")


def _file_sha256(path: Path) -> str:
    """流式计算目标文件 sha256（manifest 记录用）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scope_fingerprint(scope_result: dict) -> str:
    """scope_id 种子：query + 结果内容（文档/分数/命中概念/episode）的
    sha256 前 12 位——同一查询同一结果稳定同 id，内容变化换 id。"""
    payload = {
        "query": scope_result.get("query", ""),
        "mode": scope_result.get("mode", ""),
        "documents": [
            {
                "document_id": d.get("document_id"),
                "score": d.get("score"),
                "matched_concepts": d.get("matched_concepts"),
                "episodes": [e.get("episode_id") for e in d.get("episodes", [])],
            }
            for d in scope_result.get("documents", [])
        ],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def materialize_scope(scope_result: dict, sandbox_root, *,
                      scope_id: str = None) -> dict:
    """把 scope 结果物化成沙箱目录：每文档一个 symlink + manifest.json。

    目录布局 ``sandbox_root/<scope_id>/``：
      - ``{序号:02d}-{安全化 title}{目标扩展名}`` → 目标文档文件的
        symlink（目标不存在则跳过链接，仅在 manifest 里记录）；
      - ``manifest.json``：scope 元信息（query/mode/stats/concepts）+ 每个
        文件的命中原因（matched_concepts）/ 分数 / episode 偏移 / 目标
        文件 sha256。

    幂等：目录已存在时先清掉旧的编号链接再重建，manifest 全量覆盖。
    ``scope_id`` 缺省由 query + 结果内容的 sha256 前 12 位生成；显式传入
    时必须匹配 ``^[A-Za-z0-9_-]+$``（否则 ValueError），防止在
    ``sandbox_root`` 之外建目录。

    返回 ``{"scope_id", "path", "files", "manifest_path"}``（files 为实际
    建立链接的路径列表）。
    """
    root = Path(sandbox_root)
    if scope_id is None or not str(scope_id).strip():
        scope_id = _scope_fingerprint(scope_result)
    scope_id = str(scope_id)
    if not _SCOPE_ID_RE.match(scope_id):
        raise ValueError(
            f"非法 scope_id: {scope_id!r}（仅允许字母/数字/下划线/连字符）")
    scope_dir = root / scope_id
    scope_dir.mkdir(parents=True, exist_ok=True)

    # 幂等重建：清掉上一次物化留下的编号文件（manifest 稍后全量覆盖）
    for stale in scope_dir.iterdir():
        if stale.name == "manifest.json":
            continue
        if _NUMBERED_FILE.match(stale.name) and (stale.is_symlink() or stale.is_file()):
            try:
                stale.unlink()
            except OSError as exc:
                logger.warning("scope: 清理旧链接失败 %s: %s", stale, exc)

    files: List[str] = []
    entries: List[dict] = []
    for idx, doc in enumerate(scope_result.get("documents", []), 1):
        entry: Dict[str, Any] = {
            "index": idx,
            "document_id": doc.get("document_id", ""),
            "title": doc.get("title", ""),
            "path": doc.get("path", ""),
            "target": doc.get("file_path", ""),
            "target_exists": False,
            "filename": "",
            "sha256": "",
            "score": doc.get("score", 0.0),
            "matched_concepts": doc.get("matched_concepts", []),
            "episodes": [
                {
                    "episode_id": e.get("episode_id", ""),
                    "name": e.get("name", ""),
                    "start_offset": e.get("start_offset", 0),
                    "end_offset": e.get("end_offset", 0),
                    "matched": e.get("matched", []),
                }
                for e in doc.get("episodes", [])
            ],
        }
        target_str = str(doc.get("file_path") or "").strip()
        if target_str and Path(target_str).is_file():
            target = Path(target_str)
            filename = (f"{idx:02d}-{_safe_filename_component(doc.get('title'), doc.get('document_id'))}"
                        f"{target.suffix}")
            link = scope_dir / filename
            try:
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target)
            except OSError as exc:
                logger.warning("scope: 建链接失败 %s -> %s: %s",
                               link, target, exc)
            else:
                entry["target_exists"] = True
                entry["filename"] = filename
                entry["sha256"] = _file_sha256(target)
                files.append(str(link))
        else:
            logger.debug("scope: 目标文件不存在，跳过 symlink: %s", target_str)
        entries.append(entry)

    manifest = {
        "scope_id": str(scope_id),
        "query": scope_result.get("query", ""),
        "mode": scope_result.get("mode", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": scope_result.get("stats", {}),
        "concepts": scope_result.get("concepts", []),
        "symlinked": len(files),
        "files": entries,
    }
    manifest_path = scope_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "scope_id": str(scope_id),
        "path": str(scope_dir),
        "files": files,
        "manifest_path": str(manifest_path),
    }
