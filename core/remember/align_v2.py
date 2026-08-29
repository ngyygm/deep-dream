"""
ALIGN-V2 实验模块：簇收敛对齐引擎（环境变量 DD_ALIGN_V2=1 开启）。

对应 A/B 实验的 Variant B2：
1. step9 跨窗口并行（拆串行链）——pipeline_workers.py 读 align_v2_enabled()。
2. 窗口批量裁决升级：候选之间的等价类（duplicate groups）由 LLM 一并判定，
   本模块在窗口实体全部落库后统一应用（maybe_apply_window_cluster_dupes）。
3. 文档末全库收敛扫描（doc_end_library_sweep）：同名/子串别名 family 分组 →
   judge_entity_alignment 逐对核验 → dedup_merge_batch 批量合并。

设计语义：允许并行产生临时重复 family，靠"每次窗口触达 + 文档末兜底"收敛；
合并走既有 redirect 语义，重复执行幂等（败方已消失时跳过）。
"""
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from core.utils import entity_match_key, wprint_info

logger = logging.getLogger(__name__)

# 同一库内并发应用簇合并的互斥锁（跨文档共享一个进程内的库级处理器时生效）
_APPLY_LOCK = threading.Lock()

# ingest 期间收集的待合并等价组（键 = library_path），scope 收尾统一应用
_PENDING: Dict[Any, List[List[str]]] = {}
_PENDING_LOCK = threading.Lock()

# 收敛判定的基线/上限预算（对）。同名组对数全保留（高精度必判），别名对
# 填充剩余额度；实测 16 docs 已有 167 个子串别名候选，45 docs 满 scope 约
# 400+ 对——基线 250 对 ≈ 8-10 min 串行收尾，上限 1500 防 scope 收尾过久，
# 超出部分留给不动点轮/下次 run 收敛（redirect 幂等）。
_SWEEP_BASE_PAIRS = 250
_SWEEP_MAX_PAIRS = 1500


def _sweep_pair_budget(same_name_pair_count: int, alias_pair_count: int) -> int:
    """预算自适应：同名组对数必保，别名对共享剩余额度。"""
    return max(_SWEEP_BASE_PAIRS, min(_SWEEP_MAX_PAIRS, same_name_pair_count + alias_pair_count))

DUPES_KEY = "__duplicate_groups__"

# 配置开关：pipeline.remember.cluster_convergence=true 时由 orchestrator
# __init__ 调 set_enabled(True)（进程级；环境变量 DD_ALIGN_V2=1 仍可临时覆盖）
_CONFIG_ENABLED = False


def set_enabled(value: bool) -> None:
    global _CONFIG_ENABLED
    _CONFIG_ENABLED = bool(value)


def align_v2_enabled() -> bool:
    return _CONFIG_ENABLED or os.environ.get("DD_ALIGN_V2") == "1"


def pop_duplicate_groups(verdicts: Optional[Dict[str, Dict[str, Any]]]) -> List[List[str]]:
    """从窗口批量裁决结果中取出候选等价组（无则空列表）。键总是被移除。"""
    if not verdicts:
        return []
    groups = verdicts.pop(DUPES_KEY, None)
    if not isinstance(groups, list):
        return []
    clean: List[List[str]] = []
    for g in groups:
        if isinstance(g, list):
            fids = [str(f).strip() for f in g if str(f or "").strip()]
            if len(fids) >= 2:
                clean.append(fids)
    return clean


def apply_duplicate_family_groups(storage, groups: List[List[str]],
                                  verbose: bool = False) -> int:
    """把候选等价组合并进首个仍存在的 family（redirect 语义，幂等）。

    走非破坏性合并（merge_entity_families）：败方的 observations/mentions
    改挂到胜方 family 之下（历史可溯源），而不是 delete_entity_all_versions
    的整体删除。dedup_merge_batch 的删除路径会在败方 observation 被存活
    relation_assertions 引用（subject/object_entity_id FK）时炸 FOREIGN KEY，
    这里天然规避。
    """
    if not groups:
        return 0
    merged_total = 0
    with _APPLY_LOCK:
        for fids in groups:
            alive: List[str] = []
            for fid in fids:
                try:
                    if storage.get_entity_by_family_id(fid) is not None:
                        alive.append(fid)
                except Exception:
                    continue
            if len(alive) < 2:
                continue
            primary, sources = alive[0], alive[1:]
            try:
                result = storage.merge_entity_families(
                    primary, sources, skip_name_check=True)
                merged_total += len(result.get("merged") or [])
            except Exception as exc:
                logger.warning(
                    "align-v2 apply_duplicate_family_groups group failed: %s", exc)
    if verbose and merged_total:
        wprint_info(f"[align-v2] 簇合并：{merged_total} 个 family 并入")
    return merged_total


def maybe_apply_window_cluster_dupes(storage, verdicts: Optional[Dict[str, Dict[str, Any]]],
                                     verbose: bool = False) -> int:
    """窗口实体落库后调用：窗口裁决带出的等价组进入待合并队列（v2 关闭时为 no-op）。

    并发安全设计：ingest 期间多文档并行写库，此时执行合并（删除败方
    entity_families 行）会与其它文档的 step10 关系写入竞争——新关系引用
    刚被删除的 family 会炸 FOREIGN KEY。因此这里只入队，真正合并推迟到
    scope 收尾的 final_convergence_flush（此刻无并发写者）。
    """
    if not align_v2_enabled():
        # 键总是清理，避免极端实体名碰撞
        pop_duplicate_groups(verdicts)
        return 0
    groups = pop_duplicate_groups(verdicts)
    if not groups:
        return 0
    key = getattr(storage, "library_path", None) or id(storage)
    with _PENDING_LOCK:
        _PENDING.setdefault(key, []).extend(groups)
    return len(groups)


def _collect_sweep_candidates(storage) -> Tuple[List[List], List[List]]:
    """收集全库待收敛分组：同名组（≥2 个 family）与子串别名对。"""
    try:
        entities = storage.get_all_entities(limit=5000, exclude_embedding=True)
    except Exception:
        entities = []
    if not entities:
        return [], []

    by_name: Dict[str, List] = defaultdict(list)
    for ent in entities:
        key = entity_match_key((ent.name or "").strip())
        if key:
            by_name[key].append(ent)

    same_name_groups: List[List] = []
    alias_pairs: List[List] = []
    for key, ents in by_name.items():
        fids = {e.family_id for e in ents}
        if len(fids) > 1:
            # 同名不同 family：全部进入收敛判定组（组内以第一个为 primary 候选）
            uniq: Dict[str, Any] = {}
            for e in ents:
                uniq.setdefault(e.family_id, e)
            same_name_groups.append(list(uniq.values()))

    # 子串别名对（不同核心名、长度不同、互为包含）
    items = [(entity_match_key((e.name or "").strip()), e)
             for e in entities if entity_match_key((e.name or "").strip())]
    seen_pair = set()
    for i, (core_a, ea) in enumerate(items):
        for core_b, eb in items[i + 1:]:
            if core_a == core_b or len(core_a) == len(core_b):
                continue
            if ea.family_id == eb.family_id:
                continue
            if core_a in core_b or core_b in core_a:
                pk = tuple(sorted((ea.family_id, eb.family_id)))
                if pk not in seen_pair:
                    seen_pair.add(pk)
                    alias_pairs.append([ea, eb])
    return same_name_groups, alias_pairs


def _judge_pair(llm_client, ea, eb, match_type: str) -> Tuple[str, str, Dict]:
    try:
        result = llm_client.judge_entity_alignment(
            name_a=ea.name or "", content_a=(ea.content or "")[:500],
            name_b=eb.name or "", content_b=(eb.content or "")[:500],
            name_match_type=match_type,
        )
        return (ea.family_id, eb.family_id, result or {})
    except Exception:
        return (ea.family_id, eb.family_id, {})


def doc_end_library_sweep(processor, verbose: bool = False) -> Dict[str, int]:
    """文档末收敛入口（ingest 期间调用）。

    v2 并发安全设计：ingest 期间多文档并行写库，此处不做任何合并/判定
    （避免与其它文档的写入竞争 FK / 锁）。真正的收敛在 scope 收尾由
    final_convergence_flush 串行完成（同名组 + 别名对重判 + 应用队列）。
    保留此函数是为了兼容既有调用点——它现在是无副作用的 no-op。
    """
    return {"groups": 0, "pairs_judged": 0, "merged": 0, "seconds": 0.0}


def final_convergence_flush(processor, verbose: bool = False) -> Dict[str, int]:
    """scope 收尾收敛：应用窗口收集的等价组 + 全库扫描判定合并（串行，无并发写者）。

    幂等可重入：resume 中断后重跑会重新收集并合并（redirect 语义天然幂等）。
    """
    storage = processor.storage
    llm_client = getattr(processor, "llm_client", None)
    t0 = time.monotonic()
    key = getattr(storage, "library_path", None) or id(storage)
    with _PENDING_LOCK:
        pending = _PENDING.pop(key, [])

    total = {"groups": len(pending), "pairs_judged": 0, "merged": 0}
    total["merged"] += apply_duplicate_family_groups(storage, pending, verbose=verbose)

    prev_step = getattr(llm_client, "_current_distill_step", None) if llm_client else None
    if llm_client is not None:
        llm_client._current_distill_step = "11s_library_sweep"
    try:
        for _round in range(3):  # 合并可能级联出新别名对，最多 3 轮
            same_name_groups, alias_pairs = _collect_sweep_candidates(storage)
            if not same_name_groups and not alias_pairs:
                break
            judge_jobs: List[Tuple[Any, Any, str]] = []
            for group in same_name_groups:
                primary = group[0]
                for other in group[1:]:
                    judge_jobs.append((primary, other, "exact"))
            # 预算自适应：同名组对数（上方已全量入列）必保，别名对按剩余额度截断
            _same_name_pairs = len(judge_jobs)
            _budget = _sweep_pair_budget(_same_name_pairs, len(alias_pairs))
            _judged_alias_pairs = alias_pairs[:max(0, _budget - _same_name_pairs)]
            for pa, pb in _judged_alias_pairs:
                judge_jobs.append((pa, pb, "substring"))

            verdicts: Dict[Tuple[str, str], Dict] = {}
            if judge_jobs and llm_client is not None:
                workers = max(2, min(getattr(processor, "llm_threads", 4) or 4, 8))
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sweep") as pool:
                    futures = [pool.submit(_judge_pair, llm_client, a, b, mt)
                               for a, b, mt in judge_jobs]
                    for fut in futures:
                        fa, fb, res = fut.result()
                        verdicts[(fa, fb)] = res
            total["pairs_judged"] += len(judge_jobs)

            groups_to_merge: List[List[str]] = []
            for group in same_name_groups:
                primary = group[0]
                members = [primary.family_id]
                for other in group[1:]:
                    v = verdicts.get((primary.family_id, other.family_id)) or {}
                    if v.get("verdict") == "same":
                        members.append(other.family_id)
                if len(members) >= 2:
                    groups_to_merge.append(members)
            for pa, pb in _judged_alias_pairs:
                v = verdicts.get((pa.family_id, pb.family_id)) or {}
                if v.get("verdict") == "same":
                    groups_to_merge.append([pa.family_id, pb.family_id])

            merged = apply_duplicate_family_groups(storage, groups_to_merge, verbose=verbose)
            total["merged"] += merged
            if not merged:
                break
    finally:
        if llm_client is not None and prev_step is not None:
            llm_client._current_distill_step = prev_step

    total["seconds"] = round(time.monotonic() - t0, 2)
    if verbose:
        wprint_info(
            f"[align-v2] scope 收尾收敛：队列组 {len(pending)}｜判定 "
            f"{total['pairs_judged']} 对｜合并 {total['merged']} 个 family｜{total['seconds']}s"
        )
    return total
