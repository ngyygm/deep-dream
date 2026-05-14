"""
Remember 任务进度计算：纯函数和简单辅助函数，不依赖队列类。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Status / phase frozensets for O(1) membership tests
# ---------------------------------------------------------------------------
_TERMINAL_STATUSES = frozenset(("completed", "failed", "cancelled"))
_DONE_STATUSES = frozenset(("completed", "failed"))
_MAIN_PHASES = frozenset(("main", "phase_ab"))
_STEP910 = frozenset(("step9", "step10"))

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------
_RE_WINDOW_STEP = re.compile(r"窗口\s*(\d+)/(\d+)\s*·\s*步骤(\d+)/(\d+)")
_RE_WINDOW_ONLY = re.compile(r"窗口\s*(\d+)/(\d+)")
_RE_MAIN_1_8_DONE = re.compile(r"步骤\s*1\s*[–-]\s*8\s*/\s*10")
_RE_EXTRACT_STEP_NUM = re.compile(r"步骤\s*(\d+)")
_RE_EXTRACT_STEP_FRAC = re.compile(r"\((\d+)/(\d+)\)\s*$")


def estimate_chunk_count(text_length: int, window_size: int, overlap: int) -> int:
    if text_length <= 0:
        return 1
    stride = max(1, window_size - overlap)
    if text_length <= window_size:
        return 1
    return 1 + (max(text_length - window_size, 0) + stride - 1) // stride


def parse_window_phase_label(phase_label: str) -> Optional[tuple]:
    m = _RE_WINDOW_STEP.match(phase_label or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def intra_in_window_slice(global_p: float, g_lo: float, g_hi: float) -> float:
    span = g_hi - g_lo
    if span <= 1e-15:
        return 0.0
    return max(0.0, min(1.0, (global_p - g_lo) / span))


def intra_step9_step10(global_p: float, g_lo: float, g_hi: float, chain_id: str) -> float:
    """步骤9/10 各占单窗的 1/10（与 orchestrator 传入 extraction 的 progress_range 一致），链内 0–1。"""
    span = g_hi - g_lo
    if span <= 1e-15:
        return 0.0
    if chain_id == "step9":
        s_lo = g_lo + span * (8.0 / 10.0)
        s_hi = g_lo + span * (9.0 / 10.0)
    elif chain_id == "step10":
        s_lo = g_lo + span * (9.0 / 10.0)
        s_hi = g_hi
    else:
        return intra_in_window_slice(global_p, g_lo, g_hi)
    ss = s_hi - s_lo
    if ss <= 1e-15:
        return 0.0
    return max(0.0, min(1.0, (global_p - s_lo) / ss))


def wf_for_chain(chain_id: str, intra: float) -> float:
    """单窗内流水线权重：步骤1–8 占 8/10，步骤9 占 1/10，步骤10 占 1/10。"""
    intra = max(0.0, min(1.0, intra))
    if chain_id == "phase_ab":
        return (8.0 / 10.0) * intra
    if chain_id == "step9":
        return (8.0 / 10.0) + (1.0 / 10.0) * intra
    if chain_id == "step10":
        return (9.0 / 10.0) + (1.0 / 10.0) * intra
    return (8.0 / 10.0) * intra


def wf_win_steps_1_8(global_p: float, g_lo: float, g_hi: float) -> float:
    """单窗内步骤1–8 占窗口宽度的前 8/10；返回 [0, 8/10] 的窗口内占比（相对整窗 0–1 的片段）。"""
    span = g_hi - g_lo
    if span <= 1e-15:
        return 0.0
    return max(0.0, min(8.0 / 10.0, (global_p - g_lo) / span))


def overall_from_window_wf(win_cur: int, win_tot: int, wf: float) -> float:
    if win_tot <= 0:
        return 0.0
    wf = max(0.0, min(1.0, wf))
    return max(0.0, min(1.0, (win_cur - 1 + wf) / float(win_tot)))


def overall_chain_from_window_intra(win_cur: int, win_tot: int, intra: float) -> float:
    """链级进度条位置：按窗口累计，当前窗口内按 intra 细分。"""
    if win_tot <= 0:
        return 0.0
    intra = max(0.0, min(1.0, intra))
    return max(0.0, min(1.0, (win_cur - 1 + intra) / float(win_tot)))


def completed_chunk_fraction(done_chunks: int, total_chunks: int) -> float:
    if total_chunks <= 0:
        return 0.0
    done_chunks = max(0, min(int(done_chunks), int(total_chunks)))
    return done_chunks / float(total_chunks)


def main_chain_anchor_rank(phase_label: str, tc: int) -> tuple:
    """主滑窗 1–8 的 UI 锚点优先级：越大越应作为展示锚点（抽取步骤 2–8 / 本窗 1–8 完成 优先于 步骤1 进行中）。

    并行时主线程可能在后序窗跑步骤1，而前序窗已在步骤2–8；此时应用「更靠前」的链上位置为锚点，而非总是跟主线程窗。
    """
    pl = (phase_label or "").strip()
    if not pl:
        return (-1, 0)
    if _RE_MAIN_1_8_DONE.search(pl) and ("已完成" in pl or "缓存" in pl):
        m = _RE_WINDOW_ONLY.search(pl)
        w = int(m.group(1)) if m else 0
        if tc > 0:
            w = max(1, min(w, tc))
        return (9, w)
    parsed = parse_window_phase_label(pl)
    if parsed:
        win_cur, _wt, step_cur, _st = parsed
        if tc > 0:
            win_cur = max(1, min(win_cur, tc))
        if 2 <= step_cur <= 8:
            return (step_cur, win_cur)
        if step_cur != 1:
            return (min(step_cur, 9), win_cur)
        if "进行中" in pl:
            return (0, win_cur)
        if "完成" in pl:
            return (1, win_cur)
        return (0, win_cur)
    # 抽取步骤 2-8 标签（如 "窗口 1/1 · 步骤2a: 文本锚点召回"）不匹配 _RE_WINDOW_STEP，
    # 但包含窗口信息和步骤编号，应赋予高于步骤1的优先级。
    wm = _RE_WINDOW_ONLY.match(pl)
    if wm:
        w = max(1, min(int(wm.group(1)), tc))
        _sm = _RE_EXTRACT_STEP_NUM.search(pl)
        step_num = int(_sm.group(1)) if _sm else 2
        step_num = max(2, min(8, step_num))
        return (step_num, w)
    return (-1, 0)


def remember_callback_ui_fields(
    task,
    progress: float,
    phase_label: str,
    message: str,
    chain_id: str,
) -> Dict[str, Any]:
    """推导总进度：主滑窗链 main（步骤1–8）、步骤9/10 链各自独立进度（0–1 为链内细粒度）。"""
    parsed = parse_window_phase_label(phase_label)
    tc = max(1, int(task.total_chunks or 1))
    pc = max(0, int(task.processed_chunks or 0))
    pc_f = pc / float(tc)

    if not parsed:
        new_o = max(pc_f, max(0.0, min(1.0, float(progress))))
        pl = phase_label or ""
        if chain_id in _MAIN_PHASES and _RE_MAIN_1_8_DONE.search(pl) and (
            "已完成" in pl or "缓存" in pl
        ):
            m = _RE_WINDOW_ONLY.search(pl)
            if m:
                win_cur = max(1, min(int(m.group(1)), tc))
                wf_main = 8.0 / 10.0
                main_global = min(1.0, (win_cur - 1 + wf_main) / float(tc))
                merged_p = max(new_o, main_global, pc_f)
                new_rank = main_chain_anchor_rank(pl, tc)
                old_rank = main_chain_anchor_rank(task.main_label or "", tc)
                if task.main_label and new_rank < old_rank:
                    return {"progress": merged_p}
                _pc = (win_cur - 1) * 10 + 8
                _pt = tc * 10
                return {
                    "progress": merged_p,
                    "phase_label": phase_label,
                    "message": message,
                    "phase_current": _pc,
                    "phase_total": _pt,
                    "main_progress": main_global,
                    "main_label": phase_label or message or "",
                }
        # 抽取步骤 2–8 的标签不匹配 _RE_WINDOW_STEP（如 "窗口 1/1 · 步骤2a: 文本锚点召回"），
        # 但仍需更新 main_progress/main_label/phase_current 以避免前端进度条停滞在步骤1。
        if chain_id in _MAIN_PHASES:
            wm = _RE_WINDOW_ONLY.match(pl)
            if wm:
                win_cur = max(1, min(int(wm.group(1)), tc))
                g_lo_w = (win_cur - 1) / float(tc)
                g_hi_w = win_cur / float(tc)
                wf_main = wf_win_steps_1_8(float(progress), g_lo_w, g_hi_w)
                main_global = min(1.0, (win_cur - 1 + wf_main) / float(tc))
                # 从标签中尝试提取步骤编号（如 "步骤2a"、"步骤3.5"）
                _sm = _RE_EXTRACT_STEP_NUM.search(pl)
                estimated_step = int(_sm.group(1)) if _sm else 2
                estimated_step = max(2, min(8, estimated_step))
                # 尝试提取子步骤分数（如 "实体对齐 (3/5)"）
                _fm = _RE_EXTRACT_STEP_FRAC.search(pl)
                if _fm:
                    _sub_done = int(_fm.group(1))
                    _sub_total = max(1, int(_fm.group(2)))
                    estimated_step = max(estimated_step, min(8, estimated_step + int(_sub_done / max(1, _sub_total))))
                _pc_phase = (win_cur - 1) * 10 + estimated_step
                _pt_phase = tc * 10
                new_rank = main_chain_anchor_rank(pl, tc)
                old_rank = main_chain_anchor_rank(task.main_label or "", tc)
                if task.main_label and new_rank < old_rank:
                    return {"progress": max(new_o, main_global)}
                return {
                    "progress": max(new_o, main_global),
                    "phase_label": phase_label,
                    "message": message,
                    "phase_current": _pc_phase,
                    "phase_total": _pt_phase,
                    "main_progress": main_global,
                    "main_label": phase_label or message or "",
                }
        return {
            "progress": new_o,
            "phase_label": phase_label,
            "message": message,
        }

    # 仅用标签解析「当前第几窗」；分母必须与 task.total_chunks 一致，否则与 orchestrator
    # 传入的 progress（按 total_chunks 切片的全局坐标）错位，实体/关系条会不按本窗比例显示。
    win_cur, _win_tot_label, step_cur, _step_tot = parsed
    win_cur = max(1, min(win_cur, tc))
    win_tot_eff = tc
    g_lo = (win_cur - 1) / float(win_tot_eff)
    g_hi = win_cur / float(win_tot_eff)
    if chain_id in _STEP910:
        intra = intra_step9_step10(float(progress), g_lo, g_hi, chain_id)
    else:
        intra = intra_in_window_slice(progress, g_lo, g_hi)
    wf = wf_for_chain(chain_id, intra)
    new_o = overall_from_window_wf(win_cur, win_tot_eff, wf)

    _pc = (win_cur - 1) * 10 + step_cur
    _pt = win_tot_eff * 10

    wf_main = wf_win_steps_1_8(float(progress), g_lo, g_hi)
    main_global = min(1.0, (win_cur - 1 + wf_main) / float(win_tot_eff))

    base: Dict[str, Any] = {
        "progress": max(pc_f, new_o),
        "phase_label": phase_label,
        "message": message,
        "phase_current": _pc,
        "phase_total": _pt,
    }

    # 主滑窗（步骤1–8）：chain main 或历史 phase_ab（锚点优先抽取链上位置，而非主线程步骤1）
    if chain_id in ("main", "phase_ab"):
        base["progress"] = max(base["progress"], main_global)
        new_rank = main_chain_anchor_rank(phase_label or "", tc)
        old_rank = main_chain_anchor_rank(task.main_label or "", tc)
        merged_p = base["progress"]
        if task.main_label and new_rank < old_rank:
            return {"progress": merged_p}
        base.update(
            main_progress=main_global,
            main_label=phase_label or message or "",
        )
        return base
    if chain_id == "step10":
        step9_global = max(
            completed_chunk_fraction(task.step9_done_chunks or 0, tc),
            completed_chunk_fraction(win_cur, tc),
            float(getattr(task, "step9_progress", 0.0) or 0.0),
        )
        step10_global = max(
            completed_chunk_fraction(task.step10_done_chunks or 0, tc),
            overall_chain_from_window_intra(win_cur, win_tot_eff, intra),
        )
        base.update(
            step9_progress=step9_global,
            step10_progress=step10_global,
            step10_label=phase_label or message or "",
        )
        return base
    # step9：实体链进度按窗口累计；不要把关系链已完成进度清零。
    base.update(
        step9_progress=max(
            completed_chunk_fraction(task.step9_done_chunks or 0, tc),
            overall_chain_from_window_intra(win_cur, win_tot_eff, intra),
        ),
        step9_label=phase_label or message or "",
    )
    return base
