"""MemoryAgentBench 分层抽样：选一组覆盖全部 source family 的 scope。

背景：全量 146 个 scope 约 1.1 亿字符，ingest 成本高；先用一个 ≤ 800 万字符
的分层样本跑通 ingest→answer→official-score 链路，抽样口径与 ICLR Table-3
的 source 分组一致（scorer 对缺席 source 记 None，summary 里有每源 scope 数，
报告可据此标注"抽样"）。

选择规则（确定性）：
- TTL（6）/CR（8）全收；
- AR 每个 source family 取最小的 1 个 scope，longmemeval_s 取最小 2 个；
- LRU 取最小的 4 个 infbench + 最小的 2 个 detective_qa；
- 超出预算时循环丢弃"所选 family 内还有其他代表"的最大 scope，实在不行
  才丢最大 scope（尽量保住每个 family 的代表）。

用法（repo 根目录）::

    .venv/bin/python -m research.benchmark.mab_sample \
        --out research/.benchmark_runs/mab-sample
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.benchmark.datasets import load_benchmark

RESEARCH_ROOT = Path(__file__).resolve().parents[1]

# 与 scorer macro 的 source 分组对齐（Table-3 列粒度）。
def source_family(source: str) -> str:
    """把 official_source 归并到 Table-3 的 family 粒度。"""
    if source.startswith("icl_"):
        return "icl"
    if source.startswith("factconsolidation_sh_"):
        return "factconsolidation_sh"
    if source.startswith("factconsolidation_mh_"):
        return "factconsolidation_mh"
    if source.startswith("eventqa_"):
        return source
    if source == "longmemeval_s*":
        return "longmemeval_s"
    return source


# 每个 family 的取样个数：不在表里的 family 默认 1 个。
FAMILY_PICKS = {
    "longmemeval_s": 2,
    "infbench_sum_eng_shots2": 4,
    "detective_qa": 2,
}


def build_scope_records(items: list[Any]) -> list[dict[str, Any]]:
    """按 scope 聚合：题目数、context 总字符、family。"""
    scopes: dict[str, dict[str, Any]] = {}
    for item in items:
        record = scopes.setdefault(item.scope_id, {
            "scope_id": item.scope_id,
            "competency": item.metadata["competency"],
            "official_source": str(item.metadata.get("official_source") or ""),
            "questions": 0,
            "context_chars": 0,
            "context_documents": item.metadata.get("context_documents", 0),
            "context_sha256": item.metadata.get("context_sha256", ""),
            "_chars_counted": False,
        })
        record["questions"] += 1
        if not record["_chars_counted"]:
            record["context_chars"] = sum(len(s.text) for s in item.sessions)
            record["_chars_counted"] = True
    for record in scopes.values():
        del record["_chars_counted"]
    return sorted(scopes.values(), key=lambda r: r["scope_id"])


def select_scopes(
    records: list[dict[str, Any]], *, budget_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """分层选择 + 预算裁剪；返回 (selected, dropped)。"""
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_family.setdefault(source_family(record["official_source"]), []).append(record)

    selected: list[dict[str, Any]] = []
    for family, members in sorted(by_family.items()):
        # TTL/CR 全收：这两个 competency 的 family 各自只有少量 scope。
        keep_all = {m["competency"] for m in members} <= {"TTL", "CR"}
        picks = len(members) if keep_all else FAMILY_PICKS.get(family, 1)
        smallest = sorted(members, key=lambda m: (m["context_chars"], m["scope_id"]))
        selected.extend(smallest[:picks])

    dropped: list[dict[str, Any]] = []
    while sum(r["context_chars"] for r in selected) > budget_chars:
        families = [source_family(r["official_source"]) for r in selected]
        # 优先丢弃 family 内还有其他代表的 scope，尽量保住每个 family。
        droppable = [
            r for r in selected
            if families.count(source_family(r["official_source"])) > 1
        ] or selected
        victim = max(droppable, key=lambda r: (r["context_chars"], r["scope_id"]))
        selected.remove(victim)
        dropped.append(victim)
    return sorted(selected, key=lambda r: r["scope_id"]), dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a stratified MemoryAgentBench scope sample")
    parser.add_argument("--data-dir", type=Path, default=RESEARCH_ROOT / ".benchmark_data")
    parser.add_argument("--out", type=Path, default=RESEARCH_ROOT / ".benchmark_runs" / "mab-sample",
                        help="输出前缀：写 <out>.scopes.txt 与 <out>.stats.json")
    parser.add_argument("--budget-chars", type=int, default=8_000_000)
    args = parser.parse_args(argv)

    items, _ = load_benchmark("memoryagentbench", args.data_dir)
    records = build_scope_records(items)
    selected, dropped = select_scopes(records, budget_chars=args.budget_chars)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scopes_path = args.out.with_suffix(".scopes.txt") if args.out.suffix else args.out.parent / (args.out.name + ".scopes.txt")
    scopes_path.write_text(
        "".join(f"{r['scope_id']}\n" for r in selected), encoding="utf-8",
    )
    total_chars = sum(r["context_chars"] for r in selected)
    total_questions = sum(r["questions"] for r in selected)
    stats = {
        "dataset": "memoryagentbench",
        "budget_chars": args.budget_chars,
        "selection": (
            "TTL 全收 + CR 全收 + AR 每 family 最小 1 个（longmemeval_s 取 2）"
            "+ LRU 最小 4 个 infbench + 2 个 detective_qa；超预算先丢多代表 "
            "family 中最大 scope，再丢最大 scope"
        ),
        "scopes": [
            {**r, "family": source_family(r["official_source"])} for r in selected
        ],
        "dropped_for_budget": [
            {"scope_id": r["scope_id"], "context_chars": r["context_chars"],
             "family": source_family(r["official_source"])}
            for r in dropped
        ],
        "totals": {
            "scopes": len(selected),
            "questions": total_questions,
            "context_chars": total_chars,
            "families": sorted({source_family(r["official_source"]) for r in selected}),
        },
    }
    stats_path = args.out.parent / (args.out.name + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats["totals"], ensure_ascii=False))
    print(f"wrote {scopes_path} and {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
