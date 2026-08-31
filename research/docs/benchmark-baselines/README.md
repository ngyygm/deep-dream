# Benchmark paper baselines

本目录整理项目实际运行过的 benchmark 对应原始论文中的方法结果。数字尽量直接抄录论文主表；派生值、第三方复现和本项目结果均单独标注，避免把不同 judge、backbone、子集或指标混成一个 leaderboard。

## Index

| Benchmark | Paper | Extracted file | Project artifacts |
|---|---|---|---|
| LoCoMo | Maharana et al., ACL 2024 | [locomo.md](locomo.md) | `.benchmark_runs/locomo-*` |
| LoCoMo-Plus | Li et al., ACL 2026 / arXiv:2602.10715 | [locomo-plus.md](locomo-plus.md) | `.benchmark_runs/locomo-plus-*` |
| LongMemEval-S | Wu et al., ICLR 2025 / arXiv:2410.10813 | [longmemeval.md](longmemeval.md) | `.benchmark_runs/longmemeval-*` |
| MemoryAgentBench | Hu, Wang & McAuley, arXiv:2507.05257 | [memoryagentbench.md](memoryagentbench.md) | `.benchmark_runs/memoryagentbench-*` |
| MEME | Jung et al., arXiv:2605.12477 | [meme.md](meme.md) | `.benchmark_runs/meme-*` |
| BEAM | Tavakoli et al., ICLR 2026 / arXiv:2510.27246 | [beam.md](beam.md) | `.benchmark_runs/beam-*` |
| Mem0 paper | Chhikara et al., arXiv:2504.19413 | [mem0.md](mem0.md) | `.benchmark_runs/locomo-full-quality-v1` |
| Graphiti/Zep | Rasmussen et al., arXiv:2501.13956 | [graphiti-zep.md](graphiti-zep.md) | `.benchmark_runs/longmemeval-*` |

项目结果对照总览见：[project-comparison.md](project-comparison.md)。

## Comparison rules

- Accuracy、token-F1、retrieval recall 和 rubric/nugget score 不是同一指标。
- 论文原始结果优先；厂商网页、第三方复现和本地 run 只作为补充。
- 任何跨论文比较都必须同时写明 answer model、judge model、题目子集、prompt 和是否使用 full-context/oracle evidence。
- 当前 `.benchmark_runs/` 多数是被忽略的本地实验产物；本目录是可审阅、可版本化的论文数字摘要。
