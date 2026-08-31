# Experiment Tracker

| ID | Status | Evidence |
|---|---|---|
| E1 LoCoMo-1986 | complete | `.benchmark_runs/locomo-full-quality-v1/summary.baseline.json` |
| E2 LoCoMo-1540 | complete | `.benchmark_runs/locomo-full-quality-v1/judge_summary.kimi-agent-direct-qwen37-full-thinking-off.qwen37-mem0-current-exact-direct.json` |
| E3 LongMemEval-S-500 | complete | `.benchmark_runs/longmemeval-source-v24-full500/judge_summary.kimi-agent-direct-longmem500-memory-v24-full.qwen37-longmem-official-memory-v24-full500.json` |
| E4 LoCoMo-Plus-401 | complete | `.benchmark_runs/locomo-plus-cognitive-qwen35-cues-full-v1/judge_summary.kimi-agent-direct-locomo-plus-qwen37-qwen35-cues-full-v1.qwen37-locomo-plus-official-qwen35-cues-full-v1-final2.json` |
| E5 MEME-100 | complete, limitation | `.benchmark_runs/meme-filler32k-qwen35-full100-k3-v1/meme_official_judge_kimik3_summary.json` |
| X1 channel ablation | complete | `.benchmark_runs/locomo-full-quality-v1/channel_policy_replay.lexical-vs-*.summary.json` |
| X2 context expansion | complete | `.benchmark_runs/locomo-full-quality-v1/retrieval_ablation.source-v2-neighbor-expansion.wrong.summary.json` |
| X3 budget frontier | invalid/pending | depth10-vs20 replay violates prefix invariant on 210/210; excluded until rerun |
| X4 judge sensitivity | complete | exact/legacy/Kimi judge summaries |
| X5 update/temporal slices | complete | LongMemEval full-500 judge summary |
| X6 negative-memory | partial | LoCoMo adversarial and MEME completed; dedicated false-premise metric pending |
| X7 causal provenance ablation | pending | — |
| X8 update/delete microbenchmark | pending | — |
| X9 controlled external systems | pending | — |
| X10 cost/latency | partial | local runtime captured; common hardware protocol pending |
