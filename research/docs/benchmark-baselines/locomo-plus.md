# LoCoMo-Plus paper baselines

来源：Li et al., *LoCoMo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents*, ACL 2026. [arXiv](https://arxiv.org/abs/2602.10715) · [ACL PDF](https://aclanthology.org/2026.acl-long.1150.pdf)

主表同时报告 factual LoCoMo 和 cognitive LoCoMo-Plus。论文主表实际给出 LoCoMo 的五类分数、LoCoMo factual average、LoCoMo-Plus average 以及两者 gap；下面只摘录后面三个字段（百分比）。该论文的 cognitive evaluation 使用 constraint-consistency / judge protocol，不应与原始 LoCoMo token-F1 混排。

| Method | Backbone | LoCoMo factual avg | LoCoMo-Plus avg | Gap |
|---|---|---:|---:|---:|
| Qwen2.5-3B-Instruct | open | 42.20 | 10.82 | 31.38 |
| Qwen2.5-7B-Instruct | open | 45.31 | 9.57 | 35.74 |
| Qwen2.5-14B-Instruct | open | 63.45 | 19.24 | 44.21 |
| Qwen3-4B | open | 54.91 | 15.70 | 39.21 |
| Qwen3-8B | open | 56.86 | 17.68 | 39.18 |
| Qwen3-14B | open | 59.65 | 19.09 | 40.56 |
| gpt-5-nano | closed | 54.96 | 14.84 | 40.12 |
| gpt-4.1 | closed | 62.21 | 18.63 | 43.58 |
| gpt-4o | closed | 62.99 | 21.05 | 41.94 |
| gemini-2.5-flash | closed | 69.25 | 24.67 | 44.58 |
| gemini-2.5-pro | closed | 71.78 | 26.06 | **45.72** |
| Text-ada-embedding-002 | RAG + GPT-4o | 37.38 | 13.91 | 23.47 |
| text-embedding-3-small | RAG + GPT-4o | 37.23 | 12.29 | 24.94 |
| text-embedding-3-large | RAG + GPT-4o | 45.32 | 15.55 | 29.77 |
| Mem0 | GPT-4o | 57.24 | 15.80 | 41.44 |
| SeCom | GPT-4o | 57.53 | 14.90 | 42.63 |
| A-Mem | GPT-4o | 59.64 | 17.20 | 42.44 |

注：论文 PDF 的列布局把 factual LoCoMo 五类和三个汇总字段并排展示；此处不自行从类别数重算 average。

## Project anchors

本地 full-401 结果见 `.benchmark_runs/locomo-plus-cognitive-qwen35-cues-full-v1/`；Kimi-K3 judge 的 published-style run 为 0.9551，但 Qwen3.7 judge 的同一 run 约 0.621–0.646，说明 judge/backbone 差异必须保留。

## Project comparison

论文表中的最高 LoCoMo-Plus average 是 Gemini-2.5-Pro **26.06**；项目同一 full-401 run 在 Kimi-K3 judge 下为 **95.51%**，在 Qwen3.7-plus judge 下为 **64.59%**。这两个项目数字不能直接解释成超过论文 baseline：judge model、answer model 和运行时都不同。最可靠的结论是项目结果对 judge 极其敏感，后续比较应固定同一 judge。
