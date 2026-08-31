# MemoryAgentBench paper baselines

来源：Hu, Wang & McAuley, *Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions*, arXiv:2507.05257. [Paper](https://arxiv.org/abs/2507.05257) · [HTML full text](https://ar5iv.labs.arxiv.org/html/2507.05257)

论文 Table 2 的列为：RULER-QA、NIAH-MQ、∞Bench-QA、LongMemEval(S*)、EventQA、MCC、Recommendation、∞Bench-Sum、FactCon-SH、FactCon-MH；数值为 accuracy (%)。所有 RAG/commercial memory agents 使用 GPT-4o-mini backbone。

| Method | RULER | NIAH | ∞QA | LME(S*) | EventQA | MCC | Recom | ∞Sum | Fact-SH | Fact-MH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-4o (full context) | 61.5 | 25.0 | 55.4 | 32.0 | 77.2 | 87.6 | 12.3 | 32.2 | 60.0 | 5.0 |
| GPT-4.1-mini (full context) | 74.5 | 94.8 | 45.8 | 55.7 | 82.6 | 75.6 | 16.7 | 41.9 | 36.0 | 5.0 |
| Gemini-2.0-Flash | 73.0 | 83.8 | 53.2 | 47.0 | 67.2 | 84.0 | 8.7 | 23.9 | 30.0 | 3.0 |
| Claude-3.7-Sonnet | 65.0 | 38.0 | 50.6 | 34.0 | 74.6 | 89.4 | 18.3 | 52.5 | 43.0 | 2.0 |
| BM25 | 61.0 | 100.0 | 45.6 | 45.3 | 74.6 | 75.4 | 13.6 | 20.9 | 56.0 | 3.0 |
| Contriever | 26.5 | 2.5 | 38.1 | 15.7 | 66.8 | 70.6 | 15.2 | 21.2 | 18.0 | 7.0 |
| Text-Embed-3-Large | 49.0 | 19.5 | 50.1 | 52.3 | 70.0 | 72.4 | 16.2 | 21.6 | 28.0 | 4.0 |
| NV-Embed-v2 | 83.0 | 73.5 | 51.4 | 55.0 | 72.8 | 69.4 | 13.5 | 20.7 | 55.0 | 6.0 |
| RAPTOR | 33.5 | 15.8 | 31.3 | 34.3 | 45.8 | 59.4 | 12.3 | 13.4 | 14.0 | 1.0 |
| GraphRAG | 47.0 | 38.3 | 35.8 | 35.0 | 34.4 | 39.8 | 9.8 | 0.4 | 14.0 | 2.0 |
| HippoRAG-v2 | 71.0 | 67.5 | 45.7 | 50.7 | 67.6 | 61.4 | 10.2 | 14.6 | 54.0 | 5.0 |
| Mem0 | 28.0 | 4.8 | 22.4 | 36.0 | 37.5 | 3.4 | 10.0 | 0.8 | 18.0 | 2.0 |
| Cognee | 33.5 | 4.0 | 19.7 | 29.3 | 26.8 | 35.4 | 10.1 | 2.3 | 28.0 | 3.0 |
| Self-RAG | 38.5 | 8.0 | 28.5 | 25.7 | 31.8 | 11.6 | 12.8 | 0.9 | 19.0 | 3.0 |
| MemGPT | 39.5 | 8.8 | 20.8 | 32.0 | 26.2 | 67.6 | 14.0 | 2.5 | 28.0 | 3.0 |

论文结论是：RAG 在 accurate retrieval 较强，full-context 模型在 test-time learning / long-range understanding 较强，而 conflict resolution 尤其 multi-hop 几乎所有方法都失败。

## Project anchors

项目已有 `memoryagentbench-qwen35-full-v1` run manifest；目前未发现完整的最终项目 summary，因此这里只保存论文表，不把该 run 当作已完成结果。

## Project comparison

当前没有可引用的项目最终分数，因此不填 0，也不把 run manifest 视为失败。下一步需要先从该目录补齐按四个 capability（AR/TTL/LRU/CR）和各子任务的最终 summary，再与论文 Table 2 对齐。
