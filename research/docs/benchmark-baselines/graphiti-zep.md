# Graphiti / Zep 论文：benchmark 与原始结果

这里的“Graphiti 论文”对应 Graphiti 作为底层时序知识图谱引擎、以 Zep 系统呈现的原始论文： [Zep: A Temporal Knowledge Graph Architecture for Agent Memory（arXiv:2501.13956）](https://arxiv.org/abs/2501.13956)。

## 论文实际测评的 benchmark

原始论文测评两个 benchmark：

1. **Deep Memory Retrieval (DMR)**：500 个多 session 对话，每段 5 个 session、每个 session 最多 12 条消息；每段对话一个单跳事实检索问题。
2. **LongMemEval**：长时记忆问答，论文报告 GPT-4o-mini 和 GPT-4o 两个 answer model，以及六类问题的分项结果。

论文没有把 LoCoMo 作为主 benchmark；因此 Graphiti/Zep 与本项目的 LoCoMo 结果只能做额外参考，不能称为论文同榜。

## DMR 主表摘录

| 方法 / answer model | DMR accuracy |
|---|---:|
| Recursive Summarization / GPT-4-turbo | 35.3% |
| Conversation Summaries / GPT-4-turbo | 78.6% |
| MemGPT / GPT-4-turbo | 93.4% |
| Full-conversation / GPT-4-turbo | 94.4% |
| **Zep / GPT-4-turbo** | **94.8%** |
| Conversation Summaries / GPT-4o-mini | 88.0% |
| Full-conversation / GPT-4o-mini | 98.0% |
| **Zep / GPT-4o-mini** | **98.2%** |

## LongMemEval 主表摘录

| Answer model | System | Accuracy | Latency | 平均上下文 |
|---|---|---:|---:|---:|
| GPT-4o-mini | Full-context | 55.4% | 31.3 s | 115k tokens |
| GPT-4o-mini | **Zep** | **63.8%** | **3.20 s** | **1.6k tokens** |
| GPT-4o | Full-context | 60.2% | 28.9 s | 115k tokens |
| GPT-4o | **Zep** | **71.2%** | **2.58 s** | **1.6k tokens** |

LongMemEval 六类分项（`Full-context → Zep`）如下：

| 类别 | GPT-4o-mini | GPT-4o |
|---|---:|---:|
| Single-session preference | 30.0 → 53.3 | 20.0 → 56.7 |
| Single-session assistant | 81.8 → 75.0 | 94.6 → 80.4 |
| Temporal | 36.5 → 54.1 | 45.1 → 62.4 |
| Multi-session | 40.6 → 47.4 | 44.3 → 57.9 |
| Knowledge update | 76.9 → 74.4 | 78.2 → 83.3 |
| User | 81.4 → 92.9 | 81.4 → 92.9 |

## 与本项目的对照

| Benchmark | 论文结果 | 项目现状 | 可比性 |
|---|---:|---:|---|
| DMR | Zep 94.8% (GPT-4-turbo), 98.2% (GPT-4o-mini) | 尚未发现 DMR run | **不可比**；需要单独实现/运行 DMR |
| LongMemEval-S | Zep 71.2% (GPT-4o), 63.8% (GPT-4o-mini) | v24 **84.6% / 500**（Qwen3.7 judge）；早期 50 题 88.0% | **同题数但 cross-judge**；answer model、prompt、ingestion 仍不同 |
| LoCoMo | 原始 Zep 论文未测 | 项目 1540 exact protocol **93.18%/93.57%** | 不是论文同榜；可与 Mem0 的 LoCoMo 结果比较 |

结论是：项目当前最适合和 Zep/Graphiti 比的是 **LongMemEval**，项目已有 500 题 full run，但必须进一步固定 answer model、judge 和评分脚本并重跑双方才构成严格比较；DMR 目前没有对应项目结果。
