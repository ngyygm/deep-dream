# Project results vs paper baselines

这是项目当前本地 run 与论文结果的并列摘要。`可比` 只表示指标和大致 protocol 接近，不表示严格复现；`方向性` 表示题数、judge、answer model 或数据协议不同。

| Benchmark | Paper reference | Deep-Dream run | Comparison |
|---|---:|---:|---|
| LoCoMo original token-F1 | Human 87.9; Full-context GPT-4 Turbo 51.6; Observation-RAG 43.3 | quality-v1 baseline **64.99**, skill-agent **63.17**, thinking-on **61.33** (1,986) | **可比指标**；模型/prompt/ingestion 不同，不能称严格复现 |
| LoCoMo 12-question probe | — | **0.6780** overall | 仅项目 smoke/probe |
| LoCoMo-Plus | Best paper baseline Gemini-2.5-Pro **26.06** cognitive average（论文表）；Mem0 **15.80**、A-Mem **17.20** | Kimi-K3 judge **0.9551** / 401；Qwen3.7 judge **0.6459** / 401 | **方向性**；judge/backbone 不同，不能直接宣称超过论文 |
| LongMemEval-S | Paper GPT-4o full S **0.640** (500); oracle **0.924** | v24 memory **0.846** (500, Qwen3.7 judge); earlier gate **0.880** (50) | **同题数但 cross-judge**；仍非官方 GPT-4o head-to-head |
| MEME filler32k | Paper: six-system average Cas **0.03**, Abs **0.01**; best practical overall about **0.42** | Kimi-K3 bash-files: raw after **0.7248**, real/trivial-filtered task results available; alternate judge summary raw after **0.6225** | **方向性**；paper uses GPT-4o judge and gpt-4.1-mini, local uses Kimi-K3/Qwen3.5 pipeline |
| MemoryAgentBench | Paper Table 2 reports method-by-task scores; no single headline overall | run manifest exists, no final project score found | 尚不能做数值对比 |
| BEAM | Paper LIGHT vs RAG at 10M: **0.266 vs 0.249** (Llama-4 Maverick) | `beam-100k-qwen35-full-remember-v1` has pipeline manifest, no completed score summary found | 尚不能做数值对比 |
| Mem0 LoCoMo 1540 protocol | Mem0 paper weighted `J` ≈ **62.14**; current Mem0 repo **92.5%** | Qwen3.7 judge **93.18%**; Kimi-K3 judge **93.57%** | **协议接近但非严格同榜**；论文 `J` 与项目 binary accuracy 不是同一指标，judge/backbone/backend 仍不同 |
| Graphiti/Zep DMR | **94.8%** (GPT-4-turbo), **98.2%** (GPT-4o-mini) | 无 DMR run | 不可比，需补测 |
| Graphiti/Zep LongMemEval | **71.2%** GPT-4o / **63.8%** GPT-4o-mini | v24 full-500 **84.6%** (Qwen3.7 judge) | 同题数但 cross-judge；可作方向性对照 |

## Reading the deltas

### LoCoMo

项目的 full quality-v1 baseline 64.99 token-F1 高于论文中 GPT-4 Turbo full-context 51.6，但这不是纯粹 memory architecture 的提升：项目使用了不同的 Qwen backbone、remember pipeline、chunking 和 answer protocol。skill-agent 在同一项目协议下比 baseline 低 1.82 points，thinking-on 再低 1.84 points；这支持“先优化证据召回和 answer grounding，再增加 Agent reasoning”的判断。

### LoCoMo-Plus

项目 Kimi-K3 judge 的 95.51% 看起来远高于论文表中的 26.06%，但 Qwen3.7 judge 同一 full-401 run 只有 64.59%。这不是模型突然提升 69 points 的证据，而是 judge/backbone/protocol 敏感性的直接示例。论文 baseline 表中的 cognitive average 和项目 run 只能作为上下界/诊断，不应合并到一个排名。

### LongMemEval-S

项目 v24 full-500 为 84.6%，另有早期 50-question gate 88%。完整结果与原论文和 Zep 使用同一题数，但 answerer、judge 和 memory ingestion 不同，因此只能作为 cross-judge 结果；不能称为严格超过 GPT-4o 或 Zep。

### MEME

项目最有价值的不是 raw overall 0.7248，而是任务分解：在 filler32k + Kimi-K3 bash-files 条件下 ER/Agg/Tr 为 1.00/0.91/0.97，Del 为 0.02，Cascade 的 trivial-filtered real accuracy 0.1098，Absence 为 0.6692。与论文共同暴露出 Del/Cascade 的困难，但 Absence 明显受 judge 和实现协议影响，因此应优先看 real/trivial-filtered 数字而不是 raw pass。

### 未完成项目

MemoryAgentBench 和 BEAM 目前只确认有 run manifest / pipeline 产物，没有可引用的最终项目 score。这里特意不填 0，也不把未完成 run 当作失败。

### Mem0 与 Graphiti/Zep

项目已经有与 Mem0 最接近的 LoCoMo-1540 本地结果；但由于评测模型和 judge 不同，当前只能说“在对齐题集和协议的本地实验中表现相当/更高”，不能写成严格超过 Mem0。与 Graphiti/Zep 的直接缺口只剩 DMR：LongMemEval 已有 full-500，但仍是 cross-judge。详细论文表见 [mem0.md](mem0.md) 与 [graphiti-zep.md](graphiti-zep.md)。
