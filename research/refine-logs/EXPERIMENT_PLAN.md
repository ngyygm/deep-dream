# Deep-Dream 论文实验计划

## 目标

验证“versioned concept graph 负责导航、source-level evidence 负责事实授权、Agent 受 evidence submission 约束”是否带来跨 benchmark 的长期记忆质量、可追溯性和预算效率。

## 主表实验

| ID | 实验 | 数据 | 核心指标 | 状态 |
|---|---|---|---|---|
| E1 | LoCoMo original protocol | 1,986 QA | token-F1、retrieval recall、failure attribution | 已完成 |
| E2 | LoCoMo Mem0-compatible protocol | 1,540 QA | binary judge accuracy、四类分项、cross-judge | 已完成 |
| E3 | LongMemEval-S | 500 QA | overall、六类分项、retrieval recall | 已完成 |
| E4 | LoCoMo-Plus cognitive | 401 QA | official-compatible cognitive accuracy | 已完成；cross-judge |
| E5 | MEME filler32k | 100 episodes | ER/Agg/Tr/Del/Cas/Abs | 已完成；暴露明显短板 |

## 特色实验

### X1. Retrieval-channel decomposition（已完成）

在 LoCoMo 210 题上比较 lexical、semantic 和 lexical+semantic fused。报告 recall-any、recall-all、响应字节。目的不是证明 semantic 必然更好，而是检验多通道互补性。

### X2. Source-span expansion efficiency（已完成）

在 105 个错误/困难样本上，从精确 span 逐步扩展到 ±1/±2 neighboring turns，测量 evidence recall 与 payload 增长。该实验直接对应“fine retrieval, then source expansion”。

### X3. Evidence-budget frontier（旧 replay 失效，待重跑）

旧 depth 10 vs 20 replay 在 210/210 题上破坏了预期 prefix invariant，因此不得进入主文或正向结论。正式投稿前先修复不变量，再补 1/3/5/10/20、最终 QA accuracy 和 paired confidence intervals。

### X4. Judge/prompt sensitivity（已完成）

固定同一 1,540 个 hypothesis，仅替换 judge prompt 或 judge family：旧 prompt vs Mem0 exact prompt；Qwen3.7 vs Kimi-K3。用于说明单一 vendor leaderboard 的不稳定性。

### X5. Update and temporal slices（已完成）

使用 LongMemEval-S knowledge-update 与 temporal-reasoning 分项，检验版本链相关能力；不得把 slice performance 直接归因为版本图，除非 X7 完成。

### X6. Negative-memory stress test（已完成基础结果，需增强）

使用 LoCoMo adversarial/false-premise 和 MEME Del/Cas/Abs 检验错误前提、删除、级联和“不存在”判断。MEME 已显示 Del/Cas 是系统短板，论文应明确报告。

### X7. Provenance/graph ablation（待运行，最高优先级）

固定同一 answerer/judge，比较：

1. source-only lexical;
2. source lexical+semantic;
3. source + neighboring spans;
4. source + versioned concepts/relations;
5. full Deep-Dream + evidence-bound Agent;
6. full Deep-Dream but允许未读取的 derived memory 直接进入答案。

主指标为 QA accuracy、source evidence recall、unsupported-answer rate、平均 evidence tokens。

### X8. Update/delete causal microbenchmark（待实现）

构造每个实体 3–8 次状态变化、冲突、撤销和否定的可审计序列；在每次写入后询问 current/as-of/ever/never 四类问题。比较 static graph、overwrite store、versioned source-grounded graph。报告 current-state accuracy、historical accuracy、deletion leakage、provenance exact match。

### X9. Controlled Mem0/Zep comparison（待运行）

在同一 answer model、judge、temperature、prompt 和题集上运行 Deep-Dream、Mem0 与 Graphiti/Zep。LoCoMo 1,540 与 LongMemEval-S 500 为优先；DMR 作为补充。该实验完成前，论文只允许 cross-protocol/cross-judge 对照。

### X10. Ingestion/query cost（待补）

报告 ingest latency、LLM calls、storage size、query P50/P95、evidence tokens，并与 full-context/summary/RAG 对照。当前只有项目运行时统计，不能和 Zep 的服务端 latency 直接混排。

## 投稿最小闭环

X7、X8、X9 均为投稿硬门槛；否则论文定位为 system report / preprint，并从标题、摘要和贡献中移除对应的机制或相对性能主张。X8 的优先级高于再增加一个通用 QA benchmark。

## 计算与执行策略

- 当前阶段只聚合已有 run，不启动新的付费/GPU 实验。
- 新实验全部使用冻结数据哈希、固定 answerer/judge 和逐题 JSONL。
- 每个数字必须可回溯到 raw artifact；失败/缺失题不得静默丢弃。
