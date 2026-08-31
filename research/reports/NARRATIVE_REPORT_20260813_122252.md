# Deep-Dream：面向长期记忆 Agent 的原文锚定、可审计概念层

## Problem statement

长期记忆系统通常把对话压缩为摘要、原子事实或知识图谱。压缩降低了检索成本，却也改变了事实的授权边界：抽取错误可能成为新的“事实”，实体合并会抹平时间变化，图边或摘要又常缺少足以重建原始上下文的 provenance。结果是系统可能检索到一个语义相关的派生节点，却无法证明最终答案来自哪个原始 turn，也难以在冲突、删除和错误前提下可靠拒答。

Mem0 通过增量事实抽取与 ADD/UPDATE/DELETE/NOOP 操作维护记忆；Mem0g 与 Zep/Graphiti 增加实体关系和时间信息。Deep-Dream 选择不同的事实模型：**原始文档和对话 turn 始终是 source of truth；concept、relation、summary 和 graph 只是可重建的导航 overlay。**

## Core claim

Deep-Dream separates memory navigation from factual authority: concepts and relations guide retrieval, while submitted evidence IDs must refer to source spans actually read within the active scope. This is an ID/provenance boundary, not a semantic entailment guarantee.

当前证据支持“系统可在多个长期记忆 benchmark 上取得强 cross-judge 结果，并且 source-span expansion 和多通道融合能显著提高 evidence recall”；当前证据尚不能独立证明“版本图因果性地带来全部 QA 增益”，因为 causal provenance/graph ablation 尚未完成。

## Method summary

### 1. Document-first memory substrate

Deep-Dream 保存完整 document、document version 和 heading-aware episode。原始 Markdown/文本仍可读、可编辑；派生记忆不会取代来源。每个 entity observation 和 relation assertion 绑定 episode、source span/lineage 和 pipeline run。

### 2. Versioned concept graph

系统把稳定身份与时间变化分开：`family_id` 表示跨 episode 的概念身份，observation/assertion 表示某一来源中的版本化内容。保守 alignment 合并真正同一的概念，同时保留每次观察；relation 也以 family 和 assertion 分层。该结构允许查询“现在是什么”“过去是什么”和“哪段来源支持这一状态”。

### 3. Multi-step Remember pipeline

Remember pipeline 对 heading-aware chunks 执行实体发现、规范化、实体内容写作、质量门控、关系发现、关系内容写作、跨窗口对齐和原子发布。实体内容与关系内容的生成和 discovery 解耦；文档在全部窗口完成前保持不可搜索，避免部分 ingest 泄漏到查询。

### 4. Hybrid navigation and source expansion

查询可通过 lexical、semantic、concept、relation 和 graph neighborhood 产生候选，并通过 reciprocal-rank fusion 合并。系统优先定位细粒度 source spans，再按需扩展相邻 turns，而不是一开始返回整个 session。实验表明，邻域扩展在较小 payload 增量下显著提高困难样本的 evidence recall。

### 5. Evidence-bounded Agent interface

Agent 只能使用只读、conversation-scoped memory tools。每个问题使用 fresh context，不能访问 gold answer、gold evidence ID 或其他 conversation。最终 `submit_evidence` 只接受本题工具调用实际返回且属于 active scope 的 source/session/turn IDs。系统保存工具调用、观察、证据、延迟和运行时哈希，但不保存隐藏思维链。

## Quantitative evidence

### Main benchmark snapshot

| Benchmark | Questions | Deep-Dream | Scope |
|---|---:|---:|---|
| LoCoMo original | 1,986 | 64.99 token-F1 deterministic baseline | 与原论文同指标，但不同模型/实现 |
| LoCoMo Mem0-compatible | 1,540 | 93.18% Qwen3.7; 93.57% Kimi-K3 | exact prompt，cross-judge |
| LongMemEval-S | 500 | 84.60% Qwen3.7 | official-compatible prompt，cross-judge |
| LoCoMo-Plus cognitive | 401 | 64.59% Qwen3.7 | official cognitive prompt，cross-judge |
| MEME filler32k | 100 episodes | 62.25% raw after | Kimi-K3；Deletion 0%，明确短板 |

LongMemEval-S 分项为 knowledge-update 91.03%、multi-session 74.44%、single-session-assistant 100%、single-session-preference 53.33%、single-session-user 98.57%、temporal-reasoning 84.21%。

### Retrieval-channel ablation

在 210 个 LoCoMo development/gate 问题上，lexical/semantic/fused 的 recall-all 分别为 75.24%、70.48% 和 78.10%。semantic-only 低于 lexical，而融合在该切片上高于两者；这只是通道互补的诊断观察，不构成一般可靠性或显著性结论。

### Source-span expansion

在 105 个从既有错误中选出的诊断样本上，lexical+semantic span 的 recall-any/all 为 70.48%/52.38%；保持相同通道并扩展 ±2 turns 后达到 89.52%/73.33%，平均 payload 从 3,089 增至 3,609 bytes，即 recall-any/all 增加 19.05/20.95 points、payload 增加 16.84%。该切片不是独立 held-out test，论文将其限定为开发诊断。

### Invalidated depth replay

检索 depth 10 到 20 的旧 replay 在 210/210 题上破坏了 top-k prefix invariant。该结果仅保留在 evidence ledger 作为失效实验记录，不进入主文 figure 或结论。正式 budget curve 必须在修复 invariant 后重跑。

### Judge sensitivity

同一 1,540 个输出在旧版简化 prompt 下为 92.21%，换成 Mem0 exact prompt 后为 93.18%；换用 Kimi-K3 judge 为 93.57%。这说明约 1 point 的评分变化可由 judge 指令或模型产生，因此论文把 cross-judge 结果与官方结果分栏。

## Relation to Mem0 and Zep/Graphiti

行文结构可借鉴两篇论文，但贡献不能复制：

- 与 Mem0 类似，论文从长期记忆的成本/质量矛盾切入，随后给出可部署系统和 LoCoMo 主结果。
- 与 Zep/Graphiti 类似，论文解释 graph/time/provenance 数据模型，并报告 LongMemEval 和检索诊断；不把 Zep 的原始架构简化成“图边直接充当事实”。
- Deep-Dream 的独立定位是 auditable source authority：事实不由抽取后的 memory node 单独授权，而由可回溯 source span 支持；concept/relation overlay 负责寻找和组织证据。当前实现不保证答案与提交证据之间的语义蕴含。

Mem0 论文的 rubric `J`、Mem0 当前 binary accuracy、Zep 的 GPT-4o/GPT-4o-mini accuracy 和本项目 Qwen/Kimi cross-judge 不会混排成一个严格 leaderboard。正式对外 SOTA claim 必须等待 controlled comparison。

## Distinctive experiment inventory

### 已有、可进入当前草稿

1. lexical vs semantic vs fused channel decomposition；
2. exact span vs neighboring-turn expansion；
3. depth/response-size budget frontier（当前两个点）；
4. judge prompt/model sensitivity；
5. LongMemEval update/temporal slice；
6. MEME deletion/cascade/absence 失败剖面；
7. deterministic retrieval vs forced Agent/thinking（64.99 vs 63.17 vs 61.33 token-F1）。

### 投稿前应完成

1. **Causal provenance/graph ablation**：source-only → hybrid source → neighbor expansion → auditable overlay → evidence-gate Agent；
2. **Update/delete microbenchmark**：current/as-of/ever/never + conflict/retraction，测 historical accuracy 与 deletion leakage；
3. **Controlled Mem0/Zep run**：固定 answerer、judge、prompt、temperature 和 evidence budget；
4. **Full evidence-token/latency curve**：1/3/5/10/20 与最终 QA accuracy；
5. **Cross-scope leakage/adversarial memory test**：验证 evidence submission 是否真正阻止越权来源。

## Figure and table inventory

| ID | Artifact | Status |
|---|---|---|
| Fig. 1 | Source-of-truth layer、auditable concept overlay、hybrid retrieval、scope/evidence gate 架构图 | 需生成 |
| Fig. 2 | channel ablation + neighbor expansion 双面板 | 可由 JSON 自动生成 |
| Table 1 | Mem0/Zep/Deep-Dream design comparison | 可生成；避免跨指标排名 |
| Table 2 | main benchmark results with protocol/judge columns | 可生成 |
| Table 3 | LongMemEval category results | 可生成 |
| Table 4 | MEME limitation analysis | 可生成 |

## Limitations

1. 当前主要结果使用 Qwen3.7/Kimi-K3 cross-judge，而 Mem0/Zep 官方结果使用不同 judge/backbone。
2. 缺少在同一 harness 中运行 Mem0、Graphiti/Zep 和 Deep-Dream 的严格 head-to-head。
3. 版本化 graph 的因果增益尚未通过完整 ablation 隔离。
4. MEME 删除任务为 0%，cascade 和 absence 也明显弱，说明 append-friendly provenance 尚未转化为可靠的 negative memory semantics。
5. 项目 query latency 包含多步 Agent，当前平均耗时高，不能把质量结果解读为效率优势。
6. LoCoMo/LongMemEval 是合成或半合成长期对话，尚不能覆盖生产环境的隐私、恶意记忆和长期漂移。

## Writing handoff

- 推荐 venue：ICLR，方法/系统实证论文，9 页主文（标题、摘要、浮动体均计入页预算；引用和附录另计）。
- 当前 draft 允许写：系统设计、完整 LoCoMo/LongMemEval cross-judge 结果、检索消融、预算分析和诚实 limitations。
- 当前 draft 不允许写：严格 SOTA、显著优于 Mem0/Zep、版本图的因果提升、生产级低延迟。
- 论文完成度定位：强 preprint draft；完成 causal ablation 与 controlled baseline 后再提升为 submission-ready empirical claim。
