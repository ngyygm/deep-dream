# Deep-Dream 论文方向筛选报告

## 研究任务

在 Mem0 的“问题—方法—大规模基准—效率分析”节奏和 Zep/Graphiti 的“系统架构—时序图—检索质量—延迟”节奏基础上，为 Deep-Dream 形成独立而可验证的论文主线。所有已完成结果与待补实验必须严格分开。

## 候选方向

### 1. Source-Grounded Auditable Memory（推荐）

**核心问题**：现有长期记忆系统常把抽取后的事实、摘要或图边当成事实本身；当抽取、合并或版本更新出错时，回答无法回到原始证据，也难以判断是否应该拒答。

**方法主线**：Deep-Dream 将文档/对话原文保持为 source of truth，在其上建立带稳定 identity、版本链和 provenance 的 concept/relation overlay；查询时用 lexical、semantic、graph/provenance 多通道生成候选，Agent 只能读取作用域内的原始 source span，并通过受约束的 evidence submission 接口提交证据后回答。

**已观察信号**：

- LoCoMo 1540：93.18%（Qwen3.7 judge）/ 93.57%（Kimi-K3 cross-judge）。
- LongMemEval-S full-500：84.6%，其中 knowledge-update 91.03%、temporal 84.21%。
- LoCoMo 1986 token-F1：deterministic baseline 64.99，高于强制 agent 63.17 和 thinking-on 61.33；这些轨道同时改变了检索、编排和回答方式，因此只能说明在该配置下增加 Agent/thinking 没有自动改善 token-F1。
- 210 题通道消融：lexical 的 recall-all 75.24%，semantic 70.48%，fused 78.10%。
- 105 个从既有错误中选出的诊断样本中，保持 lexical+semantic 候选不变并扩展 ±2 turns，recall-any 从 70.48% 到 89.52%，recall-all 从 52.38% 到 73.33%，payload 增加 16.84%；该切片不是 held-out test。

**新颖性判断**：相对 Mem0 的增量事实 CRUD 和 Mem0g/Zep 的图记忆，差异不应表述为“使用图”，而应表述为“图是可重建的导航层，原始来源和可审计 provenance 才是回答边界”。这一 claim 需要用 provenance completeness、update/conflict、false-premise/abstention 和 evidence-gate 实验支持。

**结论**：选择。现有结果最充足，且能形成不同于 Mem0/Zep 的系统立场。

### 2. Memory as an Auditable Agent Workspace

把贡献重点放在只读 memory tools、conversation scope isolation、fresh context、trajectory logging 和 evidence validation 上，强调 Agent 安全与可复现性。

**优点**：系统特色鲜明；已有 trajectory、scope validation 和 runtime hash。

**不足**：主 benchmark 的质量提升不能完全归因于 workspace 约束；需要安全攻击、cross-scope leakage 和 malicious-memory 实验，当前证据不足。

**结论**：作为主线 1 的系统与可信性子贡献，不单独成文。

### 3. Adaptive Retrieval over Source, Semantic, and Graph Channels

以多通道融合、context expansion、scope/evidence checks 和按需 Agent 为核心。

**优点**：已有 replay/ablation 数字，易做强实验。

**不足**：单纯 hybrid retrieval 的新颖性不足；应服务于 source-grounded auditable memory，而不是作为唯一贡献。

**结论**：作为主线 1 的检索机制与特色实验。

## AUTO_PROCEED 决策

`AUTO_PROCEED: selected Idea 1 — Source-Grounded Auditable Memory`

## 暂定论文定位

**Working title**: *Deep-Dream: Source-Grounded Long-Term Memory with Auditable Concept Overlays*

**一句话 claim**：Deep-Dream separates memory navigation from factual authority: concept and relation overlays guide retrieval, while submitted evidence IDs remain bound to source spans that the Agent actually read within the active scope. This mechanism does not by itself guarantee semantic entailment between evidence and answer.

## 贡献设计

1. **Source-grounded memory model**：原文/对话作为不可替代的事实层，concept、relation 和 summary 是可重建 overlay；所有派生节点保留 source span provenance。
2. **Versioned concept graph**：稳定 identity 与 observation/assertion 版本链分离，支持演化事实、冲突和跨窗口对齐，而不是静态覆盖。
3. **Evidence-bounded agent retrieval**：lexical、semantic、graph/provenance 多通道检索与邻域扩展；Agent 最终提交的 evidence ID 必须来自本题实际读取且通过 scope/active-state 校验。
4. **Broad and diagnostic evaluation**：LoCoMo、LongMemEval-S、LoCoMo-Plus、MEME；除最终 QA 外，报告 retrieval recall、预算曲线、cross-judge sensitivity、update/temporal、false-premise/absence 和失败归因。

## 不允许的主张

- 不把不同 judge 的 93.18%、93.57%、Mem0 92.5% 和 Zep 94.7% 混成严格 leaderboard。
- 不把 Mem0 论文 rubric `J` 与 binary accuracy 直接比较。
- 不把尚未运行的 DMR、BEAM、MemoryAgentBench 填入主结果。
- 不声称图或版本链本身造成全部性能提升，除非对应 ablation 完成。
