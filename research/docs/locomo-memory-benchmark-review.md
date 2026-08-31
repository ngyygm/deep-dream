# LoCoMo 长期记忆系统：评测协议、公开结果与 Deep-Dream 对比基线

> 状态：论文调研基线  
> 最后核验：2026-07-21  
> 适用范围：LoCoMo QA，不包含 event summarization 与图像本体任务

## 1. 最重要的可比性结论

LoCoMo 公开结果至少存在两套不能直接混排的协议：

| Protocol | Questions | Categories | Primary metric | Typical answer/judge |
|---|---:|---|---|---|
| Original LoCoMo | 1,986 | single-hop, multi-hop, temporal, open-domain, adversarial | normalized token-F1 | 论文中的各类 reader；无统一 LLM judge |
| Mem0-style | 1,540 | 删除 446 道 adversarial，其余四类 | LLM-as-a-Judge accuracy，部分论文同时给 token-F1 | 通常固定 GPT-4o-mini answerer/judge，但不同工作仍可能改变 prompt、top-k 与 judge 次数 |

因此，Deep-Dream 当前 `0.6499` 是 **1,986 题 category-aware token-F1**；MemMachine 的 `0.9169`、Mem0 新版的 `0.925`、Zep 的 `0.947` 是不同 answerer/judge 或更新协议下的语义正确率，不能据数值直接声称优劣。论文主表必须先按协议分组。

原始数据的类别数为 single-hop 841、multi-hop 282、temporal 321、open-domain 96、adversarial 446。原始论文同时报告每段 conversation 平均约 600 turns、16K tokens，最多 32 sessions。[Maharana et al., ACL 2024](https://aclanthology.org/2024.acl-long.747/)

## 2. 原始 1,986 题 Token-F1

| Method | Answer model / reader | Overall token-F1 | Notes | Evidence class |
|---|---|---:|---|---|
| Human | human | 87.9 | 原始论文上界 | peer-reviewed |
| Full context | GPT-4 Turbo 128K | 51.6 | 全对话直接放入上下文 | peer-reviewed |
| Full context | Claude 3 Sonnet | 42.8 | 同上 | peer-reviewed |
| Full context | Gemini 1.0 Pro | 39.1 | 同上 | peer-reviewed |
| Dialog RAG top-25 | GPT-3.5 + DRAGON | 41.0 | 原始对话片段稠密检索 | peer-reviewed |
| Observation RAG top-5 | GPT-3.5 + DRAGON | 43.3 | 事实观察比长对话具有更高信噪比；继续增加 top-k 反而下降 | peer-reviewed |
| **Deep-Dream baseline** | **Qwen qwen3.6-27b-awq** | **64.99** | quality-v1 memory；thinking off；temperature 0 | local reproduced run |
| Deep-Dream skill-agent | Qwen qwen3.6-27b-awq | 63.17 | 最多 8 步，thinking off | local reproduced run |
| Deep-Dream skill-agent thinking-on | Qwen qwen3.6-27b-awq | 61.33 | Agent 决策开启 thinking | local reproduced run |

原始外部数字来自 [LoCoMo ACL 2024 论文](https://aclanthology.org/2024.acl-long.747.pdf)。Deep-Dream 与论文模型、prompt 和运行年代不同，只能作为同指标诊断，不构成严格 SOTA 结论。

### A-MEM 的同口径结果

[A-MEM](https://papers.neurips.cc/paper_files/paper/2025/file/19909c36f51abc4856b4560aff3d36d6-Paper-Conference.pdf) 把记忆组织为带关键词、标签、上下文描述和链接的原子 notes，并报告了完整五类 token-F1。下表的 overall 不是论文直接列出的 headline，而是使用 LoCoMo 官方类别数进行的加权计算：

| Method with GPT-4o-mini | Multi-hop | Temporal | Open-domain | Single-hop | Adversarial | Derived weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Full context | 25.02 | 18.41 | 12.04 | 40.36 | 69.23 | 39.75 |
| MemGPT | 26.65 | 25.52 | 9.15 | 41.04 | 43.29 | 35.45 |
| A-MEM | 27.02 | 45.85 | 12.14 | 44.65 | 50.03 | 41.98 |

这些派生值必须在论文中标为 `our weighted calculation from reported category scores`。

## 3. 1,540 题 LLM-Judge 系列

| System | Reported overall | Token-F1 if reported | Core design | Protocol/status |
|---|---:|---:|---|---|
| Mem0 | 66.88 | — | 增量事实抽取；相似记忆检索；ADD/UPDATE/DELETE/NOOP | 1,540；GPT-4o-mini judge；preprint |
| Mem0g | 68.44 | — | 实体关系图、时间戳、冲突边失效、dense + graph dual retrieval | 同上；preprint |
| MemOS | 73.31 | 44.42 | MemCube provenance/versioning、多视角记忆、混合检索和调度 | 1,540；GPT-4o-mini；preprint |
| ENGRAM | 77.55 | 21.08 | typed stores、query routing、紧凑证据聚合 | 1,540；GPT-4o-mini；ICLR 2026 |
| MemMachine | 87.47 | 22.00 | 保存完整 episode、sentence index、nucleus hit 周边扩展、rerank | 1,540；GPT-4o-mini；preprint |
| MemMachine Agent | 88.12 | 22.10 | direct / parallel / iterative 的按需路由，迭代有界 | 同上；preprint |
| MemMachine Agent | 91.69 | — | 相同 memory pipeline，answerer 改为 GPT-4.1-mini | vendor paper/preprint |
| Mem0 current | 92.5 | — | single-pass hierarchical extraction + multi-signal retrieval | 更新后的 vendor-reported 协议 |
| Zep current | 94.7 | — | temporal knowledge graph、多路候选、cross-encoder rerank、auto-search | GPT-5.4 reasoning-medium judge；vendor-reported |

主要来源：[Mem0 paper](https://arxiv.org/abs/2504.19413)、[MemOS paper](https://arxiv.org/abs/2507.03724)、[ENGRAM paper](https://arxiv.org/abs/2511.12960)、[MemMachine paper](https://arxiv.org/abs/2604.04853)、[Mem0 research page](https://mem0.ai/research)、[Zep research page](https://www.getzep.com/research/)。

LLM-Judge 与 token-F1 的排序可能显著不同。例如 ENGRAM 报告 judge 77.55，但 token-F1 只有 21.08；更长或同义改写的答案可能语义正确，却受到字面匹配惩罚。因此论文必须同时公开原始 hypothesis，并将两种指标分栏。

## 4. 方法层面对 Deep-Dream 的启示

| Design pattern | Evidence in prior work | Deep-Dream consequence |
|---|---|---|
| Raw evidence is ground truth | MemMachine 尽量减少有损 LLM extraction | concept/relation 只能导航，最终答案必须读取原始 turn |
| Retrieve fine, then expand context | MemMachine sentence hit + surrounding episode context | 先做 turn-level ranking，再扩展前后 turn，而不是直接返回整段 session |
| Hybrid candidate generation | Mem0g、MemOS、Zep | BM25、semantic、provenance、graph/relation 应统一召回后再排序 |
| Compact final evidence | ENGRAM 约 916 tokens；Mem0 约 1,764 tokens；LoCoMo observation top-5 最优 | 粗召回可以大，最终 prompt 应受显式 token budget 约束 |
| Temporal/version semantics | Mem0g、MemOS、Zep | event_time、版本和冲突关系必须进入排序与回答审计 |
| Agent only when useful | MemMachine 的 Agent 相对普通 memory 增益很小，部分设置还略降 | baseline 保持确定性；多步 Agent 应由 query router 按需启用，而非每题强制运行 |

Deep-Dream 当前 full-run 也支持最后一点：固定 baseline 为 64.99，8 步 skill-agent 为 63.17，thinking-on 为 61.33。当前优先级应是 turn-level retrieval、跨通道 reranking 和 evidence budgeting，而不是增加 Agent 步数。

## 5. Deep-Dream 当前可复现实验快照

- Run directory：`.benchmark_runs/locomo-full-quality-v1`
- Manifest schema：3
- Dataset SHA-256：`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- Recorded Git commit：`d9f299259fd06761305659f939494875f5ba177b`
- Scope/session：10 conversations / 272 active sessions
- QA：每条轨道 1,986 unique question IDs，0 runtime errors
- Remember profile：`quality-v1`
- Answer model：`qwen3.6-27b-awq`
- Embedding：`all-MiniLM-L6-v2`, CPU
- Answer configuration：thinking off，temperature 0，top-5 sessions

### Baseline category token-F1

| LoCoMo category | Count | F1 |
|---|---:|---:|
| multi-hop (1) | 282 | 35.87 |
| temporal (2) | 321 | 53.84 |
| open-domain (3) | 96 | 17.64 |
| single-hop (4) | 841 | 67.99 |
| adversarial (5) | 446 | 95.96 |
| **overall** | **1,986** | **64.99** |

去掉 adversarial 后，1,540 题的本地 token-F1 为 56.02；它仍不能与 1,540 题的 LLM-Judge accuracy 直接比较。

### Baseline retrieval

| Metric | @5 | @10 |
|---|---:|---:|
| Session Recall-any | 88.37 | 93.88 |
| Session Recall-all | 78.96 | 84.88 |
| Turn Recall-any | 59.43 | 70.66 |
| Turn Recall-all | 52.15 | 62.22 |

这说明主要改进空间集中在 turn ranking 和最终证据组合，而不是 conversation 隔离或问答运行稳定性。

## 6. 论文公平对比清单

任何主表比较必须同时固定并披露：

1. 题集是 1,986 还是 1,540，是否包含 adversarial。
2. category mapping、答案规范化和 token-F1 实现。
3. answerer 模型、版本、temperature、thinking/reasoning 参数。
4. judge 模型、prompt、重复次数及聚合方式。
5. candidate depth、最终 top-k、evidence token budget、reranker。
6. ingestion 是否使用未来信息，问题时点可见的 session 范围。
7. 是否保存完整原文，以及最终证据能否回溯到原始 turn。
8. 数据版本、SHA-256、代码 commit、错误数量和重试策略。

后续正式对外对比应保留两条并列轨道：原始 1,986 题 category-aware token-F1，以及复现 Mem0-style 的 1,540 题统一 answerer/judge；前者衡量字面与拒答兼容性，后者用于和近期 memory systems 对齐。

