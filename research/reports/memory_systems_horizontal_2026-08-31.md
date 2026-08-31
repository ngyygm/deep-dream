# Deep-Dream vs 外部记忆系统横向对比（2026-08-31）

> 目的：把四基准收官成绩放到公开生态里定位。内部纵向（v1→v2 配对）结论见
> `mab_v1_vs_v2_2026-08-30.md` 与 `four_benchmarks_summary_2026-08-29.md`；本文只做横向。
> 我们的成绩 = kimi-k3（sz-infer 端点）单一模型承担记忆构建 + 问答 + 判分。

## 〇、我们的收官快照

| 基准 | 轨道 | 成绩 | run |
|---|---|---:|---|
| MemoryAgentBench（sampled 767q/10 scopes） | pi（v2 引擎） | **0.6511**（TTL 0.46 / FC-MH 0.70） | `memoryagentbench-kimik3-sample-v2` |
| LongMemEval-S（全量，1176 docs/25 scopes） | pi（v2 引擎） | **0.926** | `longmemeval-kimik3-full-v2` |
| BigCodeBench / ALFWorld | — | 0.486 / 0.979 | 与记忆无关，不作横向 |

## 一、MemoryAgentBench 横向（主表）

对照官方论文 Table 2（ICLR 2026，arXiv:2507.05257v2；与我们用同一 scorer 仓库 commit 455306d，% 值）：

| 系统 | AR Avg | TTL Avg | LRU Avg | SF Avg | **Overall** |
|---|---:|---:|---:|---:|---:|
| **Deep-Dream v2 pi（kimi-k3）** | **92.8** | 46.0 | 41.7 | **80.0** | **65.1** |
| Deep-Dream v1 pi | 94.0 | 43.0 | 52.3 | 78.5 | 67.0* |
| GPT-4o（全文） | 58.1 | 50.0 | 54.9 | 32.5 | 48.8 |
| Claude-3.7-Sonnet（全文） | 59.7 | **53.9** | **62.2** | 22.5 | 49.6 |
| GPT-4.1-mini（长上下文） | 71.8 | 46.2 | 49.1 | 20.5 | 46.9 |
| Gemini-2.0-Flash | 65.1 | 46.4 | 41.6 | 16.5 | 42.4 |
| BM25（简单 RAG） | 60.5 | 44.5 | 35.6 | 25.5 | 41.5 |
| HippoRAG-v2 | 65.1 | 35.8 | 36.2 | 29.5 | 41.6 |
| MIRIX（4.1-mini） | 63.0 | 35.7 | 40.5 | 11.5 | 37.7 |
| MemGPT | 34.3 | 40.8 | 22.4 | 15.5 | 28.3 |
| Zep | 37.5 | 37.5 | 16.2 | 5.0 | 24.0 |
| Mem0 | 32.6 | 21.2 | 20.7 | 10.0 | 21.1 |
| Cognee | 28.3 | 22.8 | 16.0 | 15.5 | 20.6 |

\* v1 的 67.0 含 LRU Summ 单题运气分（0.212），v2 剔除该域后反超（详见配对报告）。

**读法**：Overall 65.1 领先表内最佳（Claude-3.7 全文 49.6）**15.5pp**，领先最佳记忆系统（MIRIX 37.7）**27pp+**。官方论文自己的结论——"商用记忆系统（Mem0/MemGPT）在大多数任务上表现糟糕、全文方案在 TTL 上碾压记忆系统"——在我们的数字里得到同样印证，但我们在其余三域的优势足以反超全文方案。

## 二、确定性口径逐任务对比（无 LLM judge，最硬的数字）

EventQA / DetQA / FC / MCC 用 SubEM 或 accuracy 判分，不受 judge 模型影响：

| 任务 | Deep-Dream v2 | 官方表最佳 | 差距 |
|---|---:|---:|---|
| FC-MH（多跳事实合并） | **70** | 7（Contriever/MemoRAG） | **+63pp，一个数量级** |
| FC-SH（单跳事实合并） | **90** | 60（GPT-4o） | +30pp |
| EventQA | **98** | 82.6（GPT-4.1-mini） | +15pp |
| DetQA | **83.3** | 77.5（GPT-4o） | +6pp |
| MCC（TTL 域，icl_clinic150） | 46 | CLINC 96（GPT-4o）† | **-50pp，最大短腿** |

† 我们的 MCC 任务是 icl_clinic150（数据集新版任务），官方 Table 6 逐任务为 BANKING/CLINC/NLU/TRECC/TRECF，无直接同任务对照；方向性可比——全文模型 70–98，记忆系统 5–89（MemGPT 在 CLINC/BANKING 达 83–89，靠分页保留原始历史）。

FC（事实整合/版本化）与 EventQA 的领先是设计核心（簇收敛 + 原子事实抽取 + 图检索）直接兑现的地方。

## 三、口径差异与诚实调整（必读）

1. **采样子集**：我们是 767 题/10 scopes（全量 3671 题/29 任务）。TTL 域只含 MCC 无 Recom、AR 域无 MH-QA，而这两个任务普遍低分（Recom 全场 8–18、MH-QA 43–75）。补齐估算：TTL→(46+13)/2≈29.5，AR→(92+60+88.3+98)/4≈84.6，调整后 Overall ≈ **59–61，仍领先官方表最佳约 10pp**。结论方向不变，幅度缩水。
2. **actor/judge 模型不同**：官方行用 GPT-4o/Claude/Gemini 当 agent（LME 域 GPT-4o 判分）；我们 kimi-k3 自演自判。判分域（LME(S*) 88.3 vs 他们最佳 55.7）可能含 judge 偏差；上表第二节的无 judge 域不受此影响。
3. **我们的 baseline 轨**（0.324）也高于多数记忆系统，但那是"无 agentic 检索"的退化路径，横向一律用 pi 轨。

## 四、LongMemEval 生态横向

我们的 LME-S 全量 pi **0.926**。公开生态（各家自测、口径互不服）：

| 系统/方案 | LongMemEval 成绩 | 来源与可信度 |
|---|---:|---|
| OMEGA（本地记忆） | 95.4（GPT-4.1） | 厂商自宣，待独立复现 |
| Mem0 | 93.4（GPT-4o） | 厂商自宣；被 Zep 公开质疑用全文历史抬高分数，独立复现 49–68 |
| **Deep-Dream pi（kimi-k3）** | **92.6** | 自测 + 自判（kimi judge） |
| Zep / Graphiti | 71.2（GPT-4o） | 厂商自宣，第三方用 4o-mini 复现出同值 |
| 全文 GPT-4o | ~60 | Zep 复测 |
| 全文 GPT-4.1（1M 窗口） | 56.7 | Zep 复测（"overselling long-context"） |

该领域没有中立榜，厂商分数互相打架（Letta 另以"文件系统即记忆"在 LoCoMo 拿 74% 加入战团）。我们 92.6 的定位：**第一梯队，与头部厂商自宣值同档；优势是来自严格 v1→v2 配对实验的内生数字，劣势是含自判 judge 因素**。对外表述建议用"LongMemEval-S 0.926（kimi-k3，自建评测管线）"，不做"超过 Mem0"式声明。

## 五、结论定位

1. **画像**：检索（AR）与事实合并（SF）两个域对外部系统是数量级领先（FC-MH 70 vs 7）；Overall 在披露口径调整后仍领先已发表最佳 ~10pp。
2. **短腿**：TTL（测试时学习）46，落后全文方案 30–50pp——与官方论文"所有记忆系统都追不上 full-context TTL"的结论一致。全文方案赢在把历史样本原样留在上下文里；记忆系统赢在成本。我们的改进抓手已定位（窗口内原子事实抽取密度，见配对报告第五节）。
3. **效率轴**（外部系统少报，仅 Mem0 论文口径）：我们 v2 calls/doc 329 / tokens 1.19M（MAB，45-doc 交集）。与全文方案比是天然优势（MAB 论文自己强调 agent 方案的总计算负载更低）。
4. BCB/ALFWorld 是 agent 载体基准，与记忆系统不可横向，维持回归锚点定位。

## 来源

- MemoryAgentBench 论文（ICLR 2026）：[arXiv:2507.05257](https://arxiv.org/abs/2507.05257)（Table 2 总表、Table 6 逐任务 MCC）；官方仓库 [HUST-AI-HYZ/MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
- LongMemEval：[arXiv:2410.10813](https://arxiv.org/abs/2410.10813)
- Zep：[State of the Art in Agent Memory](https://blog.getzep.com/state-of-the-art-agent-memory/)、[Is Mem0 Really SOTA?](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)、[GPT-4.1/o4-mini long-context 复测](https://blog.getzep.com/gpt-4-1-and-o4-mini-is-openai-overselling-long-context/)
- Mem0：[Zep vs Mem0](https://mem0.ai/blog/zep-vs-mem0-which-ai-memory-layer-should-you-choose)、[State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- Letta：[Benchmarking AI Agent Memory](https://www.letta.com/blog/benchmarking-ai-agent-memory/)（LoCoMo 74% 文件系统方案，战团背景）
- 我们的成绩文件：`memoryagentbench_summary.*.kimik3-official-v1.sampled.json`（两 run 目录）、`four_benchmarks_summary_2026-08-29.md`
