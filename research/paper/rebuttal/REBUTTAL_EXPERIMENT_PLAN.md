# REBUTTAL_EXPERIMENT_PLAN — 讨论期补证据 (用户已批准: D + A + B)

规则约束: ICLR 2027 讨论期**仅文字回复**, 无 PDF 修订 → 所有新数只能以 official comment 文字+表格呈现, 写作层修复一律表述为 camera-ready 承诺, 并猛指已随投稿发布的 supplement。
时间预算: >1 周 (2026-11-09 批准)。

---

## GAP-D · 实体对齐 vs ground-truth 【先跑: 零 API 成本】

回应: M-C2 (mV1D), C-C3 (cB3k "entity merges are not evaluated against ground-truth labels")

**做什么**: 在冻结工件上, 把 Deep-Dream 的 concept-family 划分与 ground-truth 实体身份对账:
- 数据源: MAB 767 配对样本的 `entity2id` 指纹 (六次运行共享) + LongMemEval 数据集实体标注; Base 与 Align+ 两代冻结库快照都在 `.benchmark_runs/`。
- 指标 (两个方向都报, 诚实优先):
  - **过合并** (over-merge): 不同 ground-truth 实体被并入同一 family 的 pairwise 错误率
  - **碎片化** (fragmentation): 同一实体被拆成多个 family 的比率 (duplicate families 340→124 是它的代理量, 这次直接算真值)
  - 分 Base / Align+ 两个引擎报告, 正好构成"延迟对齐对对齐质量的影响"的直接评估
- 输出: 一张小表 (precision/recall of co-family decisions, 或 confusion counts), 直接贴进 rebuttal。

**成功判据**: 任何数字都可发表——错误率低则直接化解 M-C2; 错误率高则窄让步 + 指出 Prop.one-sided (地址错误不污染已接受引用) 的缓冲作用。关键是"有直接评估"这件事本身。

**成本**: 纯离线分析, 0 次 API 调用。工作量 = 定位工件 + 写对账脚本。

## GAP-A · 在线图消融 (full vs no-graph, 同 agent 同预算) 【核心新实验 1】

回应: M-Q3 (mV1D 点名设计), C-C3 (cB3k "graph navigation...separately ablate")

**做什么**: 在 MAB 配对样本上新增一个臂: 完整 source-grounded 读取 agent, 但**禁用 concept/relation 图通道**——agent 只保留 lexical (BM25) + semantic + chunk/file 工具, 其余 (K3 answerer、prompt、迭代与 token 预算、源访问) 与已冻结的 source 路径完全一致。
- 对照组不需要重跑: 复用已冻结的 source-reading 结果 (71.0@1074 / 70.8@614)。
- 汇报口径: Overall + 分能力 (AR/TTL/LRU/SF) + MCC/FC-MH, 配 paired bootstrap CI (沿用论文现有方法)。
- 账本纪律: 新臂必须走 manifest + runtime-code fingerprint 流程 (与 app:ledger 一致), 否则不可引用。

**成功判据**: full > no-graph 且差异超 CI (尤其 TTL/MCC/FC-MH 这些更新与多跳任务) → 图的在线导航作用成立, M-Q3 正面化解; 无差异 → 窄让步, 与 X7 离线 null 一致地呈现 (说明增益来自源读取而非图本身——这依然是有价值的科学结论, 且与我们 T1 表述兼容)。

**成本**: 767 题 × ~10 calls × K3 ≈ 一次完整 MAB source 跑批的量级; 需确认 service_config 端点可用。

## GAP-B · 预算匹配的裸迭代 RAG 基线 【核心新实验 2】

回应: M-Q4 (mV1D "would a simpler iterative RAG system achieve comparable performance when given the same source access and computational resources"), C-C2 侧面

**做什么**: MAB 配对样本上, 一个**无记忆层**的迭代 RAG agent: 工具 = 原始语料上的 BM25 + 语义检索 + 源文件打开 (同源访问), 同 K3 answerer, calls 上限与 token 预算与 source 路径对齐 (~10.2 calls / ~328.6k tokens)。
- 这与 memory-tool 臂的关键差异: memory-tool 在**存储记录**上导航, 本臂在**裸语料**上检索——正是 mV1D 要的 "simpler iterative RAG with same source access"。
- 汇报口径: 与 33.3 / 40.0 / 71.0 / 78.7 同表并列, 附预算列。

**成功判据**: 若裸 RAG ≪ 71.0 → "收益只是算力" 被数据反驳; 若接近 → 窄让步, 重心移到写入侧贡献 (对齐/版本/删除边界) —— 越早知道越好, 不进 camera-ready 才发现。

**成本**: 同 GAP-A 一个臂的量级。可与 GAP-A 并行排。

---

## 执行顺序与里程碑

1. **D (今天起)**: 定位 entity2id + 冻结库快照 → 写对账脚本 → 出表。
2. **A/B (D 之后并行)**: 先侦察 harness 的通道开关与 agent 轨道配置 (research/benchmark/), 确认禁用图通道与裸语料模式的最小改动路径; 每臂一次完整 MAB 跑批。
3. 所有新数入账本 (manifest/fingerprint), 供 rebuttal 引用与审计。

## 立场红线 (复述)

- X7 null、MEME 0%、LoCoMo 单跑指纹缺陷照旧披露, 不反口、不淡化。
- GAP-A/B 若结果不利, 如实窄让步——论文最大资产是 cB3k 点名表扬的可信度。
