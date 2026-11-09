# STRATEGY_PLAN — ICLR 2027 Submission44989

## 0. 总体形势判断

**分数 2/1/1 ≠ 方法被否。** 三人无一直击方法本身错误; cB3k (conf 4) 明确 Soundness=3 无实质异议, 拒稿理由全落在 (a) novelty 表述 (b) 受控对比 (c) 组件隔离——三者都可被 rebuttal + 讨论期新证据动摇。mV1D 的 Presentation=1 主要因为算法规格在 supplement 而正文过抽象, 是可修复的呈现问题。f1J5 只有一件事 (PlugMem), 而 PlugMem 与本文在"压缩-保真"轴上几乎正反对立, delta 极好写。

**最高 ROI 顺序: f1J5 (单点, 2→5/6 现实) > mV1D (要规格, 材料大半已有) > cB3k (要新实验, 中低概率但 conf 4 权重最大)。**

## 1. 全局主题 (opener 用, 2-4 条)

### T1 · 地址层 vs 蒸馏层: 一根轴上的两端
统一回应 F1-C1 (PlugMem)、C-C1 (Zep 等)、M-C4 (工具流):
- Mem0/Zep/A-MEM/PlugMem 的记忆对象是**提取物** (笔记/边/命题知识), PlugMem 明言 "abstract knowledge rather than raw experience" 并以信息密度最大化为目标;
- Deep-Dream 的记忆对象是**不可变源 span + 可修订地址**: 图条目是 search handle 而非 replacement fact; 写入时不预判哪些信息日后相关 (§1 动机), 所以不丢任何一份未来可能需要的证据。
- 形式化差异是可检查的: Prop.1 非破坏性合并 (错误身份可拆回)、π_t 状态投影 (历史图配置可复原)、引用门 (地址错误只能漏证、不能换文)。Zep 的 validity interval 是边属性, 不提供这三条。
- 窄让步给 mV1D/cB3k: 在线读策略确实是 LLM 工具使用——贡献不在"新检索算法", 在表示与契约, 以及它解锁的实证分离: 同一 answerer 下 fixed 33.3 → tool 40.0 → source 71.0 (full-context 上界 78.7@614), 这是文献里少有的把"读什么"从"记什么"里剥出来的受控阶梯。

### T2 · 受控证据在能控的地方已经存在
回应 C-C2/M-Q4 的已有部分: Table 2 的跨系统行自带 answerer/judge 标注, 论文从未主张异源行是受控对比; 真正受控的是 (i) 同模型三路径阶梯, (ii) 三臂对齐研究 (byte-identical 配置), (iii) K3-substitution 行 (Mem0 原版双 judge 协议跑 Deep-Dream)。Appendix 还逐字复刻了 Mem0/Zep 原表供审计。

### T3 · 规格在 supplement, revision 将吸收进正文
回应 M-C1/M-C3: 逐字 prompt、manifest、config diff、runtime hash 已随包发布且自校验; 讨论期将补 Align/Link 伪代码与读策略 ρ 的完整规格进 revision (材料已有, 纯写作)。

### T4 · 讨论期新证据 (取决于用户决策, 见 §3)
在线图消融 (GAP-A) + 预算匹配迭代 RAG (GAP-B) + Mem0 同协议重跑 (GAP-C) + 对齐 vs ground-truth (GAP-D)。

## 2. 每审稿人回复骨架 (占全文字符预算 75-80%)

**f1J5** (预算最省, 潜在收益最大):
1. 感谢+确认 PlugMem 相关, 承认应引用并将在 revision 加入。
2. 轴定位: PlugMem 蒸馏命题/规约知识图、优化信息密度与跨任务插拔; Deep-Dream 保源文本权威、优化证据保真与身份可修订。引用 PlugMem 摘要原文 "abstract knowledge rather than raw experience" 对照本文 "compresses the search description, not the only copy of the fact" (§3.5)。
3. 信息论衔接: 附录 Prop.1 的数据处理不等式正是对蒸馏式记忆的一般性陈述; PlugMem 的"信息密度"目标与本文"保留召回上界"目标可显式对偶。
4. (可选, 若 GAP-H 跑) 受控对比一行。
→ 预期: novelty 单点被解除后, 2→5/6。

**mV1D**:
1. 规格定位: supplement 已含逐字 prompt/config/hash (app:configs, app:ledger), revision 将上伪代码; 在回复里直接给 Align/Link 决策规则摘要 (从代码/manifest 提炼, 引用具体 commit)。
2. 检索策略: 给 ρ 的通道选择启发式 + 停止准则描述 + 指向 agent-trace 图的真实调用序列。
3. 窄让步 M-C4: 同意"策略=工具使用", 重定向到表示贡献 + 受控阶梯 (T1 末条)。
4. M-Q3/M-Q4: 要么报 GAP-A/B 新数, 要么窄让步 + 指出 X7 null 与邻跳 replay 的区别 (X7 是离线单通道替换; 邻跳扩展 70.5→89.5 说明图邻居在扩展式使用中有增益——诚实呈现两者)。

**cB3k** (最难, 最长预算):
1. Zep 逐点 delta (T1 第3条) + 承认 AriGraph/G-Memory/GAM 漏引, revision 补引+对照表。
2. Table 2: 重申自认的协议异源 + 指向已有的三处受控证据 (T2); 若 GAP-C 跑了 Mem0 同协议, 直接给数。
3. C-C3: 列已有隔离证据 (三臂、X7、邻跳、cost); GAP-A/D 新数; version-links/provenance 消融若不跑 → future_work_boundary (诚实划界: 12× token 的代价已被 cost 图如实报告, 收益-代价权衡在 §5 已讨论)。
4. C-C4 图: 承诺重绘 (执行方案需用户确认, 见决策点 5)。

## 3. 证据缺口与成本估计 (AUTO_EXPERIMENT=false → 全部呈报用户)

| GAP | 回应 | 内容 | 成本估计 | 优先级 |
|---|---|---|---|---|
| A 在线图消融 | M-Q3, C-C3 | 同 agent/工具/预算, 禁用 concept/relation 通道, MAB 767 或其子集 + K3 | 本地可跑; 算力=K3 API 调用 × 若干臂 | ★★★ (两人点名) |
| B 预算匹配迭代 RAG | M-Q4 | 裸 BM25+语义迭代 RAG agent, 匹配 calls/tokens 预算, 同源访问 | 同上 | ★★★ |
| C Mem0 同协议重跑 | C-C2 | Mem0 OSS 在 LoCoMo 上, 同 K3 answerer/judge | 中 (外部系统接入) | ★★ |
| D 对齐 vs ground-truth | M-C2, C-C3 | 用 entity2id 指纹 + 数据集实体标签评 merge 质量 | **纯离线分析, 冻结工件上做, 无新调用** | ★★★ (最便宜) |
| E version/provenance 消融 | C-C3 | 各自单独关掉 | 重 | ★ |
| F 伪代码/规格进正文 | M-C1/C3 | 从代码提炼 Align/Link/ρ 伪代码 | 写作 | ★★★ (零实验) |
| G 补引+对照表 | C-C1 | AriGraph/G-Memory/GAM/PlugMem + delta 表 | 写作 | ★★★ |
| H PlugMem 对比 | F1-C2 | 其代码开源 (TIMAN-group/PlugMem), 但基准不重叠, 需适配 | 重 | ★ |
| I 图 1/3 重绘 | C-C4 | **涉及既定工作流, 需用户确认方案** | 设计 | ★★ |

## 4. 阻塞项 (provenance/commitment gate)

- GAP A/B/C/D/H 是否执行、各自预算上限 → **用户**
- ICLR 2027 讨论期是否允许修订 PDF (review 头部出现 "Revisions" 标记, 需确认); 回复字数上限 → **用户**
- 图重绘方案 → **用户** (涉及 d38e87b1 提交的刻意工作流)
- 立场红线: X7 null、MEME 0%、LoCoMo 单跑/指纹缺陷这些已披露事实**不得**在 rebuttal 里淡化或反口, 只能补充解释与新证据——论文的可信度是其最大资产 (cB3k 的 strengths 明确表扬了诚实披露)。

## 5. 字符预算草案 (待确认上限后细化)

opener 10-15% · f1J5 15% · mV1D 30% · cB3k 35% · closing 5-10%
