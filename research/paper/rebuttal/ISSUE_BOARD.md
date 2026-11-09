# ISSUE_BOARD — ICLR 2027 Submission44989

评分速览: f1J5=2/conf3 (摆动潜力高) · mV1D=1/conf3 (摆动中等) · cB3k=1/conf4 (摆动中低, 但 Soundness=3 无异议)

图例 — issue_type: novelty/baseline_comparison/ablation/clarity/reproducibility/presentation/other
severity: critical/major/minor · stance: positive/swing/negative
response_mode: direct_clarification / grounded_evidence / nearest_work_delta / narrow_concession / future_work_boundary
status: open / answered / deferred / needs_user_input

---

## Reviewer f1J5 (Rating 2, Conf 3) — 单一关切型审稿人

### F1-C1 · 漏引 PlugMem, novelty 不清
- raw_anchor: "missing some more advanced approaches... PlugMem... the idea proposed there appears to be more comprehensive"
- type: novelty / severity: major / stance: swing
- response_mode: nearest_work_delta
- 核实: **属实**——references.bib 无 PlugMem。但 delta 极清晰且可正面打:
  - PlugMem (arXiv:2603.03296, UIUC+MSR): 前提是 "decision-relevant information is concentrated as **abstract knowledge** rather than raw experience"; 把 episodic memory 蒸馏成 **propositional + prescriptive** 知识图; 追求最高信息密度; 目标是 task-agnostic 可插拔、跨任务迁移。
  - Deep-Dream: 前设是**写入时无法预知哪些信息日后决策相关**→ 源文档不可变, 图条目只是"search handle 而非 replacement fact"; 追求证据保真/可审计/身份可修订 (Prop.1 非破坏性合并、版本投影 π_t)。
  - 两者是同一根"压缩-保真"轴的两端, 附录 Prop.1 的数据处理不等式 (σ=g(O_t) ⇒ I(Z;σ)≤I(Z;O_t)) 正是对蒸馏式设计的信息论陈述。
  - 基准也不重叠: PlugMem = 长程对话QA/多跳检索/web agent; Deep-Dream = MAB/LoCoMo/LongMemEval/MEME。
- status: answered (写作层) — 见 STRATEGY_PLAN T1

### F1-C2 · 未与"更接近的方法"对比, 有效性存疑
- raw_anchor: "its effectiveness may also be questionable since there's no comparison with more closely related approaches"
- type: baseline_comparison / severity: major / stance: swing
- 核实: **部分不实**——Table 2 与 Mem0/Zep/HippoRAG-2/A-MEM/Hindsight 的已发表结果对比存在 (附录还逐表复刻 Mem0/Zep 原表), 论文明示"只有 same-model MAB 对是受控比较"; 且 K3-substitution 行已用 Mem0 原版评分协议跑 Deep-Dream (95.19%, 双 judge 交叉)。缺的只是 PlugMem 一家。
- response_mode: grounded_evidence + (可选) 新对比
- status: needs_user_input (是否补跑 PlugMem 对比/或在 MAB 上补一个受控外部队)

---

## Reviewer mV1D (Rating 1, Conf 3, Presentation=1) — 要规格与算法型

### M-C1 · §3.2 图构建算法不明 (matching criteria / decision mechanisms)
- raw_anchor: "the specific algorithms, matching criteria, and decision mechanisms underlying these operations remain unclear"
- type: clarity+reproducibility / severity: critical / stance: swing
- 核实: **论文正文属实** (只有抽象 Align/Link 算子)。但补充材料**已有**: 逐字 prompt、run manifest、engine config diff、B1/B2 byte-identical 配置、runtime-code hash (app:configs + app:ledger)。
- response_mode: grounded_evidence (指向 supplement) + 承诺 revision 加入伪代码
- status: answered (材料在, 需写入 revision)

### M-C2 · 抽取/对齐错误的影响无直接评估
- raw_anchor: "Errors in these processes may introduce incorrect cross-document connections or fragment the history... difficult to assess their reliability"
- type: empirical_support / severity: major / stance: swing
- 核实: 间接证据有 (duplicate families 340→124 / 482→99; Prop.1 单向地址错误不污染已接受引用), 但 **vs ground-truth 的实体对齐评估没有** → 与 C-C3 重叠
- response_mode: grounded_evidence + GAP-D 新分析 (冻结工件上做, 无需新 LLM 调用, 便宜)
- status: needs_user_input

### M-C3 · §3.4 检索策略未规格化 (如何选通道/构造查询/选邻居/停止)
- raw_anchor: "does not adequately explain how the agent selects among these retrieval methods... determines when to stop retrieving"
- type: clarity / severity: critical / stance: swing
- 核实: 正文有形式化 (a_{k+1}=ρ(q,M_t,C_k,u_k) ∈ spans∪{stop}) 与三 read paths, 但 ρ 的实现细节在 supplement。伪代码可从代码导出。
- response_mode: direct_clarification + revision 补伪代码
- status: answered (写作层)

### M-C4 · "更像通用 LLM 工具流而非新检索算法"
- raw_anchor: "closer to a general LLM-driven tool-use workflow than to a clearly defined novel retrieval algorithm"
- type: novelty / severity: major / stance: swing
- 核实: **可以窄让步 + 重定向**: 论文本就声明贡献是表示层与契约 (finding/reading 分离、source gate、可修订身份), 不主张新检索算法。让步"检索策略本身是工具使用", 强调表示+形式化性质+实证阶梯 (33.3→40.0→71.0) 才是贡献。A-RAG 已引用并区分。
- response_mode: narrow_concession + 重定向
- status: answered

### M-Q3 · 在线图消融 (同工具同预算, 去掉图)
- raw_anchor: "an online ablation comparing the full Deep-Dream system against an otherwise identical agentic retrieval system without the graph"
- type: ablation / severity: critical / stance: swing
- 核实: **没有**。离线 X7 channel replay 是 null 结果 (等价界 ±2.4), 论文自己承认 "online...is exercised, not ablated"。这是 mV1D/cB3k 共同的最大证据缺口 → GAP-A
- response_mode: grounded_evidence (若跑出新结果) / narrow_concession (若不跑)
- status: needs_user_input (GAP-A: 本地可跑, MAB 子集 + K3)

### M-Q4 · 资源匹配对比 (同 calls/tokens, 简单迭代 RAG 能否打平)
- raw_anchor: "would a simpler iterative RAG system achieve comparable performance when given the same source access and computational resources"
- type: baseline_comparison / severity: critical / stance: swing
- 核实: **部分有**: 同模型阶梯 (fixed 33.3 / tool 40.0 / source 71.0) 已控 answerer; cost 图披露 source 路径 10.2 calls/328.6k tokens。但**预算匹配的裸迭代 RAG 基线没有** → GAP-B。注意: tool-reading 臂已是"迭代自适应检索", 差它在存储记录上导航而非裸语料。
- response_mode: grounded_evidence + GAP-B
- status: needs_user_input

---

## Reviewer cB3k (Rating 1, Conf 4, Soundness=3) — 证据标准最严型

### C-C1 · Novelty 未立: vs A-MEM/AriGraph/G-Memory/GAM/Zep; Zep 已保 Episodes+源链接+关系历史, delayed alignment + version access 增了什么
- raw_anchor: "Zep already preserves lossless Episodes... what new capabilities delayed alignment and version access add"
- type: novelty / severity: critical / stance: negative
- 核实: Zep/A-MEM 已引但 **AriGraph/G-Memory/GAM 未引** (属实); delta 表述散落 (§2 末段、§4.4 末段、Prop.1/π_t) 未收拢成对照。真实 delta:
  1. 候选族保留而非写入即并 (Prop.1 非破坏性 coarsening, 错误合并可拆回)
  2. 版本投影 π_t(M_{t+1})=M_t —— 状态可复原 (Zep 的 validity window 是边的属性, 不是全状态投影)
  3. 引用门 (Prop. one-sided): 地址层错误只能漏证不能换文
  4. 源 span 权威, 图条目=地址非事实
- response_mode: nearest_work_delta (Zep 逐点) + 补引 AriGraph/G-Memory/GAM
- status: answered (写作层)

### C-C2 · Table 1 只比自家管线; Table 2 协议异源 → 对 Zep/HippoRAG2/A-MEM/Hindsight 无可靠优势
- raw_anchor: "Table 2 combines results with different answer models, judges, evaluation protocols, and data subsets"
- type: baseline_comparison / severity: critical / stance: negative
- 核实: **指控本身论文已自认** (published rows 各自带 answerer/judge; "only the same-model MAB pair is a controlled comparison"; LongMemEval-S 上 Deep-Dream 84.6 < MemMachine 93.0/Hindsight 91.4 都如实列出)。回应 = 重排已有受控证据 + GAP-C 补一个真正同协议外部基线 (Mem0 OSS 最可行, LoCoMo 就绪)。
- status: needs_user_input (GAP-C)

### C-C3 · 组件效应未隔离: 源读取=+12×tokens; delayed alignment 与批处理捆绑; version links/provenance 无消融; 图消融 null; 实体合并无 ground-truth
- raw_anchor: "It therefore remains unclear which new mechanisms are effective... beyond engineering integration"
- type: ablation / severity: critical / stance: swing
- 核实: 全部属实且论文自己披露过。已有可引: 三臂对齐研究 (A/B1/B2, 配置 byte-identical), X7 null, 邻跳 replay 70.5→84.8→89.5 (注意与 X7 是两种不同设置, 勿混), cost 图。缺: GAP-A (在线图消融), GAP-D (对齐 vs ground-truth), version-links/provenance 单独消融 (GAP-E, 重)
- status: needs_user_input (GAP-A/D/E 优先级)

### C-C4 · Figure 1/3 是未编辑 AI 生成 SVG, 丑
- raw_anchor: "Figure 1 and Figure 3 appears to be an unedited AI-generated SVG"
- type: presentation / severity: minor / stance: swing
- 核实: Figure 1=fig0_motivation, Figure 3=fig4_agent_trace, 均为 SVG 转 PDF, 出自提交 d38e87b1 (有 .prompt.txt, AI 生成工作流产物)。
- response_mode: 承诺 revision 重绘。**注意: 按用户规矩, 图的架构出自刻意工作流, 重画方案需用户确认, 不得擅自改动。**
- status: answered (承诺层, 执行留相机-ready)
