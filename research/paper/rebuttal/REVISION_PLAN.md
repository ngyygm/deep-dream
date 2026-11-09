# REVISION_PLAN — ICLR 2027 Submission44989 camera-ready 修订清单

- 论文: Deep-Dream (`research/paper/main.tex` / `main-zh.tex`)
- Venue: ICLR 2027 · text-only official comment（惯例 ~5000 字符/条，无硬性上限）
- Rebuttal 轮次: 初轮（讨论期），起草于 2026-11-09
- 上游: `ISSUE_BOARD.md`（12 个原子关切）· `STRATEGY_PLAN.md`（T1-T4 + GAP-A..I）
  · `REBUTTAL_DRAFT_v1.md` / `PASTE_READY.txt`
- 证据存档: `REBUTTAL_EXPERIMENTS.md`（GAP-D 已完成；GAP-A/B 评测机执行中）

规则：本清单是 rebuttal 中（显式或隐式）承诺的每一处论文修改的唯一台账。
承诺三级：`already_done`（证据已在手，只需写入）/ `approved_for_rebuttal`（用户已批）
/ `future_work_only`（只在 out-of-scope log 里，不进 checklist）。

---

## Overall Checklist

### §2 Related Work — 引用与定位（F1-C1 / C-C1）

- [ ] (F1-C1) references.bib 加 PlugMem（arXiv:2603.03296, ICML'26, openreview NWKaQIKoGp），正文 §2 引用 — commitment: `approved_for_rebuttal` — owner: author — status: pending — 依据: anti-hallucination 规则须过 DBLP/CrossRef 核对
- [ ] (F1-C1) §2 压缩-保真轴段落写入 PlugMem 前提引文（"decision-relevant information is concentrated as abstract knowledge…"）与 Prop.1 DPI 对置 — commitment: `approved_for_rebuttal` — status: pending
- [ ] (C-C1) 补引 AriGraph — commitment: `approved_for_rebuttal` — status: pending — 须 DBLP/CrossRef 核对
- [ ] (C-C1) 补引 G-Memory — commitment: `approved_for_rebuttal` — status: pending — 同上
- [ ] (C-C1) 补引 GAM — commitment: `approved_for_rebuttal` — status: pending — 同上
- [ ] (C-C1) §2 新增逐点 novelty delta 表，覆盖 Zep / A-MEM / AriGraph / G-Memory / GAM / PlugMem × 四个 delta（候选族保留非破坏合并 / π_t 全状态投影 vs 边级 validity window / 引用门单向错误 / 图条目=地址非事实）— commitment: `approved_for_rebuttal` — status: pending

### §3 方法 — 规格入正文（M-C1 / M-C3）

- [ ] (M-C1) §3.2 Align/Link 算子伪代码从代码导出并写入正文（材料已在 supplement）— commitment: `approved_for_rebuttal`（re-layout，非新工作）— status: pending
- [ ] (M-C1) §3.2 正文写明 matching thresholds：similarity 0.7、max-similar 10、relation-endpoint Jaccard/embedding 0.9/0.9（出处 = 两代冻结 engine config diff）— commitment: `approved_for_rebuttal` — status: pending
- [ ] (M-C3) §3.4 补 ρ 策略伪代码 + 停止条件实现（stop=submit_evidence、8 步上限、重复动作守卫、surfaced-evidence 门）— commitment: `approved_for_rebuttal` — status: pending

### §4 实验 + 附录 — 新证据写入（M-C2 / M-Q3 / M-Q4 / C-C2 / C-C3）

- [ ] (M-C2, C-C3) 附录加 GAP-D 合并对账：方法两行（单射断言、terminal family）+ 两表（池单射子集 637/601、名称层全量分桶）+ 诚实披露（池覆盖 2.8%/2.7%；歧义名 split 含正确消歧）— evidence: `already_done`（gapd_results.json 已冻结）— 论文写入: `approved_for_rebuttal` — status: pending
- [ ] (M-Q3, M-Q4, C-C3) §4 加 no-graph 臂结果：总体 + 分能力（AR/TTL/LRU/SF、MCC/FC-MH）+ 与 71.0/70.8 的配对 bootstrap CI + 预算列（calls/tokens 对照 10.2/328.6k）— commitment: `approved_for_rebuttal` — status: **pending（blocked on 评测机 no-graph-v1 跑批；无论结果如何必须如实写入，null 按 null 报）**
- [ ] (C-C2) 新增受控证据汇总表：阶梯 33.3→40.0→71.0（上界 78.7@614k）、三臂对齐研究、K3-substitution LoCoMo 95.19%、GAP-D、no-graph 臂 — commitment: `approved_for_rebuttal` — status: pending（依赖上一项完成）

### 图（C-C4）

- [ ] (C-C4) 重绘 Figure 1（fig0_motivation）— commitment: `approved_for_rebuttal`（承诺层已给；**执行方案须用户确认**，图架构出自 d38e87b1 刻意工作流，不得擅改）— status: pending
- [ ] (C-C4) 重绘 Figure 3（fig4_agent_trace，可对齐 supplement agent-trace 图风格）— commitment: `approved_for_rebuttal`（同上约束）— status: pending

---

## Grouped View

### 按论文位置

| 位置 | 条目 | issue |
|---|---|---|
| §2 Related Work | 6 条（PlugMem 引用+定位、AriGraph/G-Memory/GAM 补引、delta 表） | F1-C1, C-C1 |
| §3.2 构建规格 | 2 条（伪代码、阈值） | M-C1 |
| §3.4 检索规格 | 1 条（ρ 伪代码+停止条件） | M-C3 |
| §4/附录 实验 | 3 条（GAP-D、no-graph、受控证据汇总表） | M-C2, M-Q3, M-Q4, C-C2, C-C3 |
| 图 | 2 条（Fig 1、Fig 3） | C-C4 |

### 按严重度

- critical（M-C1/M-C3/M-Q3/M-Q4/C-C1/C-C2/C-C3 相关）：§3 规格入正文 3 条 + no-graph 结果 + delta 表 + 受控证据表 + GAP-D = 8 条
- major（F1-C1/F1-C2/M-C2）：PlugMem 引用与定位 2 条 + GAP-D 写入 = 3 条
- minor（C-C4）：图 2 条

### 中英同步

所有条目须同步落 `main.tex` 与 `main-zh.tex`（中文版用 xelatex，注意既有注意事项）。

---

## Commitment Summary

- `already_done`（证据在手，只欠写入）：1 条（GAP-D 分析；论文写入另计 approved）
- `approved_for_rebuttal`：14 条 checklist 全部（含上条的写入动作）
- `future_work_only`：0 条进清单（见下 log）
- `needs_user_input` 阻塞中：0 条（Figure 1/3 执行方案需用户确认，但承诺级已批准，不阻塞 rebuttal）

## Out-of-Scope Log（不触发论文修改，防静默消失）

| issue | 内容 | 不做的理由 |
|---|---|---|
| M-C4 | "ρ 是通用 LLM 工具流非新检索算法" | 窄让步：论文从未主张新检索算法，贡献在表示+契约+阶梯；rebuttal 文字让步即可，无需改论文 |
| F1-C2 | 与 PlugMem 的正面对比实验 | 用户批准范围仅 GAP-D+A+B；且双方基准零重叠（PlugMem=长程对话QA/多跳/web agent；DD=MAB/LoCoMo/LongMemEval/MEME），跑不成同协议对比 |
| C-C2 (GAP-C) | Mem0 OSS 同协议外部基线（LoCoMo 就绪） | 本轮未获批；rebuttal 以"受控证据重排 + K3-substitution 行"回应 |
| C-C3 (GAP-E) | version-links / provenance 单独消融 | 重实验，讨论期不可行；rebuttal 明言"deliberately concede"为诚实边界 |
| C-C3 (12× token 成本) | — | 论文已如实报 cost 图，rebuttal 维持披露，无需新编辑 |
| X7 null | — | 立场红线：如实保留 null 结果与"exercised, not ablated"原话；no-graph 臂是补证不是洗白 |
