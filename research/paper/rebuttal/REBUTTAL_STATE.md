# REBUTTAL_STATE — ICLR 2027 Submission44989

## 元数据
- 论文: Deep-Dream (research/paper/, 主文件 main.tex / main-zh.tex)
- Venue: ICLR 2027, OpenReview Submission44989
- 当前阶段: Phase 2-3 (拆解关切 + 策略)
- 评审结果: f1J5=2/conf3, mV1D=1/conf3, cB3k=1/conf4 (均分 ~1.33)
- 意见修改时间: 2026-11-06 20:34 (三人同时修改, 可能是 rebuttal 前的最后调整)
- Rebuttal 规则: 待确认 (ICLR 通常 text-only official comment, 无硬性字数上限但惯例 ~5000 字符/条)

## 阶段进度
- [x] Phase 0: 初始化 rebuttal/ 工作区
- [x] Phase 1: REVIEWS_RAW.md 存档 (三人逐字)
- [x] Phase 2: ISSUE_BOARD.md (15 个原子关切: F1×2, M×6, C×4+重叠)
- [x] Phase 3: STRATEGY_PLAN.md (4 全局主题 T1-T4, 9 个证据缺口 GAP-A..I)
- [x] Phase 3.5: 证据缺口已呈报用户 → **用户决策 (2026-11-09): 仅文字回复; 批准 GAP-D+A+B; 时间>1周** → REBUTTAL_EXPERIMENT_PLAN.md 已落地
- [x] GAP-D 对齐对账 **完成 (2026-11-09)**: 单射名称匹配法 (v2); 两代引擎 ground-truth 过合并均 0, 碎片化 2.3%→1.5%, 池覆盖 2.8% 如实报告; 见 REBUTTAL_EXPERIMENTS.md + gapd_results.json。v1 块主题法已证伪弃用 (MAB AR 块是文章中段段落, 维基首段摘要 avgJ=0.085)
- [x] GAP-A/B 代码就绪 **(2026-11-09)**: no-graph 工具集臂合二为一 (agentic.py/runner.py, 测试 41/41); K3 端点本机不可达 (TCP 超时) → 待评测机执行 RUNBOOK_GAP_AB.md (改动包 gapab_harness.patch)
- [ ] Phase 4: 起草 (可在等 no-graph 跑批期间先行 — GAP-D 数字已冻结可引用; no-graph 数字留占位)
- [ ] Phase 4: 起草 (规则已确认: text-only official comment; 写作层修复=camera-ready 承诺+指向 supplement)
- [ ] Phase 5: 安全校验
- [ ] Phase 6: 压力测试
- [ ] Phase 7: 定稿双版本
- [ ] Phase 8: 追加轮次

## 关键事实速查
- 论文分数构成: Presentation 3/1/2, Informativeness 2/2/2, Soundness 1/1/3
- 注意: cB3k (conf 4, 最有发言权) Soundness=3 无实质异议, 拒稿理由全在 novelty + 外部对比 + 消融 —— 都是 rebuttal 可翻的类型
- f1J5 唯一实质指控 = 漏引 PlugMem (ICML'26, openreview id NWKaQIKoGp)
- mV1D Presentation=1: 需要算法细节/伪代码级别的澄清
