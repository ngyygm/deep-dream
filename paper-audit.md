# Deep-Dream 论文严格审查报告

> **结论：不建议以当前版本直接投稿。** 未发现可证实的虚构参考文献，但发现明确的页数违规、评分协议版本混淆、若干与表格相反的解释，以及核心来源约束在当前代码中的可复现反例。高 QA 分数尚不能证明概念图层、证据门禁和版本机制的科学收益。
>
> **这不是数据造假或 AI 作者身份鉴定。** 文献真实、数值转录正确、文内算术自洽、历史实验真实、结论得到支持，是五个不同问题；本报告分别判断。

## 1. 审查范围、口径与总览

- 审查日期：**2026-09-21**。
- 仓库：`/workspace/exp/iclr/deep-dream`；分支 `main`；实测 HEAD：`66af0642c2ca6487b47d0f445674a84069faa2fd`。
- 用户明确选择按 **ICLR 2026** 审查，不改为 2027。发现原始运行材料缺失后，用户明确选择**仅审查当前可见材料**。
- 论文入口：`research/paper/main.tex`；通读全部 7 个章节输入、8 个表格、图表生成器、引用库、实际 PDF；追踪相关 `core/` 和 `research/benchmark/` 实现。
- 调用了 **8 个独立、只读、fresh-context 子 agent**：引用身份、官方政策、实验取证、实现契约、严格科学审稿、外部基线原文、PDF/模板、新颖性。主审合并、排除重复，并独立复核关键原文、算术、模板及代码反例。
- 只新增本报告；**没有修论文、改系统、重跑付费模型实验、git commit 或 push**。编译在 `st-jump-dev` 的隔离 `/tmp` 目录进行，不在本机安装重型 TeX 环境。

### 1.1 覆盖与核验结果

| 检查 | 实际覆盖 | 结论边界 |
|---|---|---|
| 引用身份 | 8/8 文献；42 个作者位置；22 个 citation-key 实例、8 个唯一 key | DeepXiv、Semantic Scholar、一手 arXiv/ACL/会议页交叉核验；没有未定义 key、没有已证虚构文献 |
| Google Scholar | 实际请求后 HTTP 429，未取得结果 | **未声称完成 Google Scholar 结果交叉确认**；Semantic Scholar 不是 Google Scholar |
| 外部数值转录 | Mem0/Zep 五张表，156/156 个外部数值单元格与所引预印本一致 | 不包括 Deep-Dream 自身数值；转录正确不等于协议一致或原作者实验已复现 |
| PDF | 13/13 页渲染检查；字体、页面、引用、表宽、元数据 | 正文延伸至 p11；现有 PDF 与源码没有可证的普遍陈旧问题 |
| 模板 | `.sty`、`.bst` 与 ICLR 官方 ZIP 逐字节一致 | 没有私改这两个模板文件；不意味着最终排版合规 |
| 洁净编译 | 隔离 Tectonic 0.17.0，去掉旧 aux/bbl 等后构建成功 | 仍 13 页，存在 overfull 警告；不等于原 pdfTeX 环境逐字节复现 |
| 文内实验算术 | 主要正确数、分类数、净增量、交叉裁判四格表、外部重加权 | 多数自洽；有明确叙述/算术错误，详见第 5 节 |
| 当前代码契约 | 主审复跑 12 个轻量行为探针，另复跑 3 个汇总/验证反例 | “探针通过”指成功重现报告的行为，**不是系统安全通过** |
| 历史逐题实验 | 原始 JSON/JSONL、数据集、冻结运行时和 DB 当前缺失 | 不能独立认证历史主分数、失败比例、实际模型调用或当时是否命中当前漏洞 |

### 1.2 严重性说明

- **P0：** 当前目标的硬性阻断，如期限与正文页限。
- **P1：** 实质影响主要论证、实验可解释性或核心实现保证，投稿前应解决。
- **P2：** 明确局部错误、重要报告缺口或格式风险。
- **建议：** 研究设计/呈现选择，不冒充 ICLR 强制条款。

| 优先级 | 主要问题 | 证据类型 | 最小行动 |
|---|---|---|---|
| P0 | 当前日期不能正常新投 ICLR 2026 | 官方时间事实 | 明确历史审查/后续稿身份，不倒填年份 |
| P0 | 正文到 p11，初投上限 9 页 | 实际 PDF＋洁净构建 | 删重复，外部背景表移附录；不缩字体/边距 |
| P1 | 2026 新 judge 被称为 2025 原论文 exact protocol | 两份固定版本一手提示词 | 分开协议；撤回“仅替换模型” |
| P1 | Mem0 重加权分数被当作已发表总体基线 | 原论文 Tables 1/2＋算术 | 并列原文 Overall J 与本稿重加权值，解释未知口径 |
| P1 | 主分数缺可复核原始证据；K3 历史指纹不完整 | 当前文件状态＋作者自述 | 发布逐题证据；诊断结果不冒充正式主结果 |
| P1 | 当前证据提交/发布边界有可复现绕过 | 实际源码＋合成反例 | 修共享验证边界，冻结后重新验证；历史影响另查 |
| P1 | overlay/gate 的主要科学收益仍未测出 | X7 表及正文承认 pending | 强源检索对照＋gate 因果实验 |
| P1 | 高一致率排除宽松 judge、共同困难类别等推论不成立 | 逻辑反证＋原始表 | 删除排除性因果语言，做独立标注 |
| P2 | X7 题数、变化“小于两点”、列反置、缺预算表/CI | 文表直接比对 | 修数值解释及生成器，补实际结果展示 |
| P2 | 漏引、双段摘要、表越界、图字过小、页眉缺失 | 原文/模板/PDF | 小范围修正并重编复查 |

## 2. 引用是否存在幻觉

### 2.1 逐条身份核验台账

下面的“通过”仅指身份核验，不代表该文献支持 Deep-Dream 的所有推论。8/8 条均获得 DeepXiv brief、Semantic Scholar 最终 HTTP 200，以及一手身份记录。姓名缩写/数据库作者归并以原论文为准。

| BibTeX key／本地位置 | 正确标题与链接 | 核验结论与年份处理 |
|---|---|---|
| `maharana2024locomo`，`references.bib:1–9` | [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://aclanthology.org/2024.acl-long.747/) | 通过；ACL 2024，pp. 13851–13870；DOI `10.18653/v1/2024.acl-long.747`；arXiv [2402.17753](https://arxiv.org/abs/2402.17753)，2024-02-27 |
| `chhikara2025mem0`，`:11–17` | [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) | 通过；arXiv v1 2025-04-28。另有 ECAI 2025 正式记录，DOI `10.3233/FAIA251160`；保留明确的预印本引用并非错误 |
| `rasmussen2025zep`，`:19–25` | [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956) | 通过；arXiv v1 2025-01-20；本轮未另核得正式出版 DOI |
| `wu2025longmemeval`，`:27–33` | [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) | 通过；首版 2024-10-14，正式 [ICLR 2025](https://iclr.cc/virtual/2025/poster/28290)。**2025 会议年正确**，不能因 arXiv/S2 首年为 2024 判错 |
| `jung2026meme`，`:35–41` | [MEME: Multi-entity & Evolving Memory Evaluation](https://arxiv.org/abs/2605.12477) | 通过；v1 2026-05-12。论文真实，但晚于 ICLR 2026 主会，须区分当前稿与历史提交稿 |
| `hu2025memoryagentbench`，`:43–49` | [Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions](https://arxiv.org/abs/2507.05257) | 通过；v1 2025-07-07；正式 [ICLR 2026](https://iclr.cc/virtual/2026/poster/10010781)。2025 预印本引用合法，但要固定版本 |
| `tavakoli2026beam`，`:51–57` | [Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs](https://arxiv.org/abs/2510.27246) | 通过；v1 2025-10-31、v2 2026-02-21；正式 [ICLR 2026](https://iclr.cc/virtual/2026/poster/10006595)。当前“arXiv载体＋2026年＋无版本URL”含混，**不是虚构年份** |
| `xu2025amem`，`:59–65` | [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110) | 通过；v1 2025-02-17；正式 [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/119020)。会议字段建议规范化 |

作者顺序已核对如下，合计 **42 个作者位置**，未发现错配：

- LoCoMo：Adyasha Maharana；Dong-Ho Lee；Sergey Tulyakov；Mohit Bansal；Francesco Barbieri；Yuwei Fang。
- Mem0：Prateek Chhikara；Dev Khant；Saket Aryan；Taranjeet Singh；Deshraj Yadav。
- Zep：Preston Rasmussen；Pavlo Paliychuk；Travis Beauvais；Jack Ryan；Daniel Chalef。
- LongMemEval：Di Wu；Hongwei Wang；Wenhao Yu；Yuwei Zhang；Kai-Wei Chang；Dong Yu。
- MEME：Seokwon Jung；Alexander Rubinstein；Arnas Uselis；Sangdoo Yun；Seong Joon Oh。
- MemoryAgentBench：Yuanzhe Hu；Yu Wang；Julian McAuley。
- BEAM：Mohammad Tavakoli；Alireza Salemi；Carrie Ye；Mohamed Abdalla；Hamed Zamani；J. Ross Mitchell。
- A-MEM：Wujiang Xu；Zujie Liang；Kai Mei；Hang Gao；Juntao Tan；Yongfeng Zhang。

**版本不可混用：** LongMemEval/A-MEM 宜用 `@inproceedings` 和 `booktitle`。BEAM 应明确引 2025 v1、2026 v2，还是 ICLR 2026 会议版。MemoryAgentBench 当前最新版使用 selective forgetting；截止前 v1 的对应术语为 conflict resolution，不能把最新全文当作 2025 v1。LoCoMo arXiv 与 ACL 摘要中的平均长度/会话规模也有版本变化。不要仅机械对齐年份。

### 2.2 引用内容与归属的实质问题

1. **LoCoMo-Plus 实际用于主实验却漏引。** `sections/4_experiments.tex:5,13`、`TABLE_main_results.tex:16` 报告 401 题，但 bib 没有其来源。正确来源：[Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents](https://aclanthology.org/2026.acl-long.1150/)，Yifei Li 等，ACL 2026，pp. 25085–25100，DOI `10.18653/v1/2026.acl-long.1150`；[arXiv:2602.10715](https://arxiv.org/abs/2602.10715)，v1 2026-02-11。代码：[xjtuleeyf/Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus)。它不是原 LoCoMo 的别名，需写清 cues、过滤、数据和提示词版本。
2. **MemoryBank、MemGPT 直接归属缺引。** `sections/2_related_work.tex:7` 提及两者但没有对应条目。补 [MemoryBank](https://arxiv.org/abs/2305.10250)（Wanjun Zhong、Lianghong Guo、Qiqi Gao、He Ye、Yanlin Wang；AAAI 2024，DOI `10.1609/aaai.v38i17.29946`）和 [MemGPT](https://arxiv.org/abs/2310.08560)（Charles Packer、Sarah Wooders、Kevin Lin、Vivian Fang、Shishir G. Patil、Ion Stoica、Joseph E. Gonzalez；2023 预印本）。DMR 的起源也不应只通过后来的 Zep 间接归属。
3. **真实引用不支持错误解释。** Mem0 open-domain、Zep preference、“exact protocol”等关键错误见第 4 节。不是补一个引用就能修复。
4. **相关工作遗漏了对手已有的 source provenance。** Zep 原文已明确 raw episodes、双向来源索引和 citation/quotation；A-MEM 保存 original interaction content。只把它们概括为压缩事实，会夸大本文区别，详见第 8 节。
5. **缩略词大小写保护。** 现有 `.bbl` 有 “Meme”“Longmemeval”“A-mem”“llm”。使用 `{MEME}`、`{LongMemEval}`、`{A-MEM}`、`{LLM}` 等保护专名。属于格式问题，不是虚假引用。

### 2.3 检索方法与未成功通道

实际命令包括 `deepxiv search 'LoCoMo Plus' --limit 3 --format json`、`deepxiv paper <id> --brief`（8 条原引用＋LoCoMo-Plus），以及 Semantic Scholar `/graph/v1/paper/ARXIV:<id>?fields=title,authors.name,year,venue,publicationVenue,publicationDate,externalIds,url`。DeepXiv TLDR 仅辅助发现，最终身份采用原始 metadata；不以 AI 摘要当原文。

- Google Scholar 请求 `https://scholar.google.com/scholar?q=%22LoCoMo-Plus%22` 返回 **429**，正文为 unusual traffic；未绕过验证码，未取得搜索结果。
- 原 bib 的 S2 **8/8 最终成功**；新增 LoCoMo-Plus 的 S2 查询退避后仍 429，改由 arXiv＋ACL＋DeepXiv 核验身份。
- OpenReview API 四个题名请求返回 **403 ChallengeRequiredError**；会议身份通过 ICLR/NeurIPS 官方程序页验证，不声称读取了被阻拦的评审历史。
- Mem0 出版商页面 403；正式 DOI 元数据由 [Crossref](https://api.crossref.org/works/10.3233%2FFAIA251160) 核验，不声称检查了出版商全文。
- 部分网页搜索返回 503，`source_check` 自动语义判定不可用。决定性结论均使用实际取得的一手 PDF、固定 commit 源码或官方正文人工核对，不把检索成功/自动分数当语义证明。

## 3. ICLR 2026 模板、格式、政策与时间线

### 3.1 时间：当前只能作历史规范审查，不能正常新投该届【P0】

[Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide) 明确摘要截止 **2025-09-19 AoE**、全文及补充材料截止 **2025-09-24 AoE**，且不为错过期限作例外。[官方回顾（2026-03-31）](https://blog.iclr.cc/2026/03/31/a-retrospective-on-the-iclr-2026-review-process/) 已写选稿 “has fully concluded”。本次日期为 2026-09-21。

稿件仍多次写 “Before submission”，并使用 MEME（2026-05-12 首公开）、LoCoMo-Plus（2026-02-11）、2026-05-13 judge commit 和当代模型。**不能把当前文件当作已经证明在 2025 截止前存在的冻结稿。** 若是已有投稿的后续版本，应提供 OpenReview ID、初投文件和更新记录；若只是沿用 2026 模板的研究稿，应明确这一点。本报告不擅自改目标会议。

这不意味着“截止后 arXiv 的论文都不能引”：BEAM 本身确实是 ICLR 2026 论文，可以先匿名投稿、后公开，也可以在允许的讨论期补引。问题是当前稿的时间身份，不是用 arXiv 首发日推断不端。

### 3.2 格式核查表

| 项目 | 官方要求／核验结果 | 判定 |
|---|---|---|
| 初投正文页数 | Author Guide 与官方模板均为 **≤9 页**；讨论/终稿 ≤10 页 | **不通过：结论 p10 开始、p11 结束**。References 同在 p11，不能把整页扣掉；附录 p12–13 |
| 官方样式 | `.sty`、`.bst` 与 [官方 ZIP](https://raw.githubusercontent.com/ICLR/Master-Template/master/iclr2026.zip) 字节一致 | 通过；无私改样式证据 |
| 纸张 | 13/13 页均 612×792 PDF points，即 US Letter | 通过 |
| 字体 | 18/18 去重字体资源有嵌入字节；未发现 Type3；正文符合约10pt | 基本通过；不等于图中字够大 |
| 主结果表 | `TABLE_main_results.tex:7–19` 不换行的最后列撑大表格 | **越界**：宽452.25bp，对比396bp版心，超56.25bp＝14.20%；新编译报56.46097 TeX pt overfull |
| 摘要 | 官方模板：“The abstract must be limited to one paragraph.” | **不通过**：`0_abstract.tex:1–3` 为两段，PDF可见 |
| 图中文字 | 官方要求 clean/legible，不杜撰硬性最小字号 | 图2部分 tick/legend 实际约 **5.805pt**，图1小字约6.5pt，可读性风险 |
| 页眉 | 官方样式配置 “Under review as a conference paper at ICLR 2026” | 现有与新构建均 **0/13页显示**，仅横线；应查依赖/分组行为，根因未确定 |
| 图表引用 | 无 `??`；13个被引用的唯一label全部解析 | 语法通过；预算表指向内容错误仍不通过 |
| 文献引用 | 22个 key 实例，8个唯一key，Bib/Bbl均8条 | 无未定义 citation；不等于无漏引 |
| 匿名 | p1为 Anonymous authors；PDF author/title等为空，XMP空，无附件/普通批注 | 所查PDF未发现身份泄露；补充包、落地页、OpenReview表单未认证 |
| 源包卫生 | `.fls/.fdb_latexmk` 留有 `/Users/admin/...` 构建路径 | 打包时剔除缓存；该路径本身不等于已识别作者 |

**两项独立事实：** 能用已有 TeX 表和图 PDF 编译，不等于能从缺失的实验 ledger 重生图表。当前 `.fdb_latexmk` 中可访问本地输入 **27/27 记录 MD5 相符**，因此不应把上述问题一概归咎于“PDF忘了更新”。

洁净构建在 CPU 工作机隔离目录 `/tmp/deep-dream-audit-20260921/clean-paper`，采用 Tectonic 0.17.0，经历 TeX、BibTeX 和引用收敛重跑，成功生成13页；没有把新PDF覆盖回仓库。首次依赖下载超时后使用缓存继续成功。新日志明确包含表1 overfull、若干 underfull 和包兼容提示；**成功不等于零警告**。

### 3.3 LLM、伦理与复现披露

- [Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide) 要求 significant research ideation/writing usage 独立披露；[2025-08-26 官方政策](https://blog.iclr.cc/2025/08/26/policies-on-large-language-model-usage-at-iclr-2026/) 更广，明确 **“Any use of an LLM must be disclosed”**，写作辅助应同时在论文及投稿表单说明。作者始终对事实、引用、实验负责，LLM 不能署名为作者。
- `main.tex:2` 有 `% Generated by ARIS paper-write skill`，报告目录也有自动流水线记录；这是**追问真实使用范围的线索**，不是可靠的 AI 作者检测。当前可见论文没有研究/写作辅助的独立 LLM usage 声明；写 answerer/judge 的实验型号不能代替该声明。应如实披露构思、代码、实验分析、写作、核验中的实际角色。本次没有取得投稿表单，因此不能断言表单未披露。
- **Ethics Statement 和 Reproducibility Statement 是推荐/可选，不是每篇统一强制的 checklist。** 必须遵守并在提交时确认 Code of Ethics；不可把缺少某个标题直接判为违规。可选伦理段≤1页不计正文，可选复现导航段也不计正文。
- 本题涉及个人对话、第三方模型 API、删除、历史保留。建议说明许可、数据来源、隐私/保留期、脱敏、API传输、撤回与物理清除的边界。不能把 MEME deletion 0%直接等同于法律意义的擦除违规。
- 本次在 PDF 文本轨迹中未发现透明/不可见文本跨度或所查“ignore instructions／强制好评”等注入语句；这只是有限检查，不是对所有隐写形式的保证。

**官方内部措辞冲突已保留：** Author Guide 后部 FAQ 有“submission version (10 pages)”的混写，但前部主规则与模板都明确初投9、修订10；本稿即使按10也超限。最终通知页面存在 Jan22/Jan25差异，不影响现已关闭的结论。

## 4. 最重要的外部基线问题：不是同一个评分协议

### 4.1 Mem0 2025 原论文与本稿固定的 2026 代码不同【P1】

`sections/4_experiments.tex:15,24,77` 和 `A_appendix.tex:33` 反复写 “the only substitution/change is the model”，甚至称 “the prompt, dataset, library … are Mem0's”。这不仅忽略 Deep-Dream 自己的记忆库/agent运行，也被**固定版本提示词直接反证**。

| 维度 | [Mem0 2025 v1，Appendix A p18](https://arxiv.org/pdf/2504.19413v1)／同期代码 | [本稿固定的 memory-benchmarks@4b61c5d](https://github.com/mem0ai/memory-benchmarks/blob/4b61c5d31b9c668a12b4f5e78064248a02c82d2b/benchmarks/locomo/prompts.py#L217-L245) |
|---|---|---|
| 日期 | “same date or time period”；“same date” | “Dates within **14 days** … CORRECT” |
| 时长 | 未定义50%误差规则 | “Durations within **50%** are CORRECT” |
| 列表 | 无显式任意一项即整题正确规则 | “**AT LEAST ONE** correct item”；“1 out of 2, 2 out of 4 … always acceptable” |
| 版本 | 2025-04-28论文；[2025-04-29同期代码](https://github.com/mem0ai/mem0/blob/393a4fd5a6cfeb754857a2229726f567a9fadf36/evaluation/metrics/llm_judge.py) | [commit 日期 **2026-05-13**](https://github.com/mem0ai/memory-benchmarks/commit/4b61c5d31b9c668a12b4f5e78064248a02c82d2b) |

2025 原提示词本身也要求 generous grading，**不能描述成绝对严格真值**；关键是两版本的明文标准不同。新版还对 open-domain gold 做分号前截断，与所查旧版直接传答案的路径不同。官方仓库不等于原论文实验归档。

**可保留的说法：** 本地使用某个明确 commit 的 binary CORRECT/WRONG rubric。**必须撤回的说法：** 只是把 GPT-4o-mini 换成 K3，便与 Mem0 2025 发表结果完全同协议。

建议改写：

> We apply the binary judging rubric from memory-benchmarks@4b61c5d to Deep-Dream-generated answers. This 2026 rubric differs from the prompt in the 2025 Mem0 paper; the memory implementation, answerer, judge and execution protocol also differ. Published scores are contextual references, not controlled comparisons.

### 4.2 重加权 62.14/61.36 不是原论文的 Overall J【P1】

稿件采用人数 `841/282/96/321` 对 Mem0 Table 1 的分类均值重加权：

- Mem0：`(841×67.13 + 282×51.15 + 96×72.93 + 321×55.51)/1540 = 62.1432597`。
- Mem0ᵍ：同理 `61.3620649`。

**这两项算术正确**，但 [原论文 Table 2 p11](https://arxiv.org/pdf/2504.19413v1) 明确报告：

| 方法 | 原文直接报告 Overall J | 本稿按分类人数重加权 |
|---|---:|---:|
| Mem0 | **66.88±0.15%** | 62.1433% |
| Mem0ᵍ | **68.44±0.17%** | 61.3621% |

差异约4.74和7.08个百分点，连排序也相反，不可能只由分类百分比的两位小数舍入解释。**不能把较低重加权值默默替代成“已发表总体分数”**，也不能凭此指控原论文造假。缺少原作者逐题/逐轮聚合说明，来源内部口径尚未对齐。

如保留，标签必须写“本稿据Table1重算；非原文Table2 Overall J”，并同时披露原文总体值及不可解释差异。更稳妥的是不构建混协议的总体排名。

### 4.3 原表转录虽准确，几处解释错误

外部数值核对分母：Mem0 F1/BLEU **88**格、分类J **24**格、Zep DMR **8**格、LongMemEval主表 **12**格、分类 **24**格，共 **156**；均与所引 arXiv v1 一致。需要改的是：

- **Mem0 open-domain 不是最弱，而是最高。** `4_experiments.tex:24` 写72.93低于其余类别；实际为 **67.13/51.15/72.93/55.51**。最低是 multi-hop 51.15。不能说与 Deep-Dream“共享最弱类别”。
- **Preference 不是所有 Zep 配置最弱。** `TABLE_zep_longmemeval_cat.tex:3` 的 all systems 错；Zep/4o-mini 的 multi-session **47.4 < preference 53.3**。原表4个配置仅3个的最低类是preference。
- 由类似类别排序推出“benchmark难度而非protocol主导”也不成立，即使排序完全相同也排除不了协议影响。
- **“verbatim tables”不准确。** 删了24个J标准差、4个延迟IQR、DMR两项转引†注脚，转置并加入本地行。应称保留数值的精简/改编表。
- caption 声称 J std 在附录，但实际附录 F1/BLEU 表未列 J std。最大std≤0.75确实成立，但去处承诺不成立。
- Zep LongMemEval 原文评估 judge 是 **GPT-4o**，两种 answerer 行都应标明；DMR原文只说明 LLM judge，本次未核定型号，不要沿用 LongMemEval 的型号。
- DMR原文是 **single-turn fact-retrieval**；不要未验证就强化为所有题都 single-hop。Recursive Summarization 35.3 与 MemGPT 93.4 是 Zep 表中转引原工作，不是所有行都在同一500题设置重跑。35.3不落在单次500题0.2pp网格上应提示口径，而非据此判错。
- `TABLE_zep_longmemeval_baselines.tex:8,14` 的 **Answerer/System 列反置**：本地行应是 `Qwen3.7-plus | Deep-Dream`。
- `TABLE_mem0_f1_bleu_appendix.tex:3` 写“Deep-Dream不计算token-overlap”，与本稿64.99 token-F1矛盾；应限定为K3-1540轨道未报告该指标。

原文确支持 Mem0 使用二元J、GPT-4o-mini、报告10次独立运行 mean±std。**不要再额外推断每次都重建记忆、重新生成全部答案**，原文该说明位于评分随机性语境。无论重复范围如何，当前单次诊断均不应包装成同样的重复实验。

## 5. 实验结果、统计和可复现性

### 5.1 主结果算术台账：内部一致，不等于运行认证

LoCoMo类别顺序 S/M/O/T，分母 **841/282/96/321**，合计1540。根据打印百分比反推的整数唯一且自洽：

| Answerer / judge | 分类正确数 S/M/O/T | 总正确数 | 复算 |
|---|---|---:|---:|
| Qwen / Qwen | 802/259/71/303 | 1435/1540 | 93.1818% →93.18% |
| Qwen / K3 | 808/256/73/304 | 1441/1540 | 93.5714% →93.57% |
| K3 / K3 | 819/264/77/306 | 1466/1540 | 95.1948% →95.19% |
| K3 / Qwen | 815/262/74/300 | 1451/1540 | 94.2208% →94.22% |
| LongMemEval-S | 当前论文未打印完整类别分母 | 423/500 | 84.60% |
| LoCoMo-Plus | cognitive子集 | 259/401 | 64.5885% →64.59% |
| MEME | 100 episodes产生异质after-checks | 432/694 raw | 62.2478% →62.25% |

64.99 token-F1是连续分数的平均，不应反推“正确题数”。原 LoCoMo 1986 与去掉446 adversarial后的1540轨道不能混用；原评分实现还有类别特例，建议准确说明而非称全部题都是同一普通F1公式。single-hop占1540题的 **54.61%**，应补宏平均或各类误差分析，但不改变已声明的微平均。

相同类别人数不证明相同 question IDs、题干、gold和会话版本；2026官方代码直接给出了类别映射，无须仅“按人数猜”。应公开逐题清单及数据revision。

### 5.2 Cross-judge：一致率正确，有效性结论过强【P1】

K3答案上的四格表：

| | Qwen判正确 | Qwen判错误 |
|---|---:|---:|
| K3判正确 | 1447 | 19 |
| K3判错误 | 4 | 70 |

`(1447+70)/1540 = 98.5065%`，23题分歧，两judge总分差15题＝**0.9740pp**，均算对。但共同接受的1447题可能有共同偏差，尤其共享同一个宽松rubric。因此 `4_experiments.tex:15` 的 “not an artifact of a lenient same-model judge” **不受支持**。

应只说“固定该rubric下，对所测两个judge模型的敏感性较小”，并对共同正确、共同错误和全部分歧分层盲审。跨供应商不是误差统计独立；正确答案judge也不是citation entailment测量。同一个Qwen同时回答与评分应称 same-model judge/rejudge，不能暗示跨模型独立验证。

**“变化小于两点”也错误：** `4_experiments.tex:60` 同时包含92.21、93.18、95.19。基于打印值，93.18→95.19为2.01pp、92.21→95.19为2.98pp；若旧92.21确为1420/1540，按整数计算分别为2.01299、2.98701pp。只保留固定答案的prompt/judge变化才适合描述为较小评分敏感性；answerer重新生成是另一个实验。

### 5.3 210/105 开发诊断、X7 和预算

| 配置 | Recall-any成功数 | Recall-all成功数 |
|---|---:|---:|
| 210 lexical | 194/210 | 158/210 |
| 210 semantic | 187/210 | 148/210 |
| 210 fused | 195/210 | 164/210 |
| 105 lexical+semantic span | 74/105 | 55/105 |
| 105 span±1 | 89/105 | 69/105 |
| 105 span±2 | 94/105 | 77/105 |
| X7 arm1 | 152/210 | 108/210 |
| X7 arm2 | 152/210 | 108/210 |
| X7 arm3 | 150/210 | 107/210 |
| X7 arm4 | 152/210 | 109/210 |

- fused相对lexical净增any 1题、all 6题正确；净增不等于只有这些题发生翻转，需paired四格表。
- ±2相对span：any净增20/105＝19.05pp，all净增22/105＝20.95pp，正确。打印均值3089→3609 bytes给出16.8339%；论文16.84%可能来自未舍入均值，**暂不判错**，补足精度即可。
- **X7最大差为2题，不是“最多1题”。** `4_experiments.tex:67` 的all范围也应 **50.95–51.90%**，不是51.43–51.90%。
- 旧fused all 78.10%与X7 full @10的51.90%不能直接串成同一测量。要列 retrieval profile、k、query重构、融合方式、候选/证据定义和代码/库版本差异；不把差异本身判作造假。
- 210是development/gate、105从prior errors选择，正文已诚实披露，应保留；但还缺具体ID清单、哪次baseline错误、选择规则、排除项、与主测试重叠及调参历史。不能外推为held-out QA提升。

**统计问题：** `replay_budget_frontier.py:145–159` 的2000次bootstrap是在问题行上重采样、分别给均值边际CI；同seed可产生相同索引，但**没有报告arm差值的paired CI**。CI重叠不证明等效。题目共享conversation/library，应报告会话数，做conversation-cluster配对差值和敏感性分析；cluster很少时也要说明区间不稳定。

**X7消融标签问题：** `replay_provenance_ablation.py:126–141` 只改候选通道，所有arm共用hybrid-v2；`retrieval.py:600–631` 仍做embedding、0.35权重semantic特征和语义MMR。arm1应称 **lexical candidate channels + shared semantic reranker**，不是端到端纯词法系统。固定从原agent轨迹重建的query可能已适应full-overlay返回，不等价于去掉overlay后重新运行agent。

**预算证据缺失于论文：** `4_experiments.tex:52–54` 称Table diagnostics给每个k的结果/CI及invalid replay，实际 `TABLE_diagnostics.tex:10–16` 一项也没有；`latex_includes.tex` 只include图1/2，图3虽存在却没进入稿件。不能写预算gate已经满足。

**预算含义过强：** 新replay按最大深度取结果、固定排序切前缀，单调性是构造性质，不能证明原在线limit=10/20接口已修复。`_evidence_payload_bytes`只累计假想top-k `[turn_id] text` 字节，遗漏实际多次工具响应、元数据、重复读取、邻接扩展和序列化；应称 **offline fixed-ranking payload proxy**，不是agent实际总消费。真实效率主张需要实际tokens、工具次数、端到端时间/费用、摄取成本及QA–cost曲线。

### 5.4 原始证据与 K3 指纹【P1】

`A_appendix.tex:5` 说ledger是checked-in且逐字节重生成核验，然而当前：

- `research/paper/results/benchmark_summary.json` **不存在、未被git跟踪**；该目录仅 `aggregate_existing.py`。
- `research/.benchmark_runs/`、`.benchmark_data/`、`.benchmark_runtime/` 当前不存在。
- `.gitignore` 的 `benchmark_*.json` 也忽略了ledger。大数据目录本就声明忽略，缺失不证明实验从未发生，但**“当前仓库checked-in”说法不实**。
- 汇总器需要15份summary/comparison/assessment JSON、3份dataset JSON、12份JSONL，共30项来源；本次执行在第一份 `summary.baseline.json` 即 `FileNotFoundError`，未生成ledger。
- 即使补回这些，当前hash清单仍不足以覆盖原始F1逐题文件、MEME逐项文件、所有运行manifest和历史runtime。现在算hash只是当前文件身份，不是生成时的可信记录。

K3诊断自述 **1529/1540＝99.29%** 的最终答案没有生成当时的runtime-code指纹。2063行后来都有hash、`answer_rows_unchanged=true`、`library_certified=true`不能自动补上历史执行链。`2019+33+11=2063`自洽，但2063到1540的523行差额要给明确的去重/重试/最终行选择规则；不能把它们当额外独立试验，也不能自动当挑分。

**处理：** 主表95.19%应就地标 `diagnostic / legacy provenance incomplete`，或移附录。正文已披露caveat，不应说作者完全隐瞒；但主表粗体且不就地说明仍会误导。完整指纹新跑只能修历史provenance缺口，不能同时修评分协议、模型映射、统计和机制对照。

### 5.5 当前汇总器的确定风险，历史影响未知

1. `judging.py:100–147` 的overall分母是 **completed**；`aggregate_existing.py:153–159,190–201` 却用 `round(overall×total)`推正确数。主审反例：2题，1题completed且正确、1题error，得到overall=1.0、total=2，ledger推2正确，实际仅1。正式结果必须断言覆盖完整，或分别报告expected/observed/completed/errors并直接逐题求和。
2. `latest_by_question`按文件顺序保留最后行，不是最高分；重试本身不能指控挑分。但resume只按qid/status、没有联合绑定答案和judge配置hash，存在旧评分被复用的风险。
3. gate loader保留重复qid，合成2行相同qid得到2个样本。必须明确唯一ID集合及重复处理，否则bootstrap有效样本数也受影响。
4. X7 `_validate_unsurfaced_evidence_flag`捕获任意异常当gate拒绝。合成gate-on抛无关 `RuntimeError`、gate-off接受，仍给 `plumbing_ok=true`。应断言正确的错误类型/原因，基础设施失败另计。

**以上证明代码可发生这些行为，不证明历史1435/1540等实际由它们造成。** 没有原始逐题文件，不能报受影响题数或修正后的历史分数。

### 5.6 MEME：0%删除必须保留，raw/real必须拆开

100 episodes的after-check数为 ER/Agg/Tr/Del/Cas/Abs＝**100/100/100/100/164/130**，合计694。删除 **0/100** 明确阻止“可靠负向记忆”主张，但尚不能定位是存储、agent指令、数据适配、任务时序还是judge所致。

正文混用ER/Agg/Tr的raw与Del/Cas/Abs的real。根据打印值可反推 `100+80+97+0+76+77=430`，不等于raw总正确432；**这不是已经证明的加总错误，因为口径本来不同**。需要每任务raw/real分子分母、before/after定义、逐项评分和episode级不确定性，不能用混合分项解释raw aggregate。

## 6. 方法是否与当前代码一致

以下来自当前 HEAD、真实函数/SQL提取和内存 SQLite/模拟桥接；主审复跑后确认。**不是对缺失的历史冻结运行时作推断。** 不要因为找到现有漏洞，就宣布全部历史成绩无效；应先回查实际轨迹和执行版本。

### 6.1 非拒答答案可绕过 accepted-submit【P1】

- 论文：`3_method.tex:19–28,40` 要求提交证据符合资格，算法应拒绝不合格项。
- 当前实现：`kimi_runtime.py:614–642` 将无submit普通文本包装成 `plain-text-without-evidence`，并放过accepted检查；更严重，合法模型JSON可自行填写同名 `format_fallback` 字段，混入可信控制状态。
- 主审两项重现：仅返回文本 `Paris`、0工具调用，或JSON `{"answer":"Paris","format_fallback":"plain-text-without-evidence"}`，均正常返回；后续 `kimi_benchmark.py:162–181,207–220` 可记completed及实质答案。
- `agentic.py:509–515` 的步数耗尽路径也可直接用surfaced材料回答，而非通过统一成功submit。

**修正：** 非拒答success必须绑定某一个真实accepted call；模型不能设置内部fallback状态；无submit标为失败或明确拒答，不照常充当契约成功。允许空证据拒答本身合理，**空集合不是这一漏洞的充分证明**。

### 6.2 processing／父版本失效并非所有路径都隐藏【P1】

- `core/cli/_helpers.py:160–166,189–203` 使用 `LEFT JOIN v_document_files`，找不到已发布父文档仍返回 `ep.source_text`。
- `schema_v15.py:647–676` 的 `v_episodes`仅过滤episode active；`agentic.py:185–191`不完整检查父document/version/ingestion。提交时的所谓validate也没有补齐这些条件。
- 内存重现：processing文档在 `v_document_files` 为 **0/1**，provenance查询仍返 **1/1** 源文本，可read并submit；仅令父version superseded但episode active，也仍被接受。
- 正面边界：`repositories/search.py:122–138` 的FTS路径确有父状态过滤；不是所有入口都错。标准更新是否会产生“父失效、子仍active”需另查，但论文声称的提交时防御未实现。

**修正：** source eligibility共享函数/SQL覆盖全部search/read/explore/submit入口；current与historical读取明确分开；read后失效的ID也重新检查。

### 6.3 “所有窗口完成”被 episode 存在替代【P1】

`orchestrator_pipeline.py:97–138` 按active episode的chunk index计完成。episode在step1已存在，`:431–444`允许其后pause/cancel；最终发布helper不要求该窗口完成extraction/alignment。

主审重现：**total_chunks=1、successful_window_indices=[]、failed_window_indices=[]**，但已有step1 episode，发布结果仍为 `active, complete_windows=1, missing_windows=[]`。即确认完成 **0/1**，发布声称 **1/1**。

应持久化真正的阶段完成标记，区分单窗口写入事务、整篇查询发布和全流程完成；不能把单次数据库事务称为整篇处理原子性证明。

### 6.4 ID暴露、文本暴露和答案输入不是同一个集合【P1/P2】

- `agentic.py:40–50,297–307` 先把字符串截到4000字符，但保留全部turn_ids。内部runner可能接受一个ID，而对应文本根本没返回。
- 公共MCP分页又发生在截断之后。构造第一turn超过4000字符、第二turn为`SECOND`，指定读第二turn返回空text、空IDs、`has_more=false`。这是实际源内容不可达，不是正常分页。
- `contexts_for_submission`（`:320–363`）只提交t1却把完整session（含t2）送给下游answerer。精确提交ID的预算/recall因此不一定代表实际答案输入。

应先对原文分页再截断、保持ID与实际返回文本一致；扩展全文/邻居要记录其实际可见范围及tokens。论文应区分 **ID曾出现、文本被暴露、模型使用文本、文本蕴含答案、事实仍有效**，不要用第一项替代后四项。

### 6.5 算法名称、评分解析及其他边界

- benchmark legacy有加权RRF；hybrid-v2有额外特征/MMR；生产explore是多通道返回；replay跨query按best rank融合，**不是统一同一RRF算法**。两query `[a,z]`、`[b,z]`时，best-rank为`[a,b,z]`，RRF为`[z,a,b]`。请在方法中给运行profile矩阵。
- LoCoMo `_parse_label`（`judging.py:178–190`）遇非JSON取第一个标签；`Not CORRECT. Final verdict: WRONG` 被判True，已重现。LongMemEval `scoring.py:59–74` 使用 `"yes" in text.lower()`，也有明显混合输出风险。需严格解析、保留raw/parse_error；是否与官方弱parser相同不消除鲁棒性问题，历史影响应离线重解析后量化。
- family只要有任意allowed来源，不一定保证返回的specific observation内容来自allowed scope。当前每conversation独立library若确实完全隔离可规避；本轮仅列共享库场景的静态风险，**未证明历史跨scope泄漏**。
- 当前适配器有fresh session、无内建写工具、thought过滤、dataset hash等正面防护；未发现可直接证明的gold答案注入模型prompt。gold用于评测或开发集选择不等于泄露给模型，但筛题/调参必须披露。
- “工具只读”不等于底层数据库以只读方式打开；schema初始化会执行DDL/commit。后续取证应只读副本并验证真实 `library/library.db`，不要用CLI显示0或误找`graph.db`作结论。

## 7. 是否存在 AI 幻觉痕迹

### 7.1 已确认的是事实/论证错误，不是作者身份

本稿最值得警惕的不是某些“像AI”的连接词，而是以下**可验证的无依据自信陈述**：

| 原稿说法 | 实际证据 | 判断 |
|---|---|---|
| 只换模型，Mem0原论文exact protocol | 新旧rubric日期/时长/列表规则不同 | 已证协议幻觉/版本混淆 |
| Mem0 open-domain比其余类别低 | 72.93最高 | 已证解释方向错误 |
| preference是所有系统最弱 | Zep-mini multi-session更低 | 已证过度概括 |
| X7最多差一题 | 152与150 | 已证计数错误 |
| 预算每k/CI已在表中 | 表中0/5档，图3未include | 已证证据指向错误 |
| 98.51%一致说明非宽松judge假象 | 不能排除共享rubric和共同错误 | 推断不成立 |
| ledger已checked-in | 当前不在仓库 | 当前发布状态陈述错误 |
| 当前scope/source状态门禁全路径保证 | 有复现反例 | 当前实现不足以支撑保证 |

这些都需要改，不应只做“去AI味”润色。现有 `PAPER_ACCEPTANCE_CONTRACT.md` 仍为proposed、0轮、reviewer pending，其部分要求与正文互相矛盾；过去自动审稿给出的分数或“已完成”记录不能取代本次直接证据。

### 7.2 已排除或未证实的怀疑

- **没有发现虚构的8条参考文献。** BEAM/LongMemEval年份差异存在真实会议/预印本版本原因。
- **Kimi-K3、Qwen3.7-plus不是本次可判虚构名称。** 当前 [Kimi官方quickstart](https://platform.moonshot.ai/docs/guide/kimi-k3-quickstart) 和 [Alibaba Cloud文本模型文档](https://www.alibabacloud.com/help/en/model-studio/text-generation-model) 正面列出 `kimi-k3`、`qwen3.7-plus`。主审再次读取确认。
- 这些动态页面只证明本次公共名称存在，**不证明论文实验实际调用哪个snapshot，也不证明2025可用**。[K3官方技术博客](https://www.kimi.com/blog/kimi-k3) 内容涉及2026年7月发布；Qwen3.7首发日期本轮未确认。`Qwen3.6-27B`等其他运行字符串也需独立供应商映射，不能用确认两个名字替代整张模型表。
- 未获得原始运行数据，不能据高分、重复行、后补hash、目录缺失推断造假、污染或挑最好答案。
- 未运行风格式AI检测；不能可靠从文风判断AI作者比例。应披露真实流程并逐句核验，避免伪精确“AI生成概率”。

## 8. 新颖性与更有说服力的论文故事

### 8.1 需要正面比较的先行工作

下列原始版本均有2025-09-24前公开证据；推荐作为机制对照，不意味着ICLR强制跑这8个系统。官方 [Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide) 明确：没有SOTA并非独立拒稿理由；对2025-07-24及以后正式发表、以及仅arXiv工作的缺少比较也有特殊边界。此处建议来自本文主张，而非杜撰会规。

| 工作 | 已有机制／与本文的重叠 | 应回答的差异 |
|---|---|---|
| [Zep/Graphiti v1](https://arxiv.org/html/2501.13956v1) | non-lossy raw episodes、派生实体边、双向source索引、双时间失效；原文明确 “sources for citation or quotation” | 新点不能只是保存原文；提交时强制资格检查比其来源链接多保证什么？ |
| [Mem0 v1](https://arxiv.org/html/2504.19413v1) | ADD/UPDATE/DELETE/NOOP；图关系失效/历史保留 | 哪种错误是简单source-preserving wrapper或同gate不能解决的？ |
| [A-MEM v1](https://arxiv.org/html/2502.12110v1) | original interaction content＋生成keywords/tags/context＋links | 哪些字段可变、原文如何保留、对齐污染如何影响答案？ |
| [MemGPT v1](https://arxiv.org/html/2310.08560v1) | 完整未压缩事件历史、分页搜索、agent上下文管理 | 多步原文检索而非图结构本身贡献多少？ |
| [LongMemEval方法研究 v1](https://arxiv.org/html/2410.10813v1) | fact-augmented keys＋original values，粒度与time-aware query expansion | 简单派生key导航＋原文value读取是否已达到相同效果？ |
| [HippoRAG 2 v1](https://arxiv.org/html/2502.14802v1) | phrase/passage图、contains边、筛选＋PPR回原文passage | graph→source不是宽泛首创；为何需要本文overlay？ |
| [ALCE v1](https://arxiv.org/html/2305.14627v1) | citation precision/recall、statement–passage entailment测量 | 用这些或明确人工标准测语义支持，而非只测ID成员资格 |
| [MemoryAgentBench v1](https://arxiv.org/html/2507.05257v1) | 增量交互、冲突/更新、不同memory机制 | 截止前已能做状态更新测试，不必只靠后来的MEME |

原论文未写某个gate不证明所有历史/当前实现都没有。应固定对手版本，公平描述其能力。原文保留、RRF、窗口扩展、scope过滤、数据库事务分别都不宜包装成全新算法；组合工程可以有价值，但需可测增量。

### 8.2 建议集中成一个可证伪问题

> **已有系统能保留原文，但“可导航的信息”不应自动成为“当前问题可提交的证据”。显式执行这一边界，能否在不明显损害QA与成本的前提下，减少不合资格引用和无依据回答？**

这是建议研究问题，**不是当前已证明的结论**。最低可信叙事可分成：

1. 现实失败模式：派生事实正确/相关，不保证当前用户、版本、实际可见文本和答案支持都正确。
2. 明确窄契约：记录级资格检查与可核验轨迹；语义蕴含另测。
3. 最小控制：强原文检索＋相同gate vs 完整overlay；不把强answerer或多轮搜索收益归给graph。
4. 结果：分别报告QA、资格错误、语义支持、拒答、成本；保留null result与deletion失败。
5. 结论：只有实验支持的收益写入贡献，未测维度留作局限，不写“已证明但待完成”。

**若补实验后overlay仍无增益：** 可以如实转为来源约束/审计系统研究或有价值的负结果研究；不能因recall无增益就自动宣布provenance收益成立。更不能靠换标题或增加“auditable”来替代证据。

### 8.3 建议的9页主文组织（编辑建议，不是模板规定）

| 内容 | 约页数 | 取舍 |
|---|---:|---|
| 摘要＋问题/贡献 | 1.25 | 单段摘要；删反复强调95.19与“同列”的文字 |
| 相关工作 | 0.75 | 聚焦source preservation、graph navigation、runtime gate区别 |
| 契约、方法与架构图 | 2.0 | 明确eligible/surfaced/text/entailment；图1靠近方法而非p8 |
| 实验协议与最小控制 | 1.0 | 一张角色/数据/预算矩阵，替代多段辩解 |
| 主要机制与QA结果 | 2.0 | 本地受控结果优先，含null/negative findings |
| 错误分析、状态/删除、成本 | 1.25 | 展示能改变结论的失败模式 |
| 局限与结论 | 0.75 | 不把未完成gate当贡献，结论p9内结束 |

外部完整复刻表、DMR（本系统未运行）、逐类数字、旧无效replay、历史指纹细节和大段长路径移附录。引用外部表只是背景，不是Deep-Dream的实验覆盖。不要为容纳它们缩小字体。

## 9. 最能提高可信度与录用机会的行动顺序

本节是**建议及未来验收条件**，不是声称本次审稿已经替作者做完这些实验，也不是保证录用。

### 阶段A：先消除确定错误与信任损伤

- 明确ICLR 2026历史/后续稿定位；修页限、表越界、摘要、错列及缺图表引用。
- 分开2025/2026 judge rubric；并列原文Overall J与自行重加权，不再写only model differs。
- 修Mem0/Zep困难类别、X7计数、less-than-two-points、F1自相矛盾、std附录指向。
- 把诊断K3行就地标明；补LoCoMo-Plus/MemoryBank/MemGPT引用及真实LLM使用披露。
- 撤掉 “not an artifact”“difficulty, not protocol, dominates”“引用已替代受控实验” 等不成立推论。

### 阶段B：恢复证据链并修关键实现

- 冻结数据、模型角色/snapshot、提示词、工具schema、代码、库与运行参数；公开全长hash，不只8位前缀。
- 按预先定义规则从逐题记录重算，不从summary乘总数反推；强制expected-ID覆盖、唯一性、errors/pending、resume指纹一致性。
- 修accepted-submit、publication completion、source资格、分页/可见ID一致性与judge解析；保留当前12个行为探针中的必要反例为正式回归测试。
- 对历史输出先做离线轨迹核验和重新解析，再决定哪些必须完整重跑；不要不加区分丢弃所有历史数据，也不要把后补hash当全部认证。

### 阶段C：最小有辨识力的实验矩阵

| 实验 | 控制条件 | 主要指标／完成标准 |
|---|---|---|
| 强source baseline vs overlay | 相同answerer/judge、embedder/reranker、可见原文、总工具/token预算；独立held-out conversations | QA与paired evidence recall、实际成本；报告所有配置，不只最优run |
| gate ON/OFF | 固定可见材料及预算；派生内容暴露作为独立factor | 未读ID、跨scope、inactive/read后失效ID拒绝；正常提交误拒；unsupported-answer与拒答变化 |
| source-text与citation consistency | 对共同判正确/共同判错/分歧分层盲审，标注者不知方法 | claim-level support、citation precision/recall、abstention precision；公开规则与分歧裁决 |
| current/as-of/ever/never | 可手验的更新、冲突、迟到事件、撤回、logical delete与historical读取 | 检索泄漏和最终答案泄漏分开；同时测基础门禁与模型行为，不以某一正向QA分数替代 |
| 实际预算曲线 | 各预算端到端重新运行agent，累计全部工具返回与上下文 | QA/unsupported/recall vs actual tokens、latency、API cost；另报摄取/维护成本 |
| 外部同条件对照（有相对优越性主张时） | 固定Mem0/Graphiti版本、answerer/judge/prompt、温度、数据revision与预算 | 只有此时才形成共享排行榜；否则原文数字继续作为分区背景 |

安全性质的确定性单元反例应全部关闭；语义效果是经验指标，不要求凭空承诺零错误。统计应报告配对差值和会话/episode级不确定性，说明独立运行/seed范围；不机械要求所有论文恰好10次运行，也不将边际CI重叠当等效。

### 最低匿名可复现包

1. 数据许可、revision/hash、scope/qid清单及过滤/开发集选择规则。
2. 最终答案、raw judge输出、label、失败/重试记录；2063→1540的确定性选择映射。
3. 实际发送与响应model字段、脱敏provider/alias映射、日期/snapshot、temperature/reasoning/max_tokens。
4. 生成时code/runtime/config/library指纹，原始与回溯归属分开；不能事后覆盖旧日志。
5. 工具实际返回文本、IDs、offset/version/scope及accepted-submit事件，可核对文本与证据一致性。
6. 从逐题记录生成summary/ledger/图表的命令与断言；MEME raw/real逐任务分子分母；选题和bootstrap代码。
7. 匿名、可下载的归档入口；不用本地绝对路径代替交付。公开来源数据可用获取脚本与hash，敏感数据需许可和脱敏，不能为了复现泄露隐私。

## 10. 模拟严格审稿结论

**建议：Reject / 暂不投稿；纯科学内容的非官方假设评分约3/10。** 这不是大会真实评分、概率预测或录用承诺。事实与当前代码判断置信度高，历史运行能力判断受原始证据缺失限制。格式页限另有明确desk-reject风险，不应与科学评分混为一谈。

**优点：** 问题重要；navigation/authority区分清楚；源码和仪器化投入真实可见；作者主动披露错误子集、无效replay、null result和删除0分，比隐藏失败更值得保留。

**主要弱点：** 最接近先行工作区分不充分；核心机制的agent-level效果未完成；协议等价性说法错误；高QA受answerer/rubric/预算等混杂；当前实现不满足所声称的全路径来源保证；历史证据尚不能独立复核。

**最关键的审稿问题：**

1. 为什么“强原文检索＋邻接扩展＋同一answerer＋简单ID allowlist”不够？
2. overlay没有明显recall收益时，在什么独立测量上值得其复杂度和摄取成本？
3. 门禁约束的是ID、可见文本、版本资格还是语义支持？各自反例与指标是什么？
4. 双judge共同接受的答案中，有多少不被提交来源支持？
5. 2063行如何变成1540个最终答案；失败、重试、旧评分和后补指纹如何处理？
6. 删除0/100与“still-valid source”的叙述如何协调？时间、失效与物理删除如何定义？
7. 当前稿的基准、模型和评分代码与ICLR 2026历史提交时点是什么关系？

**提高录用机会的重点不是进一步美化外部数字，而是把一个窄、重要、可证伪的机制问题真正闭合。**

## 附录A. 可重复的审查记录

### A.1 当前文件指纹

| 文件 | SHA-256 |
|---|---|
| `research/paper/main.pdf` | `81151ff3b925e14d1e8350950cb5836e7c819f33410521ef09aeffaa4e99befc` |
| `research/paper/main.tex` | `be6286ea670d91795a19ed309fa7d803a4c6d611f2280486d5fc6a03b5fbc602` |
| `research/paper/references.bib` | `c6cfdf21fb03e1c30180ebbdac5e0131213e2ac4de80970ce6e501beda166e4c` |
| `iclr2026_conference.sty` | `a4852f68e080d6c5245057ca2039100b409e31727898aa93c03d78ddb84374a3` |
| `iclr2026_conference.bst` | `2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5` |
| 隔离Tectonic构建 `main.pdf` | `fde90345018a6ea8bcac1ce197a37f19287f5b617e05c3350a7e0da96c4de34b` |

原PDF metadata生成时间是2026-08-13，不是投稿时间证明。上述hash用于标识本次审查对象，不认证实验真实性。

### A.2 已执行检查与结果

```text
git rev-parse HEAD                         -> 66af0642...（见报告完整值）
git ls-files research/paper/results        -> 仅 aggregate_existing.py
aggregate_existing.py --output /tmp/...    -> FileNotFoundError，缺首个summary
官方ZIP vs 本地 .sty/.bst                 -> byte equality: True / True
现有输入 .fdb_latexmk MD5                  -> 27/27匹配
现有PDF / 隔离新PDF                        -> 13页 / 13页；正文均延伸p11
Tectonic 0.17.0 clean build                -> 成功；非零排版警告
Table1 overfull                            -> 56.46097 TeX pt（≈56.25 PDF bp）
12个当前代码行为探针                       -> 全部重现；不是安全通过
completed/total反例                        -> total=2, completed=1, overall=1.0，错误推correct=2
gate重复qid反例                            -> 2行、1唯一qid，loader仍保留2条
plumbing任意异常反例                       -> 无关RuntimeError也得plumbing_ok=true
```

主审读取了临时探针脚本后重新执行；探针通过AST使用实际函数，SQLite只在内存中，模型桥接使用StringIO，不启动LLM/真实服务。不对缺失运行库进行写初始化；没有运行全库pytest或重新评分真实JSONL。

### A.3 小型自包含复核片段

在仓库根目录执行以下stdlib片段即可复核关键算术和一个真实parser反例，不依赖本次 `/tmp` 脚本：

```python
from pathlib import Path
import ast, json, re

root = Path("research/paper")
n = [841, 282, 96, 321]
assert sum(n) == 1540
assert round(1466 / 1540 * 100, 2) == 95.19
assert round(1517 / 1540 * 100, 2) == 98.51
assert round(432 / 694 * 100, 2) == 62.25
assert round(152 / 210 * 100, 2) == 72.38
assert round(150 / 210 * 100, 2) == 71.43
assert 152 - 150 == 2
j = [67.13, 51.15, 72.93, 55.51]
assert j[2] == max(j)  # Mem0 open-domain不是其最弱类别
assert round(sum(a*b for a, b in zip(n, j)) / 1540, 2) == 62.14
print("ledger exists:", (root / "results/benchmark_summary.json").exists())
source = Path("research/benchmark/judging.py").read_text()
tree = ast.parse(source)
node = next(x for x in tree.body
            if isinstance(x, ast.FunctionDef) and x.name == "_parse_label")
ns = {"json": json, "re": re}
exec(compile(ast.Module(body=[node], type_ignores=[]), "actual-parser", "exec"), ns)
assert ns["_parse_label"]("Not CORRECT. Final verdict: WRONG")[0] is True
print("已重现旧parser的误判；这不是正确行为验收。")
```

该反例应在修复后不再成立；不要为了继续通过审查脚本保留漏洞。

### A.4 子任务与取证位置

工作流 `df9c91ae-043f-420e-82f5-14fbb4b4d906`，8/8子任务完成；完整独立意见的运行绑定输出目录：

`/home/dinotty/.pi/agent/sessions/--workspace-exp-iclr-deep-dream--/subagent-artifacts/outputs/df9c91ae-043f-420e-82f5-14fbb4b4d906/audit/`

| 文件 | 子任务 run ID |
|---|---|
| `citation-identities.md` | `cc084e9b-e5d6-4f78-a7d0-b023e9aff429` |
| `iclr-policy.md` | `4bbb984d-4ffa-4d14-9a22-866af16080a2` |
| `experiment-evidence.md` | `1cd27d2c-6622-45ec-9e8e-d61ed8da383f` |
| `implementation-contract.md` | `5a0d4b16-583b-4505-b16e-1fa1be18ea9a` |
| `strict-narrative.md` | `1fb13855-269d-4f08-b143-2695ffcff4db` |
| `published-baselines.md` | `c675104a-7fcf-405c-80e6-df5a7584f317` |
| `pdf-format.md` | `f253c71b-3965-4fb8-8ae9-0395a51c703b` |
| `novelty-positioning.md` | `9a67acb2-37b5-49af-b5b5-d08fb15d4d7a` |

运行receipt：`/tmp/pi-subagents-uid-1000/async-subagent-runs/df9c91ae-043f-420e-82f5-14fbb4b4d906/workflow-receipt.json`。本机临时网页取证在 `/tmp/dd-citation-audit/`，PDF渲染/官方模板在 `/tmp/deep-dream-pdf-audit/`，主审编译输出与探针日志在 `/tmp/deep-dream-audit-20260921/`。这些属于保留策略管理/临时材料，**报告关键结论、原句、来源链接、分母和复核代码已内嵌，不依赖临时路径长期存在**。

## 附录B. 最终证据边界

- 本报告已完成用户指定的当前材料审稿，不代表论文已达到投稿要求；修论文、修系统与补实验是后续工作。
- Google Scholar未取得结果；没有把S2冒称Google Scholar，没有声称绕过OpenReview访问限制。
- 原始实验未提供，历史正确率、污染、失败比例、供应商snapshot、代码指纹与真实运行的一致性仍**未独立认证**。
- 当前代码反例来自合成小环境，不覆盖真实并发/全流水线，不能给出历史受影响比例。
- 本次洁净编译仅认证已有TeX/图可构建；不认证从原始数据重生图表，也不认证原pdfTeX环境逐字节复现。
- PDF匿名通过限于所检文件；没有审核未来提交包、作者表单、匿名仓库或所有外部链接行为。
- 没有作者不端、AI身份或录用概率结论。严格审稿应优先纠正可证实错误，而不是对未知事项作最坏猜测。
- 收尾另调用顾问两次，但仅返回不完整片段（`You`、`##`），没有可用审查意见；**未将其计为终审通过**。交付依据为8个完整独立子任务、主审原文复核、实际构建与可运行检查。
