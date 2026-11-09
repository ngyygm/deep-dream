# REBUTTAL_EXPERIMENTS — 讨论期新证据存档 (provenance)

规则: 仅文字回复 → 所有数字以 official comment 文字表格呈现; 每项证据须可溯源到冻结工件 + 脚本。

---

## GAP-D · 实体合并 vs ground-truth 【已完成 2026-11-09, 纯离线 0 API 调用】

回应: M-C2 (mV1D 抽取/对齐错误无直接评估), C-C3 (cB3k "entity merges are not evaluated
against ground-truth labels")

### 工件与方法

- 脚本: `research/paper/rebuttal/gapd_alignment_audit.py` (v2, 单射名称匹配法)
- 结果: `research/paper/rebuttal/gapd_results.json`
- 输入 (全部冻结, 只读):
  - Base 引擎库: `research/.benchmark_runs/memoryagentbench-kimik3-sample-v1/libraries/mab-ar-*`
    (commit 73d9c5e)
  - Align+ 引擎库: `research/.benchmark_runs/memoryagentbench-kimik3-sample-v2/libraries/mab-ar-*`
    (commit 8fa9651, 同 runtime_policy fc73d07e)
  - Ground-truth 实体池: `research/.benchmark_data/memoryagentbench/entity2id.json`
    (31,161 DBpedia URI, MAB 官方随数据集分发)
- 方法 (v2, 弃用 v1 块主题法):
  - 一个 mention 的真值身份当且仅当其表面形内容词集合与池中**恰好一个** URI 的名称
    (slug, 括号后缀剥离, 两次注册) 精确相等时被断言; 歧义名 (如 "Victoria") 一律排除,
    不做猜测。34,958 个单射名集 / 808 个歧义名集。
  - terminal family = entity_redirects 链追到不动点 (含环保护)。
  - v1 曾尝试 "块主题 = 维基摘要 Jaccard 反推", 已证伪并删除: MAB AR 的 "Document N:"
    块是**文章中段段落**而非首段摘要 (例: "The upper Rhine region was changed
    significantly by a rhine straightening program…"), REST summary 只返回首段,
    avgJ=0.085 全军覆没。维基抓取缓存 (gapd_wiki_cache.json, 395 URI) 留档不再使用。

### 结果 (5 个 AR library, Base 22,830 mentions / Align+ 21,967)

池单射子集 (真值标签可无争议断言的 mention; 池覆盖率 2.8%/2.7%, 如实报告):

| 引擎 | 单射 resolved | 真值实体 | 碎片化 (≥2 terminal families) | 过合并 family (对数) |
|---|---|---|---|---|
| Base (73d9c5e) | 637 | 349 | **8 (2.3%)** | **0 (0)** |
| Align+ (8fa9651) | 601 | 330 | **5 (1.5%)** | **0 (0)** |

名称层全量检查 (不依赖池; 同一表面名落入 ≥2 terminal families 的比率, 按歧义分层):

| 名桶 | Base 名数 (split%) | Align+ 名数 (split%) |
|---|---|---|
| 池无歧义实体名 | 349 (2.3%) | 330 (1.5%) |
| 池歧义名 (split 可能是正确消歧) | 80 (13.8%) | 72 (6.9%) |
| 实体大小写、不在池 | 4,864 (2.1%) | 4,852 (1.2%) |
| 小写非专名 | 11,708 (1.2%) | 11,375 (0.8%) |

### Rebuttal 用法 (要点)

1. **首次给出 merge 质量 vs 真值标签的直接评估** — C-C3 的"无 ground-truth"指控被化解;
   指标定义 (单射断言、terminal family) 两行说清, 不可反驳。
2. **两代引擎 ground-truth 过合并均为 0**; 碎片化 2.3%→1.5% — 延迟对齐**改善**对齐质量,
   同时为 M-C2 提供 "错误率低且有方向性改善" 的正面证据。
3. 与论文 duplicate-family 代理 (340→124) 的趋势一致, 且这里是 mention 级直接测量。
4. 诚实披露 (必须保留): 池覆盖率 2.8% (DBpedia 池是 MAB 自带切片, 不含 ar 语料全部实体,
   如 Warsaw/European Union 不在池中); 歧义名单独分层, 其 split 含正确消歧, 不作错误率解读。
5. 局限的缓冲论述: 即使按最坏解读 (池外实体名 2.1% 全为碎片化), Prop.1 单向地址错误
   (错并/错拆只影响检索召回, 不污染已接受引用的源文本) 仍是缓冲。

---

## GAP-A/B · no-graph 在线消融 + 预算匹配迭代 RAG 【代码就绪, 待评测机执行】

回应: M-Q3 (mV1D 点名在线图消融), M-Q4 (同预算裸迭代 RAG), C-C3 (组件隔离)

- 设计: **一个新臂同时回答两问** — 工具集裁剪为 {search_documents, read_episode,
  read_session, submit_evidence}。因 Deep-Dream 源文档无损存储, search_documents 的
  raw-document 通道 (= 文件级词法检索) 在裸语料上检索, 该臂同时是 (a) 去图导航的完整
  agent (GAP-A) 与 (b) 同源访问的迭代 RAG (GAP-B)。K3 answerer/prompt 预算/max_steps=8/
  answer_top_k=5/normalized-v1 与冻结 source 轨逐项一致; 唯一自由度 = 工具菜单。
- 实现: `research/benchmark/agentic.py` (AGENT_TOOLSETS full/no-graph + 动态 TOOL_PROMPT
  + allowed_tools 贯穿), `runner.py` (装配点传 toolset; manifest 变体记录含
  agent_toolset), 测试 41/41 绿 (含 no-graph 回归测试)。
- 阻塞: K3 端点 (sz-infer.x2robot.cn) 仅评测机可达; 本机 TCP 80/443 均超时。
- 执行: 见 `RUNBOOK_GAP_AB.md` (评测机 /Users/admin/Desktop/exa/deep-dream)。
- 对照 (冻结, 无需重跑): v2 source-reading 71.0@1074 / 70.8@614; v1 同库可做引擎间对照。
- 账本纪律: 新臂走 --result-tag no-graph-v1 → manifest track_variants 自动记录
  agent_toolset/config/created_at; 计分用官方 score_memoryagentbench_official.py
  (judge kimi-k3, 与论文同协议)。
- 成功判据: full > no-graph 超配对 CI → M-Q3 正面化解; ≈ → 窄让步 (增益来自源读取
  而非图导航本身 — 与 X7 null 一致的诚实呈现, 论点退到写入侧贡献)。
