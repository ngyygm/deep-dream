# LongMemEval-S 全量 V1 vs V2 最终对比（2026-08-29）

> 结论先行：**V2（ALIGN-V2 簇收敛引擎）全面胜出，已执行主树默认切换。**
> 效率口径（主指标）：calls/doc **−65.5%**（63.8→22.0）、tokens/doc **−19.6%**（139.0k→111.7k）；
> 质量口径：三轨准确率全升（+0.04/+0.04/+0.037）、检索 recall@10 +8pp、重复家族率 0.8%→0.2%。

## 1. 实验框架

| | V1（对照） | V2（候选） |
|---|---|---|
| run 目录 | `longmemeval-kimik3-full-v1`（2026-08-26） | `longmemeval-kimik3-full-v2`（2026-08-27~29） |
| 数据 | LME-S 1176 docs / 25 scopes，逐 scope 独立库 | 同左（dataset_sha256 一致） |
| 评测题集 | baseline 25 / skill-agent 25 / pi 27 | 同左（同 question_ids 配对） |
| LLM | kimi-k3 @ sz-infer，同一端点 | 同左 |
| 引擎 | **B1-equivalent**：window_batch_alignment=on、cluster_convergence=off（当时的 DEFAULTS 即此状态） | **B2**：window_batch + 簇收敛（align_v2：窗口等价组收集 + step9 跨窗口并行 + scope 末全库收敛） |
| 运行时 | w16/w32 调度 | 同左 + 四项吞吐修复（§5） |

**B1-equivalent 说明**：V1 主跑时 DEFAULTS 已显式开 window_batch（"主 run 实证等效 B1 档"），故 V1≠裸 A（逐实体对齐），而是 A/B 实验（2026-08-25，`dd-exp/research/reports/align_ab_experiment_2026-08-25.md`）中的 B1 臂。A/B 已证 B1 单独使用同名重复家族反而最多（12 vs A 的 8）、不收敛；本对比实际验证的是 **B1 → B2 的增量**（补上收敛引擎）。

**V1 库 489 篇修复分界标注**：V1 灌库中途 489 docs 处经历一次 runner 修复重启（resume/UNIQUE 链，见记忆 deep-dream-reingest-unique-pitfall）。该修复只影响断点续跑正确性、不改变单文档处理成本（calls/doc 前后段无系统差），故本对比不对 V1 内部分层，V1 计入全部 940 有统计文档。

**pi Total 27 vs 25 口径核对**：pi 轨道因题目资格规则（session-evidence 可判）多收 2 题，两轮均为 27，与其它两轨的 25 不是缺题或丢题；跨轨只比同轨自身 V1→V2。

## 2. 效率（主口径：per-doc LLM calls/tokens，不用墙钟）

数据源：run_manifest 的逐文档 `llm_call_stats` 聚合。**覆盖度诚实标注**：V1 940/1176、V2 1020/1176 文档有统计（进程被调度器定时 kill 后未 flush 的条目丢失；kill 由 23:00/11:00 定时器驱动、与单文档成本无关，对 per-doc 比值无偏）。两侧 DB active docs 均 1176/1176，scope 集 25/25 完全一致。

### 全量

| 指标 | V1 | V2 | Δ |
|---|---:|---:|---:|
| LLM calls/doc | 63.8 | 22.0 | **−65.5%** |
| tokens/doc（prompt+completion） | 139.0k | 111.7k | **−19.6%** |
| ├ prompt tokens/doc | 68.1k | 43.0k | −36.9% |
| └ completion tokens/doc | 70.9k | 68.7k | −3.1% |

### 按修复分界分层（V2 内部）

V2 中途部署了四项吞吐修复，分界 @ 2026-08-29 05:19（ingest log line 356111）。按 scope 完成序切分，与 V1 同 scope 集配对：

| 分层 | scopes | V1 calls/doc | V2 calls/doc | Δ | V2 tok/doc |
|---|---:|---:|---:|---:|---:|
| v2-pure（修复前） | 15 | 66.2 | 23.2 | −64.9% | 114.9k（V1 143.9k，−20.2%） |
| v2-tuned（修复后） | 9 | 59.8 | 20.9 | −65.1% | 108.8k（V1 130.0k，−16.3%） |
| gpt4_78cf46a3 | 1 | — | — | 跨分界混合 scope，分层对比中剔除（全量表中计入） | |

引擎本体（window_batch+簇收敛）贡献 ~−65% calls/doc；吞吐修复再削 ~10%（23.2→20.9，主要是 JSON 重试消除与补裁帽）。

### 分步 calls（全量）

| step | V1 | V2 | 说明 |
|---|---:|---:|---|
| 02s_onepass_extract | 2,775 | 3,003 | 窗口数 ≈2.95/doc 两侧一致（V2 覆盖文档更多） |
| 09s_window_batch_entities | — | 6,335 | V2 新：窗口批量裁决 |
| 06_entity_alignment | 48,866 | 4,912 | V1 主成本；V2 仅歧义带兜底 |
| 07_relation_alignment | 1,729 | 1,679 | 持平 |
| unlabeled（评测腿等） | 6,621 | 6,509 | 持平 |

对齐腿合计 48,866 → 11,247（**−77%**）：V1 的逐实体 LLM 对齐被窗口批裁决 + 快路径 + 歧义带兜底替代，这正是 V2 的设计目标。

## 3. 质量（三轨准确率，配对同题集）

| track | V1 | V2 | Δ |
|---|---:|---:|---:|
| baseline | 0.6800 | 0.7200 | +0.0400 |
| skill-agent | 0.6800 | 0.7200 | +0.0400 |
| pi | 0.8889 | 0.9259 | +0.0370 |

- 每轨净 +1 题（25/27 题样本下单个 +1 无统计功效；但三轨同向 + 检索大升构成一致证据链）。
- skill-agent − baseline = 0.0 **在两轮都为 0**——该基准配对下检索侧改动不翻转最终答案，属基准特征而非 V2 异常。
- pi by_type：V2 在 knowledge-update / multi-session / temporal-reasoning / single-session-user 全 1.0；短板 single-session-assistant 0.5、preference 0.5。

### 检索指标（baseline 轨）

| 指标 | V1 | V2 | Δ |
|---|---:|---:|---:|
| session_evidence_recall@1 | 0.447 | 0.513 | +6.7pp |
| session_evidence_recall@5 | 0.727 | 0.827 | +10.0pp |
| session_evidence_recall@10 | 0.767 | 0.847 | +8.0pp |
| session_evidence_recall@30 | 0.780 | 0.860 | +8.0pp |
| session_ndcg_any@1 | 0.720 | 0.800 | +8.0pp |
| session_ndcg_any@10 | 0.714 | 0.800 | +8.5pp |

（skill-agent 轨混合：@1 0.567→0.513 −5pp，@10 0.78→0.80 +2pp。）

## 4. 簇收敛性

| 指标 | V1 | V2 |
|---|---:|---:|
| entity_families 总数（25 scopes） | 59,892 | 55,878 |
| 同名重复家族行数 | 482 | **99** |
| 重复家族率 | 0.008 | **0.002** |
| 修复分界后段（9 scopes 配对） | 0.012 | 0.002 |

- V2 每 scope 收尾收敛 flush 裁 2,000–4,500 对、并 2–199 个家族；manifest 全部 25 scopes `converged=true`。
- 全量尺度复核了 A/B 结论：B1（=V1）不收敛、B2 是唯一真收敛臂；且 `merge_entity_families` 非破坏合并全量零 FK 事故。

## 5. 吞吐修复与墙钟（附录，非对比口径）

V2 运行中部署的四项修复（详见记忆 deep-dream-v2-throughput-fixes）：① `llm.json_object_mode`（response_format 终于到 OpenAI 分支，4xx-only 自动降级）；② `window_align_llm_cap=12`（歧义带补裁帽，合成 create_new conf 0.7，仅触发 4 次）；③ runner converged-marker（消灭重启税：重启一次曾重复收敛全部存量 scope ~2.8h，这是 36→14.3 docs/h 劣化的主因）；④ 同 hash active 碰撞硬删重插（UNIQUE 崩溃绝根）。修复后 14.3→**52.4 docs/h**（3.7×），ingest 1176/1176 完成，eval 链 31 分钟自动跑完（与 V1 当时的节奏一致）。

## 6. 判定与切换

**V2 胜出，维度无短板**：效率（−65% calls / −20% tokens）、质量（三轨全升+检索大升）、收敛（重复率 −75%）、稳定性（UNIQUE 绝根、重启税清零）。已执行：

1. `core/server/config.py` DEFAULTS：`cluster_convergence` **False→True**（v2 成默认引擎；`json_object_mode` 入 DEFAULTS 默认关）。
2. v1 复现配置 `service_config.kimi-w16/w32.json` 显式钉住 `cluster_convergence: false`（防默认翻转后失真）。
3. 主树提交：v2 引擎 + 四项吞吐修复 + benchmark runner 支持（显式文件清单提交）。

原始数据：`research/.benchmark_runs/longmemeval-kimik3-full-{v1,v2}/`（summary.*.json / run_manifest.json / *.ingest.log）。
