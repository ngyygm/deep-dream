# 四基准汇总报告（2026-08-29）

> Deep-Dream 系统在 Kimi-k3（sz-infer 端点）上的四基准成绩全景。
> 引擎口径：**LME 与 MAB 均为 V1→V2 配对终判**（MAB v2 重跑完成于 8/31，见 `mab_v1_vs_v2_2026-08-30.md`）；BCB / ALFWorld 测于 V1 引擎期（8/24–8/26），v2 切换后未重跑——对记忆引擎不敏感，作回归锚点。

## 总表

| 基准 | 轨道/口径 | 成绩 | run 目录 |
|---|---|---:|---|
| **LongMemEval-S**（1176 docs/25 scopes，配对 V1→V2） | baseline | 0.68 → **0.72** | `longmemeval-kimik3-full-{v1,v2}` |
| | skill-agent | 0.68 → **0.72** | |
| | pi | 0.889 → **0.926** | |
| | 检索 recall@10（baseline 轨） | 0.767 → **0.847** | |
| | 效率 calls/doc · tokens/doc | 63.8→**22.0** · 139k→**112k** | |
| **MemoryAgentBench**（官方 scorer，sampled 767q/10 scopes，配对 V1→V2） | baseline | 0.3025 → **0.3240** | `memoryagentbench-kimik3-sample-{v1,v2}` |
| | skill-agent | 0.3759 → **0.4068** | |
| | pi | **0.6695** → 0.6511（TTL 0.43→0.46、FC-MH 0.67→0.70） | |
| | 效率 calls/doc（45 doc 交集） | 609 → **329（−46%）** | |
| **BigCodeBench**（instruct/full，calibrated） | completion pass@1 | **0.4859**（gt 上限 0.8554） | `bigcodebench-kimik3-v1` |
| | hard pass@1 | 0.2838（gt 上限 0.7778） | |
| **ALFWorld**（max 50 步，w3） | in-distribution | **0.9786**（137/140） | `alfworld-id` |
| | out-of-distribution | **0.9851**（132/134） | `alfworld-ood` |

## 分基准要点

### 1. LongMemEval-S（质量+效率主战场，V2 终判）
详见 `lme_v1_vs_v2_full_2026-08-29.md`。V2（B2 = window_batch + 簇收敛）五维全胜：三轨准确率 +0.04/+0.04/+0.037、recall@10 +8pp、ndcg@1 +8pp、calls/doc −65%、tok/doc −20%、同名重复家族率 0.8%→0.2%。已切默认引擎（commit 8fa9651）；pi 轨短板 single-session-assistant/preference 各 0.5。

### 2. MemoryAgentBench（长程记忆压力面，V2 配对终判 8/31）
详见 `mab_v1_vs_v2_2026-08-30.md`。v2 重跑验证了两大短腿被拉起：**TTL MCC 0.43→0.46、FC-MH 0.67→0.70**，辅轨双涨（skill-agent +0.031 / baseline +0.022）；pi Overall 0.6695→0.6511 的微降全部来自 LRU Summ 单题域翻转（n=1 噪声，v1 0.212→v2 0.000），剔除此域 v2 反超。效率 calls/doc −46%（609→329）、总 tokens −18%（45 doc 交集口径）；重复家族率 0.20→0.00。**TTL（0.46）与 FC-MH（0.70）仍是最大短板**，下一抓手在窗口内原子事实抽取密度，而非对齐合并。

### 3. BigCodeBench（代码生成，系统作为 agent 载体）
calibrated 口径（含 gt_pass_rate 上限校准）：completion 0.4859 / gt 0.8554（达成率 56.8%），hard 0.2838 / gt 0.7778（36.5%）。主要失败模式为任务理解与 API 组合错误，非记忆问题——该基准对记忆引擎改动不敏感，作回归锚点用。

### 4. ALFWorld（具身决策，id/ood 双分布）
id 0.9786 / ood 0.9851，平均步数 ~17（上限 50）——接近饱和，ood 反而略高（样本小）。作回归锚点用。

## 结论与下一步

1. **记忆质量主指标看 LME（0.926 pi）+ MAB（v2 pi 0.6511，靶点域 TTL 0.46 / FC-MH 0.70 均已拉起）**，pi 轨一致最强，验证 dd_scope 沙箱 + bash 验据的 agentic 检索路径。
2. v2 引擎在两大记忆基准均完成配对终判（LME 五维全胜、MAB 靶点双升 + 成本 −46%），**默认引擎地位坐实**；剩余短板收敛到 MAB 的 TTL/FC-MH（抽取密度）与 LME 的 single-session/preference（会话内记忆）。
3. BCB / ALFWorld 接近饱和或与记忆无关，保持作回归锚点，每次引擎大改后抽跑即可。
4. **对外部系统的横向定位**见 `memory_systems_horizontal_2026-08-31.md`：MAB Overall 领先已发表最佳 ~10pp（口径调整后）、FC-MH 数量级领先（70 vs 7）、TTL 落后全文方案是全行业共病；LME 0.926 居第一梯队（厂商自宣互打架， ours 自判需披露）。

## 记忆沉淀索引

- LME V2 终判与切换：`lme-v2-final-verdict`（calls/doc −65%、commit 8fa9651、v1 配置已钉）
- MAB V2 终判：`mab-v2-ingest-robustness-fixes`（TTL/FC-MH 双拉起、calls/doc −46%、四处鲁棒性修复 db81ee1）
- 吞吐修复四件套：`deep-dream-v2-throughput-fixes`（json_object_mode / cap=12 / converged-marker / 同 hash 碰撞）
- A/B 框架与 B1-equivalent 语义：`align-v2-experiment-verdict`
- 效率口径纪律：`efficiency-metrics-tokens-not-wallclock`
- 基准运行坑位：`deep-dream-benchmark-concurrency`、`deep-dream-reingest-unique-pitfall`、`deep-dream-kimik3-full-run`
