# A/B 对齐实验报告与迭代决策(2026-08-25)

三臂:LME-S 3 scopes(42+49+46 = 137 docs/臂),kimi-k3 真实端点,全部完成 ingest + 三轨评测。
所有数字经独立复核(效率 <0.15% 偏差,收敛/覆盖 0% 偏差)。

## TL;DR 决策

**采纳 v2 引擎(B2)作为目标架构**;B1 证明的 window_batch 对齐是最大 token 节省来源,直接默认开启。
主 run 继续用现状引擎跑完(保基线完整性),v2 落地主树后在后续 run 验证。

## 三臂设置

| 臂 | 引擎 | 开关 |
|---|---|---|
| A | 现状系统 | 无 window_batch、无 v2 |
| B1 | 窗口批量对齐 | `pipeline.remember.window_batch_alignment: true`(纯配置) |
| B2 | 簇收敛引擎 v2 | window_batch + `DD_ALIGN_V2=1`(并行 step9 + 延迟簇收敛 + 非破坏合并) |

## 轴 1:效率(主指标,per-doc tokens/calls,口径=manifest 文档数)

| 指标/doc | A | B1 | B2 | B2 vs A |
|---|---|---|---|---|
| calls | 68.4 | **24.2** | 29.4 | **-57%** |
| prompt_tokens | 72,528 | **37,148** | 40,207 | -45% |
| completion_tokens | 72,850 | 71,319 | 71,647 | -2% |
| total_tokens | 145,378 | **108,467** | 111,853 | **-23%** |

- 改进集中在 step9:A 的 `06_entity_alignment` 每文档 55.6 calls / 55,242 ptok,B1/B2 的 `09s_window_batch_entities` 只要 6.6-6.8 calls / 18-20K ptok(**prompt -67%,calls -88%**)
- completion 基本不变:抽取(`02s_onepass_extract` ~40K ctok)是内容蒸馏本体,三臂一致——对齐优化的正是"重复决策"而非"内容理解"
- B2 比 B1 贵 ~3%(+3.4K tokens/doc):收敛扫描的判定调用(unlabeled 桶 1,685 vs 858 calls)——**用 3% 换真收敛,值**
- 分母敏感性:改用 137 DB 口径则三臂一致缩水 5.8-8%,相对结论不变

## 轴 2:收敛(三 scope 合计,v2 的核心价值)

| 指标 | A | B1 | B2 |
|---|---|---|---|
| entity_families | 7,070 | 7,076 | **6,861**(-3%) |
| entity_redirects | 0 | 1 | **245** |
| relation_families | 12,045 | 11,977 | **11,135**(-7.5%) |
| 同名重复组 | 8 | 12 | **6** |
| entity_observations | 8,040 | 8,062 | 8,070(≈持平) |

- **只有 B2 收敛**:245 个 redirect 来自 245 个不同 source family 汇入 227 个 target,单 target 最多 4(长尾,无异常值);observations 不掉——非破坏合并(merge_entity_families)保住了历史可溯源,**FK bug 修法在全量负载下零爆炸验证通过**
- B1 不做收敛(redirects=1),同名组反而最多(12,其中 8 个集中在 scope3):window_batch 单独用会略微加重跨窗口重复——**window_batch 必须和收敛引擎一起上**
- B2 全程收敛:9 轮扫描共判 6,000 对、合并 182 个 family(scope1: 51+6+2+2、scope2: 65+5+1+0、scope3: 50)
- 缺陷:750 对/轮预算两 scope 都打满(候选池 > 预算);scope3 只跑了一轮未到不动点(78cf46a3 末轮 merged=0 已收敛,af6db32f 末轮仍 merged=2)

## 轴 3:质量(无回归检验,统计功效弱)

三臂 × 三轨全部 **1.0**(baseline 3/3、skill-agent 3/3、pi 4/4)。
LME-S 每 scope 恰 1 题 → n=3-4,只能得出"无明显回归",不能区分优劣。质量轴按设计让位于效率+收敛轴。

## 鲁棒性账本(全天运行记录)

| 事件 | A | B1 | B2 | 说明 |
|---|---|---|---|---|
| step9-None(出现次数) | 236 | 16 | 50 | 三臂共性缺陷(stock A 最多),端点拥堵时 step9 返回 None、step10 raise;靠链 retry 自愈 |
| episode-UNIQUE 重灌冲突 | 6 | 5 | 6 | 三臂共性:`alignment.py:359 _update_cache → save_episode` 直插绕过 ingest 去重 |
| shutdown-wait 楔死 | 多次 | 1 | 5 | 崩溃后进程 0% CPU 挂 15-30 min 不退出,链无法重试;需人工精确杀 PID(今日 4 次) |
| 15:00-16:00 端点 502 宕机 | — | — | — | 全臂靠 retry 预算(~200 圈)自愈 |

结论:v2 的额外活动件没有引入新故障模式——所有失败都是现状系统的老问题。

## 迭代决策

1. **采纳 v2(B2)**:效率 -23% tokens/-57% calls + 唯一真收敛 + 质量无回归 + 无新增故障模式
2. **主 run 不切换**:继续现状引擎跑完 500 scopes(#12 基线完整性);v2 落地后另起 run 验证(可先 3-scope 快验再全量)

## 主树落地清单(benchmark 全部结束后执行,按序)

1. `DEFAULTS` 中 `pipeline.remember.window_batch_alignment: true`(最大 token 节省,纯配置)
2. 移植 `core/remember/align_v2.py` + pipeline_workers 并行 step9 + runner 的 scope 收尾 flush 钩子(先挂 `pipeline.remember.cluster_convergence` 开关)
3. `cross_window.py`:dedup_merge_batch 删除路径 → merge_entity_families(FK bug,见记忆 deep-dream-dedup-merge-fk-bug)
4. 修 episode-UNIQUE:save_episode 直插路径幂等化(INSERT OR IGNORE on (document_version_id, chunk_index, chunk_hash))或走 ingest.py 去重
5. Runner 加固:fail-fast 后 `pool.shutdown(wait=False, cancel_futures=True)` + 立即退出,消灭 20-min 楔死
6. 收敛预算:_SWEEP_MAX_PAIRS 750 → 按库规模自适应;补不动点确认轮(末轮 merged=0 才停)

## 局限

- n=3 scopes(137 docs/臂):效率/收敛趋势可信(大 n),质量轴无功效
- 三臂共享端点跑,墙钟不可比(用户定的 tokens 口径已规避)
- B2 的收敛 flush 在 resume 时对已完成 scope 无条件重跑(每次重试 ~16 min 开销)——落地时应跳过已收敛 scope
