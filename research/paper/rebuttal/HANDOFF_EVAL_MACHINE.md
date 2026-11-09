# HANDOFF_EVAL_MACHINE — 评测机全量交接（ICLR rebuttal 补证据）

更新: 2026-11-09 · 代码基线: `feat/fullctx-anchor` @ **4099a0a8** · 文件服务器: `http://219.223.187.56:8899/`
对象: 评测机 `/Users/admin/Desktop/exa/deep-dream`（Mac, admin）
为什么在这台机: K3 端点 `sz-infer.x2robot.cn` 只在评测机网络可达。

---

## 0. 背景：评委要什么（30 秒版）

ICLR 2027 Submission44989（Deep-Dream 论文）三人评审 f1J5=2 / mV1D=1 / cB3k=1。
可实验解决的核心质疑只有一组，**全部落在下面 P0 这一个新臂上**：

| 评委原话（大意） | 一句话 | 谁来做 |
|---|---|---|
| mV1D-Q3 / cB3k: "online ablation … without the graph" | 同一 agent、同预算、同冻结库，把 5 个图工具从菜单里拿掉，分数掉多少 | **评测机 P0** |
| mV1D-Q4: "would a simpler iterative RAG … same source access and computational resources" | 同预算裸迭代 RAG 能否打平 | **同一个 P0 臂**（源无损存储 ⇒ 库上检索=裸语料 RAG） |
| mV1D-C2 / cB3k: 合并无 ground-truth 评估 | 已在本机完成（两代引擎过合并均 0，碎片化 2.3%→1.5%） | 已做完，无需跑 |
| 漏引 PlugMem/AriGraph/G-Memory/GAM、伪代码入正文、图重绘 | 写作层，camera-ready 承诺 | 论文侧，不占评测机 |
| 与 PlugMem 正面对比 / Mem0 外部基线 / version-provenance 消融 | **本轮明确不做**（基准零重叠 / 未设计 / 讨论期内不可行），见仓库 `REVISION_PLAN.md` out-of-scope log | 无人做，非遗漏 |

---

## 1. 任务（按顺序做）

### P0 · no-graph 臂（必做；产出直接进 rebuttal 回帖）

**第 0 步 · 拉代码到 4099a0a8 并自检**

```bash
cd /Users/admin/Desktop/exa/deep-dream
git status --short                      # 须干净
curl -fL -o /tmp/dd.bundle http://219.223.187.56:8899/dd-full.bundle
git pull /tmp/dd.bundle feat/fullctx-anchor
git log --oneline -1                    # 必须是 4099a0a8
python3 -m pytest research/benchmark/tests/test_benchmark_harness.py -q   # 须 41 passed
```
拉不动 bundle 就改用 `curl -O http://219.223.187.56:8899/gapab_harness.patch` + `git apply gapab_harness.patch`（等价）。

**第 1 步 · 造 no-graph 配置**（在 v2 主轨 service_config 基础上只加一个键）

```bash
python3 - <<'EOF'
import json
cfg = json.load(open("research/service_config.json"))   # ← 换成 v2 主轨当时实际用的配置文件名
cfg.setdefault("llm", {})["agent_toolset"] = "no-graph"
json.dump(cfg, open("research/service_config.nograph.json", "w"), indent=2)
print("ok:", cfg["llm"]["agent_toolset"], cfg["llm"]["base_url"])
EOF
```
（配置要求: model kimi-k3 · base_url `…/infer/inf-dddgcurffgz3366q/v1` · think false · temperature 0 · max_tokens 16384）

**第 2 步 · 跑**（对 **v2** 冻结库；绝不碰 v1 库；断点续跑=同命令再执行）

```bash
PYTHONPATH=. python3 -m research.benchmark.cli evaluate \
  research/.benchmark_runs/memoryagentbench-kimik3-sample-v2 \
  --config research/service_config.nograph.json \
  --track skill-agent --result-tag no-graph-v1 \
  --max-agent-steps 8 --context-k 5 --answer-profile normalized-v1 \
  --answer-workers 4 --resume
```

**第 3 步 · 官方计分**（judge kimi-k3，与论文同协议）

```bash
PYTHONPATH=. python3 -m research.benchmark.score_memoryagentbench_official \
  research/.benchmark_runs/memoryagentbench-kimik3-sample-v2 \
  --track skill-agent-no-graph-v1 --sampled
```

**第 4 步 · 校验 + 登记**

```bash
cd research/.benchmark_runs/memoryagentbench-kimik3-sample-v2
jq -r 'select(.status!="completed") | .question_id' results.skill-agent-no-graph-v1.jsonl | wc -l   # 须为 0
cd /Users/admin/Desktop/exa/deep-dream
git add -A && git commit -m "bench(eval): no-graph-v1 arm artifacts + config"   # 记下 commit hash
```

### P1 · 回传（两种方式二选一）

**方式 A（首选）· scp 直写本机收件箱**（若 Mac 能 ssh 通 219.223.187.56）：

```bash
cd research/.benchmark_runs/memoryagentbench-kimik3-sample-v2
tar czf /tmp/nograph-v1.tgz \
  results.skill-agent-no-graph-v1.jsonl \
  retrieval.skill-agent-no-graph-v1.jsonl \
  memoryagentbench_summary.skill-agent-no-graph-v1.kimik3-official-v1.sampled.json \
  memoryagentbench_scores.skill-agent-no-graph-v1.kimik3-official-v1.sampled.jsonl \
  run_manifest.json
scp /tmp/nograph-v1.tgz linkco@219.223.187.56:/home/linkco/deep-dream-inbox/
# 顺带回传 git 登记: 在仓库根 git bundle create /tmp/return.bundle 4099a0a8..HEAD
#   然后 scp /tmp/return.bundle linkco@219.223.187.56:/home/linkco/deep-dream-inbox/
```

**方式 B · 打包走任意渠道**（ssh 不通时）: 把 `/tmp/nograph-v1.tgz` + `/tmp/return.bundle`
（git 登记同上）通过网盘/微信发回，收件人放到 `219.223.187.56:/home/linkco/deep-dream-inbox/` 或告知路径即可。

---

## 2. 数据落盘位置（精确路径）

| 什么 | 在哪 |
|---|---|
| 评测机产物（5 个文件） | `research/.benchmark_runs/memoryagentbench-kimik3-sample-v2/` 下 `results./retrieval./summary./scores.` 的 `*no-graph-v1*` + `run_manifest.json` |
| 回传落地（本机收件箱） | `linkco@219.223.187.56:/home/linkco/deep-dream-inbox/` |
| 校验后正式入库（**由论文侧做，评测机不要直接写**） | 同 run 目录 + 分析结果写 `research/paper/rebuttal/` |
| 回传后论文侧做什么 | 与冻结 71.0@1074 / 70.8@614 的配对 bootstrap CI；calls/tokens 预算列（对照 10.2 calls / 328.6k）；分能力 AR/TTL/LRU/SF 与 MCC/FC-MH 拆分；回填 rebuttal 占位 |

## 3. 对照锚点（已冻结，不许改不许重跑）

source 臂 71.0@1074 / 70.8@614 · tool 臂 40.0 · fixed 臂 33.3 · full-ctx 上界 78.7@614k
· 成本 10.2 calls / 328.6k tokens · X7 离线 null ±2.4

## 4. 红线（必读）

1. v1/v2 已冻结的任何 results/retrieval/manifest 字段**不许动**；新臂只走 `no-graph-v1` 标签。
2. 只在 **v2** 库上跑（`memoryagentbench-kimik3-sample-v2`），v1 库（`…-sample-v1`）不要碰。
3. 结果无论好坏都必须如实回传——null 结果也是结果，论文侧按 null 如实写。
4. 中途要停：保留已完成部分，报实际完成题数即可（`--resume` 支持续跑）。
5. git 登记不可省：评测机上的 commit hash（或 return.bundle）必须随数据回传，否则新臂不可引用。

## 5. 故障与联系

- 8899 端口拉不动 → 告知论文侧（备选: ssh 22 直拉，或 patch 走别的渠道）
- K3 端点失败/限流 → 同命令 `--resume` 重试；持续失败就停下报错
- 判分中断 → 计分命令可直接重跑（按题幂等）
