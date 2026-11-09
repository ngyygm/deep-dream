# RUNBOOK_GAP_AB — no-graph 臂评测机执行手册

评测机: `/Users/admin/Desktop/exa/deep-dream` (Mac, admin)
为什么在这台机器: K3 端点 `sz-infer.x2robot.cn` 只在评测机网络可达 (本机 TCP 80/443 均超时)。
主轨端点 `inf-dddgcurffgz3366q` (32k 窗) — 与 v2 冻结主轨一致; 计分器默认即此, 无需改。

---

## 0. 前置检查 (一次性)

```bash
cd /Users/admin/Desktop/exa/deep-dream
git status --short                 # 确认工作区干净
git log --oneline -1               # 记下当前 commit (账本用)
```

**把 harness 改动带过来** (二选一):

- 若本仓库改动已推送到远端: `git pull`
- 否则把 `research/paper/rebuttal/gapab_harness.patch` 拷到评测机后:
  ```bash
  git apply gapab_harness.patch
  python3 -m pytest research/benchmark/tests/test_benchmark_harness.py -q   # 须 41 passed
  ```

改动内容: `agentic.py` (AGENT_TOOLSETS/动态 TOOL_PROMPT/allowed_tools), `runner.py`
(装配点 + manifest 记录 agent_toolset), 测试 (+no-graph 回归)。默认行为不变 (full),
不触碰任何冻结结果。

## 1. 造 no-graph 配置

复制 v2 当时用的 service_config (主轨那份: model kimi-k3, base_url
`…/infer/inf-dddgcurffgz3366q/v1`, think false, temperature 0, max_tokens 16384),
只加一个键:

```bash
python3 - <<'EOF'
import json
cfg = json.load(open("research/service_config.json"))   # ← 换成 v2 主轨实际用的配置文件
cfg.setdefault("llm", {})["agent_toolset"] = "no-graph"
json.dump(cfg, open("research/service_config.nograph.json", "w"), indent=2)
print("ok:", cfg["llm"]["agent_toolset"], cfg["llm"]["base_url"])
EOF
```

## 2. 跑 no-graph 臂 (对 v2 冻结库, 不重新 ingest)

```bash
PYTHONPATH=. python3 -m research.benchmark.cli evaluate \
  research/.benchmark_runs/memoryagentbench-kimik3-sample-v2 \
  --config research/service_config.nograph.json \
  --track skill-agent --result-tag no-graph-v1 \
  --max-agent-steps 8 --context-k 5 --answer-profile normalized-v1 \
  --answer-workers 4 --resume
```

- 产物: `results.skill-agent-no-graph-v1.jsonl` (+retrieval 缓存), manifest 的
  track_variants 自动记 `agent_toolset: no-graph`。
- 断点续跑: 同命令重复执行即可 (--resume 按 question_id 跳过已完成)。
- 中断检查: `jq -r 'select(.status!="completed") | .question_id' \
  research/.benchmark_runs/memoryagentbench-kimik3-sample-v2/results.skill-agent-no-graph-v1.jsonl | wc -l` 须为 0。
- 注意: v1 库 (`…-sample-v1`) 上**不要**跑 — 对照只用 v2 冻结的 71.0/70.8。
- 并发: 4 workers 是保守值; 整机空可升 8 (ar-000 建议 ≤4)。

## 3. 官方计分 (与论文同协议: kimi-k3 judge, 双口径与 frozen 数同表)

```bash
PYTHONPATH=. python3 -m research.benchmark.score_memoryagentbench_official \
  research/.benchmark_runs/memoryagentbench-kimik3-sample-v2 \
  --track skill-agent-no-graph-v1 --sampled
```

- 产物: `memoryagentbench_summary.skill-agent-no-graph-v1.kimik3-official-v1.sampled.json`
- 若 judge 端点要显式指定, 加 `--base-url http://sz-infer.x2robot.cn/infer/inf-dddgcurffgz3366q/v1`

## 4. 回传 (回本机后我来做对照分析)

打包三个文件回传:

```bash
cd research/.benchmark_runs/memoryagentbench-kimik3-sample-v2
tar czf /tmp/nograph-v1.tgz \
  results.skill-agent-no-graph-v1.jsonl \
  retrieval.skill-agent-no-graph-v1.jsonl \
  memoryagentbench_summary.skill-agent-no-graph-v1.kimik3-official-v1.sampled.json \
  memoryagentbench_scores.skill-agent-no-graph-v1.kimik3-official-v1.sampled.jsonl \
  run_manifest.json
```

(回传后本机做: 与 71.0/70.8 的配对 bootstrap CI; calls/tokens 预算列
(retrieval_prompt_tokens/completion_tokens 均值, 对照 paper 的 10.2 calls/328.6k);
分能力 AR/TTL/LRU/SF 与 MCC/FC-MH 拆分。)

## 5. 红线

- 不要动 v1/v2 已冻结的任何 results/retrieval/manifest 字段 (新臂全部走 -no-graph-v1 标签)。
- 跑完把评测机上 git apply 的改动**提交或记入 manifest** (commit hash 写进
  track_variants.created_at 旁), 否则新臂不可引用。
- 若中途改主意要停: 已完成部分保留, 告知实际完成题数即可。
