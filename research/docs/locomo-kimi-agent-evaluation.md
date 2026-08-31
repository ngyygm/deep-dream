# LoCoMo Kimi Agent 评测运行说明

本文记录 Deep-Dream 将 Kimi CLI 作为外部 Agent 运行时、以 Qwen3.7-plus 自主查询 LoCoMo 记忆库的实验协议。Kimi 源码和用户全局配置不进入项目；固定版本安装在 Git 忽略的隔离目录中。

## 实验轨道

- `kimi-agent-direct`：Kimi Agent 查询 Deep-Dream 后直接给出答案。
- `kimi-agent-evidence`：Kimi Agent 只负责提交可回溯证据，再由独立 Qwen3.7-plus `normalized-v1` 回答器作答。

两条轨道共享冻结的 10 个 conversation、272 个 active session。Agent 不可见 QA category、参考答案、官方 evidence ID 或其他 conversation。

## 安装与预检

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[benchmark,benchmark-agent]'
deep-dream benchmark runtime install --runtime kimi --version 1.49.0
deep-dream benchmark runtime check --runtime kimi --version 1.49.0
```

运行时位于 `.benchmark_runtime/kimi-cli/1.49.0/`。安装命令生成入口文件 SHA-256、来源和环境规范哈希，之后不会自动升级。DashScope 密钥通过 `SCMR_QWEN_API_KEY` 环境变量或 macOS Keychain 注入；日志、manifest 和 trajectory 均不得保存密钥。

`service_config.qwen37.local.json` 至少应配置：`llm.model=qwen3.7-plus`、`llm.base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`、`llm.api_key_env=SCMR_QWEN_API_KEY`、`llm.answer_extra_body.enable_thinking=false`、`llm.temperature=0`，以及 `all-MiniLM-L6-v2` CPU embedding。该文件是本地配置且不应写入密钥。Kimi 1.49.0 的 `openai_legacy` provider 没有暴露 Agent temperature override，因此 manifest 将 Agent 的有效温度明确记为 `provider-default`；独立回答器仍使用配置中的 temperature 0。

若没有冻结 run，先执行 `deep-dream benchmark prepare --dataset locomo`，再用 `benchmark ingest` 建立隔离 library。本文实验使用的冻结数据是 `.benchmark_data/locomo10.json`（其 SHA-256 与 Git commit 都记录在 `run_manifest.json`），正式比较不得混用重新下载但哈希不同的数据。

## 查询、复答与恢复

```bash
deep-dream benchmark agent-query .benchmark_runs/locomo-full-quality-v1 \
  --runtime kimi \
  --config service_config.qwen37.local.json \
  --mode direct --mode evidence \
  --agent-model qwen3.7-plus \
  --agent-thinking off \
  --max-agent-steps 12 \
  --qa-workers 2 \
  --result-tag kimi-qwen37-thinking-off \
  --resume

deep-dream benchmark answer .benchmark_runs/locomo-full-quality-v1 \
  --source-track kimi-agent-evidence-kimi-qwen37-thinking-off \
  --config service_config.qwen37.local.json \
  --answer-profile normalized-v1 \
  --result-tag qwen37-answer-v1 \
  --resume
```

`agent-evaluate` 是以上两阶段的快捷命令。查询结果逐题追加写入 JSONL；`--resume` 只重试缺失或失败题目。证据轨道可以反复更换回答 prompt，而不会重新运行 Kimi 查询。

每题使用独立 `KIMI_SHARE_DIR`，custom agent 的内置工具列表为空，只注册 conversation-scoped 的九个只读 Deep-Dream MCP 工具。`submit_evidence` 会拒绝没有被本题工具调用实际返回、属于其他 scope、或来自非 active/incomplete 文档的 ID。

## 审计产物

- `retrieval.kimi-agent-evidence-*.jsonl`：证据、session/turn 排名和可回溯原文。
- `results.kimi-agent-direct-*.jsonl`：Agent 直接答案。
- `results.kimi-agent-evidence-*-qwen37-answer-v1.jsonl`：固定证据的规范化答案。
- `trajectories/<variant>/`：工具调用、脱敏观察、token、延迟、退出码和停止原因；不保存隐藏思维文本。
- `run_manifest.json` schema v5：Kimi 版本、入口哈希、MCP/策略/适配器哈希、模型、thinking 和最大步数。

报告同时保留 Token-F1 和 Qwen3.7-plus cross-judge。cross-judge 只用于内部一致 A/B，不能冒充采用 GPT-5 系列 Judge 的 Mem0/Zep 官方可比成绩。

## 全量结果快照（2026-07-23）

本次正式运行使用 `kimi-agent-direct-qwen37-full-thinking-off`，即 Kimi CLI
内的 `qwen3.7-plus` Agent、thinking off、每题全新上下文。冻结数据哈希为
`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`，
运行起点 Git commit 为 `d9f299259fd06761305659f939494875f5ba177b`。

| 轨道 / 评分协议 | 样本数 | 总分 | Multi-hop | Temporal | Open-domain | Single-hop |
|---|---:|---:|---:|---:|---:|---:|
| Kimi direct，LoCoMo Token-F1（含 adversarial） | 1,986 | 22.7520 | 25.9590 | 19.6684 | 18.4144 | 35.0579 |
| Kimi evidence → Qwen3.7 normalized answer，Token-F1 | 1,986 | 68.3699 | — | — | — | — |
| Kimi direct，旧版缩写 Judge prompt，Qwen3.7-plus | 1,540 | 92.2078 | 90.4255 | 91.9003 | 72.9167 | 95.1249 |
| Kimi direct，Mem0-current 精确 Judge prompt，Qwen3.7-plus | 1,540 | **93.1818** | **91.8440** | **94.3925** | **73.9583** | **95.3627** |

精确 prompt 共判对 1,435/1,540 题，0 个运行错误，平均 Judge 延迟
1.6903 秒。相对旧版缩写 prompt，总分提高 0.9740 个百分点：16 题由错变对，
1 题由对变错。分类变化分别为 multi-hop +1.4184、temporal +2.4922、
open-domain +1.0417、single-hop +0.2378 个百分点。

精确 Judge prompt 来自 `mem0ai/memory-benchmarks` commit
`4b61c5d31b9c668a12b4f5e78064248a02c82d2b` 的 unified
no-evidence 模式，逐字 prompt SHA-256 为
`16bed642e415fb8f5c8550cd733dcf19985ad157427e40ab5dac209e7b837a56`。
这次 A/B 说明 Judge 指令宽松度确实会影响分数，但影响约为 1 个百分点，
不足以单独解释 90% 以上的结果。

可复现产物：

- 精确 prompt 逐题结果：`.benchmark_runs/locomo-full-quality-v1/judge_results.kimi-agent-direct-qwen37-full-thinking-off.qwen37-mem0-current-exact-direct.jsonl`
- 精确 prompt 汇总：`.benchmark_runs/locomo-full-quality-v1/judge_summary.kimi-agent-direct-qwen37-full-thinking-off.qwen37-mem0-current-exact-direct.json`
- 旧版缩写 prompt 汇总：`.benchmark_runs/locomo-full-quality-v1/judge_summary.kimi-agent-direct-qwen37-full-thinking-off.qwen37-kimi-full.json`
- 供 GPT-5 外部评测的 1,540 题包：`.benchmark_runs/locomo-full-quality-v1/gpt5-judge-export/locomo1540-kimi-direct-mem0-current-gpt5-judge.json`

上述 93.1818% 是 **Mem0-current 精确 prompt + Qwen3.7-plus Judge** 的
cross-judge 结果，不是 GPT-5 官方可比成绩。GPT-5 API 冒烟请求因项目额度
不足（HTTP 429 `insufficient_quota`）而未启动全量评测。

## Direct 轨道究竟如何产生答案

每道题在外层只执行一次隔离的 Kimi Agent 任务，但它不是“一次 `find` 后立即
回答”。任务内部由 Agent 根据前一步结果自主进行多步工具调用，可使用
`search_documents`、`explore_memory`、`search_concepts`、
`trace_concept`、`expand_neighbors`、`relation_evidence`、
`read_episode` 和 `read_session`，最后必须调用 `submit_evidence`。

同一次 Agent 执行同时产生：

1. `direct`：直接保存 Agent 最终结构化输出中的 `answer`。
2. `evidence`：保存 Agent 提交并通过作用域校验的 session、episode、turn
   及原文，供独立回答器在不重新检索的情况下复答。

因此 direct 的准确描述是“**单题一次 Agent 会话，内部自主多步检索并回答**”；
不是固定的一次性数据库召回，也不是检索 Agent 与 direct 回答 Agent 各跑一遍。
常驻 8-worker 只复用 Kimi 进程和环境以减少启动开销，每道题仍创建
`fresh_context=true` 的新会话，题目之间不共享对话上下文。
