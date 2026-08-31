# research/ — 实验测评与论文工作区

这里收纳 Deep-Dream 仓库中**与系统本体无关**的两类内容：长期记忆基准测评（LoCoMo / LongMemEval-S / MEME / MemoryAgentBench 等）和 ICLR 论文工程。Deep-Dream 系统本体在仓库根目录（`core/`、`library/`、`docs/` 等），提交系统时不需要本目录。

## 目录结构

| 路径 | 内容 |
|------|------|
| `benchmark/` | 测评 harness 代码（原 `core/benchmark/` + 原 `deep-dream benchmark` CLI），依赖系统包 `core.*`，反向无依赖 |
| `benchmark/tests/` | harness 单元测试（`pytest research/benchmark/tests`） |
| `docs/` | 调研文档：LoCoMo/Kimi 评测记录、记忆基准综述、`.bib` 文献库、基线调研（`benchmark-baselines/`）、研究提案 |
| `paper/` | ICLR 2026 LaTeX 工程（`main.tex`、`sections/`、`figures/`、`results/`） |
| `reports/` | 论文流水线阶段报告：MANIFEST、NARRATIVE_REPORT、PAPER_PLAN、PAPER_ACCEPTANCE_CONTRACT（含时间戳副本） |
| `idea-stage/` `refine-logs/` `review-stage/` | 论文流水线的 idea / 实验计划与跟踪 / 自动评审阶段产物 |
| `service_config.kimi*.json` | 基准机器专用配置（内部端点，勿提交） |
| `.benchmark_data/` | 数据集文件（locomo、longmemeval、meme、beam 等） |
| `.benchmark_runs/` | 历次评测运行输出（34GB，含 judge 结果与 frozen library） |
| `.benchmark_runtime/` | 外部运行时（kimi-cli、`code-snapshots/` 冻结代码快照——**取证存档，勿改**） |
| `.benchmark_refs/` | 基线参考实现（内嵌 git clone：MemoryAgentBench、MEME-public） |
| `.aris/` | 自动评审 / 论文流水线的运行痕迹 |
| `.etc-b3-host-migration-evidence-*/` | etc-b3 主机迁移实验的取证快照 |

数据目录均被根 `.gitignore` 按目录名忽略（在任意层级都生效）。

## 跑测评

在**仓库根目录**执行（harness 导入 `core.*` 系统包，系统配置 `service_config*.json` 也是按根目录相对路径解析）：

```bash
# 安装 harness 依赖（系统本体先 pip install -e .）
pip install "nltk>=3.8" "regex>=2023.0" "tqdm>=4.66"   # 基础
pip install "fastmcp==3.2.4"                            # MCP agent 轨道
pip install acp                                          # kimi_persistent_bridge 可选

# 原 `deep-dream benchmark ...` 子命令的等价入口：
python -m research.benchmark.cli --help

# 数据集准备 / 摄取 / 评测（数据目录默认在本目录下的 .benchmark_data、.benchmark_runs）
python -m research.benchmark.cli prepare --dataset locomo
python -m research.benchmark.cli ingest --dataset locomo --run-dir research/.benchmark_runs/locomo-conv26 \
  --config service_config.local.json --scope-id conv-26 --session-limit 6

# replay / 消融脚本（输出到 research/.benchmark_runs/locomo-full-quality-v1/）
python -m research.benchmark.retrieval_replay
python -m research.benchmark.replay_budget_frontier 16
```

路径约定：代码内数据目录一律锚定 `research/`（`Path(__file__).resolve().parents[1]`），系统配置按 CWD（仓库根）解析。
