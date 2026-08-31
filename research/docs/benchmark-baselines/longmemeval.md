# LongMemEval-S paper baselines

来源：Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, ICLR 2025. [arXiv](https://arxiv.org/abs/2410.10813) · [official project page](https://xiaowu0162.github.io/long-mem-eval/)

原始 benchmark 有五类能力：information extraction、multi-session reasoning、knowledge updates、temporal reasoning、abstention；LongMemEval-S 每题约 115K tokens，500 questions。论文的主 pilot 是 97-question commercial comparison 和 LongMemEval-S long-context comparison。

## Paper pilot numbers

| Setting | Model | Oracle evidence | Full LongMemEval-S | Drop |
|---|---|---:|---:|---:|
| No Chain-of-Note | GPT-4o | 0.870 | 0.606 | 30.3% |
| No Chain-of-Note | Llama 3.1 8B Instruct | 0.744 | 0.334 | 55.1% |
| No Chain-of-Note | Phi-3 Medium 128K | 0.702 | 0.380 | 45.9% |
| Chain-of-Note | GPT-4o | **0.924** | **0.640** | 30.7% |
| Chain-of-Note | Llama 3.1 8B Instruct | 0.848 | 0.286 | 66.3% |
| Chain-of-Note | Phi-3 Medium 128K | 0.722 | 0.344 | 52.4% |

Commercial pilot on 97 questions: offline reading GPT-4o 0.9184; ChatGPT with GPT-4o 0.5773 (GPT-4o-mini 0.7113); Coze with GPT-4o 0.3299 (GPT-3.5-turbo 0.2474).

## Paper design findings

- round-level indexing outperformed session-level storage;
- expanding keys with extracted user facts improved recall@k by about 4% and downstream QA by about 5 points;
- time-aware indexing/query expansion improved temporal recall by roughly 7–11%;
- Chain-of-Note and structured JSON reading improved answer accuracy by up to 10 points.

## Project anchors

`.benchmark_runs/longmemeval-source-v22-gate50/` contains local 12/20/50-question gates. In addition, the v24 run has a completed 500-question judge summary at `.benchmark_runs/longmemeval-source-v24-full500/judge_summary.kimi-agent-direct-longmem500-memory-v24-full.qwen37-longmem-official-memory-v24-full500.json`.

## Project comparison

项目 v24 memory 的完整 500 题结果为 **0.846**（423/500，Qwen3.7-plus cross-judge）；六类分别为 knowledge-update 91.03%、multi-session 74.44%、single-session-assistant 100%、single-session-preference 53.33%、single-session-user 98.57%、temporal-reasoning 84.21%。另有早期 full-50 gate 为 0.88。原论文 GPT-4o full LongMemEval-S 为 **0.640**（500 题），Zep 论文报告 GPT-4o 为 **0.712**。项目使用不同 answerer/judge 与 ingestion，因此应写成“同一 500 题数据上的 cross-judge 结果”，不能宣称严格超过官方 GPT-4o/Zep 成绩。
