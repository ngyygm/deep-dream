# MEME paper baselines

来源：Jung et al., *MEME: Multi-entity & Evolving Memory Evaluation*, arXiv:2605.12477. [Paper](https://arxiv.org/abs/2605.12477) · [dataset](https://huggingface.co/datasets/meme-benchmark/MEME)

MEME 有 100 episodes、六类任务：Exact Recall (ER)、Aggregation (Agg)、Tracking (Tr)、Deletion (Del)、Cascade (Cas)、Absence (Abs)。论文基线统一使用 `gpt-4.1-mini` 做内部记忆处理和回答，GPT-4o 做 judge。

## Systems evaluated in the paper

| Paradigm | Systems |
|---|---|
| Raw retrieval | BM25, text-embedding-3-small |
| LLM-processed memory | Mem0, Graphiti |
| File-based agent | Karpathy Wiki, MD-flat |

论文主结论（六系统平均）：Cascade **0.03**、Absence **0.01**；Aggregation 平均约 **0.23**；六系统中最高 overall 约 **0.42**（MD-flat）。唯一明显突破依赖推理的是 MD-flat + Claude Opus 4.7：Cascade **0.32**、Absence **0.59**，但成本约为 baseline 的 70×，并伴随 Exact Recall / Tracking 回退。

## Intervention values reported in the paper

| System | top-k | Cas | Abs |
|---|---:|---:|---:|
| BM25 | 5 / 10 / 20 / 40 | 0.02 / 0.00 / 0.02 / 0.02 | 0.07 / 0.15 / 0.24 / 0.21 |
| text-embedding-3-small | 5 / 10 / 20 / 40 | 0.02 / 0.02 / 0.00 / 0.00 | 0.15 / 0.19 / 0.23 / 0.15 |
| Mem0 | 5 / 10 / 20 / 40 | 0.00 / 0.00 / 0.02 / 0.00 | 0.04 / 0.02 / 0.02 / 0.02 |

Answerer swap `gpt-4.1-mini → Sonnet 4` only modestly improves Absence and does not solve Cascade: BM25 0.00→0.12 Abs, dense 0.00→0.16 Abs, Mem0 0.00→0.00 Abs; Cascade remains 0.01–0.04.

## Project anchors

项目的 `meme-filler32k-qwen35-full100-k3-v1` 保存了 100-episode before/after 运行和 judge summaries。其结果必须和论文 baseline 分开，因为 judge、answer model 和 pipeline condition 不同。

## Project comparison

在 filler32k 条件下，Kimi-K3 bash-files summary 的 raw after accuracy 是 **0.7248**；任务分解为 ER **1.00**、Agg **0.91**、Tr **0.97**、Del **0.02**、Cas real/trivial-filtered **0.1098**、Abs **0.6692**。同一目录的另一份 judge summary 为 raw after **0.6225**，体现了 judge/protocol 敏感性。与论文共同的稳定信号是 Deletion/Cascade 困难；Absence 不应只看 raw pass，而应优先看 trivial-filtered/real accuracy。
