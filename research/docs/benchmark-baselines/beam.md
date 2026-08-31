# BEAM paper baselines

来源：Tavakoli et al., *Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs*, ICLR 2026. [arXiv](https://arxiv.org/abs/2510.27246) · [project page](https://mohammadtavakoli78.github.io/beam-light/)

BEAM 有 100 conversations、2,000 validated questions、10 memory abilities和四个规模层级：100K、500K、1M、10M。指标是 LLM rubric/nugget mean（0/0.5/1），不能与 LoCoMo/LongMemEval 的 binary accuracy 直接换算。

论文方法为 LIGHT；其对比方法是 Vanilla full-context 和 RAG。论文 project page 给出四种 backbone 的平均结果：

| Tier | Backbone | Vanilla | RAG | LIGHT |
|---|---|---:|---:|---:|
| 128K | Qwen 2.5 | 0.280 | 0.269 | 0.311 |
| 128K | Llama 4 Maverick | 0.240 | 0.323 | 0.358 |
| 128K | Gemini 2 Flash | 0.242 | 0.280 | 0.294 |
| 128K | GPT-4.1-nano | 0.239 | 0.309 | 0.345 |
| 500K | Qwen 2.5 | 0.200 | 0.291 | 0.316 |
| 500K | Llama 4 Maverick | 0.283 | 0.330 | 0.359 |
| 500K | Gemini 2 Flash | 0.257 | 0.267 | 0.292 |
| 500K | GPT-4.1-nano | 0.194 | 0.314 | 0.335 |
| 1M | Qwen 2.5 | 0.193 | 0.285 | 0.309 |
| 1M | Llama 4 Maverick | 0.259 | 0.307 | 0.336 |
| 1M | Gemini 2 Flash | 0.199 | 0.271 | 0.284 |
| 1M | GPT-4.1-nano | 0.191 | 0.302 | 0.336 |
| 10M | Qwen 2.5 | 0.133 | 0.211 | 0.238 |
| 10M | Llama 4 Maverick | 0.104 | 0.249 | 0.266 |
| 10M | Gemini 2 Flash | 0.122 | 0.216 | 0.192 |
| 10M | GPT-4.1-nano | 0.109 | 0.218 | 0.226 |

论文摘要报告 LIGHT 相对最强 baseline 的平均提升约 3.5%–12.69%，并通过 episodic memory、working memory 和 scratchpad 的 ablation 验证三者贡献。

## Published 10M reference table

为便于对照项目的 `beam-100k` 运行，公开 BEAM baseline 汇总为：RAG **24.9%**、LIGHT **26.6%**、Honcho **40.6%**、Hindsight **64.1%**（10M）。这组扩展方法不是 BEAM 原论文主实验全部内容，已单独标为 published reference。

## Project comparison

项目 `beam-100k-qwen35-full-remember-v1` 目前只有 remember pipeline manifest 和停止状态，没有完成的 BEAM rubric/nugget score。因此暂不与论文的 128K/10M 数字做虚假比较；补齐 score summary 后再按相同 tier、backbone 和 judge 对齐。
