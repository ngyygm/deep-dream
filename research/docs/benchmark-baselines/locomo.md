# LoCoMo paper baselines

来源：Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM Agents*, ACL 2024. [Paper](https://aclanthology.org/2024.acl-long.747/) · [PDF](https://aclanthology.org/2024.acl-long.747.pdf)

原始 LoCoMo protocol 使用 1,986 道题，五类为 single-hop、multi-hop、temporal、open-domain、adversarial，主要指标是 normalized token-F1。

| Method | Reader | Overall token-F1 |
|---|---|---:|
| Human | human | 87.9 |
| Full context | GPT-4 Turbo 128K | 51.6 |
| Full context | Claude 3 Sonnet | 42.8 |
| Full context | Gemini 1.0 Pro | 39.1 |
| Dialog RAG top-25 | GPT-3.5 + DRAGON | 41.0 |
| Observation RAG top-5 | GPT-3.5 + DRAGON | 43.3 |

论文观察：observation-level RAG 比直接检索长对话更有效；继续增加 top-k 可能引入噪声并降低分数。

## Protocol caveat

后来大量工作使用 1,540 题（去掉 adversarial）和 LLM-as-a-judge accuracy；这些数不能与上表的 token-F1 直接比较。项目已有更完整的协议审计和 A-MEM/ Mem0/ MemOS/ ENGRAM/ MemMachine 汇总：[locomo-memory-benchmark-review.md](../locomo-memory-benchmark-review.md)。

## Project anchors

`locomo-six-session-probe`：12 题，overall 0.6780；`locomo-conv26-clean-current/comparison.md`：baseline 0.5818、skill-agent 0.6641（均为本地 protocol，不是论文复现）。

## Project comparison

项目另有 quality-v1 full run（1,986 题、同为 token-F1）：baseline **64.99**、skill-agent **63.17**、skill-agent thinking-on **61.33**。相对论文的 GPT-4 Turbo full-context 51.6，数值上更高，但使用了不同 backbone、remember pipeline 和 answer prompt，只能作为同指标诊断。固定项目协议下 skill-agent 比 baseline 低 1.82 points。
