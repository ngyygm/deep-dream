# Experiment Results

## Main benchmark results

| Benchmark | Questions | Deep-Dream result | Evaluation scope |
|---|---:|---:|---|
| LoCoMo | 1,540 | 93.18% Qwen3.7; 93.57% Kimi-K3 | Mem0-compatible question set/prompt; cross-judge |
| LongMemEval-S | 500 | 84.60% | official-compatible prompt; Qwen3.7 cross-judge |
| LoCoMo-Plus | 401 | 64.59% | official cognitive prompt; Qwen3.7 cross-judge |
| MEME filler32k | 100 episodes | 62.25% raw after | Kimi-K3 judge; deletion 0%, therefore limitation rather than headline |

LongMemEval-S category scores are 91.03% knowledge-update, 74.44% multi-session, 100% single-session-assistant, 53.33% single-session-preference, 98.57% single-session-user, and 84.21% temporal-reasoning.

## Diagnostic experiments

### Retrieval channels (LoCoMo, 210 questions)

| Channels | Recall-any | Recall-all |
|---|---:|---:|
| Lexical | 92.38% | 75.24% |
| Semantic | 89.05% | 70.48% |
| Lexical + semantic | **92.86%** | **78.10%** |

Fusing channels improves recall-all by 2.86 points over lexical, while semantic-only underperforms lexical. This supports complementary candidate generation rather than embedding-only retrieval.

### Source-span expansion (105 difficult LoCoMo cases)

| Retrieval unit | Recall-any | Recall-all | Mean gold recall | Mean response bytes |
|---|---:|---:|---:|---:|
| Lexical + semantic span | 70.48% | 52.38% | 60.92% | 3,089 |
| Span + ±1 turn | 84.76% | 65.71% | 73.97% | 3,456 |
| Span + ±2 turns | **89.52%** | **73.33%** | **80.88%** | 3,609 |
| Legacy context-3 | 86.67% | 64.76% | 75.19% | 3,425 |

Holding lexical+semantic channels fixed, ±2 expansion gains 19.05 points recall-any and 20.95 points recall-all for 16.84% more response bytes. This is a 105-case prior-error development diagnostic, not a held-out final QA result.

### Invalidated retrieval-depth replay (LoCoMo, 210 questions)

| Depth | Recall-any | Recall-all | Mean response bytes |
|---:|---:|---:|---:|
| 10 | 92.86% | 77.62% | 3,986 |
| 20 | 95.24% | 84.29% | 7,367 |

The replay violates the expected top-k prefix invariant on 210/210 questions (`primary_result_invariant_passed=false`). It is retained as a negative integrity record and excluded from paper claims and figures until rerun.

## Integrity notes

- All figures above are copied or deterministically derived from the SHA-256-bound artifacts listed in `paper/results/benchmark_summary.json`.
- External Mem0/Zep values are not mixed into the same metric column unless answerer, judge, prompt and subset match.
- MEME Del/Cas/Abs results reveal weaknesses and are retained; they are not omitted from the paper.
