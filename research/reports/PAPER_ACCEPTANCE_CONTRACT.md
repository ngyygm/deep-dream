# Paper Acceptance Contract

status: proposed  
rounds: 0  
reviewer: pending

## Testable assertions

1. The title and abstract describe Deep-Dream as source-grounded/auditable and do not claim strict state of the art or semantic entailment guarantees.
2. Every numeric claim in the abstract, main text, figures, tables, and appendix appears in `paper/results/benchmark_summary.json` or a cited external primary table, with numerator/denominator, metric, subset, answerer, judge, prompt/config provenance, and a SHA-256-listed per-item artifact where available.
3. LoCoMo 1,540 results are labeled binary cross-judge accuracy and never compared numerically with Mem0 paper rubric `J` as if they were the same metric.
4. LongMemEval-S 84.60% is labeled as a 500-question Qwen3.7 cross-judge result; Zep's 63.8/71.2 figures retain their GPT-4o-mini/GPT-4o labels.
5. The paper never uses “outperforms Mem0”, “outperforms Zep”, “state of the art”, or an equivalent superiority claim without a controlled same-answerer/judge experiment.
6. The 105-question source-expansion experiment is identified as a difficult/error subset, not the full LoCoMo test set.
7. Figure 2 values exactly match the channel and source-expansion entries in `paper/results/benchmark_summary.json`; the expansion comparison holds lexical+semantic channels fixed and labels the 105 cases as selected prior errors.
8. No result with a failed expected integrity invariant enters a figure, headline table, abstract, or positive conclusion; the failed depth replay remains only as an appendix integrity note.
9. Before submission, evidence-gate tests must cover unread IDs, cross-scope IDs, inactive versions, false-premise/unanswerable cases, and answer-evidence consistency; otherwise “evidence-bounded” cannot appear as a validated contribution.
10. The method section defines source evidence, derived memory, version family, observation/assertion, scope, and evidence submission, and states that scope-valid IDs do not guarantee semantic entailment.
11. Before submission, current/as-of/ever/never, conflict, retraction, deletion leakage, and source-only→overlay ablations must be completed before versioning can be a headline contribution; otherwise versioning remains an implementation detail.
12. MEME deletion accuracy of 0% and cascade/absence limitations appear in the main limitations discussion or a referenced main-text table, not only in supplementary material.
13. Latency is reported only for Deep-Dream's measured environment and is not visually compared with Zep service latency as a hardware-controlled result.
14. Every citation key in the final LaTeX resolves to a verified primary bibliographic record, and every cited source supports its surrounding claim.
15. The final PDF contains no author identity, TODO/FIXME/VERIFY markers, undefined references, or undefined citations.
16. The main body fits the ICLR 9-page limit through the end of Conclusion; references and appendix are outside this count.
17. The appendix records selection rules for the 210/105 samples, gold-access policy, dataset hashes, answerer and judge identifiers, temperatures, judge prompt provenance, code/runtime/library hashes, error counts, and public/supplement artifact paths sufficient to audit every main table.
18. Submission is blocked until causal provenance/evidence-gate tests, the update/delete microbenchmark, and a controlled Mem0/Graphiti comparison are complete, or until the corresponding mechanism and relative-performance claims are removed from the title, abstract, and contributions.
19. MEME is labeled as 694 heterogeneous after-task checks drawn from 100 episodes; each task denominator is shown and raw versus real scoring is never conflated.
20. Channel diagnostics are described only for the 210-question development/gate set; no general reliability or significance claim is made without paired uncertainty analysis.
21. Main-text floats are limited to Figure 1, Figure 2, one main result table, and one core mechanism/safety table; other protocol, category, and MEME tables move to the appendix.

## Disputed

None at proposal time.
