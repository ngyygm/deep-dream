# Paper Plan

**Title**: *Deep-Dream: Source-Grounded Long-Term Memory with Auditable Concept Overlays*  
**Venue**: ICLR  
**Type**: Method/system empirical paper  
**Date**: 2026-08-13  
**Page budget**: 9 pages to the end of Conclusion, excluding references and appendix  
**Section count**: 6 main sections + appendix

## One-sentence contribution

Deep-Dream separates memory navigation from factual authority: concept overlays guide retrieval, while submitted evidence IDs must refer to source spans actually read within the active scope; semantic answer support remains a separate evaluation target.

## Claims-Evidence Matrix

| Claim | Evidence | Status | Section |
|---|---|---|---|
| A source-grounded auditable overlay can support strong long-term-memory QA without treating extracted nodes as ground truth. | LoCoMo-1540 93.18/93.57 cross-judge; LongMemEval-S-500 84.60; architecture/code provenance | Supported as system-level observation; no strict causal attribution | §3, §4 |
| Hybrid candidate generation is more reliable than semantic-only retrieval. | 210-question channel ablation: fused recall-all 78.10 vs lexical 75.24 and semantic 70.48 | Supported | §4.3 |
| Neighboring source turns improve coverage on a development diagnostic when channels are held fixed. | 105 prior-error cases: lexical+semantic span vs +±2 turns yields +19.05 recall-any, +20.95 recall-all, and +16.84% bytes | Supported on selected development cases; final QA effect untested | §4.4 |
| The recorded depth replay is invalid as a prefix ablation. | prefix invariant failed 210/210 | Negative integrity finding; excluded from main results | Appendix |
| Evidence-bounded evaluation is more trustworthy than an unqualified single judge score. | exact-prompt +0.97 points; Qwen/Kimi gap 0.39 points; all raw outputs retained | Supported diagnostic observation | §5.1 |
| Versioning improves update/delete semantics. | LongMemEval update 91.03 and temporal 84.21; MEME deletion 0% | **Not causally supported**; current evidence exposes a gap | §5.2, limitations |
| Deep-Dream strictly outperforms Mem0/Zep. | No controlled same-answerer/judge run | Unsupported; prohibited claim | — |

## Structure

### §0 Abstract (180–220 words)

- Problem: compressed memories and graph edges can become unverifiable pseudo-facts.
- Approach: source-of-truth documents, auditable concept/relation overlay, hybrid navigation, scope-checked evidence submission.
- Evidence: LoCoMo 1,540 and LongMemEval-S 500 cross-judge scores; source-span expansion result.
- Scope sentence: comparisons to Mem0/Zep are not strict head-to-head because judges/backbones differ.
- Memorable result: on a prior-error diagnostic subset, adding ±2 turns to the same lexical+semantic span retrieval increases recall-all by 20.95 points for 16.84% more payload.

### §1 Introduction (1.3 pages)

- Hook: memory quality is not only whether an answer sounds right, but whether the system can show which still-valid source authorized it.
- Gap: fact CRUD and temporal graphs improve organization but often conflate navigation and factual authority.
- Insight: make graph-derived memory disposable/rebuildable; keep source spans and versions authoritative.
- Contributions: source-grounded data model; evidence-bound retrieval interface; multi-benchmark diagnostic evaluation.
- Results preview with explicit cross-judge qualifier.
- Figure 1 appears here.

### §2 Related Work (1.0 page)

1. Conversational long-term memory benchmarks: LoCoMo, LongMemEval, MEME.
2. Stateful memory systems: MemoryBank, MemGPT, A-MEM, Mem0.
3. Graph and temporal memory: Graphiti/Zep, Mem0g.
4. Source-grounded RAG/provenance: position Deep-Dream as separating navigation index from evidence authority.

### §3 Deep-Dream (2.0 pages)

#### 3.1 Memory model

- Document, document version, episode/source span.
- Stable concept/relation family vs versioned observation/assertion.
- Provenance and active/incomplete publication state.

#### 3.2 Remember pipeline

- heading-aware chunking; multi-step entity/relation discovery; quality gates; conservative alignment; atomic publication.
- Avoid presenting LLM extraction as factual truth.

#### 3.3 Hybrid retrieval

- lexical, semantic and graph/provenance channels; reciprocal-rank fusion; fine retrieval then neighboring-turn expansion.

#### 3.4 Evidence-bounded Agent

- scope isolation, fresh context, read-only tools, submit-evidence validation, trajectory/runtime hashing.
- Pseudocode for source-bounded query execution.

### §4 Evaluation (3.0 pages)

#### 4.1 Setup and comparability

- Dataset versions, question counts, answerer/judge, prompt source, metric distinction.
- Strict separation between token-F1, rubric J and binary accuracy.

#### 4.2 Main results

- Table 1: LoCoMo original + Mem0-compatible and LongMemEval-S full-500.
- Table 2: protocol-aware external reference values; no visual SOTA ranking across judges.
- LongMemEval category breakdown.

#### 4.3 Retrieval channels

- lexical vs semantic vs fused on 210 questions.

#### 4.4 Source expansion

- exact span, ±1, ±2 and legacy context-3 on 105 difficult cases.

### §5 Analysis and Limitations (1.0 page)

#### 5.1 Judge sensitivity

- exact vs abbreviated prompt; Qwen vs Kimi; implications for memory leaderboards.

#### 5.2 Negative memory and updates

- LongMemEval update/temporal strengths and MEME deletion/cascade weakness.
- State clearly that update slice performance is not a versioning ablation.

#### 5.3 Latency and scope

- multi-step Agent is slow; no production latency claim.
- synthetic benchmark limitations and required controlled Mem0/Zep run.

### §6 Conclusion (0.35 page)

- Reframe result: an auditable memory system can retain strong protocol-aligned results while keeping source evidence authoritative; no cross-system superiority claim is made.
- Concrete next steps: causal provenance ablation, update/delete microbenchmark, controlled external systems.

### Appendix

- Complete data schema and tool contracts.
- Prompt/judge hashes and dataset hashes.
- Full category tables and MEME limitations.
- Additional failure examples and proposed controlled experiment protocol.

## Figure Plan

| ID | Type | Description | Data source | Priority |
|---|---|---|---|---|
| Fig. 1 | Hero architecture | Bottom: immutable/source-of-truth documents and turns; middle: auditable concept/relation overlay with provenance arrows; top: lexical/semantic/graph navigation feeding a scope gate, then source reads and evidence submission. Contrast with a left-side generic “compressed fact becomes authority” risk schematic, not a claim about Zep. Caption must say overlay guides retrieval but cannot authorize an answer. | code/schema; deterministic diagram | HIGH |
| Fig. 2 | Two-panel grouped bar | (a) lexical/semantic/fused recall-any/all, n=210; (b) exact/±1/±2/context-3 recall-any/all, n=105 difficult cases. | `paper/results/benchmark_summary.json` | HIGH |
| Table 1 | Main results | Deep-Dream benchmark score, N, metric, answerer/judge, prompt protocol. | raw summaries + result ledger | HIGH |
| Table 2 | Prior systems | Mem0 and Zep paper results grouped by paper/protocol; no single sorted ranking. | benchmark baseline docs | HIGH |
| Table 3 | LongMemEval categories | six category scores and counts. | full-500 judge summary | MEDIUM |
| Table 4 | Negative-memory profile | MEME ER/Agg/Tr/Del/Cas/Abs, highlighting Del=0. | MEME summary | MEDIUM/appendix |

## Citation Plan

- §1: Mem0, Zep/Graphiti, LoCoMo, LongMemEval.
- §2: MemoryBank, MemGPT, ReadAgent, A-MEM, Mem0, Zep/Graphiti, LoCoMo, LongMemEval, MEME.
- §3: cite prior systems only when contrasting data models; method description otherwise references implementation artifacts.
- §4: cite each benchmark's original paper and the Mem0/Zep tables from which external numbers are copied.
- All metadata must come from ACL Anthology, arXiv primary records, DBLP/CrossRef, or the existing verified bibliography; no memory-generated bibliography.

## Required reviewer checks

1. Does “source-grounded/auditable” have a checkable definition and matching mechanism tests?
2. Are the main scores clearly labeled cross-judge?
3. Does the paper avoid causal versioning claims before X7?
4. Are the selected 105 prior-error cases identified as a development diagnostic rather than a full-test improvement?
5. Is MEME deletion failure visible enough to prevent a misleading “robust updates” story?

## Next steps

- [x] Independent plan/contract review.
- [x] Generate Figures 1–2 from the frozen evidence ledger; keep the invalid depth replay out of main-text figures.
- [x] Draft English LaTeX.
- [x] Compile and run numerical/citation checks.
- [ ] Before submission: run X7 causal ablation and X9 controlled Mem0/Zep comparison.
