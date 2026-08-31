# Deep-Dream autonomous review loop

run_id: run_20260813_ddreview  
reviewer: gpt-5.6-sol / same-family provisional  
status: continuing after Round 1 revisions

## Round 1 (2026-08-13 12:38)

### Assessment

- Score: 4.5/10
- Verdict: revise / not ready
- Critical issues: core mechanism not causally tested; semantic entailment is not enforced; MEME deletion is 0%; depth replay invariant fails; controlled Mem0/Zep comparison absent; ledger and subset labels need strengthening.

### Actions taken

- Renamed the paper from a versioned/evidence-bounded headline to source-grounded, auditable concept overlays.
- Reframed `submit_evidence` as a scope/source-ID boundary, explicitly not a semantic entailment guarantee.
- Changed source-expansion comparison to the fixed lexical+semantic span baseline and updated the expected deltas to +19.05 recall-any, +20.95 recall-all, +16.84% payload.
- Excluded the depth replay from paper figures and positive claims because the prefix invariant failed on all 210 questions.
- Added LoCoMo original, LoCoMo-Plus, per-item JSONL, dataset and raw-summary hashes to the generated evidence ledger; verified exact regeneration.
- Made evidence-gate tests, update/delete tests, and controlled Mem0/Graphiti comparison submission hard gates in the acceptance contract.
- Reduced the main figure plan to two figures; moved invalid budget replay and detailed tables to appendix/integrity notes.

### Results

- `python paper/results/aggregate_existing.py --output paper/results/benchmark_summary.json` followed by stdout comparison passes exact regeneration.
- Updated Figure 1 and Figure 2 PDFs were rendered with the project `.venv` and visually inspected; Figure 2 legends and Figure 1 routing were corrected.

### Status

Continuing to Round 2: re-review the revised framing, evidence ledger, and contract. New experiments are not launched in this turn.

## Round 2 (2026-08-13 14:20)

### Assessment

- Score: 7.0/10
- Verdict: almost, but not submission-ready
- Confirmed: lexical+semantic-span baseline, exact ledger regeneration, invalid depth exclusion, source-ID boundary wording, and explicit hard gates.
- Remaining blockers: X7 evidence-gate tests, X8 update/delete microbenchmark, X9 controlled Mem0/Graphiti comparison, repaired budget replay, and semantic answer--evidence checks.

### Actions taken after the review

- Added explicit numerator/denominator fields for LongMemEval categories, channel recall, and source-expansion recall.
- Added SHA-256 entries for Kimi-K3 per-item judgments and channel/source-expansion JSONL artifacts.
- Removed stale Figure 3 planning text and wired the two generated figures through `figures/latex_includes.tex`.
- Drafted all paper sections and abstract, regenerated tables, fixed bibliography escaping, installed missing TinyTeX packages, and compiled the eight-page PDF.
- Kept the invalid depth replay and MEME deletion failure visible as negative integrity findings.

### Status

Review loop stops provisionally at the positive threshold (`almost`, 7.0/10), with the acceptance contract still blocking submission until X7/X8/X9 and the repaired budget replay are complete.
