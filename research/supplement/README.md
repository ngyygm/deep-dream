# Supplementary material — artifact ledger and recomputation package

Anonymous submission. This package makes the paper's reported numbers auditable:
it ships every size-bounded frozen artifact the paper cites, the two scripts
that turn those artifacts into the paper's tables and macros, the frozen
outputs of those scripts, and SHA-256 pins for everything too large to
redistribute. The directory layout mirrors the authors' research tree, so both
scripts resolve their input paths unmodified when the package root is used as
the tree root.

## Layout

| Path | Contents |
|---|---|
| `paper/results/evidence_ledger.json` | the machine-readable ledger: 63 sources with SHA-256, per-run track scores, library-snapshot hashes, self-check values |
| `paper/results/aggregate_existing.py` | generates the ledger from the artifact tree |
| `paper/results/recompute_from_logs.py` | regenerates `recomputed_values.json` and `RECOMPUTED_MACROS.tex` from frozen per-question outputs |
| `paper/results/recomputed_values.json` | frozen recompute output (machine-readable, every paper number) |
| `paper/figures/RECOMPUTED_MACROS.tex` | frozen recompute output (LaTeX macros used by the paper) |
| `.benchmark_runs/<run>/` | per-run manifests, summary JSONs, per-item score JSONLs, judge outputs — exactly the files the ledger hashes |
| `.benchmark_data/` | benchmark-side dataset files that are ledger sources (`locomo10.json`, `locomo-plus/locomo_plus.json`, `memoryagentbench/entity2id.json`) |
| `evidence/` | integrity-evidence bundles for the repaired tracks (full-context anchor runs, engine patch) |
| `reports/` | two dated run-comparison reports underlying the v1/v2 engine narrative |
| `verify_shipped.py` | self-verification script (below) |

## What you can verify in this package

**1. File identity (self-verifying).**

```bash
python3 verify_shipped.py
```

Every file shipped here that the ledger lists is checked against its recorded
SHA-256. The only listed-but-absent entry is the 277 MB LongMemEval-S dataset
file (pinned below); everything else must match or the check fails.

**2. Per-question scores and answers.** Each `.benchmark_runs/<run>/` directory
contains the per-item score JSONLs (`*_scores.*.jsonl`), judge outputs
(`judge_results.*.jsonl`), and run manifests (dataset/prompt/model/runtime
provenance, code hashes, exit status) for the runs behind the paper's tables.
The 70.5 MB file
`.benchmark_runs/locomo-k3-agent-judge-diagnostic-v1/results.*.jsonl` is the
complete per-question record of the LoCoMo K3-as-answerer diagnostic.

**3. Script logic.** The two scripts are the exact code that produced the
ledger and the paper's recomputed numbers; both are deterministic (no
timestamps, no network access, fixed iteration order).

## What requires the full artifact tree

Both scripts run to completion only on the authors' full tree; this package
pins the SHA-256 of every absent input:

- `aggregate_existing.py` additionally hashes 70 `library.db` snapshots
  (2.6 GB) under the four run families' `libraries/` directories; their hashes
  and entity statistics are recorded inside the shipped ledger at
  `four_benchmark.engine_comparison.{lme,mab}.library_dbs`.
- `recompute_from_logs.py` additionally reads the raw agent trajectories
  (`results.*.jsonl`) for rung-cost accounting.

| Absent input | Size | SHA-256 |
|---|---|---|
| `.benchmark_data/longmemeval_s_cleaned.json` | 277.4 MB | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `.benchmark_runs/longmemeval-kimik3-full-v1/results.skill-agent.jsonl` | 3.8 MB | `676f0037daac6c292a076a418b4098c0dd36029536dec98e34f97dda80fed1d6` |
| `.benchmark_runs/longmemeval-kimik3-full-v2/results.skill-agent.jsonl` | 3.5 MB | `b7f24ea159090606af1b6f0bce31b14852549e3d7881d5b142c0ff7146096011` |
| `.benchmark_runs/memoryagentbench-kimik3-sample-v1/results.baseline.jsonl` | 1,129.7 MB | `adb37730fa5cfc53e0f9c9ba9ffe72e775a79c6adae4bdf0f9fc95e96233e698` |
| `.benchmark_runs/memoryagentbench-kimik3-sample-v1/results.pi.jsonl` | 2.5 MB | `fce638e263c6e6195e6e83ebb9cd8911ad3a1ea71e1bdd58b4e22bb270ad6879` |
| `.benchmark_runs/memoryagentbench-kimik3-sample-v1/results.skill-agent.jsonl` | 254.4 MB | `72df6ee40b751073406a4bab1a15f43164a93d2df421d800de7c4680abc5e2b3` |
| `.benchmark_runs/memoryagentbench-kimik3-sample-v2/results.baseline.jsonl` | 2,557.0 MB | `207345dc845959180791faa0eccf5955908cced29951dd8421265f7ba8a90fd1` |
| `.benchmark_runs/memoryagentbench-kimik3-sample-v2/results.pi.jsonl` | 5.0 MB | `0b82f8b80c363427e515a11266e2291a6233c8924f849254687e5b06f4b7e643` |
| `.benchmark_runs/memoryagentbench-kimik3-sample-v2/results.skill-agent.jsonl` | 439.7 MB | `a562b26b06be2a8e072dd6c06ab9802e0289ad8e8236f60491a65899e26ab9d9` |

Both regenerations were verified **byte-for-byte** on the full artifact tree
when this package was assembled (2026-09-22): `aggregate_existing.py --output`
reproduces the shipped `evidence_ledger.json` exactly (twice, idempotently),
and `recompute_from_logs.py` reproduces the shipped `recomputed_values.json`
and `RECOMPUTED_MACROS.tex` exactly. The absent inputs above are available from
the authors on request; with them, both verifications repeat end to end.

## Dataset provenance

| File | Origin |
|---|---|
| `.benchmark_data/locomo10.json` | public LoCoMo release (Maharana et al., 2024), as consumed by the runs |
| `.benchmark_data/locomo-plus/locomo_plus.json` | LoCoMo-Plus cognitive-memory set (Li et al., 2026) |
| `.benchmark_data/memoryagentbench/entity2id.json` | MemoryAgentBench official scorer fingerprint file (Hu et al., 2025), pinned at the scorer commit cited in the paper |
| LongMemEval-S 500-question file | not redistributed (277 MB, pinned above); derived from the public LongMemEval release (Wu et al., 2024) |

## Anonymity note

Run manifests record absolute paths of the original execution machine
(`/Users/admin/...` — a generic macOS account name). The package was scanned
for author names, emails, institutional handles, local usernames of the
authors' machines, and API credentials: none are present.
