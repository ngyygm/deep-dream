# Remember V1/V2 evidence audit (2026-09-18)

## Verified version identity

Verified against development commit 8fa9651, reports `lme_v1_vs_v2_full_2026-08-29.md` and `mab_v1_vs_v2_2026-08-30.md`, and current implementation in `align_v2.py` and `pipeline_workers.py`.

- V1 is the B1-equivalent predecessor. The development report says window-batch alignment was already enabled via defaults. It must not be called the original per-entity-only A arm. The stored LME V1 config has `window_batch_alignment=null`; default inheritance needs the original runtime snapshot to be independently reconstructed.
- V2 is B2 / ALIGN-V2: Step-9 entity alignment becomes concurrent across windows; window batch decisions produce candidate equivalence groups; ingestion queues those groups, and scope-end processing merges families after concurrent writers finish. Observations and mentions are reassigned through non-destructive merge semantics.
- MAB V2 also changed context/output budgets, retries, timeout inheritance and document path persistence. LME V2 received four throughput/reliability fixes during its run. These are additional changes, not the definition of V1 versus V2.
- Therefore V1/V2 is a bundled system revision comparison, not a component-isolating ablation. LLM-call reduction is not itself a wall-clock speedup.

## Artifacts already on the paper server

Both LME and MAB V1/V2 result directories, per-run manifests, per-question scores/results, ingest logs, and per-scope database snapshots are present. LME has 25 databases per version; MAB has 10. No need to retransmit those unchanged artifacts.

LME manifests contain 940 (V1) and 1020 (V2) document records with latency and call statistics, out of 1176 source documents each. These published averages are not a document-paired comparison. MAB call-cost figures use the 45-document intersection. Existing call figures are 63.8 to 22.0 (LME), and 609 to 329 (MAB).

Both LME manifests record the same Git commit, `73d9c5e6996aad48c981068056b34d15388f99f9`, despite different Remember implementations. A commit alone cannot reconstruct uncommitted code, runtime defaults, environment switches or mid-run hotfixes. Do not label these runs independent frozen-code replications without recovering that evidence.

## Minimal original-machine handoff

1. `align_ab_experiment_2026-08-25.md` and its raw A/B1/B2 arm artifacts, runner and aggregation scripts. This report is referenced by the LME development report but is absent from the checked server repository. Preserve the actual arm definitions; do not infer them from names.
2. Per-run executed source snapshots or base commit plus dirty patch, including untracked source modules; effective configurations after merging defaults, relevant environment switches, concurrency limits, prompts/models and dependencies. Redact credentials. Include hotfix timestamps and the affected scopes/documents.
3. Any unsynced per-stage event traces: run/scope/document/window ID, stage, start/end timestamps, worker/concurrency and outcome; include retries, rate-limit waits, restarts, cache hits and scope-end convergence time. Existing ingest logs and per-document latency records will be checked first. Never fabricate missing timestamps or reconstruct a Gantt chart from summed concurrent durations.
4. Alignment decision and merge traces, convergence before/after snapshots if not already synced: candidate groups, confidence/thresholds, create/merge/redirect actions, evidence links and conflicts. Include any human-labelled identity pairs or false-merge/missed-merge analysis. Duplicate-name counts alone do not measure entity-linking accuracy.
5. An inventory mapping each artifact to its run and checksum, marking absent artifacts as absent. A present-day rerun must be labelled as a new experiment.

## Follow-up experiment if historical A/B cannot isolate components

Use one codebase and identical data/order, models/prompts, effective context/output/retry settings, hardware and API limits. Vary batching, cross-window concurrency and deferred convergence as explicit factors where supported; keep document-level concurrency fixed for a cross-window parallelism comparison. Measure total ingestion elapsed time including convergence, document throughput, call/token use, alignment errors/duplicate proxies, source-link integrity and paired downstream QA. Repeat timing runs; keep proposed experiments distinct from completed evidence.
