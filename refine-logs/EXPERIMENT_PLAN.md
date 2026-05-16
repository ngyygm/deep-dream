# Experiment Plan

**Problem**: Temporal knowledge graph (TKG) reasoning methods struggle with open-world entities, non-local temporal dependencies, and lack of uncertainty modeling. Current LLM-TKG integration uses LLMs as frozen encoders rather than temporal reasoning agents.
**Method Thesis**: A diffusion-based temporal reasoning framework with selective attention that treats LLMs as temporal reasoning agents over evolving knowledge graphs, enabling open-world TKG completion with calibrated uncertainty.
**Date**: 2026-05-15

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1: Diffusion + selective attention outperforms deterministic TKG methods on open-world reasoning | Open-world reasoning is the key unsolved frontier; generative models naturally handle unseen entities | MRR/Hit@1 improvements on ICEWS18, GDELT, and at least one open-world split | B1, B2 |
| C2: LLM-as-temporal-reasoner provides meaningful gains beyond frozen-encoder usage | Current LLM-TKG work uses LLMs passively; active reasoning is the gap | Ablation showing LLM reasoning agent > LLM encoder-only on temporal inference tasks | B3, B4 |

**Anti-claim to rule out**: "The gains come only from the diffusion model's capacity, not from the LLM reasoning or selective attention mechanism."

## Paper Storyline

- **Main paper must prove**: Diffusion + selective attention + LLM reasoning agent = better open-world TKG completion than existing deterministic or frozen-encoder methods
- **Appendix can support**: Scaling behavior, additional datasets, hyperparameter sensitivity, qualitative examples
- **Experiments intentionally cut**: Full RAG integration (separate paper), spatio-temporal extension (out of scope)

## Experiment Blocks

### Block 1: Main Anchor Result
- **Claim tested**: C1
- **Why this block exists**: Core proof that the method solves the actual bottleneck (open-world TKG reasoning)
- **Dataset / split / task**: ICEWS18, GDELT, YAGO -- standard TKG completion splits + open-world entity splits (unseen entities at test time)
- **Compared systems**: RE-NET, CyGNet, TANGO, EvolveGCN, PT2KGC, Diffusion-TKG (ECML-PKDD 2025)
- **Metrics**: MRR, Hit@1, Hit@3, Hit@10 (decisive); calibration error Brier score (secondary)
- **Setup details**: 4x A100 GPUs, 3 seeds, AdamW optimizer, learning rate grid {1e-4, 5e-5, 1e-5}, sequence length 10 snapshots, batch size 256
- **Success criterion**: MRR improvement of >= 3% over best baseline on open-world splits; comparable or better on standard splits
- **Failure interpretation**: If standard splits improve but open-world does not, the diffusion component is not addressing the right gap -- pivot to closed-world analysis
- **Table / figure target**: Table 1 (main results), Figure 2 (calibration curves)
- **Priority**: MUST-RUN

### Block 2: Novelty Isolation (Selective Attention Ablation)
- **Claim tested**: C1 (selective attention specifically)
- **Why this block exists**: Prove that selective attention in the diffusion process is the key innovation, not just "use diffusion"
- **Dataset / split / task**: Same as B1, open-world splits only
- **Compared systems**: (a) Our method (full), (b) Our method without selective attention (uniform attention), (c) Vanilla diffusion (no attention mechanism), (d) Best baseline from B1
- **Metrics**: MRR, Hit@1, Hit@10
- **Setup details**: Same as B1; identical hyperparameters across variants
- **Success criterion**: Selective attention variant >= 2% MRR over uniform attention; full method >= 1% over no-attention diffusion
- **Failure interpretation**: If selective attention does not help, the contribution reduces to "diffusion for TKG" -- need to reframe the paper
- **Table / figure target**: Table 2 (ablation study)
- **Priority**: MUST-RUN

### Block 3: Frontier Necessity Check (LLM Reasoning Agent)
- **Claim tested**: C2
- **Why this block exists**: Prove the LLM reasoning agent is genuinely useful, not decorative
- **Dataset / split / task**: ICEWS18, GDELT open-world splits; plus a temporal analogy dataset
- **Compared systems**: (a) Full method with LLM reasoning agent, (b) Frozen LLM encoder (PT2KGC-style), (c) No LLM (pure GNN + diffusion), (d) Fine-tuned small LM (DeBERTa) as encoder
- **Metrics**: MRR, Hit@1; plus temporal consistency score (novel metric: does the method make temporally coherent predictions?)
- **Setup details**: GPT-4o-mini as reasoning agent (API), 3 seeds, same training budget
- **Success criterion**: LLM reasoning agent >= 2% MRR over frozen encoder; >= 1% improvement in temporal consistency
- **Failure interpretation**: If frozen encoder matches reasoning agent, the LLM contribution is overstated -- downplay or remove C2
- **Table / figure target**: Table 3 (LLM ablation), Figure 3 (temporal consistency visualization)
- **Priority**: MUST-RUN

### Block 4: Simplicity Check
- **Claim tested**: C1 (elegance)
- **Why this block exists**: Defend that the method is not overbuilt
- **Dataset / split / task**: ICEWS18 open-world split
- **Compared systems**: (a) Final method, (b) Overbuilt variant (extra GNN layers + multi-head attention + auxiliary losses), (c) Minimal variant (diffusion only, no LLM, no selective attention)
- **Metrics**: MRR, parameter count, inference time per query
- **Setup details**: Single GPU, same training epochs
- **Success criterion**: Final method within 1% MRR of overbuilt variant but with < 50% parameters; minimal variant at least 2% worse
- **Failure interpretation**: If overbuilt variant is significantly better, the method needs more capacity -- reconsider architecture
- **Table / figure target**: Table 4 (simplicity comparison)
- **Priority**: NICE-TO-HAVE

### Block 5: Failure Analysis
- **Claim tested**: Both C1 and C2 (honest limits)
- **Why this block exists**: Reviewers expect clear-eyed analysis of what the method cannot do
- **Dataset / split / task**: Error analysis on ICEWS18 test set
- **Compared systems**: N/A (qualitative)
- **Metrics**: Error categorization (temporal out-of-range, unseen relation, entity ambiguity), confidence calibration on errors vs correct
- **Setup details**: Manual analysis of 200 prediction errors
- **Success criterion**: Clear categorization of failure modes; evidence that diffusion uncertainty correlates with errors
- **Failure interpretation**: N/A (this is analysis, not a pass/fail experiment)
- **Table / figure target**: Table 5 (error taxonomy), Figure 4 (confidence vs correctness scatter)
- **Priority**: NICE-TO-HAVE

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0: Sanity | Data pipeline, metric correctness, one quick overfit on toy split | R001-R003 | Metrics computed correctly; method overfits small split | 2 GPU-hrs | Low |
| M1: Baselines | Reproduce TANGO, PT2KGC, Diffusion-TKG on ICEWS18 | R004-R006 | Baseline numbers match published results within 1% | 8 GPU-hrs | Medium (reproducibility) |
| M2: Main method | Full method on ICEWS18 + GDELT + YAGO, standard + open-world | R007-R012 | MRR >= best baseline + 3% on open-world | 24 GPU-hrs | High |
| M3: Ablations | Selective attention ablation + LLM ablation | R013-R018 | Selective attention >= 2% over uniform; LLM agent >= 2% over frozen | 16 GPU-hrs | Medium |
| M4: Polish | Simplicity check + failure analysis | R019-R021 | Clear story for Tables 4-5 | 4 GPU-hrs | Low |

## Compute and Data Budget

- **Total estimated GPU-hours**: ~54 hours (4x A100)
- **Data preparation needs**: ICEWS18 and GDELT are public; YAGO needs open-world split construction (novel contribution to benchmarks)
- **Human evaluation needs**: 200 error categorizations (~4 hours manual work)
- **Biggest bottleneck**: LLM API costs for reasoning agent in B3 (estimate ~$200-500 depending on API pricing)

## Risks and Mitigations

- **Risk**: Diffusion-TKG (ECML-PKDD 2025) already covers diffusion for TKG -- we need to differentiate via selective attention + LLM reasoning agent
  - **Mitigation**: Position selective attention as the key differentiator; run direct comparison
- **Risk**: LLM reasoning agent may be too slow for practical use
  - **Mitigation**: Include latency analysis; propose distillation as future work
- **Risk**: Open-world splits may not show enough difference from standard splits
  - **Mitigation**: Construct harder open-world splits (lower entity overlap); report both
- **Risk**: Baseline reproducibility issues
  - **Mitigation**: Use official code where available; report re-implementation numbers alongside published numbers

## Final Checklist

- [x] Main paper tables are covered (Tables 1-3 = MUST-RUN)
- [x] Novelty is isolated (B2 selective attention ablation)
- [x] Simplicity is defended (B4 simplicity check)
- [x] Frontier contribution is justified (B3 LLM necessity check)
- [x] Nice-to-have runs are separated from must-run runs (B4, B5 = NICE-TO-HAVE)
