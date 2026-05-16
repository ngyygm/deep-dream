# Paper Plan

**Title**: Diffusion-Based Temporal Reasoning with Selective Attention and LLM Agents for Open-World Knowledge Graph Completion
**One-sentence contribution**: We propose a diffusion-based temporal reasoning framework that combines selective attention mechanisms with LLM reasoning agents to enable open-world entity prediction on temporal knowledge graphs, achieving significant improvements over deterministic and frozen-encoder methods while providing calibrated uncertainty estimates.
**Venue**: ICLR
**Type**: method
**Date**: 2026-05-15
**Page budget**: 9 pages (main body to Conclusion end, excluding references & appendix)
**Section count**: 6

---

## Claims-Evidence Matrix

| Claim | Evidence | Status | Section |
|-------|----------|--------|---------|
| C1: Diffusion + selective attention outperforms deterministic TKG methods on open-world reasoning | Main experiments on ICEWS18, GDELT, YAGO standard + open-world splits; MRR/Hit@1/3/10 | Planned (needs experiment) | §4 |
| C2: LLM-as-temporal-reasoner provides gains beyond frozen-encoder usage | Ablation: LLM reasoning agent vs frozen encoder vs no LLM on open-world splits; temporal consistency score | Planned (needs experiment) | §4, §5 |
| C3: Selective attention is the key differentiator vs vanilla diffusion | Ablation: selective vs uniform vs no attention on ICEWS18/GDELT | Planned (needs experiment) | §5 |
| C4: Method is not overbuilt -- final design is Pareto-optimal vs overbuilt and minimal variants | Parameter efficiency comparison; MRR vs parameter count | Planned (needs experiment) | §5 |
| C5: Diffusion provides calibrated uncertainty on predictions | Brier score + reliability diagrams; confidence correlates with correctness | Planned (needs experiment) | §4, §5 |

---

## Structure

### §0 Abstract
- **What we achieve**: A generative (diffusion-based) framework for open-world temporal knowledge graph completion that uses selective attention and LLM reasoning agents.
- **Why it matters / is hard**: Real-world knowledge graphs evolve constantly and encounter new entities not seen during training. Existing TKG methods are deterministic and closed-world -- they cannot handle unseen entities or quantify prediction uncertainty.
- **How we do it**: We formulate TKG completion as a denoising diffusion process over entity embeddings, where selective attention focuses the model on temporally relevant historical context, and an LLM reasoning agent provides semantic guidance for open-world inference.
- **Evidence**: State-of-the-art results on ICEWS18, GDELT, and YAGO open-world splits; calibrated uncertainty estimates; ablations confirming each component's contribution.
- **Most remarkable result**: >= 3% MRR improvement over best baseline on open-world TKG splits with calibrated confidence.
- **Estimated length**: 200 words
- **Self-contained check**: Yes -- states the problem (open-world TKG), the approach (diffusion + selective attention + LLM agent), and the key result.

### §1 Introduction
- **Opening hook**: Knowledge graphs are living structures -- entities and relations evolve daily, yet most TKG completion methods assume a fixed entity vocabulary and produce point predictions without uncertainty.
- **Gap / challenge**: Three limitations of current TKG methods: (1) closed-world assumption prevents handling new entities; (2) deterministic predictions cannot quantify confidence; (3) LLM integration uses frozen encoders rather than active temporal reasoning.
- **One-sentence contribution**: We propose a diffusion-based temporal reasoning framework with selective attention and LLM reasoning agents for open-world TKG completion with calibrated uncertainty.
- **Approach overview**: The framework has three components: (a) a denoising diffusion process over entity/relation embeddings for generative prediction; (b) a selective attention mechanism that identifies temporally relevant historical context; (c) an LLM reasoning agent that provides semantic guidance for open-world inference.
- **Key questions**: (1) Can generative models outperform deterministic methods for TKG completion? (2) Is selective attention critical, or does vanilla diffusion suffice? (3) Does LLM reasoning (not just encoding) help temporal inference?
- **Contributions**:
  1. First diffusion-based framework for open-world TKG completion with calibrated uncertainty (C1, C5)
  2. Selective attention mechanism that identifies temporally relevant context for the diffusion process (C3)
  3. LLM-as-temporal-reasoner paradigm: LLM actively reasons about temporal patterns rather than passively encoding (C2)
  4. Comprehensive evaluation on open-world splits showing significant improvements (C1, C4)
- **Results preview**: >= 3% MRR improvement on open-world splits; calibrated uncertainty (Brier score improvement).
- **Hero figure**: Figure 1 showing (left) a temporal KG evolving over time with new entities appearing (open-world challenge), (center) the diffusion process with selective attention highlighting relevant snapshots and LLM agent providing semantic guidance, (right) comparison of uncertainty-calibrated predictions vs. deterministic baselines.
- **Estimated length**: 1.5 pages
- **Key citations**: [VERIFY] RE-NET (Jin et al., 2020), TANGO (Ding et al., 2024), PT2KGC (2025), Diffusion-TKG ECML-PKDD 2025, TKG Survey (arXiv 2403.04782)
- **Front-loading check**: Title + abstract + intro first paragraph + hero figure make the three-component contribution clear before the method section.

### §2 Related Work
- **Subtopics**:
  1. Temporal KG embedding and completion (RE-NET, CyGNet, TANGO, RE-GCN, HiSMatch) -- direct competitors for link prediction
  2. Generative models for graphs (diffusion models, VAEs, GANs for graph generation) -- methodological lineage
  3. LLM + knowledge graph integration (PT2KGC, LLM reasoning over KGs, RAG+KG) -- LLM integration paradigm
  4. Open-world KG reasoning (few-shot entity prediction, inductive KG reasoning) -- open-world setting
- **Positioning**: We differ from TKG methods by using generative (diffusion) rather than discriminative prediction; differ from graph diffusion by adding temporal selective attention and LLM reasoning; differ from LLM+KG work by using the LLM as an active temporal reasoner rather than a passive encoder.
- **Minimum length**: 1 full page (4 paragraphs with synthesis)
- **Organization rule**: Organized by methodological family, not paper-by-paper
- **Must NOT be just a list** -- each paragraph synthesizes a family and positions our contribution

### §3 Method
- **Notation**:
  - $\mathcal{G}_t = \{(e_s, r, e_o, t)\}$ -- KG snapshot at time $t$
  - $\mathcal{E}_{train}$ -- entity set seen during training; $\mathcal{E}_{test} \setminus \mathcal{E}_{train}$ -- open-world entities
  - $\mathbf{x}_0 \in \mathbb{R}^d$ -- target entity embedding
  - $\mathbf{x}_T \sim \mathcal{N}(0, I)$ -- noise at diffusion step $T$
  - $\mathbf{A}_t$ -- selective attention weights at time $t$
- **Problem formulation**: Given historical snapshots $\mathcal{G}_{1}, ..., \mathcal{G}_{t}$, predict $(e_s, r, ?, t+1)$ where $?$ may be an unseen entity.
- **Method description**:
  1. **Historical Encoding**: Encode each snapshot $\mathcal{G}_i$ into a graph embedding via a temporal GNN (shared backbone)
  2. **Selective Attention**: Compute attention weights over historical snapshots based on query relevance, focusing on temporally and structurally relevant context
  3. **LLM Reasoning Agent**: Given the query and selectively attended context, the LLM agent generates a semantic reasoning trace that conditions the diffusion process
  4. **Diffusion Process**: Forward process adds Gaussian noise to the target entity embedding; reverse process denoises conditioned on (a) selectively attended historical context and (b) LLM reasoning trace
  5. **Training**: Optimize the denoising network with a simplified ELBO objective; train selective attention end-to-end
  6. **Inference**: Sample from the reverse process; map denoised embedding to entity via nearest neighbor (supports open-world: can match to new entity embeddings)
- **Formal statements**: Diffusion ELBO bound for TKG completion; selective attention concentration lemma
- **Estimated length**: 2 pages

### §4 Experiments
- **Figures planned**:
  - Fig 1 (Hero): System overview + open-world challenge + comparison (described in §1)
  - Fig 2: Calibration reliability diagrams comparing our method vs. baselines
  - Table 1: Main results -- MRR, Hits@1/3/10 on ICEWS18, GDELT, YAGO (standard + open-world splits) vs. RE-NET, CyGNet, TANGO, EvolveGCN, PT2KGC, Diffusion-TKG
  - Table 2: LLM ablation -- full reasoning agent vs. frozen encoder vs. no LLM vs. small LM
- **Data source**: ICEWS18, GDELT (standard benchmarks); YAGO with constructed open-world splits
- **Baselines**: RE-NET, CyGNet, TANGO, EvolveGCN, PT2KGC, Diffusion-TKG (ECML-PKDD 2025)
- **Estimated length**: 2.5 pages

### §5 Ablation / Analysis
- **Selective attention ablation**: Full method vs. uniform attention vs. no attention (vanilla diffusion)
- **Simplicity check**: Final method vs. overbuilt variant (extra GNN layers + multi-head attention + aux losses) vs. minimal variant (diffusion only)
- **Failure analysis**: Categorization of prediction errors (temporal out-of-range, unseen relation, entity ambiguity); correlation between diffusion uncertainty and error
- **Hyperparameter sensitivity**: Number of diffusion steps $T$, number of attended snapshots, LLM reasoning depth
- **Estimated length**: 1 page

### §6 Conclusion
- **Restatement**: We introduced a diffusion-based framework with selective attention and LLM reasoning agents for open-world TKG completion, demonstrating that generative prediction with temporal context awareness significantly outperforms existing deterministic approaches.
- **Limitations**: LLM reasoning agent adds inference latency; open-world splits require careful construction; diffusion sampling is slower than single-pass methods.
- **Future work**: (1) Distill LLM reasoning agent into a small model for efficiency; (2) extend to continuous-time TKGs; (3) explore RAG integration for real-time KG updates.
- **Estimated length**: 0.5 pages

---

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | Hero/Architecture | Left: temporal KG with new entities; Center: diffusion + selective attention + LLM agent pipeline; Right: calibrated prediction comparison | manual + experiment results | HIGH |
| Fig 2 | Reliability diagram | Calibration curves: our method vs. baselines showing uncertainty quality | calibration experiment | HIGH |
| Fig 3 | Bar chart | Selective attention ablation: MRR for full vs. uniform vs. no attention | ablation experiment | MEDIUM |
| Fig 4 | Scatter plot | Diffusion confidence vs. prediction correctness (failure analysis) | error analysis | MEDIUM |
| Table 1 | Comparison table | Main results: MRR, H@1/3/10 on 3 datasets x 2 splits (standard + open-world) | experiment results | HIGH |
| Table 2 | LLM ablation table | Reasoning agent vs. frozen encoder vs. no LLM vs. small LM | ablation experiment | HIGH |
| Table 3 | Simplicity table | Final vs. overbuilt vs. minimal: MRR, params, latency | simplicity experiment | MEDIUM |

**Hero Figure (Fig 1) Detail**:
- Left panel: Timeline showing a TKG evolving over snapshots $t_1$ through $t_5$. New entities (highlighted in red) appear at $t_4$ and $t_5$, illustrating the open-world challenge. A query at $t_6$ asks about a new entity.
- Center panel: The diffusion pipeline. Historical snapshots are encoded by a temporal GNN. Selective attention (shown as weighted arrows) focuses on $t_3$-$t_5$. The LLM reasoning agent generates a reasoning trace (shown as a text bubble). The diffusion process (shown as a denoising sequence $x_T$ to $x_0$) is conditioned on both.
- Right panel: Horizontal bar chart comparing MRR on the open-world split: Our method (highlighted in green) vs. RE-NET, CyGNet, TANGO, PT2KGC, Diffusion-TKG. Error bars show confidence intervals.
- Caption draft: "Our diffusion-based framework addresses open-world TKG completion by combining selective temporal attention with LLM-guided reasoning. New entities (left, red) are handled by the generative diffusion process (center), achieving significant MRR improvements on open-world splits (right)."
- Why it helps skim readers: The figure conveys the entire paper story -- the open-world problem, the three-component solution, and the quantitative result -- in a single visual.

---

## Citation Plan

- §1 Intro: [VERIFY] RE-NET (Jin et al., 2020), TANGO (Ding et al., 2024), PT2KGC (ScienceDirect 2025), Diffusion-TKG (ECML-PKDD 2025), TKG Survey (arXiv 2403.04782)
- §2 Related Work:
  - Temporal KG: [VERIFY] RE-NET (2020), CyGNet (Zhu et al., 2021), RE-GCN (2022), TANGO (2024), EvolveGCN (Pareja et al., 2020), HiSMatch (2021)
  - Graph generative models: [VERIFY] GDSS (Jo et al., 2022), DiGress (Vignac et al., 2023), GraphGDP (2023)
  - LLM + KG: PT2KGC (2025), [VERIFY] LLM reasoning over KGs survey, [VERIFY] KG-RAG (2024)
  - Open-world KG: [VERIFY] Inductive KG reasoning (Teru et al., 2020), [VERIFY] Few-shot entity prediction
- §3 Method: Ho et al. (2020) -- DDPM; Vaswani et al. (2017) -- attention; [VERIFY] Song et al. (2021) -- score-based generative modeling

---

## Reviewer Feedback

*To be generated via Codex MCP cross-review with gpt-5.4 model. Skipped in this plan-generation step since Codex MCP may not be configured in the current environment.*

---

## Next Steps
- [ ] /outline-agent to refine the section structure
- [ ] /paper-figure to generate figure specifications
- [ ] /section-writing-agent to draft individual sections
- [ ] /citation-audit to verify all [VERIFY] flagged citations
- [ ] /paper-write to draft full LaTeX
- [ ] /paper-compile to build PDF
