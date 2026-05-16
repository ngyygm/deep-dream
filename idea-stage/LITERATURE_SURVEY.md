# Literature Survey: Temporal Knowledge Graphs

**Direction**: temporal knowledge graphs
**Date**: 2026-05-15

## Literature Table

| Paper | Venue/Source | Method | Key Result | Relevance | Source |
|-------|-------------|--------|------------|-----------|--------|
| A Survey on Temporal Knowledge Graph (2403.04782) | arXiv 2024 | Comprehensive survey of TKG representation learning | Taxonomy of embedding, reasoning, and completion methods | Foundational landscape map | web |
| TKGQA Survey (2406.14191v3) | arXiv 2024 | Survey on Temporal KG Question Answering | Classifies explicit/implicit temporal questions; discusses normalization | QA is a key application area | web |
| Survey on TKG Embedding (ScienceDirect) | Knowledge-Based Systems 2024 | Review of embedding models with temporal data | Covers embedding models and related techniques | Core technique space | web |
| TKG Completion Survey | IJCAI 2023 | KGC for predicting missing links | Link prediction is critical for real-world KGs | Baseline task formulation | web |
| Spatio-Temporal KG Survey | ResearchGate 2024 | Review of spatio-temporal KG models | Traces origins in static, temporal, and spatial methodologies | Extended scope: spatial+temporal | web |
| TKG Diffusion Model for Open-World Reasoning | Science China Info Sciences 2025 | Diffusion model for open-world TKG reasoning | Novel generative approach to open-world reasoning | Emerging: generative models | web |
| PT2KGC | ScienceDirect 2025 | Pre-trained LM for TKG embedding | PLMs improve TKG completion and extrapolation | LLM integration trend | web |
| Hierarchical TKG Reasoning | ECML-PKDD 2025 | Contrastive learning merging historical/non-historical dependencies | Enhanced temporal reasoning via contrastive signals | Novel training paradigm | web |
| TKG Reasoning with Evolutionary Patterns | ACM/Springer 2025 | Inferring from historical snapshots | Structural + temporal reasoning combined | Core reasoning approach | web |
| Temporal Reasoning over Evolving KGs (2509.15464) | arXiv 2025 | LLM + KG for temporal reasoning | LLMs combined with evolving KGs improve factual reasoning | Key trend: LLM integration | web |
| Subgraph Reasoning on TKGs | MDPI Mathematics 2025 | Subgraph-based forecasting beyond sequential enforcement | Flexible subgraph reasoning for future event prediction | Architectural innovation | web |
| Diffusion with Selective Attention for TKG | ECML-PKDD 2025 | Diffusion models with selective attention for TKG completion | Generative prediction of missing entities at future timesteps | Emerging: diffusion for TKG | web |

## Narrative Landscape Summary

**The field of temporal knowledge graphs (TKGs)** has matured significantly, with a clear taxonomy emerging around four core tasks: embedding, completion, reasoning, and question answering. The 2024 surveys (arXiv 2403.04782, ScienceDirect KBS survey) provide a solid foundation, categorizing approaches into tensor decomposition, graph neural networks, and sequence-based methods.

**Three major trends define the 2025 frontier:**

1. **LLM Integration**: Multiple papers now combine pre-trained language models with TKGs (PT2KGC, arXiv 2509.15464). The key insight is that PLMs provide rich semantic representations that complement the structural/temporal signals in TKGs. However, most approaches use LLMs as frozen encoders rather than reasoning engines.

2. **Generative Models**: Diffusion models are entering the TKG space (Science China 2025, ECML-PKDD 2025). This is early-stage but promising -- generative approaches can model uncertainty in future predictions, which deterministic methods cannot. The selective attention mechanism for diffusion is particularly novel.

3. **Beyond Sequential Reasoning**: Subgraph reasoning (MDPI 2025) and hierarchical contrastive learning (ECML-PKDD 2025) move beyond the dominant sequential/autoregressive paradigm. These approaches capture non-local temporal dependencies.

**Key gaps and open problems:**

- **Open-world TKG reasoning** remains underexplored -- most methods assume a closed set of entities/relations.
- **Scalability** is a recurring concern -- TKG methods struggle with large-scale graphs due to the temporal dimension multiplying computational cost.
- **LLM-as-reasoner** for TKGs is nascent -- current work uses LLMs as encoders but not as temporal reasoning agents.
- **Benchmark diversity** is limited -- most evaluation is on ICEWS and GDELT; broader benchmarks are needed.
- **RAG + TKG** integration (Medium 2025) is a practical direction with industry relevance but lacks rigorous academic evaluation.
