# Literature Survey: Deep-Dream A-Conference Paper Opportunities

**Date**: 2025-06-10
**Scope**: Automated prompt optimization, LLM-based KG construction, hybrid retrieval, episodic memory for agents

---

## 1. Research Landscape Map

### 1.1 Automated Prompt Optimization

| Paper | Venue/Year | Method | Key Result | Relevance |
|-------|-----------|--------|------------|-----------|
| **DSPy / MIPROv2** | ICML 2024 | Bayesian optimization over instruction + few-shot candidates | Joint optimization of multi-step pipelines | ⭐⭐⭐ Direct competitor to TreePrompt. Uses **flat search** (propose → evaluate → select), no tree backtracking |
| **GEPA Optimizer** (DSPy) | 2025 | Auto-budget prompt optimization with LM-generated variants | Simplified API over MIPROv2 | ⭐⭐ Evolution of DSPy's optimizer family |
| **MCTS-OPS** | arXiv 2025.08 | MCTS for prompt sequence optimization in code generation | Incremental code construction guided by tree search | ⭐⭐⭐ Closest to TreePrompt — uses MCTS for prompts but targets **code generation**, not multi-step NLP pipelines |
| **MCTS for Heuristic Design** | ICML 2025 | Progressive widening + exploration decay for LLM heuristic evolution | Comprehensive exploration of heuristic space | ⭐⭐ Tree search for program synthesis, not prompt optimization per se |
| **ReST-MCTS*** | NeurIPS 2024 | Process reward + MCTS for LLM self-training | Reinforced self-training via tree search | ⭐⭐ Tree search for training, not prompt optimization |
| **Survey: Auto Prompt Optimization** | arXiv 2025.02 | Taxonomy of optimization methods by where/what/criteria | Comprehensive landscape | ⭐⭐ Useful for related work positioning |

**Gap identified**: No existing work combines **tree-search prompt optimization** with **LLM-as-Judge evaluation** specifically for **multi-step NLP pipelines with step-specific quality criteria**. DSPy uses flat Bayesian search; MCTS-OPS targets code generation. TreePrompt's niche — tree search + judge + multi-step pipeline — remains open.

### 1.2 LLM-as-Judge

| Paper | Venue/Year | Method | Key Result | Relevance |
|-------|-----------|--------|------------|-----------|
| **Learning LLM-as-a-Judge** | ICLR 2025 | DPO training with contrastive judgment pairs | Better reasoning in judgment | ⭐⭐ Focuses on training judge models, not using them for optimization |
| **Active-Sampling Prompt Opt for Judge** | ACL 2025 | Active sampling for judge prompt optimization | Auto-optimization of judge prompts themselves | ⭐⭐⭐ Very relevant — optimizes judge prompts, but doesn't use judge to optimize other prompts |
| **Auto Concept Discovery for Judge** | OpenReview 2025 | Concept extraction for bias discovery | Automated bias identification | ⭐ Interesting but orthogonal |
| **Survey: LLM-as-a-Judge** | arXiv 2024.11 | Comprehensive taxonomy of judge paradigms | Formal classification | ⭐⭐ Background reference |

**Gap identified**: Existing work uses LLM-as-Judge for **evaluation** or **self-improvement of the judge itself**. No work uses a strong-model judge to **guide tree-search prompt optimization** for a multi-step pipeline executed by a weaker model (the asymmetric executor-judge architecture).

### 1.3 Knowledge Graph Construction from Text

| Paper | Venue/Year | Method | Key Result | Relevance |
|-------|-----------|--------|------------|-----------|
| **GraphRAG** (Microsoft) | arXiv 2024.04 | Hierarchical KG extraction + community summarization | Better broad-question answering | ⭐⭐⭐ Major competitor. But: no document provenance, no versioning, no concept identity tracking |
| **LLM-Empowered KG Construction Survey** | arXiv 2025.10 | Comprehensive survey of LLM+KG | Taxonomy of approaches | ⭐⭐ Background reference |
| **Nature: Refined KG Extraction** | Nature Sci. Rep. 2026 | Domain-adapted LLMs + multimodal fusion | Domain-specific KG | ⭐ Less relevant |
| **ACM: LLMs for KG Construction** | WWW Journal 2024 | Quantitative evaluation of LLM KG capabilities | Benchmark results | ⭐⭐ Evaluation methodology reference |

**Gap identified**: Most KG construction systems (including GraphRAG) treat extraction as a **stateless one-shot process**. Deep-Dream's **document-first, observation-based** architecture with versioned concept identity (family_id), episodic memory traces, and source-level provenance is architecturally novel. No system we found maintains **both** document-level source of truth AND graph-level semantic overlay with bidirectional traceability.

### 1.4 Hybrid Retrieval

| Paper | Venue/Year | Method | Key Result | Relevance |
|-------|-----------|--------|------------|-----------|
| **HybridRAG** (NVIDIA/BlackRock) | 2024 | KG + vector retrieval fusion | KG+vector > either alone | ⭐⭐⭐ Validates the hybrid approach. Uses 2 channels (KG + vector) |
| **Graph-Augmented Multi-Stage Reranking** | 2024 | Dense+sparse fusion → graph reranking | High-fidelity chunk retrieval | ⭐⭐⭐ Similar architecture but different reranking strategy |
| **RRF Hybrid RAG** | CEUR-WS 2024 | RRF(BM25 + TFIDF + Ngram) | RRF improves recall over single channel | ⭐⭐ Validates RRF approach |
| **GraphRAG Survey** | ACM Computing Surveys 2025 | Three-stage taxonomy: indexing → retrieval → generation | Comprehensive framework | ⭐⭐ Background reference |

**Gap identified**: Most systems fuse 2 channels. Deep-Dream's **3-channel fusion** (BM25 + vector + graph-BFS) with **post-fusion reranking** (node-degree boosting, MMR diversity, confidence weighting, temporal decay) is more sophisticated than any single published approach. The "concept fading" temporal decay mechanism is particularly novel.

### 1.5 Episodic Memory for LLM Agents

| Paper | Venue/Year | Method | Key Result | Relevance |
|-------|-----------|--------|------------|-----------|
| **AriGraph** | IJCAI 2025 | KG world models with episodic memory for agents | Agents build memory graphs while exploring environments | ⭐⭐⭐ Closest relative — agents construct episodic KG. But: designed for game environments, not document processing |
| **Graph-based Agent Memory Survey** | arXiv 2025.02 | Taxonomy of graph memory for agents | KG, temporal graphs, structured memory | ⭐⭐ Good positioning reference |
| **Time-Aware PKGs** | Medium 2025 | Lifespan events + factual/episodic split | Personal knowledge management with time | ⭐ Conceptually similar to Deep-Dream's Episode model |

**Gap identified**: AriGraph (IJCAI 2025) is the closest published work but targets **interactive game environments**. Deep-Dream targets **document understanding** with a fundamentally different architecture: document-first, observation-based, with source traceability and version evolution. The two systems share the episodic+semantic memory insight but solve different problems.

---

## 2. Structural Gaps & Opportunities

### Gap A: Tree-Search Prompt Optimization with LLM-as-Judge (HIGH NOVELTY)
- **What's missing**: No published work uses LLM-as-Judge to guide tree-search prompt optimization for multi-step NLP pipelines
- **TreePrompt's niche**: Asymmetric executor (cheap model) + judge (strong model) + tree search with backtracking
- **Risk**: DSPy is evolving fast; MCTS-OPS (Aug 2025) is close but targets code generation
- **Conference fit**: ACL, EMNLP, NeurIPS, ICLR

### Gap B: Document-First Concept Graph with Provenance (MEDIUM-HIGH NOVELTY)
- **What's missing**: No system maintains both document-level source of truth AND graph semantic overlay with bidirectional traceability
- **Deep-Dream's niche**: Observation-based architecture, family_id versioning, episode-level provenance
- **Risk**: GraphRAG is dominant; need clear differentiation
- **Conference fit**: WWW, KDD, ACL, EMNLP

### Gap C: 3-Channel Hybrid Retrieval with Multi-Stage Reranking (MEDIUM NOVELTY)
- **What's missing**: Most published systems use 2-channel fusion; 3-channel with MMR + temporal decay is unexplored
- **Deep-Dream's niche**: BM25 + vector + graph-BFS + node-degree + MMR + confidence + temporal decay
- **Risk**: Engineering contribution more than research contribution; hard to publish alone
- **Conference fit**: SIGIR, WWW, ECIR

### Gap D: Conversational Refinement for KG Extraction (MEDIUM NOVELTY)
- **What's missing**: Multi-round "find more" extraction with orphan recovery and adversarial refinement
- **Deep-Dream's niche**: Conversational extraction + orphan recovery + cross-window dedup
- **Risk**: Could be seen as an engineering trick; needs rigorous evaluation
- **Conference fit**: EMNLP, NAACL

### Gap E: Unified Concept Model (MEDIUM NOVELTY)
- **What's missing**: Entities/relations/observations as roles of the same primitive with shared versioning
- **Deep-Dream's niche**: Concept unification with family_id + version chain
- **Risk**: Too architectural/conceptual; hard to evaluate in isolation
- **Conference fit**: ISWC, K-CAP

---

## 3. Recommended Direction

**Primary**: Gap A (TreePrompt) — Highest novelty, clearest research questions, strongest A-conference fit.
**Strong alternative**: Gap A + Gap B combined — TreePrompt framework evaluated on a document-first KG pipeline (Deep-Dream itself as the testbed).
**Fallback**: Gap B (document-first KG) with Gap C (hybrid retrieval) as supporting contribution.

---

## Key References

1. DSPy / MIPROv2 — [dspy.ai](https://dspy.ai/api/optimizers/MIPROv2/)
2. MCTS-OPS — [arxiv.org/html/2508.05995v1](https://arxiv.org/html/2508.05995v1)
3. GraphRAG — [arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130)
4. AriGraph — [ijcai.org/proceedings/2025/0002.pdf](https://www.ijcai.org/proceedings/2025/0002.pdf)
5. Learning LLM-as-a-Judge — [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/09fd990b19b2e69cc4d20e9969e43f09-Paper-Conference.pdf)
6. Active-Sampling Judge Prompt Opt — [ACL 2025](https://aclanthology.org/2025.acl-industry.67.pdf)
7. HybridRAG (NVIDIA/BlackRock) — [community.netapp.com](https://community.netapp.com/t5/Tech-ONTAP-Blogs/Hybrid-RAG-in-the-Real-World-Graphs-BM25-and-the-End-of-Black-Box-Retrieval/ba-p/464834)
8. Auto Prompt Optimization Survey — [arxiv.org/html/2502.18746v2](https://arxiv.org/html/2502.18746v2)
9. Graph-based Agent Memory Survey — [arxiv.org/html/2602.05665v1](https://arxiv.org/html/2602.05665v1)
10. MCTS for Heuristic Design — [ICML 2025](https://openreview.net/forum?id=Do1OdZzYHr)
