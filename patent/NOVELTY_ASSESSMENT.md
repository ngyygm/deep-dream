## Patentability Assessment

### Invention Summary

A method and system for real-time knowledge graph reasoning that uses multi-head attention mechanisms to selectively score and prune candidate reasoning paths *during* graph traversal (rather than post-hoc), combined with path-level cross-attention scoring that produces interpretable confidence scores and explanation maps. The core inventive concept is attention-guided incremental path expansion with a learned dynamic threshold for pruning at each hop.

### Overall Assessment
**PATENTABLE WITH AMENDMENTS** -- The core concept of applying attention *during* traversal as a pruning mechanism (rather than as a post-hoc scoring layer) appears novel. However, several dependent features overlap with known techniques and the independent claims may need narrowing to distinguish from RL-based path-finding methods (MINERVA family) and attention-based path scoring methods (PathCon family, MASKGR).

### Prior Art References Identified

| Ref ID | Reference | Type | Key Teaching |
|--------|-----------|------|--------------|
| Ref 1 | US20190362246A1 -- "Multi-hop KG Reasoning with Reward Shaping" (Das et al., 2018) | Patent | RL agent walks on KG; action drop-out prunes non-relevant paths during traversal |
| Ref 2 | MINERVA -- "Go for a Walk and Arrive at the Answer" (Das et al., 2018) | Paper | RL-based path finding on KG; agent navigates step-by-step selecting relations |
| Ref 3 | PathCon -- "Relational Message Passing for KG Completion" (Wang et al., KDD 2021) | Paper | Combines relational context + relational paths via message passing for link prediction |
| Ref 4 | MASKGR -- "Multi-hop Reasoning Over Sparse KGs with Deep Attention" (2025) | Paper | Attention-based dynamic completion with mask/pruning strategies for multi-hop KG reasoning |
| Ref 5 | CogKR -- "Cognitive Graph for Multi-hop Knowledge Reasoning" (Du et al., IEEE TKDE 2021) | Paper | Attention flow mechanism for end-to-end training; deterministic, differentiable path reasoning |
| Ref 6 | "Multihop KG Reasoning Incorporating Semantic Attention" (SPIE 2024) | Paper | RL-based multi-hop reasoning framework incorporating semantic attention mechanisms |

---

### Step 1: Preliminary Claim Elements

**Independent Claim 1 (Method):**
- (A) Receiving a knowledge graph query comprising a head entity and a query relation
- (B) Initializing a path set comprising the head entity
- (C) At each hop during graph traversal, computing attention scores over candidate next-relations conditioned on the query relation and partial path context
- (D) Retaining only top-k candidates above a learned dynamic threshold
- (E) Computing a final reasoning score for each retained path using cross-attention between the query relation representation and the full path representation
- (F) Outputting reasoning results with confidence scores and attention-based explanations

**Independent Claim 8 (System):**
- Graph storage module, query input module, attention-based path selector, cross-attention scoring module, output module

**Independent Claim 10 (CRM):**
- Non-transitory computer-readable medium storing instructions for Claim 1 method

---

### Step 2: Anticipation Analysis (Novelty)

#### Independent Claim 1 vs. Each Reference

| Claim Element | Ref 1 (US20190362246A1) | Ref 2 (MINERVA) | Ref 3 (PathCon) | Ref 4 (MASKGR) | Ref 5 (CogKR) |
|--------------|------------------------|-----------------|-----------------|----------------|----------------|
| (A) Receiving KG query | Yes -- query entity+relation input | Yes -- same | Yes -- head/tail pair | Yes -- query input | Yes -- query input |
| (B) Initialize path set | Yes -- agent starts at head entity | Yes -- same | No -- uses message passing, not traversal | Partial -- uses attention masking | Yes -- path initialization |
| (C) Attention scores over next-relations conditioned on query+partial path | **No** -- uses RL policy network, not attention scoring; action drop-out is stochastic, not attention-conditioned | **No** -- uses RL policy (LSTM+MLP), not attention over candidate relations | **No** -- message passing over neighborhoods, not attention-guided traversal expansion | **Partial** -- uses attention for dynamic completion but does not condition on partial path context during expansion | **Partial** -- attention flow mechanism but applied to cognitive graph structure, not for pruning candidate next-relations during traversal |
| (D) Retaining top-k above dynamic threshold | **Partial** -- action drop-out prunes but uses fixed probability, not learned dynamic threshold | **No** -- no top-k pruning; explores full action space | **No** -- no traversal pruning | **Partial** -- mask/pruning but strategy differs from dynamic threshold | **No** -- no pruning mechanism described |
| (E) Cross-attention scoring (query rep x full path rep) | **No** -- reward is binary (correct/incorrect target entity) | **No** -- reward is binary | **No** -- scoring via message passing aggregation | **No** -- scoring via attention-based completion, not cross-attention between query and path | **No** -- uses attention flow, not cross-attention for scoring |
| (F) Output with confidence + attention explanation | **No** -- outputs predicted entity only | **No** -- outputs predicted entity only | **No** -- outputs link prediction score | **No** -- no explicit explanation output | **Partial** -- attention weights provide some interpretability |

#### Verdict Per Reference

| Reference | Anticipates Claim 1? | Missing Elements |
|-----------|---------------------|------------------|
| Ref 1 (US20190362246A1) | **NOT ANTICIPATED** | Elements (C), (D), (E), (F) -- uses RL policy, not attention-based expansion with dynamic threshold and cross-attention scoring |
| Ref 2 (MINERVA) | **NOT ANTICIPATED** | Elements (C), (D), (E), (F) -- RL-based, no attention mechanism for path expansion |
| Ref 3 (PathCon) | **NOT ANTICIPATED** | Elements (B), (C), (D), (E), (F) -- fundamentally different approach (message passing, not traversal) |
| Ref 4 (MASKGR) | **NOT ANTICIPATED** | Elements (C) in full, (D) in full, (E), (F) -- attention for completion, not for guided path expansion during traversal |
| Ref 5 (CogKR) | **NOT ANTICIPATED** | Elements (C) in full, (D), (E) -- attention flow differs from attention-guided pruning during traversal |

**Conclusion: Independent Claim 1 is NOT ANTICIPATED by any single reference. Novel.**

#### Independent Claim 8 (System) vs. Each Reference

| Reference | Anticipates Claim 8? | Notes |
|-----------|---------------------|-------|
| Ref 1 | **NOT ANTICIPATED** | No "attention-based path selector" or "cross-attention scoring module" |
| Ref 2-6 | **NOT ANTICIPATED** | None describe a system with the specific attention-based path selector performing guided expansion |

**Conclusion: Independent Claim 8 is NOT ANTICIPATED. Novel.**

#### Independent Claim 10 (CRM)
Follows Claim 1. Since Claim 1 is novel, Claim 10 is also novel.

---

### Step 3: Obviousness Analysis (Inventive Step)

#### Key Combinations

| # | Primary Ref | Secondary Ref(s) | Missing Elements | Motivation to Combine | Obvious? |
|---|------------|-------------------|-----------------|----------------------|----------|
| 1 | Ref 2 (MINERVA) | Ref 4 (MASKGR) | (C) attention over candidates, (E) cross-attention scoring | Both address multi-hop KG reasoning; combining RL path-finding with attention is a known trend | **ARGUABLY OBVIOUS** -- Examiner could argue that replacing MINERVA's RL policy with attention scoring (as in MASKGR) is a known substitution. However, the *during-traversal pruning with dynamic threshold* aspect is not taught. |
| 2 | Ref 1 (US20190362246A1) | Ref 5 (CogKR) | (C) attention-conditioned expansion, (E) cross-attention scoring | Same field (KG reasoning); CogKR's attention flow could be argued as substitutable for RL policy | **WEAK OBVIOUSNESS** -- Different mechanisms (RL policy vs. attention flow); combining them requires non-trivial redesign. |
| 3 | Ref 4 (MASKGR) | Ref 3 (PathCon) | (C) attention over next-relations conditioned on partial path, (E) cross-attention scoring | Both use attention for KG reasoning; combining path context (PathCon) with pruning (MASKGR) is logical | **MODERATE RISK** -- Examiner may argue combining known attention-based pruning with known path context encoding is obvious. However, the *incremental expansion with dynamic threshold* is not taught. |

#### Obviousness Assessment Summary

The **highest risk** is Combination 1 (MINERVA + MASKGR), where an examiner could argue:
- MINERVA teaches incremental path expansion on KGs
- MASKGR teaches using attention for multi-hop KG reasoning with pruning
- A POSITA would naturally combine them by using attention instead of RL policy for path selection

**Counter-argument** (strengthens patentability):
- Neither MINERVA nor MASKGR teaches applying attention *during* traversal as a pruning mechanism with a *learned dynamic threshold*
- The combination would require redesigning the attention mechanism to operate incrementally on partial paths rather than on complete subgraphs
- The cross-attention scoring for explanation generation is not taught or suggested by either reference
- The dynamic threshold (mean + alpha * std) is a specific inventive feature not found in any reference

---

### Step 4: Cross-Model Examiner Review

Cross-model validation via `mcp__codex__codex` (gpt-5.4) was not available. This section should be completed with external examiner simulation before filing.

---

### Step 5: Jurisdiction-Specific Assessment

#### Under 35 USC 102/103 (US)
- **Novelty (102)**: **PASS** -- No single reference anticipates all claim elements
- **Non-obviousness (103)**: **PASS WITH RISK** -- The MINERVA + MASKGR combination poses moderate obviousness risk. The "during-traversal attention pruning with dynamic threshold" feature is the key distinguishing element. If this element is adequately supported in the specification, the claim should survive.

#### Under Article 22 CN Patent Law (CN)
- **新颖性 (Novelty)**: **通过 (PASS)** -- No single reference discloses all elements
- **创造性 (Inventive Step)**: **通过 (PASS)** -- The combination of attention-guided expansion + dynamic threshold + cross-attention scoring constitutes a non-obvious technical contribution over existing RL-based and embedding-based approaches

#### Under Article 54/56 EPC (EP)
- **Novelty (Art. 54)**: **PASS** -- No single prior art document discloses the invention
- **Inventive Step (Art. 56, Problem-Solution Approach)**:
  - **Objective technical problem**: Reducing latency of multi-hop KG reasoning while maintaining accuracy and interpretability
  - **Solution**: Attention-guided incremental path expansion with dynamic threshold pruning
  - **Not obvious**: The cited prior art does not suggest applying attention during traversal as a pruning mechanism rather than as a scoring or aggregation layer
  - **Verdict**: **PASS**

---

### Step 6: Recommended Claim Amendments

1. **Strengthen Claim 1 Element (C)**: Explicitly recite that the attention scores are computed *incrementally at each hop during traversal* (not on pre-computed paths). This distinguishes from MASKGR which operates on complete subgraph representations.
   - Suggested language: "computing, at each hop of an incremental graph traversal, attention scores over a set of candidate next-relations, wherein said attention scores are conditioned on both a representation of the query relation and an encoded representation of the partial path traversed thus far"

2. **Strengthen Claim 1 Element (D)**: Recite the dynamic threshold formula to add a specific technical feature not found in any prior art.
   - Suggested language: "retaining only candidate next-relations whose attention scores exceed a dynamic threshold, wherein said dynamic threshold is computed as a function of a statistical distribution of attention scores for the current hop"

3. **Add fallback dependent claim**: Claim that narrows to the specific dynamic threshold computation (mean + alpha * std) to provide a fallback if the broader dynamic threshold language is found obvious.

4. **Consider adding a "wherein" clause** to Element (E): specifying that the cross-attention produces both a confidence score AND an attention weight map serving as an explanation -- this dual-output feature is not taught in any reference.

5. **For system claims (Claim 8)**: Ensure the "attention-based path selector" is described with sufficient structural specificity in the specification to avoid means-plus-function interpretation under 35 USC 112(f).

---

### Risk Factors

1. **Highest risk**: An examiner finding MINERVA + MASKGR combination renders Claim 1 obvious, arguing that substituting attention for RL policy is a known optimization. Mitigation: emphasize the *incremental, during-traversal* nature of the attention mechanism and the dynamic threshold.

2. **Moderate risk**: An examiner finding the dynamic threshold is an obvious optimization of beam search (selecting top-k is well known; the threshold is arguably an obvious variant). Mitigation: emphasize that the threshold is *learned* and *data-adaptive*, not a fixed hyperparameter.

3. **Specification risk**: The specification must provide detailed enablement for the dynamic threshold computation and the cross-attention scoring mechanism. If only described at a high level, enablement rejections (112(a)) are possible.

4. **Prior art search completeness**: This assessment is based on publicly available references. A professional prior art search (e.g., via PatBase, Derwent) may uncover additional relevant patents or applications, particularly from major tech companies (Google, Microsoft, Baidu) that may have pending applications in this space.

5. **Freedom-to-operate**: Even if patentable, US20190362246A1 (Google/DeepMind) may cover aspects of the RL-based path traversal. If the invention can be practiced without infringing that patent's claims, this is not a blocking issue, but FTO analysis is recommended.
