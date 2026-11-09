# REVIEWS_RAW — ICLR 2027 Submission44989 (Deep-Dream)

- 来源: OpenReview, 用户于 2026-11-09 粘贴
- 评审意见发布/修改时间: 2026-10-20 ~ 10-31, 集体修改于 2026-11-06 20:34
- 总分: f1J5=2 (weak rejection, conf 3) / mV1D=1 (clear rejection, conf 3) / cB3k=1 (clear rejection, conf 4)

---

## Official Review of Submission44989 by Reviewer f1J5

Rating: 2: Weak rejection
Confidence: 3: I am familiar with this topic and its relevant literature.
Presentation: 3: Clear and easy to follow.
Informativeness: 2: I learned something interesting or useful.
Soundness: 1: I have significant concerns about the paper's soundness that do not appear easy to address.

### Paper Summary:
The paper proposed and studied a new approach to management of the long-term memory of agents. The proposed approach was motivated by the difficult tradeoff when designing the memory system: If we pre-process the information to extract the most relevant information in advance before seeing a question, we might miss critical information needed for answering a question (later), but coping the entire history would be too expensive. The proposed approach addresses this challenge by keeping both representation of the original information and extracted information (linked to the original information). At query time, the agent can access both layers of representation via the links, thus being able to both access the most relevant information and increase recall as needed by reaching the original information. Experiment results show that the proposed approach outperforms multiple state of the art methods.

### Critical Strengths:
- The memory management problem for intelligent agent is an important yet difficult problem. Although many approaches were proposed, there is no clear winner yet.
- The proposed approach is conceptually simple and general
- Evaluation was done on multiple datasets with comparison with multiple baselines.

### Critical Weaknesses:
The main weakness is that the discussion of related work is not comprehensive, missing some more advanced approaches that may be similar to the proposed approach. As an example, the following ICML'26 paper is closely related and the idea proposed there appears to be more comprehensive: PlugMem: A Task-Agnostic Plugin Memory Module for LLM Agents Ke Yang*, Zixi Chen*, Xuan He*, Xuan He*, Jize Jiang*, Michel Galley, Chenglong Wang, Jianfeng Gao, Jiawei Han, and ChengXiang Zhai https://openreview.net/forum?id=NWKaQIKoGp As such the novelty of the work isn't so clear and its effectiveness may also be questionable since there's no comparison with more closely related approaches.

### Questions For Authors:
Can you explain the similarity and difference of your work and PlugMem?

Flag For Ethics Review: No

---

## Official Review of Submission44989 by Reviewer mV1D

Rating: 1: Clear rejection
Confidence: 3: I am familiar with this topic and its relevant literature.
Presentation: 1: Difficult to follow despite careful reading.
Informativeness: 2: I learned something interesting or useful.
Soundness: 1: I have significant concerns about the paper's soundness that do not appear easy to address.

### Paper Summary:
This paper proposes Deep-Dream, an agentic long-term memory framework designed to preserve historical evidence while supporting evolving entity identities and adaptive information retrieval. Deep-Dream leaves source documents unchanged, and maintains concepts, relations, observations, and versions as a linked access layer. At query time, the agent follows these links across documents, opens the source spans that are needed, and can continue to a multi-hop trail when the first evidence is insufficient. During ingestion, uncertain mentions are kept as candidate entity families instead of being forced into one identity; later observations and relation neighborhoods can align or redirect those candidates without deleting the original evidence.

### Critical Strengths:
The paper addresses important challenges in agentic long-term memory, particularly preserving historical evidence while accommodating evolving entity identities and supporting efficient retrieval under limited context windows. The proposed separation between immutable source observations and revisable concept, relation, and version links is conceptually reasonable. Maintaining access to original evidence rather than relying exclusively on compressed memory representations may help preserve details needed for historical queries and multi-document reasoning.

### Critical Weaknesses:
A fundamental concern is that the paper does not sufficiently explain how the proposed Temporal Concept Graph is constructed in practice. Section 3.2 states that entities and relations are extracted from Episodes and that candidate identities are compared against existing concept families. However, the specific algorithms, matching criteria, and decision mechanisms underlying these operations remain unclear.

This is a significant limitation because accurate relation extraction and entity alignment are essential to the proposed framework. Errors in these processes may introduce incorrect cross-document connections or fragment the history of the same entity. Without a clear description and direct evaluation of these mechanisms, it is difficult to assess their reliability, reproducibility, and generalizability.

Another major concern is the lack of a clearly specified retrieval and expansion strategy. Section 3.4 states that the agent may use lexical, semantic, concept, or relation-based retrieval and continue searching when the available evidence is insufficient. However, the paper does not adequately explain how the agent selects among these retrieval methods, constructs subsequent search queries using previously retrieved information, chooses which graph neighbors to explore, or determines when to stop retrieving.

Consequently, the proposed progressive retrieval procedure appears closer to a general LLM-driven tool-use workflow than to a clearly defined novel retrieval algorithm. The paper should better distinguish its methodological contribution from existing agentic retrieval approaches.

### Questions For Authors:
1. Could the authors provide a detailed description of the entity and relation extraction procedures, including the models, prompts, matching criteria, and algorithms used to construct concept families and relation links? In particular, how are implicit or ambiguous relationships across different documents identified and validated?

2. Could the authors clarify how the agent chooses between lexical search, semantic search, concept exploration, and relation-based expansion? How are subsequent queries generated from previously retrieved evidence, and what criteria determine whether sufficient evidence has been collected? A concrete algorithm or detailed pseudocode would help clarify the actual technical contribution beyond general LLM-driven tool use.

3. What is the independent contribution of the Temporal Concept Graph to retrieval performance? Given that the offline graph-channel ablation does not demonstrate consistent recall improvements, could the authors conduct an online ablation comparing the full Deep-Dream system against an otherwise identical agentic retrieval system without the graph? Both systems should have access to the same original documents, language model, retrieval tools, and computational budget.

4. How much of the reported performance improvement is attributable to additional inference resources? Could the authors provide resource-matched comparisons that control the number of LLM calls, total token consumption, and retrieval budget? In particular, would a simpler iterative RAG system achieve comparable performance when given the same source access and computational resources as Deep-Dream?

Flag For Ethics Review: No

---

## Official Review of Submission44989 by Reviewer cB3k

Rating: 1: Clear rejection
Confidence: 4: I have substantial direct research experience on this topic.
Presentation: 2: Generally understandable, but requires substantial effort in places.
Informativeness: 2: I learned something interesting or useful.
Soundness: 3: I have no substantive concerns about the paper's soundness.

### Paper Summary:
This paper proposes Deep-Dream. It preserves source documents and exact text locations, while organizing concepts, relations, observations, and versions into a revisable access layer. It allows the agent to navigate and read source text over multiple steps. The system also uses delayed entity alignment. It merges candidate entities or redirects links after more context becomes available. The authors report results on several memory benchmarks. However, the experiments mainly compare different reading pipelines within the system. They do not establish the independent value of the new designs through fair external comparisons and sufficient component ablations.

### Critical Strengths:
- The paper separates original evidence, entity interpretations, and retrieval results. Exact source locations and access traces make it easier to audit the evidence behind an answer.
- The authors disclose budget differences, the lack of gains in the graph retrieval ablation, and deletion failures. These details help readers understand the limits of the results.

### Critical Weaknesses:
1. The novelty is not clearly established. Graphs have been widely used for memory organization, associative retrieval, and dynamic updates. Examples include A-MEM (NeurIPS 2025), AriGraph (IJCAI 2025), G-Memory (NeurIPS 2025), and GAM (ACL 2026). In particular, Zep already preserves lossless Episodes, links semantic records back to their sources, and tracks relation history using both system time and fact validity time. Preserving source text, source links, and history is therefore not enough to establish novelty on its own. Although the paper cites some closely related work, it does not clearly explain what new capabilities delayed alignment and version access add.

2. Missing direct comparisons with key external baselines under a shared setup. Table 1 compares the paper's own reading configurations. It cannot establish an advantage over existing memory methods. Table 2 combines results with different answer models, judges, evaluation protocols, and data subsets. It therefore does not establish a reliable advantage over Zep, HippoRAG 2, A-MEM, or Hindsight. The current experiments cannot determine whether the gains come from the memory design, model capabilities, source access, or inference budgets.

3. The effects of individual components are not sufficiently isolated. Source reading adds tools, more iterations, and roughly 12 times the token usage. Delayed alignment is bundled with batching and parallel execution. Version links and provenance checks lack separate ablations. The offline graph ablation shows no consistent recall gains, and entity merges are not evaluated against ground-truth labels. It therefore remains unclear which new mechanisms are effective and whether these storage and access designs provide measurable benefits beyond engineering integration.

### Questions For Authors:
1. What is the distinct contribution over the graph-based memory methods listed above? In particular, what capabilities do delayed entity alignment and version links add beyond Zep's source tracing and relation history?

2. Could the authors directly compare Deep-Dream with Zep, HippoRAG 2, A-MEM, and Hindsight using the same answer model, judge, data subset, source access, and call/token budgets?

3. Could the authors separately ablate graph navigation, version links, delayed alignment, and provenance checks? Could they also evaluate entity alignment against ground-truth labels and test both current-state and historical-state queries? These experiments would help show which mechanisms drive the gains, rather than extra reading compute or batching improvements.

4. Figure 1 and Figure 3 appears to be an unedited AI-generated SVG and is visually unappealing. I suggest that the authors revise it to improve its visual quality.

Flag For Ethics Review: No
