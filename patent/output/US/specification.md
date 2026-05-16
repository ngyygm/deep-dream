# METHOD AND SYSTEM FOR REAL-TIME KNOWLEDGE GRAPH REASONING USING ATTENTION-GUIDED PATH EXPANSION

## CROSS-REFERENCE TO RELATED APPLICATIONS

[Not Applicable]

## STATEMENT REGARDING FEDERALLY SPONSORED RESEARCH

[Not Applicable]

## FIELD OF THE INVENTION

The present invention relates generally to knowledge graph processing and automated reasoning, and more particularly to methods and systems for performing multi-hop reasoning over knowledge graphs in real-time using attention-guided path expansion and scoring mechanisms.

## BACKGROUND OF THE INVENTION

Knowledge graphs are structured representations of information comprising entities connected by relational edges. Reasoning over knowledge graphs -- the task of inferring new facts or answering queries by traversing relational paths -- is a fundamental operation in applications including question answering, recommendation systems, semantic search, and automated knowledge discovery.

Existing approaches to knowledge graph reasoning can be broadly categorized into two families, each presenting significant technical limitations.

The first family comprises graph traversal-based methods. These methods, including breadth-first search (BFS) and depth-first search (DFS) with path ranking, enumerate candidate reasoning paths by systematically expanding paths from a query entity through the graph. While these methods provide interpretable reasoning paths and guaranteed completeness within a given search depth, they exhibit computational complexity that scales exponentially with path length. Specifically, for a knowledge graph having an average entity degree d, the number of candidate paths of length k grows as O(d^k). For knowledge graphs containing millions or billions of triples -- such as those used in production search and recommendation systems -- this exponential growth renders real-time multi-hop reasoning infeasible.

Reinforcement learning (RL)-based traversal methods have been proposed to address the scalability limitation. These methods train an RL agent to navigate the knowledge graph by selecting among outgoing relations at each entity, thereby reducing the effective search space. However, RL-based approaches present their own limitations: training instability due to sparse and delayed reward signals, difficulty generalizing to unseen query types, and sensitivity to hyperparameter selection. Furthermore, the RL agent's policy network typically selects actions based on a learned value function without providing transparent, attention-weighted explanations of why specific paths were preferred over others.

The second family comprises embedding-based methods. These methods, including translational models and neural network-based models, learn vector representations of entities and relations in a continuous embedding space. Inference is performed by computing similarity scores in the embedding space, which can be evaluated in constant time. While embedding-based methods achieve low-latency inference, they lack explicit reasoning paths and provide no structural interpretability. Additionally, embedding-based methods tend to degrade in accuracy on multi-hop queries that require compositional reasoning over multiple relational steps, because the embedding space compresses relational structure into fixed-dimensional vectors that may not preserve the compositional semantics of multi-step paths.

Attention mechanisms have been applied to knowledge graph tasks for scoring and aggregation. For example, graph attention networks have been used to weight entity neighborhoods for link prediction, and attention-based path scoring has been applied to pre-computed candidate paths. However, these applications use attention as a post-hoc scoring or aggregation mechanism applied to paths or subgraphs that have already been fully enumerated or constructed. They do not apply attention during the traversal process itself as a mechanism for dynamically pruning the search space.

No existing method jointly achieves real-time latency, multi-hop reasoning accuracy, and interpretable path-level explanations for knowledge graph reasoning at scale.

## BRIEF SUMMARY OF THE INVENTION

### Technical Problem

The technical problem to be solved is how to perform multi-hop reasoning over large-scale knowledge graphs in real-time (sub-second latency) given the combinatorial explosion of candidate reasoning paths that scales exponentially with path length, while maintaining reasoning accuracy comparable to exhaustive traversal methods and providing interpretable explanations for each reasoning result.

### Technical Solution

In accordance with one or more embodiments of the present invention, a method for real-time knowledge graph reasoning is provided that applies multi-head attention mechanisms to selectively score and prune candidate reasoning paths during incremental graph traversal, rather than exhaustively enumerating all paths before scoring or applying attention as a post-hoc scoring layer.

In one aspect, the present invention provides a method for knowledge graph reasoning, the method comprising: receiving a knowledge graph query comprising a head entity and a query relation; initializing a path set comprising the head entity; at each hop during an incremental graph traversal, computing attention scores over a set of candidate next-relations, wherein the attention scores are conditioned on both a representation of the query relation and an encoded representation of a partial path traversed thus far; retaining only candidate next-relations whose attention scores exceed a dynamic threshold, thereby expanding the path set; after a predetermined number of hops or upon satisfaction of a termination condition, computing a final reasoning score for each retained path using cross-attention between a learned representation of the query relation and a representation of the full path, wherein the cross-attention produces both a confidence score and an attention weight map; and outputting one or more reasoning results comprising target entities, confidence scores, and attention-based explanations.

In another aspect, the present invention provides a system for knowledge graph reasoning, the system comprising: a graph storage module configured to store a knowledge graph comprising a plurality of entities and relational edges; a query input module configured to receive a reasoning query comprising a head entity and a query relation; an attention-based path selector configured to perform incremental graph traversal by, at each hop, computing attention scores over candidate next-relations conditioned on the query relation and a partial path context, and retaining only candidate next-relations whose attention scores exceed a dynamic threshold; a cross-attention scoring module configured to compute reasoning scores for retained paths using cross-attention between a representation of the query relation and representations of the retained paths; and an output module configured to provide reasoning results comprising target entities, confidence scores, and attention weight maps as explanations.

In a further aspect, the present invention provides a non-transitory computer-readable storage medium storing instructions that, when executed by a processor, cause the processor to perform the method described above.

### Advantages

The present invention provides the following advantages over existing approaches.

Due to the attention-guided pruning during traversal, the method explores at most O(k x b) paths where k is the number of hops and b is a beam width, instead of O(d^k) paths where d is the average graph degree. This structural property enables sub-second inference on knowledge graphs exceeding ten million triples.

Because the attention mechanism is trained to prioritize semantically relevant relations conditioned on the query and partial path context, the method maintains reasoning accuracy within a margin of exhaustive path ranking methods while achieving substantially lower latency.

Due to the cross-attention scoring mechanism that produces attention weight maps over path constituents, the method provides human-interpretable explanations for each reasoning result, enabling verification and debugging of reasoning outputs.

Because the dynamic threshold is computed as a function of the attention score distribution at each hop, the pruning mechanism adapts its aggressiveness to query difficulty, enabling early termination for queries where high-confidence paths are found quickly.

## BRIEF DESCRIPTION OF THE DRAWINGS

FIG. 1 is a block diagram showing an overall system architecture for real-time knowledge graph reasoning in accordance with an embodiment of the present invention.

FIG. 2 is a flowchart illustrating a method for attention-guided knowledge graph reasoning in accordance with an embodiment of the present invention.

FIG. 3 is a data flow diagram showing an attention computation module for computing attention scores over candidate next-relations during path expansion in accordance with an embodiment of the present invention.

FIG. 4 is a sequence diagram showing an example reasoning trace over a sample knowledge graph, illustrating path expansion and pruning at each hop in accordance with an embodiment of the present invention.

## DETAILED DESCRIPTION OF PREFERRED EMBODIMENTS

The present invention will now be described in detail with reference to the accompanying drawings. While the invention is described in connection with specific embodiments, it should be understood that the description is not intended to limit the invention to the embodiments described, but rather to cover all alternatives, modifications, and equivalents as may be included within the scope of the appended claims.

### System Architecture (FIG. 1)

Referring to FIG. 1, a system 100 for real-time knowledge graph reasoning using attention-guided path expansion is shown. The system 100 comprises a graph storage module 102, a query input module 104, an attention-based path selector 106, a cross-attention scoring module 108, and an output module 110.

The graph storage module 102 is configured to store a knowledge graph comprising a plurality of entities and a plurality of relational edges connecting the entities. Each relational edge represents a semantic relationship between two entities. In one embodiment, the knowledge graph is stored as a set of triples (head entity, relation, tail entity). The graph storage module 102 may comprise a graph database, a triple store, or an indexed relational database configured for efficient neighbor lookup operations.

The query input module 104 is configured to receive a reasoning query comprising a head entity and a query relation. The reasoning query represents a question of the form "given head entity h and query relation r, what is the target entity t such that (h, r, t) holds in the knowledge graph?" The query input module 104 may receive queries from external applications via an application programming interface (API).

The attention-based path selector 106 is configured to perform incremental graph traversal from the head entity by, at each hop, computing attention scores over a set of candidate next-relations outgoing from a current entity in a partial path. The attention scores are conditioned on a representation of the query relation and an encoded representation of the partial path traversed thus far. The attention-based path selector 106 retains only candidate next-relations whose attention scores exceed a dynamic threshold, thereby expanding the path set while keeping its size bounded. The attention-based path selector 106 will be described in further detail below with reference to FIG. 3.

The cross-attention scoring module 108 is configured to compute a final reasoning score for each retained path using a cross-attention mechanism between a learned representation of the query relation and a representation of the full path. The cross-attention scoring module 108 produces both a confidence score indicating the likelihood that the path leads to a correct answer entity, and an attention weight map indicating the relative contribution of each path constituent to the confidence score. The attention weight map serves as a human-interpretable explanation for the reasoning result.

The output module 110 is configured to provide one or more reasoning results comprising target entities, confidence scores, and attention weight maps. The output module 110 may rank the results by confidence score and return the top-N results to the querying application.

### Method for Attention-Guided Knowledge Graph Reasoning (FIG. 2)

Referring to FIG. 2, a method 200 for real-time knowledge graph reasoning using attention-guided path expansion is shown. The method 200 may be performed by the system 100 of FIG. 1 or by a computing device comprising a processor and a memory storing instructions executable by the processor.

At step 202, the method 200 begins with receiving a knowledge graph query comprising a head entity and a query relation. For example, the query may comprise a head entity "Albert_Einstein" and a query relation "place_of_birth," and the goal of the reasoning process is to identify the target entity satisfying this relation.

At step 204, the method 200 initializes a path set. The path set is initialized to comprise a single path containing only the head entity. In one embodiment, each path in the path set is represented as a sequence of (entity, relation) pairs. At initialization, the single path comprises only the head entity with no relation.

At step 206, the method 200 determines whether a termination condition is satisfied. The termination condition may comprise one or more of: a predetermined maximum number of hops has been reached; a maximum path set size has been reached; or a confidence threshold has been met by one or more paths in the path set. If the termination condition is satisfied, the method proceeds to step 214. If the termination condition is not satisfied, the method proceeds to step 208.

At step 208, the method 200 computes attention scores over candidate next-relations for each path in the path set. For each path in the path set, the current entity (i.e., the last entity in the path) is identified, and the set of outgoing relations from the current entity in the knowledge graph is retrieved from the graph storage module 102. For each candidate next-relation, an attention score is computed. The attention score computation will be described in further detail below with reference to FIG. 3.

In one embodiment, the attention score for a candidate next-relation r_i given a query relation r_q and a partial path P is computed as follows. The query relation r_q is mapped to a learned query representation vector q. The partial path P is encoded into a path context vector c using a path encoder comprising a recurrent neural network or a transformer-based encoder that processes the sequence of (entity, relation) pairs in the partial path. The candidate next-relation r_i is mapped to a learned relation representation vector k_i. The attention score is computed as a dot product between a combined query-path representation (e.g., a concatenation or sum of q and c) and the relation representation k_i, scaled by the square root of the representation dimension. In a multi-head attention embodiment, this computation is performed independently by each of a plurality of attention heads, and the scores from each head are averaged or concatenated and projected to produce a final attention score.

At step 210, the method 200 applies a dynamic threshold to the attention scores to prune candidate next-relations. For each path in the path set, only candidate next-relations whose attention scores exceed the dynamic threshold are retained. In one embodiment, the dynamic threshold T for a given hop is computed as a function of the statistical distribution of attention scores at that hop. Specifically, T may be computed as T = mu + alpha * sigma, where mu is the mean of the attention scores for the current set of candidates, sigma is the standard deviation of the attention scores, and alpha is a learned parameter. The learned parameter alpha may be optimized during training to balance exploration (retaining more candidates) against computational efficiency (retaining fewer candidates). Alternatively, a fixed beam width b may be used to retain only the top-b scoring candidates.

At step 212, the method 200 expands the path set by extending each path in the path set with the retained candidate next-relations and the entities reached by those relations. The expanded path set replaces the previous path set, and the method returns to step 206 to evaluate the termination condition for the next hop.

At step 214, after the termination condition is satisfied, the method 200 computes a final reasoning score for each path in the path set using cross-attention between the query relation representation and the full path representation. For each path P_j, the full path is encoded into a path representation vector p_j using the path encoder. A cross-attention score is computed between the query representation q and the path representation p_j. In one embodiment, the cross-attention score is computed as s_j = softmax(q^T W_c p_j), where W_c is a learned cross-attention weight matrix. The cross-attention mechanism produces both a scalar confidence score for the path and an attention weight map over the constituents (entity-relation pairs) of the path, which serves as an explanation for why the path was scored highly.

At step 216, the method 200 outputs one or more reasoning results. Each reasoning result comprises a target entity (the terminal entity of the path), a confidence score, and an attention weight map. The results are ranked by confidence score, and the top-N results are returned.

### Attention Computation Detail (FIG. 3)

Referring to FIG. 3, the attention computation module 300 used by the attention-based path selector 106 is shown. The attention computation module 300 receives as input a query relation representation 302, a partial path encoding 304, and a set of candidate next-relation representations 306.

The query relation representation 302 is a learned vector representation of the query relation r_q. In one embodiment, each relation in the knowledge graph is associated with a learned embedding vector of dimension d (e.g., d = 256), which is learned during training of the system.

The partial path encoding 304 is produced by a path encoder 310. The path encoder 310 receives the sequence of (entity, relation) pairs constituting the partial path and produces a fixed-dimensional vector encoding. In one embodiment, the path encoder 310 comprises a bidirectional long short-term memory (BiLSTM) network that processes the sequence of entity and relation embedding vectors and outputs the final hidden state as the path encoding. In another embodiment, the path encoder 310 comprises a transformer encoder with positional encoding, where the positional encoding incorporates the hop distance from the query entity, so that the encoder is aware of the position of each (entity, relation) pair within the path.

A combiner 312 combines the query relation representation 302 and the partial path encoding 304 to produce a combined context vector. In one embodiment, the combiner 312 concatenates the two vectors and projects the concatenation through a learned linear transformation followed by a nonlinear activation function. In another embodiment, the combiner 312 computes an element-wise sum of the two vectors.

A multi-head attention module 314 computes attention scores between the combined context vector and each candidate next-relation representation 306. The multi-head attention module 314 comprises H attention heads (e.g., H = 8), where each head independently computes scaled dot-product attention between a projection of the combined context vector and projections of the candidate next-relation representations. The outputs of the H attention heads are concatenated and projected through a learned linear transformation to produce the final attention scores 316 for each candidate next-relation.

In one embodiment, different attention heads are configured to specialize in different relation types (e.g., structural relations, semantic relations, temporal relations). This specialization is learned during training and enables the multi-head attention module 314 to capture diverse aspects of relational relevance.

A threshold module 318 receives the attention scores 316 and computes the dynamic threshold T. As described above, T may be computed as T = mu + alpha * sigma, where mu and sigma are the mean and standard deviation of the attention scores, and alpha is a learned parameter. The threshold module 318 outputs a pruning mask 320 indicating which candidate next-relations have attention scores exceeding the dynamic threshold T.

### Example Reasoning Trace (FIG. 4)

Referring to FIG. 4, an example reasoning trace 400 is shown for a query (Albert_Einstein, place_of_birth, ?) over a sample knowledge graph.

At hop 0 402, the path set is initialized with a single path: [Albert_Einstein].

At hop 1 404, the outgoing relations from Albert_Einstein are retrieved. Suppose the outgoing relations include "studied_at," "published," "was_born_in," "citizen_of," and "worked_at." The attention-based path selector 106 computes attention scores for each candidate relation conditioned on the query relation "place_of_birth" and the partial path context. The relations "was_born_in" and "citizen_of" receive high attention scores because they are semantically related to the concept of birthplace. The dynamic threshold is computed, and the relations "studied_at," "published," and "worked_at" are pruned because their attention scores fall below the threshold.

At hop 2 406, the paths [Albert_Einstein, was_born_in, Germany] and [Albert_Einstein, citizen_of, Germany] are further expanded. From "Germany," outgoing relations such as "has_city" and "has_region" are evaluated. The relation "has_city" receives a high attention score, leading to expansion to entities such as "Ulm" and "Berlin." From "Germany" via "has_region," similar expansion occurs.

At hop 3 408, the terminal entities are reached. The cross-attention scoring module 108 scores each complete path against the query relation "place_of_birth." The path [Albert_Einstein, was_born_in, Germany, has_city, Ulm] receives the highest confidence score because the sequence of relations "was_born_in" followed by "has_city" is highly predictive of the birthplace entity. The attention weight map shows that the "was_born_in" relation received the highest attention weight, which is consistent with the reasoning logic.

### Embodiment Variations

In one embodiment, the path encoder 310 uses a transformer encoder with sinusoidal positional encoding. The positional encoding for the i-th (entity, relation) pair in the path is computed as PE(i, 2j) = sin(i / 10000^(2j/d)) and PE(i, 2j+1) = cos(i / 10000^(2j/d)), where d is the embedding dimension and j indexes the embedding dimension.

In another embodiment, the dynamic threshold is computed using a percentile-based approach instead of the mean-plus-standard-deviation formula. Specifically, the threshold T is set to the value at the (100 - p)-th percentile of the attention score distribution, where p is a learned parameter controlling the percentage of candidates retained.

In a further embodiment, the system includes an attention score cache configured to store previously computed attention scores for frequently traversed subgraph patterns. When a new query involves a partial path that overlaps with a cached subgraph pattern, the cached attention scores are reused to reduce computation.

In a further embodiment, the system supports batch inference mode, wherein multiple queries that share common subgraph regions are processed together, and the attention scores for shared subgraph regions are computed once and shared across queries.

In a further embodiment, the attention scores are temperature-scaled by dividing by a temperature parameter tau before applying the dynamic threshold, where tau controls the sharpness of the attention distribution. A lower tau produces a sharper distribution (more aggressive pruning), while a higher tau produces a flatter distribution (more exploration).

### Training

The system 100 is trained end-to-end using a training dataset comprising knowledge graph queries and corresponding ground-truth target entities. The training objective comprises a ranking loss that encourages the system to assign higher confidence scores to paths that lead to correct target entities and lower scores to paths that lead to incorrect entities. In one embodiment, the training objective is a margin-based ranking loss: L = max(0, margin + s_negative - s_positive), where s_positive is the cross-attention score for a correct path and s_negative is the cross-attention score for an incorrect path.

The learned parameters of the system include: the entity and relation embedding vectors, the parameters of the path encoder 310, the parameters of the multi-head attention module 314, the cross-attention weight matrix W_c, the dynamic threshold parameter alpha, and the temperature parameter tau (if used).

### Computing Environment

The system 100 may be implemented on a computing device comprising at least one processor and at least one memory. The memory stores instructions executable by the processor to perform the method 200. The computing device may be a server computer, a desktop computer, a laptop computer, or a distributed computing system. The knowledge graph may be stored in the memory or in a separate storage device accessible to the processor. In embodiments where the system is deployed for real-time serving, the computing device may include one or more graphics processing units (GPUs) or tensor processing units (TPUs) to accelerate the attention computation and path encoding operations.

## ABSTRACT

A method and system for real-time knowledge graph reasoning using attention-guided path expansion is disclosed. The method receives a knowledge graph query comprising a head entity and a query relation, and performs incremental graph traversal by computing, at each hop, multi-head attention scores over candidate next-relations conditioned on the query relation and a partial path context. Candidate next-relations whose attention scores fall below a learned dynamic threshold are pruned, bounding the number of explored paths. After traversal, a cross-attention mechanism scores each retained path against the query relation, producing a confidence score and an attention weight map that provides a human-interpretable explanation. The system comprises a graph storage module, a query input module, an attention-based path selector, a cross-attention scoring module, and an output module. The attention-guided pruning during traversal reduces computational complexity from O(d^k) to O(k times b), enabling sub-second inference on knowledge graphs with millions of triples while maintaining reasoning accuracy.
