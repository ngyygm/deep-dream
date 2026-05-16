# CLAIMS

1. A method for knowledge graph reasoning, comprising:
   receiving a knowledge graph query comprising a head entity and a query relation;
   initializing a path set comprising the head entity;
   at each hop during an incremental graph traversal, computing multi-head attention scores over a set of candidate next-relations, wherein said attention scores are conditioned on both a learned representation of the query relation and an encoded representation of a partial path traversed thus far;
   retaining only candidate next-relations whose attention scores exceed a dynamic threshold, thereby expanding the path set;
   after a termination condition is satisfied, computing a final reasoning score for each retained path using cross-attention between the learned representation of the query relation and a representation of the full path, wherein the cross-attention produces both a confidence score and an attention weight map; and
   outputting one or more reasoning results comprising target entities, the confidence scores, and the attention weight maps.

2. The method of claim 1, wherein the dynamic threshold is computed as a function of a statistical distribution of the attention scores at each hop.

3. The method of claim 2, wherein the dynamic threshold T is computed as T = mu + alpha * sigma, where mu is a mean of the attention scores for a current set of candidates, sigma is a standard deviation of the attention scores, and alpha is a learned parameter.

4. The method of claim 1, wherein the computing of multi-head attention scores is performed by a multi-head attention module comprising a plurality of attention heads, wherein at least one attention head is configured to specialize in a structural relation type and at least one attention head is configured to specialize in a semantic relation type.

5. The method of claim 1, wherein the encoded representation of the partial path is produced by a path encoder comprising a transformer encoder with positional encoding, wherein the positional encoding incorporates a hop distance from the query entity for each entity-relation pair in the partial path.

6. The method of claim 1, wherein the computing of the final reasoning score using cross-attention comprises computing a cross-attention score as s_j = softmax(q^T W_c p_j), where q is the learned representation of the query relation, p_j is a representation of a path, and W_c is a learned cross-attention weight matrix.

7. The method of claim 1, further comprising caching previously computed attention scores for frequently traversed subgraph patterns and reusing the cached attention scores when a new query involves a partial path that overlaps with a cached subgraph pattern.

8. The method of claim 1, further comprising processing a plurality of queries in a batch inference mode, wherein attention scores for shared subgraph regions are computed once and shared across the plurality of queries.

9. A system for knowledge graph reasoning, comprising:
   a graph storage module configured to store a knowledge graph comprising a plurality of entities and a plurality of relational edges connecting the entities;
   a query input module configured to receive a reasoning query comprising a head entity and a query relation;
   an attention-based path selector configured to perform incremental graph traversal by, at each hop, computing multi-head attention scores over candidate next-relations conditioned on the query relation and a partial path context, and retaining only candidate next-relations whose attention scores exceed a dynamic threshold;
   a cross-attention scoring module configured to compute reasoning scores for retained paths using cross-attention between a representation of the query relation and representations of the retained paths, wherein the cross-attention scoring module produces both a confidence score and an attention weight map; and
   an output module configured to provide reasoning results comprising target entities, the confidence scores, and the attention weight maps.

10. The system of claim 9, wherein the attention-based path selector comprises a multi-head attention module comprising a plurality of attention heads, a combiner configured to combine a query relation representation and a partial path encoding into a combined context vector, and a threshold module configured to compute the dynamic threshold as a function of a statistical distribution of the attention scores.

11. A non-transitory computer-readable storage medium storing instructions that, when executed by a processor, cause the processor to perform operations comprising:
    receiving a knowledge graph query comprising a head entity and a query relation;
    initializing a path set comprising the head entity;
    at each hop during an incremental graph traversal, computing multi-head attention scores over a set of candidate next-relations, wherein said attention scores are conditioned on both a learned representation of the query relation and an encoded representation of a partial path traversed thus far;
    retaining only candidate next-relations whose attention scores exceed a dynamic threshold, thereby expanding the path set;
    after a termination condition is satisfied, computing a final reasoning score for each retained path using cross-attention between the learned representation of the query relation and a representation of the full path, wherein the cross-attention produces both a confidence score and an attention weight map; and
    outputting one or more reasoning results comprising target entities, the confidence scores, and the attention weight maps.
