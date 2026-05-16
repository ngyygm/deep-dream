## Summary of the Invention

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
