# Idea: Episodic Memory Networks for Knowledge Graph Reasoning

## Core Contribution
We propose Episodic Memory Networks (EMN), a memory-augmented reasoning framework for knowledge graphs that encodes temporal event sequences as episodic traces. Inspired by cognitive episodic memory, EMN organizes temporal KG facts into structured episodic memory slots for improved temporal reasoning.

## Problem
Knowledge graphs are dynamic — entities and relations evolve over time. Current temporal KG completion methods (RE-NET, CyGNet, RE-GCN) model time as a global index or recurrence signal, but fail to capture the episodic structure of events: that related events cluster into coherent episodes with causal and temporal dependencies.

## Method Overview

### Event Encoder
Encode each temporal fact (e_s, r, e_o, t) into a dense representation h_i = f(e_s, r, e_o) + p_i.

### Temporal Position Encoder
Dual encoding combining:
- Sinusoidal absolute time encoding (following Transformer positional encoding)
- Learned relative time interval encoding

### Episodic Memory Write
Segment facts into episodes via temporal clustering. Write to memory slots (M in R^{K x d}) using content-based addressing inspired by Neural Turing Machines.

### Episodic Memory Read
For a query (e_s, r, ?, t_q), retrieve relevant episodes via gated attention over the memory matrix.

### Reasoning Head
Combine retrieved episode representations with query embedding to predict the missing entity.

## Key Innovation
The episodic structuring of KG facts — rather than treating facts as isolated timestamped triples, we group temporally adjacent and causally related events into coherent episodic traces that can be efficiently retrieved and reasoned over.

## Notation
- G = {(e_s, r, e_o, t)} — temporal KG quadruples
- E — entity set, R — relation set
- M in R^{K x d} — episodic memory matrix with K slots
- h_i in R^d — embedding of fact i
- p_i in R^d — temporal position encoding

## Problem Formulation
Given G_{<=t} (all facts up to time t), predict (e_s, r, ?, t+1).

## Datasets
- ICEWS14 — Integrated Crisis Early Warning System, 2014 events
- ICEWS18 — Integrated Crisis Early Warning System, 2018 events
- GDELT — Global Database of Events, Language, and Tone

## Metrics
- MRR (Mean Reciprocal Rank)
- Hits@1, Hits@3, Hits@10

## Expected Results
5-8% MRR improvement over best prior temporal KG method on ICEWS14.

## Claims
1. Episodic memory traces improve link prediction on temporal KGs over static baselines
2. EMN outperforms existing temporal KG methods (RE-NET, CyGNet, HiSMatch) on benchmarks
3. The episodic encoder captures meaningful temporal patterns that are interpretable
4. Ablation shows both the event encoder and temporal position encoder contribute meaningfully
5. EMN scales efficiently — sub-linear memory growth with KG size
