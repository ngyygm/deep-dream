# Citation Audit Report

**Date**: 2026-05-15
**Bib file**: workspace/refs.bib
**Paper file**: workspace/drafts/paper.tex
**Total entries**: 10
**Total citation uses**: 15

## Summary
| Verdict | Count |
|---------|------|
| KEEP    | 9    |
| FIX     | 1    |
| REPLACE | 0    |
| REMOVE  | 0    |

---

## Priority Fixes

### FIX: sukhbaatar2015memnn
- **Issue**: Author list is incomplete. The bib entry lists 3 authors (Sukhbaatar, Weston, Fergus), but the canonical NeurIPS 2015 publication has 4 authors: **Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus**. Missing author: **Arthur Szlam**.
- **Verification**: NeurIPS 2015 proceedings confirm 4 authors.
- **Context check**: All uses are appropriate. Cited correctly for "multi-hop reasoning over memory slots" in Related Work.
- **ACTION**: Add Arthur Szlam to author list in refs.bib.

---

## All-Clean Entries (no action needed)

### KEEP: bordes2013transe
- **Title**: Translating Embeddings for Modeling Multi-relational Data
- **Venue**: NeurIPS 2013 (Advances in Neural Information Processing Systems)
- **Authors verified**: Bordes, Usunier, Garcia-Duran, Weston, Yakhnenko -- CORRECT
- **Year**: 2013 -- CORRECT
- **Context**: Cited for "models relations as translations in embedding space" -- SUPPORTS (this is exactly what TransE does)
- **Used in**: paper.tex:43, paper.tex:155

### KEEP: dettmers2018conve
- **Title**: Convolutional 2D Knowledge Graph Embeddings
- **Venue**: AAAI 2018 (32nd AAAI Conference on Artificial Intelligence)
- **Authors verified**: Dettmers, Minervini, Stenetorp, Riedel -- CORRECT
- **Year**: 2018 -- CORRECT
- **Context**: Cited for "applies convolutional neural networks to capture richer interaction patterns" -- SUPPORTS
- **Used in**: paper.tex:43, paper.tex:155

### KEEP: sun2019rotate
- **Title**: RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
- **Venue**: ICLR 2019
- **Authors verified**: Sun, Deng, Nie, Tang -- CORRECT
- **Year**: 2019 -- CORRECT
- **Context**: Cited for "models relations as rotations in complex space" -- SUPPORTS
- **Used in**: paper.tex:43, paper.tex:155

### KEEP: jin2020renet
- **Title**: Recurrent Event Network: Autoregressive Structure Inference over Temporal Knowledge Graphs
- **Venue**: EMNLP 2020
- **Authors verified**: Jin, Qu, Jin, Ren -- CORRECT
- **Year**: 2020 -- CORRECT
- **Context**: Cited for "model temporal sequences with RNNs" and "employs recurrent neural networks to model sequences of KG snapshots" -- SUPPORTS
- **Used in**: paper.tex:26, paper.tex:47, paper.tex:155

### KEEP: zhu2021cygnet
- **Title**: Learning from History: Modeling Temporal Knowledge Graphs with Sequential Copy-Generation Network
- **Venue**: AAAI 2021
- **Authors verified**: Zhu, Chen, Fan, Cheng, Zhang -- CORRECT
- **Year**: 2021 -- CORRECT
- **Context**: Cited for "leverages historical fact repetition through copy mechanisms" -- SUPPORTS (this is exactly what CyGNet's copy-generation mechanism does)
- **Used in**: paper.tex:26, paper.tex:47, paper.tex:155

### KEEP: graves2014ntm
- **Title**: Neural Turing Machines
- **Venue**: arXiv preprint arXiv:1410.5401
- **Authors verified**: Graves, Wayne, Danihelka -- CORRECT
- **Year**: 2014 -- CORRECT
- **Context**: Cited for "introduce differentiable external memory with content-based addressing" -- SUPPORTS
- **Used in**: paper.tex:53

### KEEP: graves2016dnc
- **Title**: Hybrid Computing Using a Neural Network with Dynamic External Memory
- **Venue**: Nature, Volume 538, 2016
- **Authors verified**: Graves, Wayne, Reynolds, Harley, Danihelka, Grabska-Barwinska, Gomez Colmenarejo, Grefenstette, Ramalho, Agapiou, and others -- CORRECT (bib uses `and others` for 20 total authors, acceptable)
- **Year**: 2016 -- CORRECT
- **Context**: Cited for "introduce differentiable external memory with content-based addressing" and "content-based addressing" in Method section -- SUPPORTS
- **Used in**: paper.tex:53, paper.tex:105

### KEEP: vaswani2017attention
- **Title**: Attention Is All You Need
- **Venue**: NeurIPS 2017 (Advances in Neural Information Processing Systems)
- **Authors verified**: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin -- CORRECT
- **Year**: 2017 -- CORRECT
- **Context**: Cited for "sinusoidal encoding scheme" and "dot-product attention" -- SUPPORTS (both are core Transformer contributions)
- **Used in**: paper.tex:85, paper.tex:126

### KEEP: loshchilov2018adamw
- **Title**: Decoupled Weight Decay Regularization
- **Venue**: arXiv preprint arXiv:1711.05101 (published at ICLR 2019)
- **Authors verified**: Loshchilov, Hutter -- CORRECT
- **Year**: 2018 (arXiv) / 2019 (ICLR) -- NOTE: bib lists 2018 (arXiv year); canonical venue is ICLR 2019. This is a common convention and acceptable.
- **Context**: Cited for "AdamW optimizer" -- SUPPORTS
- **Used in**: paper.tex:143
- **NOTE**: Consider updating venue to `booktitle={International Conference on Learning Representations}, year={2019}` for the published version.

---

## Verification Sources
- DBLP (dblp.org) for venue and author verification
- arXiv for preprint metadata
- AAAI Proceedings (ojs.aaai.org) for ConvE and CyGNet
- NeurIPS Proceedings (papers.neurips.cc) for TransE, Attention Is All You Need
- OpenReview (openreview.net) for ICLR papers (RotatE, AdamW)
- Semantic Scholar for cross-reference
