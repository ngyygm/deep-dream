# Remember Pipeline Timing Report

Test date: 2026-05-14 | Text: 三国演义·长坂坡片段 (~800 chars/window, 1 window)

---

## 1. Pipeline Overview

```
Step 1 ── Cache check (LLM: embedding)
   │
Step 2 ── Entity extraction (LLM: extraction_client)
   │         ├─ initial extract (1 call)
   │         └─ refine rounds   (up to entity_rounds calls)
   │
Step 3 ── Entity dedup (CPU)
   │
┌──Step 4 ── Entity content writing (LLM: llm_client, parallel with Step 6)
│     │
│  Step 5 ── Entity quality gate (CPU)
│     │
│  ════════ entity_content_done[i].set() ════════
│     │
│  Step 9a ── Entity alignment Phase A (LLM: llm_client, overlaps Steps 6-8)
│             Per-entity exact_match calls
│
│     ┌──Step 6 ── Relation discovery (LLM: extraction_client, parallel with Step 4)
│     │     │        ├─ initial pair extraction (1 call)
│     │     │        ├─ orphan pair recovery   (1 call)
│     │     │        └─ refine rounds          (up to relation_rounds calls)
│     │     │
│     │  Step 7 ── Relation content writing (LLM: extraction_client)
│     │     │
│     │  Step 8 ── Relation quality gate (CPU)
│     │
│     ════════ extract_done[i].set() ════════
│
├──Step 9b ── Entity alignment Phase B (CPU, attaches relations)
│
Step 10 ── Relation processing + DB write
            ├─ 10a: fetch existing entities/relations (DB)
            ├─ 10b: relation matching loop (LLM: llm_client)
            ├─ 10c: refresh edges (DB)
            ├─ orphan cleanup (DB)
            ├─ entity/relation mentions (DB)
            └─ dream corroboration
```

### Dual-Client Architecture

| Client | Semaphore | Steps | Pool Size |
|---|---|---|---|
| `extraction_client` | Separate | 2-3, 6-8 | `max_concurrency: 3` |
| `llm_client` | Separate | 4-5, 7, 9-10 | `max_concurrency: 3` |

Steps 4 and 6 use **different** clients → no slot contention, true parallelism.

---

## 2. Run 1 — Empty Graph (First Write)

### 2.1 Per-Step Timing

| Step | Duration | Type | LLM Calls | Detail |
|---|---|---|---|---|
| **Step 1** cache check | **3.272s** | LLM | 1 | Embedding call for cache lookup |
| **Step 2** entity extract | **6.749s** | LLM | 4 | initial 3.2s→39 items, r1 1.3s→+7, r2 1.2s→+4, r3 1.0s→+6 |
| **Step 3** entity dedup | **0.000s** | CPU | — | Trivial for 56 items |
| **Step 4** entity content | **14.269s** | LLM | ~5-6 | 56 entities, batched writes |
| **Step 5** entity quality | **0.000s** | CPU | — | Dedup + validation |
| **Step 6** relation discovery | **30.706s** | LLM | 6 | initial 8.9s→56 pairs, orphan 2.6s→9 pairs, r1 3.8s→+14, r2 4.5s→+1, r3 6.1s→+0, r4 5.0s→+0 |
| **Step 7** relation content | **14.672s** | LLM | 4 | 78 relations, batched |
| **Step 8** relation quality | **0.000s** | CPU | — | Dedup + validation |
| **Step 9** alignment Phase A | **1.581s** | LLM | 1 | Only '文档名' needed exact_match; rest: no_candidates |
| **Step 9** alignment Phase B | **0.000s** | CPU | — | Attach relations to alignment result |
| **Step 9** entity mentions | **0.040s** | DB | — | Batch write |
| **Step 10a** fetch entities | **0.061s** | DB | — | Read existing entities |
| **Step 10a** read relations | **0.011s** | DB | — | Read existing relations |
| **Step 10b** process loop | **0.172s** | LLM | 0 | No existing relations to match |
| **Step 10c** refresh edges | **0.451s** | DB | — | Neo4j edge refresh |
| **Step 10** orphan cleanup | **0.251s** | DB | — | Remove orphaned nodes |
| **Step 10** relation mentions | **0.020s** | DB | — | Batch write |
| **Step 10d** dream corroboration | **0.007s** | CPU | — | No dream data |

### 2.2 Time Breakdown

| Category | Time | % |
|---|---|---|
| **LLM calls** (extraction_client) | ~52.1s | 59.8% |
| **LLM calls** (llm_client) | ~19.1s | 21.9% |
| **DB operations** | ~0.83s | 1.0% |
| **CPU processing** | ~0.17s | 0.2% |
| **Parallelism overlap** (Steps 4+6 concurrent) | saved ~14.3s | — |
| **Perceived total** | **~56.4s** | — |

### 2.3 Critical Path

```
Step 1 (3.3s) → Step 2 (6.7s) → [Step 4 ‖ Step 6] (max(14.3, 30.7)=30.7s)
→ Step 7 (14.7s) → Step 9 (1.6s) → Step 10 (0.7s) ≈ 57.0s
```

Steps 4+6 overlap saves **14.3s** vs sequential.

---

## 3. Run 2 — Existing Graph (311 entities)

### 3.1 Per-Step Timing

| Step | Duration | Type | LLM Calls | Detail |
|---|---|---|---|---|
| **Step 1** cache check | **0.003s** | DB | 0 | Cache hit, no LLM needed |
| **Step 2** entity extract | **7.027s** | LLM | 4 | initial 3.3s→42 items, r1 1.2s→+1, r2 1.7s→+1, r3 0.8s→+2 |
| **Step 3** entity dedup | **0.000s** | CPU | — | Trivial for 46 items |
| **Step 4** entity content | **12.039s** | LLM | 5 | 46 entities, batched |
| **Step 5** entity quality | **0.000s** | CPU | — | Dedup + validation |
| **Step 6** relation discovery | **32.103s** | LLM | 7 | initial 12.0s→47 pairs, orphan 1.0s→2 pairs, r1 4.8s→+9, r2 3.1s→+0, r3 3.6s→+2, r4 3.7s→+1, r5 3.8s→+0 |
| **Step 7** relation content | **8.897s** | LLM | 3 | 60 relations, batched |
| **Step 8** relation quality | **0.000s** | CPU | — | Dedup + validation |
| **Step 9** alignment Phase A | **40.128s** | LLM | 46 | Per-entity exact_match, ~2s each, max_concurrency=3 |
| **Step 9** alignment Phase B | **0.000s** | CPU | — | Attach relations |
| **Step 9** entity mentions | **0.085s** | DB | — | Batch write |
| **Step 10a** fetch entities | **0.047s** | DB | — | Read existing entities |
| **Step 10a** read relations | **0.035s** | DB | — | Read existing relations |
| **Step 10b** process loop | **18.242s** | LLM | ~30 | Per-relation matching, ~480-490 in / ~60 out tokens |
| **Step 10c** refresh edges | **0.664s** | DB | — | Neo4j edge refresh |
| **Step 10** orphan cleanup | **0.151s** | DB | — | Remove orphaned nodes |
| **Step 10** relation mentions | **0.017s** | DB | — | Batch write |
| **Step 10d** dream corroboration | **0.007s** | CPU | — | No dream data |

### 3.2 Step 9 Alignment — Outlier Analysis

| Entity | Duration | Note |
|---|---|---|
| 张飞 | 8.2s | Common name, many candidates |
| 据水断桥 | 9.0s | Uncommon phrase, slow LLM response |
| 喝退曹兵 | 9.1s | Uncommon phrase, slow LLM response |
| 诸葛亮 | 3.5s | Very common name, many candidates |
| Others (avg) | ~0.9s | Most hit no_candidates fast path |
| Candidate projections+encode | 0.478s (total) | CPU, one-time for all entities |

### 3.3 Time Breakdown

| Category | Time | % |
|---|---|---|
| **LLM calls** (extraction_client) | ~48.0s | 45.9% |
| **LLM calls** (llm_client) | ~70.4s | 67.3% |
| **DB operations** | ~1.00s | 1.0% |
| **CPU processing** | ~0.48s | 0.5% |
| **Parallelism overlap** (Step 9A overlaps Steps 6-8) | saved ~32.1s | — |
| **Total wall** | **78.6s** | — |

### 3.4 Critical Path

```
Step 2 (7.0s) → [Step 4 ‖ Step 6] (max(12.0, 32.1)=32.1s)
→ Step 9A (40.1s, started early after Step 5, but extends past extract_done)
→ Step 10 (19.1s) ≈ 78.6s
```

Step 9 Phase A starts after Step 5 (~19s in) but takes 40s total, becoming the bottleneck.

---

## 4. LLM Call Inventory

### 4.1 Per-Step LLM Call Count

| Step | Run 1 Calls | Run 2 Calls | Avg Duration/Call | Purpose |
|---|---|---|---|---|
| Step 1 cache | 1 | 0 | 3.3s / 0s | Embedding |
| Step 2 entity extract | 4 | 4 | ~1.7s | Extract + 3 refine |
| Step 4 entity content | ~5-6 | 5 | ~2.4s | Batch content writing |
| Step 6 relation discovery | 6 | 7 | ~4.6s | Pair extraction + orphan + refine |
| Step 7 relation content | 4 | 3 | ~3.3s | Relation content writing |
| Step 9 alignment | 1 | 46 | ~0.9-2.0s | Per-entity exact_match |
| Step 10 relation match | 0 | ~30 | ~0.6s | Per-relation dedup check |
| **Total LLM calls** | **~21** | **~95** | — | — |

### 4.2 LLM Time by Client

| Client | Run 1 | Run 2 | Contention? |
|---|---|---|---|
| `extraction_client` (Steps 2,6,7) | ~52.1s | ~48.0s | Independent pool |
| `llm_client` (Steps 4,9,10) | ~19.1s | ~70.4s | Independent pool |

---

## 5. Non-LLM Time

| Operation | Run 1 | Run 2 | Notes |
|---|---|---|---|
| Entity dedup (Step 3) | 0.000s | 0.000s | O(n log n) for ~50 items |
| Entity quality gate (Step 5) | 0.000s | 0.000s | Validation pass |
| Relation quality gate (Step 8) | 0.000s | 0.000s | Validation pass |
| Alignment Phase B (Step 9b) | 0.000s | 0.000s | CPU-only relation assembly |
| Candidate projections+encode | — | 0.478s | One-time vector projection |
| DB: entity mentions | 0.040s | 0.085s | Batch Cypher write |
| DB: fetch entities (Step 10a) | 0.061s | 0.047s | Read |
| DB: read relations (Step 10a) | 0.011s | 0.035s | Read |
| DB: refresh edges (Step 10c) | 0.451s | 0.664s | Neo4j refresh |
| DB: orphan cleanup | 0.251s | 0.151s | Delete |
| DB: relation mentions | 0.020s | 0.017s | Batch write |
| Dream corroboration | 0.007s | 0.007s | No-op |
| **Total non-LLM** | **~0.84s** | **~1.48s** | <2% of total |

**Conclusion**: Non-LLM time is negligible. The pipeline is **>98% LLM-bound**.

---

## 6. Parallelism Analysis

### 6.1 What Runs Concurrently

```
Timeline (Run 2):
t=0s    Step 1 (cache hit, instant)
t=0s    Step 2 (entity extract, 7.0s)
t=7.0s  Step 3 (dedup, instant)
t=7.0s  ┌─ Step 4 (entity content, 12.0s, llm_client)
t=7.0s  └─ Step 6 (relation discovery, 32.1s, extraction_client)
t=19.0s Step 5 done → Step 9 Phase A starts (llm_client, 40.1s)
        ╎  Step 4 ends → llm_client slots free for Step 9
t=39.1s Step 6 done → Step 7 starts (extraction_client, 8.9s)
t=48.0s Step 7 done → Step 8 (instant) → extract_done fires
t=48.0s Step 9 Phase B (instant, attaches relations from extract_results)
t=48.0s Step 9 entity mentions (DB, 0.085s)
t=57.1s Step 9 Phase A completes (was running since t=19s)
t=57.1s Step 10 starts (relation processing, 19.1s)
t=59.1s Step 9 entity mentions done (DB)
        ╎  Step 10b: ~30 LLM calls for relation matching
t=76.2s Step 10 completes
        ═══ Total: ~78.6s (including Step 10c refresh edges)
```

### 6.2 Overlap Gains

| Overlap | Saved | How |
|---|---|---|
| Step 4 ‖ Step 6 | ~12.0s | Dual client parallelism |
| Step 9A ‖ Steps 6-8 | ~32.1s | Early entity signal (Phase A starts after Step 5) |

### 6.3 Slot Utilization

**`extraction_client`** (max_concurrency=3):
- Active during Steps 2, 6, 7
- Idle during Steps 4, 5, 9, 10

**`llm_client`** (max_concurrency=3):
- Active during Steps 4, 9, 10
- Step 9 alignment (46 entities × ~2s each, concurrency=3) → ~31s wall time for 46 calls
- Step 10b relation match (~30 calls × ~0.6s each, concurrency=3) → ~6s wall time

---

## 7. Bottleneck Identification

### 7.1 Run 1 (Empty Graph)

| Rank | Bottleneck | Time | % of Total |
|---|---|---|---|
| 1 | Step 6 relation discovery | 30.7s | 54.4% |
| 2 | Step 7 relation content | 14.7s | 26.0% |
| 3 | Step 4 entity content | 14.3s | 25.3% |
| 4 | Step 2 entity extract | 6.7s | 11.9% |

**Primary bottleneck**: Relation discovery (Step 6) — 6 LLM calls, 4 of which are refine rounds that yield diminishing returns (r2: +1, r3: +0, r4: +0).

### 7.2 Run 2 (Existing Graph)

| Rank | Bottleneck | Time | % of Total |
|---|---|---|---|
| 1 | Step 9 entity alignment | 40.1s | 51.0% |
| 2 | Step 6 relation discovery | 32.1s | 40.8% |
| 3 | Step 10b relation match | 18.2s | 23.2% |
| 4 | Step 4 entity content | 12.0s | 15.3% |
| 5 | Step 7 relation content | 8.9s | 11.3% |

**Primary bottleneck**: Entity alignment (Step 9) — 46 per-entity LLM calls with concurrency=3. Grows linearly with entity count.

### 7.3 Growth Scaling Concerns

| Component | Scaling | Concern |
|---|---|---|
| Step 9 alignment | O(entities) × LLM call | **Critical** — 46 entities × 2s = 92s theoretical, 40s actual with concurrency=3 |
| Step 6 refine rounds | O(rounds) × LLM call | Diminishing returns after r1-r2; patience counter helps |
| Step 10b relation match | O(relations) × LLM call | ~30 calls × 0.6s, manageable |
| Step 4 entity content | O(entity_batches) × LLM call | Batched, moderate |
| DB operations | O(1) | Negligible |

---

## 8. Recommendations

### 8.1 High Impact

1. **Step 9 alignment batching** — Current: 1 LLM call per entity. Propose: batch multiple entities into a single alignment call (e.g., 5-10 entities/call). Could reduce 46 calls → ~5-10 calls, cutting Step 9 from 40s to ~10-15s.

2. **Step 6 refine early termination** — Run 1 r2-r4 added 0 net new pairs; Run 2 r2-r5 added 3 pairs across 4 rounds. The patience counter (2 consecutive empty) helps, but consider also stopping when `consecutive_low_yield >= 2` (e.g., ≤1 new item per round for 2 rounds).

3. **Step 9 `no_candidates` fast path** — In Run 1, 55/56 entities hit `no_candidates` instantly. In Run 2 with 311 graph entities, fewer hit this path. Consider: if graph has >N entities, skip the candidate projection for entities with low-frequency names.

### 8.2 Medium Impact

4. **Step 10b relation matching batching** — Currently 1 LLM call per relation. Could batch 5-10 relations per call. Would reduce ~30 calls → ~3-6 calls.

5. **Increase `max_concurrency`** — Both clients at 3. If LLM provider supports higher throughput, increasing to 5-6 would proportionally reduce Step 9 (40s → ~24s) and Step 6 (32s → ~19s).

### 8.3 Low Impact (Already Fast)

6. **DB operations** — Total <1s. No optimization needed.
7. **CPU processing** — Total <0.5s. No optimization needed.
8. **Cache check** — Already near-instant on cache hit (0.003s).

---

## 9. Configuration Reference

| Parameter | Value | Notes |
|---|---|---|
| `context_window_tokens` | 32,000 | Both clients |
| `window_size` | 800 chars | Source text chunk size |
| `overlap` | 100 chars | Window overlap |
| `entity_rounds` | 3 | Max refine rounds for entity extraction |
| `relation_rounds` | 5 | Max refine rounds for relation discovery |
| `max_concurrency` | 3 | Both clients (separate pools) |
| `max_tokens` | 16,000 | LLM max output tokens |
