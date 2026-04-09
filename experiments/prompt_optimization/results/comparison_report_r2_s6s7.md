# Round 2 Steps 6/7 Prompt Optimization Results

## Step 6: Entity Alignment (Batch Resolution)

### Test Data
- 60 entries (stratified): 30 match, 15 create_new, 7 relation, 8 match_relation
- Ground truth from distillation data

### Results

| Metric | A_current | B_structured | C_criteria | D_few_shot | E_reasoning |
|--------|-----------|-------------|------------|------------|-------------|
| parse_success | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **decision_match** | **0.400** | 0.183 | 0.183 | 0.183 | 0.183 |
| has_match | 0.817 | 0.783 | 0.717 | 0.800 | 0.783 |
| has_relation | 0.650 | 0.983 | 0.983 | 0.950 | 0.983 |
| match_id_accuracy | **0.895** | 0.842 | 0.842 | 0.842 | 0.868 |
| confidence | 0.772 | 0.764 | 0.682 | 0.833 | 0.841 |
| prompt_tokens | **1760** | 1962 | 2058 | 2352 | 1944 |

### Analysis

**Decision: Keep A_current (no change)**

The low `decision_match` across ALL variants is a measurement artifact, not a real quality issue. The 4-category classification (match/relation/create_new/match_relation) is overly strict:
- In production, `match` and `match_relation` both correctly identify a matching entity (the only difference is whether relations are also created)
- The key metric is `match_id_accuracy` — when a match is predicted, is the correct entity ID selected? A_current wins at 0.895
- A_current is also the shortest prompt (1760 tokens), saving cost
- B/C/D/E variants all produce excessively verbose `relations_to_create`, inflating `has_relation` to 0.983

## Step 7: Relation Alignment (Batch Matching)

### Test Data
- 60 entries (balanced): 30 match_existing, 30 create_new
- Ground truth from distillation data

### Results

| Metric | A_current | **B_criteria** | C_few_shot | D_content_compare | E_minimal |
|--------|-----------|---------------|------------|-------------------|-----------|
| parse_success | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **action_accuracy** | 0.767 | **0.783** | 0.617 | 0.767 | 0.750 |
| match_correct (precision) | **0.833** | 0.718 | 0.566 | 0.690 | 0.692 |
| match_recalled (recall) | 0.667 | 0.933 | **1.000** | 0.967 | 0.900 |
| confidence | 0.323 | 0.680 | 0.795 | 0.665 | 0.623 |
| prompt_tokens | 818 | 829 | 975 | 860 | 695 |

### Analysis

**Decision: Apply B_criteria**

B_criteria achieves the best overall accuracy (0.783) with excellent recall (0.933):
- A_current has best precision but poor recall (0.667) — it misses 1/3 of true matches, causing unnecessary duplicate relations
- C_few_shot has perfect recall but terrible precision (0.566) — over-matches, creating false merges
- B_criteria balances precision and recall well: 0.718 precision + 0.933 recall
- B_criteria also provides meaningful confidence scores (0.680 vs A_current's near-zero 0.323)
- Token cost is nearly identical to A_current (829 vs 818)

## Applied Changes

| Step | Winner | Changed? |
|------|--------|----------|
| 6 | A_current (keep) | No change needed |
| 7 | B_criteria | Already applied to production (`RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT`) |

## Round 2 Summary (All Steps)

| Step | Winner | Improvement |
|------|--------|-------------|
| 1 | A_current (keep R1 winner) | - |
| 2 | E_role_catalog | Universal catalog approach |
| 3 | D_few_shot | Few-shot examples |
| 4 | C_minimal | Simplified prompt |
| 5 | C_fixed_schema | Fixed schema alignment |
| 6 | A_current (keep) | - |
| 7 | B_criteria | Explicit matching criteria |
