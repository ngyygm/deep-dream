# Round 2 Prompt Optimization Experiment Report

Generated: 2026-04-04
Test windows: 20 (distill: 红楼梦 6, 三国演义 6, 水浒传 4, 西游记 4)
Cache continuity: 6 scenarios (same-doc, cross-doc, cross-type, resource sequences)
LLM: GLM-4-Flash

---

## Step 1: Memory Cache Update (3 variants)

| Metric | A_current (R1 winner) | B_entity_first | C_timeline |
|--------|----------------------|----------------|------------|
| parse_success | 1.000 | 1.000 | 1.000 |
| prompt_tokens | 1179 | 1152 | **1134** |
| summary_length | 513 | 541 | 515 |
| summary_verbatim_ratio | **0.208** | 0.208 | 0.211 |
| anchor_coverage | 0.850 | 0.850 | 0.850 |
| thinking_specificity | 0.000 | 0.000 | 0.000 |

**Analysis**: All three variants perform identically within noise margin. The cache format differences (entity list vs timeline) don't significantly affect summary quality for GLM-4-Flash.

**Winner**: `A_current` (keep R1 winner) — C_timeline is shortest prompt but no quality advantage.

## Step 2: Entity Extraction (5 variants)

| Metric | A_current | B_one_line_def | C_full_taxonomy | D_few_shot | **E_role_catalog** |
|--------|-----------|---------------|----------------|-----------|-------------------|
| parse_success | 1.000 | 1.000 | 0.950 | 1.000 | **1.000** |
| entity_count | 30.8 | 29.0 | 28.8 | 23.6 | **31.9** |
| noise_entity_ratio | 0.004 | **0.002** | 0.052 | 0.016 | **0.000** |
| avg_content_length | **19.6** | 15.0 | 14.1 | 12.6 | 15.9 |
| concept_diversity | 5.9 | 0.1 | 5.7 | 6.8 | **9.0** |
| name_type_coverage | 0.455 | 0.004 | 0.949 | 0.950 | **0.998** |
| prompt_tokens | 1299 | 1394 | 1410 | 1720 | **1320** |

**Analysis**:
- **E_role_catalog** is the clear winner: **zero noise**, highest entity count (31.9), highest concept diversity (9.0), near-perfect type coverage (0.998), and shortest prompt among new variants.
- **D_few_shot** has the highest prompt cost (1720 tokens) but produces fewer entities with less diversity.
- **C_full_taxonomy** has 5% parse failures and 5.2% noise — the taxonomy list may confuse GLM-4-Flash.
- **B_one_line_def** has near-zero type coverage (0.004) — the model ignores the concept definition without structured output requirements.

**Winner**: `E_role_catalog` — Zero noise, most diverse entities, best type coverage, efficient prompt.

## Step 3: Relation Extraction (5 variants)

| Metric | A_current | B_open_describe | C_matrix_10types | **D_few_shot** | E_triple_form |
|--------|-----------|----------------|-----------------|---------------|---------------|
| parse_success | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| relation_count | 31.0 | 32.3 | 30.6 | 29.3 | 27.9 |
| avg_content_length | 17.9 | **20.2** | 17.4 | 14.6 | 0.0* |
| mention_pattern_ratio | 0.015 | 0.068 | 0.028 | 0.085 | **0.000** |
| generic_pattern_ratio | 0.006 | **0.000** | 0.001 | 0.004 | **0.000** |
| endpoint_validity | 0.048 | 0.000 | 0.796 | **0.885** | 0.582 |
| prompt_tokens | 1445 | **1357** | 1500 | 1757 | 1452 |

*E_triple_form uses `relation`+`detail` fields instead of `content`, so avg_content_length=0 is misleading.

**Analysis**:
- **D_few_shot** has the best endpoint validity (0.885) — few-shot examples help the model correctly match entity names.
- **B_open_describe** has the longest descriptions but highest mention patterns (0.068) — removing constraints lets through "X提到Y" noise.
- **E_triple_form** has zero mention/generic patterns and uses structured `relation` verb phrases, but its endpoint validity is moderate (0.582).
- **A_current** (6-type constraint) has low mention patterns (0.015) but terrible endpoint validity (0.048) — the `[关系类型]` prefix format confuses entity matching.

**Winner**: `D_few_shot` — Best endpoint validity (0.885), zero parse failures, good balance. Though mention patterns are elevated (0.085), the few-shot format ensures entity names are correctly used.

**Runner-up**: `E_triple_form` — Zero mention patterns, structured verb phrases. If endpoint matching can be improved post-hoc, this is a strong alternative.

## Step 4: Supplement Entities (3 variants)

| Metric | A_current | B_concept_def | **C_minimal** |
|--------|-----------|--------------|---------------|
| parse_success | 1.000 | 0.900 | **1.000** |
| content_depth | 16.4 | 33.1 | **34.9** |
| name_match_rate | **0.950** | 0.944 | **0.950** |
| prompt_tokens | 1689 | 1321 | **1204** |

**Analysis**:
- **C_minimal** wins decisively: shortest prompt (1204 tokens), 100% parse success, deepest content (34.9 chars), best name matching.
- **B_concept_def** adds concept definition but causes 10% parse failures — GLM-4-Flash produces too verbose output that sometimes breaks JSON.
- The minimal prompt paradox: fewer instructions → model focuses on the task itself → better results.

**Winner**: `C_minimal` — Shortest prompt, 100% parse success, deepest content, best name matching. Less is more.

## Step 5: Entity Enhancement (5 variants)

| Metric | A_current | B_plain_text | **C_fixed_schema** | D_bullet_enhanced | E_few_shot |
|--------|-----------|-------------|-------------------|------------------|-----------|
| parse_success | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 |
| enhanced_length | 126.7 | 72.3 | **178.4** | 139.5 | 148.0 |
| content_expansion_ratio | 0.158 | 0.090 | **0.223** | 0.174 | 0.185 |
| novelty_ratio | 0.513 | 0.465 | **0.524** | 0.559 | 0.503 |
| section_alignment | 0.000 | 0.250 | **1.000** | 0.250 | 0.725 |
| prompt_tokens | 1365 | **1248** | 1401 | 1308 | 1612 |

**Analysis**:
- **C_fixed_schema** is the definitive winner: **perfect section alignment (1.0)**, longest output (178.4 chars), highest expansion ratio (0.223), 100% parse success. This fixes the `###` heading bug from Round 1.
- **A_current** has 0.0 section alignment — confirms the `###` vs `##` heading mismatch bug.
- **E_few_shot** achieves 0.725 section alignment — good but not perfect, since the few-shot examples use `##` headings which the model sometimes deviates from.
- **D_bullet_enhanced** has 5% parse failures — the bullet format sometimes confuses the JSON parser.

**Winner**: `C_fixed_schema` — Perfect schema alignment, longest output, fixes the h3 bug.

## Cache Continuity (3 variants, 6 scenarios)

| Metric | A_current | B_entity_first | C_timeline |
|--------|-----------|----------------|------------|
| parse_success | 1.000 | 1.000 | 1.000 |
| entity_retention_rate | 0.000 | 0.000 | 0.000 |
| new_info_coverage | 0.030 | **0.050** | 0.016 |
| cache_length_growth | 1.083 | 0.963 | **1.389** |
| verbatim_ratio_w1 | 0.214 | 0.206 | **0.181** |
| verbatim_ratio_w2 | 0.228 | 0.245 | **0.183** |
| verbatim_ratio_w3 | 0.191 | **0.190** | 0.187 |
| cross_doc_retention | 0.000 | 0.000 | 0.000 |

**Analysis**:
- All variants show **entity_retention_rate = 0.0** — this is expected because the cache is a prose summary, not an explicit entity list. The 2-4 char Chinese name matching proxy is too strict for summary text.
- **C_timeline** has the best verbatim scores (lowest copy-paste) and highest growth (1.389x), suggesting it produces more distinctive summaries.
- **B_entity_first** has the best new_info_coverage (0.050) — explicitly listing entities helps retain new information.
- Overall cache performance is similar across variants. The cache format matters less than expected for GLM-4-Flash.

**Conclusion**: Keep `A_current` for cache — no significant improvement from alternatives.

---

## Overall Winners (Round 2)

| Step | R1 Winner | R2 Winner | Change? | Key Improvement |
|------|-----------|-----------|---------|-----------------|
| 1 | B_streamlined | A_current (=B_streamlined) | No | No improvement from entity/timeline formats |
| 2 | C_two_tier | **E_role_catalog** | **Yes** | Zero noise, +53% diversity, +99% type coverage |
| 3 | C_type_constrained | **D_few_shot** | **Yes** | +18.5x endpoint validity (0.885 vs 0.048) |
| 4 | A_baseline | **C_minimal** | **Yes** | -29% prompt tokens, +113% content depth, 100% parse |
| 5 | C_structured (bug) | **C_fixed_schema** | **Yes** | Fixes h3 bug, perfect section alignment, +41% length |
| 6 | B_trimmed | (not retested) | No | R1 winner stands |
| 7 | B_with_criteria | (not retested) | No | R1 winner stands |
| Cache | A_current | A_current | No | No significant difference |

## Action Items

1. Apply to `processor/llm/prompts.py`:
   - Step 2: Replace with `E_role_catalog` (S2_R2_E_ROLE_CATALOG)
   - Step 3: Replace with `D_few_shot` (S3_R2_D_FEW_SHOT)
   - Step 4: Replace with `C_minimal` (S4_R2_C_MINIMAL)
   - Step 5: Replace with `C_fixed_schema` (S5_R2_C_FIXED_SCHEMA)
2. Step 3 evaluator: Update to handle `relation`+`detail` fields from E_triple_form (for future rounds)
3. Cache continuity: entity_retention_rate metric needs refinement — use semantic matching instead of exact substring
4. Future: Test E_role_catalog and D_few_shot on diverse resource inputs (code, logs, chat)
