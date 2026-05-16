## Patent Filing Documents -- Output Summary

### Generated Files

#### US (USPTO)
| File | Description | Status |
|------|-------------|--------|
| claims.md | Claims in US format (11 claims: 3 independent + 8 dependent) | Complete |
| specification.md | Full specification in USPTO order (Title through Abstract) | Complete |
| abstract.md | Abstract (US, ~150 words) | Complete |
| ads_template.md | Application Data Sheet skeleton | Complete |

### Claim Structure

| Claim | Type | Scope |
|-------|------|-------|
| 1 | Independent (Method) | Broadest: attention-guided path expansion + cross-attention scoring |
| 2 | Dependent on 1 | Dynamic threshold as function of score distribution |
| 3 | Dependent on 2 | Specific threshold formula (mean + alpha * std) |
| 4 | Dependent on 1 | Multi-head attention with relation-type-specific heads |
| 5 | Dependent on 1 | Path encoder with positional encoding |
| 6 | Dependent on 1 | Specific cross-attention formula |
| 7 | Dependent on 1 | Attention score caching |
| 8 | Dependent on 1 | Batch inference mode |
| 9 | Independent (System) | System with all modules |
| 10 | Dependent on 9 | Path selector architecture detail |
| 11 | Independent (CRM) | Non-transitory medium storing method instructions |

### Consistency Check

- [x] All 11 claims present in US format
- [x] Reference numerals consistent across specification sections (100-series system, 200-series method, 300-series attention module, 400-series example trace)
- [x] Language is English (correct for US jurisdiction)
- [x] Abstract is within 150-word limit
- [x] No "improved" or "new" in title
- [x] All claim elements have specification support (verified in SPECIFICATION_INDEX.md)
- [x] "comprising" used as transitional phrase in all independent claims
- [x] Antecedent basis correct throughout claims
- [x] FIG. references use correct format ("FIG. 1" not "Figure 1")

### Files from Previous Pipeline Stages

| File | Stage | Status |
|------|-------|--------|
| patent/INVENTION_DISCLOSURE.md | invention-structuring | Complete |
| patent/NOVELTY_ASSESSMENT.md | patent-novelty-check | Complete |
| patent/specification/*.md | specification-writing | Complete (7 sections + index) |
| patent/output/US/*.md | jurisdiction-format (US) | Complete (4 files) |

### Notes

- Cross-model validation (gpt-5.4 via Codex MCP) was skipped for all stages due to availability constraints. External review recommended before filing.
- No CLAIMS.md was pre-existing; claims were generated from the INVENTION_DISCLOSURE.md dependency map during this stage.
- Figures are described but not rendered. Actual figure drawings must be created separately (black and white line drawings per USPTO requirements).
- CN and EP formats were not generated (jurisdiction = US only). Use jurisdiction-format skill with CN or EP arguments to generate those formats.
