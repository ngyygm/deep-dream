---
name: reality-driven-testing
description: >
  Use when testing or verifying any CLI, API, or tool system. Drives testing from
  real data and real user questions rather than synthetic examples. The core idea:
  first understand what data actually exists, then design real questions users would
  ask, then run each command against real data, compare output to ground truth, and
  fix every gap. Not just "does it not crash" but "is the output actually correct?"
---

# Reality-Driven Testing

Test systems against **real data with real questions**, not just "does it run without
crashing." Every output must be compared to ground truth.

## Trigger

- User says "test the CLI", "verify it works", "run real tests"
- After implementing a new CLI, API, or tool
- When user asks to "reality test" or "dogfood" a system

## Philosophy

> A command that doesn't crash but returns wrong data is worse than one that crashes
> with a clear error. Wrong data silently misleads; crashes demand attention.

The five principles:

1. **Ground truth first** — Before testing any command, independently verify what the
   real data looks like. Never trust the system under test as your only data source.
2. **Real questions, not synthetic ones** — Design test scenarios from actual user
   needs. "What characters appear in all three books?" beats `--limit 5 --role entity`.
3. **Output quality, not just exit code 0** — Check every field: are names correct?
   Counts accurate? Fields populated? Meaningful, not empty?
4. **Fix root causes, not symptoms** — When a command returns wrong data, trace to the
   storage layer. A display bug is often a data access bug in disguise.
5. **Layer-by-layer verification** — If CLI shows X, verify with SQL. If API returns Y,
   verify in the database. Never trust a single layer.

## Testing Procedure

### Phase 1: Establish Ground Truth (5 min)

Before touching the CLI, independently query the real data:

```
# Check actual database tables, row counts, sample data
SQL: SELECT count(*) FROM <table>
SQL: SELECT * FROM <table> LIMIT 3
SQL: Check schema with PRAGMA table_info(<table>)

# Record:
- Which tables exist and how many rows
- Column names (they often differ from what code assumes!)
- Sample data: real names, real IDs, real content
- Known data quality issues (duplicates, missing fields, zero values)
```

**Critical:** Do this directly against the database/filesystem, NOT through the CLI
you're testing. The CLI is the suspect, not the witness.

### Phase 2: Design Real Questions (5 min)

Think about what a real user would actually want to do with this system. For each
major capability, write 1-2 concrete questions:

```
For a knowledge graph system:
  - "Find character X" → tests entity search
  - "What is the relationship between A and B?" → tests relation lookup
  - "What documents mention X?" → tests document search
  - "What appears in all of [collection]?" → tests cross-document aggregation
  - "Show me the history/versions of X" → tests versioning
  - "How healthy is the data?" → tests diagnostics

For a file management CLI:
  - "List my recent files" → tests listing
  - "Find files containing X" → tests search
  - "Show me what changed" → tests diff/change tracking
```

### Phase 3: Run Commands and Compare (main loop)

For each test scenario:

```
1. Run the CLI command
2. Compare output against ground truth from Phase 1
3. Check EVERY dimension:

   ✓ Correctness — are the values factually right?
   ✓ Completeness — are expected results missing?
   ✓ Precision — are there irrelevant/bogus results?
   ✓ Field population — are name/ID/confidence/role actually filled?
   ✓ Format — is the output human-readable? Machine-parseable?
   ✓ Error handling — does it fail gracefully with a helpful message?
```

### Phase 4: Fix and Re-verify

For every issue found:

```
1. Diagnose root cause — is it CLI display? Storage query? Data schema mismatch?
2. Fix at the right layer — don't patch display if the query is wrong
3. Re-run the SAME test — verify the fix didn't break something else
4. Check related commands — the same root cause often affects multiple commands
```

### Phase 5: Cross-Command Coverage

Ensure every command group has been tested with real data:

```
For each command:
  - Run with a valid input that should return data
  - Run with an invalid input that should error gracefully
  - Run with --json and verify JSON structure
  - Compare counts/IDs to SQL ground truth
```

## Common Failure Patterns

These patterns recur across systems. Watch for them:

### Pattern 1: "Shows 0 but data exists"

**Symptom:** Command returns empty or 0 counts when data is known to exist.
**Root cause:** Usually a path/filename mismatch — the code looks for `graph.db` but
the file is `library.db`. Or column name mismatch — code reads `concept_family_count`
but the view returns `entity_count`.
**Fix:** Always verify file paths and column names against reality.

### Pattern 2: "Runs but returns wrong type of data"

**Symptom:** Search for "叶文洁" returns "三体1疯狂年代.txt" instead.
**Root cause:** The search indexes one layer (episodes) but the user expects results
from another layer (entities). FTS on episodes returns document names, not entity names.
**Fix:** Add a search mode that targets the user-expected layer directly.

### Pattern 3: "Fields show but values are empty"

**Symptom:** Table has a "Name" column but all values are blank.
**Root cause:** The query returns data under column name X but display code reads
column Y. View aliases, ORM field maps, and dict key names often drift apart.
**Fix:** Print the raw return value (keys + values) to see what's actually there,
then fix the column mapping.

### Pattern 4: "Markup in user content crashes rendering"

**Symptom:** Rich MarkupError on `[some text]` in content.
**Root cause:** User content (source_text, descriptions) contains square brackets
that get interpreted as Rich markup tags.
**Fix:** Escape all user-provided content before rendering. Do this at the output
layer, not per-command.

### Pattern 5: "DTO vs dict confusion"

**Symptom:** `AttributeError: 'Entity' object has no attribute 'get'`
**Root cause:** Storage layer returns DTO objects in some methods, plain dicts in
others. Callers assume one type but get the other.
**Fix:** Add a normalization helper (`_to_dict()`) at the boundary between storage
and display code.

### Pattern 6: "Neighbors empty because low-value edges dominate"

**Symptom:** `get_neighbors` returns 0 despite 700+ edges existing.
**Root cause:** A LIMIT clause fetches only the most common edge type (MENTIONS)
which has empty source_family_id. The meaningful edges (RELATES) are never reached.
**Fix:** Either increase the limit, or prioritize the meaningful edge types in the
query/sort order.

## Checklist

Before declaring "testing complete":

- [ ] Every command group tested with real data
- [ ] At least one cross-command workflow tested end-to-end
- [ ] All counts/IDs verified against independent SQL queries
- [ ] Empty results verified as genuinely empty (not a bug)
- [ ] JSON output mode tested and structurally correct
- [ ] Error cases produce helpful messages with hints
- [ ] User content with special characters renders without crashes
- [ ] All fields in tables/panels are populated (or explicitly noted as N/A)
