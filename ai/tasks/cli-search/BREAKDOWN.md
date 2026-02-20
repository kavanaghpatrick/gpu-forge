---
id: cli-search.BREAKDOWN
module: cli-search
priority: 2
status: failing
version: 1
origin: spec-workflow
dependsOn: [schema.BREAKDOWN]
tags: [gpu-forge, temporal-relevance]
testRequirements:
  unit:
    required: true
    pattern: "tests/unit/temporal.bats"
---

# BREAKDOWN: CLI Search -- Generation-Filtered Search

## Context

The `kb search` command is the primary read path for both agents and humans. This module adds two new flags (`--gen <generation>` and `--include-superseded`) and a new `gen` column to search output. The implementation uses post-FTS JOIN WHERE clauses -- the FTS5 MATCH narrows results first, then WHERE filters the small result set by generation and temporal status.

Critical constraint: existing `kb search` invocations without new flags must produce identical results (except for the new `gen` column in output). The 50 existing golden queries must pass unchanged.

Reference: TECH.md Section 3.3 (Search Command Modification), UX.md Section 1.1 (kb search), Section 4.1 (Search Results Output)

## Tasks

### T-001: Add --gen/-g flag parsing

Extend the existing `while/shift` argument parsing loop in the `search)` case to handle:
- `--gen <value>` / `-g <value>`: Set generation filter
- Validate generation value: `m1|m2|m3|m4|m5|universal` or fail with actionable error

### T-002: Add --include-superseded/-S flag parsing

Add boolean flag parsing:
- `--include-superseded` / `-S`: Set include_superseded=1
- Default: include_superseded=0 (superseded findings excluded)

### T-003: Build generation WHERE clause

When `--gen` is specified:
```sql
AND (f.gpu_generation = '<gen>' OR f.gpu_generation = 'universal' OR f.gpu_generation IS NULL)
```
- Includes specified generation, universal, AND NULL (unclassified) per UX Q&A Q4
- Without `--gen`, no generation clause added (all generations returned)

### T-004: Build temporal status WHERE clause

When `--include-superseded` is NOT specified (default):
```sql
AND (f.temporal_status = 'current' OR f.temporal_status IS NULL)
```
- Excludes superseded findings by default
- With `--include-superseded`, no status clause (all statuses returned)

### T-005: Add gen column to SELECT

Always include generation column in output:
```sql
COALESCE(f.gpu_generation, '') as gen
```
- Blank for NULL (not literal "NULL" string) per UX spec -- reduces visual noise
- Gen column always visible per UX Q&A Q1

### T-006: Add status column when --include-superseded

When `--include-superseded` is active, add status column:
```sql
CASE WHEN f.temporal_status = 'superseded' THEN 'SUPERSEDED' ELSE 'current' END as status
```
- SUPERSEDED uppercase to draw attention per UX spec
- Only shown when `--include-superseded` flag is used

### T-007: Supersession-aware ORDER BY

When `--include-superseded` is active, sort superseded after current:
```sql
ORDER BY CASE WHEN f.temporal_status = 'superseded' THEN 1 ELSE 0 END,
         bm25(findings_fts, 10.0, 5.0, 2.0, 1.0)
```

### T-008: Graceful degradation for pre-migration DB

Check `pragma_table_info('findings')` for gpu_generation column existence. If not present:
- Ignore `--gen` flag silently (or warn)
- Skip temporal_status filter
- Omit gen column from output

## Acceptance Criteria

1. `kb search "GPU"` without new flags returns results (backward compatible)
2. Search output includes `gen` column header
3. `kb search "GPU" --gen m4` returns m4-tagged, universal-tagged, and NULL findings
4. `kb search "GPU" --gen m4` excludes findings tagged m1, m2, m3, m5
5. `kb search "GPU" --gen m6` fails with "Invalid generation" error message
6. `kb search "GPU"` (default) excludes superseded findings
7. `kb search "GPU" --include-superseded` includes superseded findings with SUPERSEDED status
8. Superseded findings appear after current findings when --include-superseded used
9. `-g` short flag works identically to `--gen`
10. `-S` short flag works identically to `--include-superseded`
11. All 50 existing golden queries pass unchanged

## Technical Notes

- Reference: spec/TECH.md Section 3.3 (Search Command Modification)
- Reference: spec/UX.md Section 1.1 (kb search flags), Section 4.1 (Output Formatting)
- Reference: spec/UX.md Section 6.1 (Argument Parsing)
- Test: spec/QA.md Section 4.2 (Search --gen Tests S-1 through S-9), Section 4.3 (Search --include-superseded Tests SS-1 through SS-3)
- Performance: FTS5 MATCH narrows to <100 rows; WHERE on indexed columns adds <1ms (NFR-1: <100ms)
- Backward compat: existing tests use `--partial` matching, unaffected by new gen column
