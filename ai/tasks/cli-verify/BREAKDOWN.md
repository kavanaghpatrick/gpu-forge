---
id: cli-verify.BREAKDOWN
module: cli-verify
priority: 4
status: failing
version: 1
origin: spec-workflow
dependsOn: [schema.BREAKDOWN, cli-search.BREAKDOWN]
tags: [gpu-forge, temporal-relevance]
testRequirements:
  unit:
    required: true
    pattern: "tests/unit/temporal.bats"
---

# BREAKDOWN: CLI Verify -- Verify, Add, and Help Updates

## Context

This module updates three existing `kb` commands to integrate temporal awareness: `kb verify` gains a new Check 6 for missing gpu_generation on benchmark findings, `kb add` gains an optional 11th positional parameter for gpu_generation, and the help text gains a "Temporal commands" section. These are low-risk, additive changes that preserve all existing behavior.

The verify check is gated on `pragma_table_info` so it only runs after migration, ensuring backward compatibility with pre-migration databases. The add command's new parameter is fully optional. The help text addition is positioned after "Quality commands" and before "Investigation tracking."

Reference: TECH.md Sections 3.9-3.11 (Verify, Add, Help), UX.md Sections 1.7-1.9

## Tasks

### T-001: Add Check 6 to `kb verify`

Insert after existing Check 5 (findings without source URLs), before the final summary block:

1. Check if temporal columns exist via `pragma_table_info` (graceful degradation)
2. If columns exist, count benchmark/empirical_test findings with NULL gpu_generation
3. If count > 0: WARNING with list of untagged findings (limit 10) and fix command
4. If count == 0: "OK: All benchmark/empirical findings have gpu_generation tags"
5. Add count to `issues` total

Per PM Q&A Q4: WARNING severity initially, promote to ERROR after backfill proves system works. The transition is a one-line change: `echo "WARNING:"` to `echo "ERROR:"`.

### T-002: Modify `kb add` for gpu_generation parameter

Add optional 11th positional parameter (`${11}`) for gpu_generation:

1. If provided, validate against m1|m2|m3|m4|m5|universal
2. If valid, add to INSERT column list and VALUES
3. If not provided, gpu_generation defaults to NULL (column default)
4. Backward compatible: existing calls with 9-10 positional args work unchanged

### T-003: Update help text

Add "Temporal commands" section to the help/default case output:

1. List all temporal commands: supersede, unsupersede, tag-gen, freshness, migrate-temporal
2. Document generation values: m1, m2, m3, m4, m5, universal (NULL = unclassified)
3. Document search filters: --gen, --include-superseded
4. Position after "Quality commands" and before "Investigation tracking"

## Acceptance Criteria

1. `kb verify` after migration includes benchmark/empirical gpu_generation check (either OK or WARNING)
2. `kb verify` on pre-migration DB skips temporal check gracefully (no error)
3. WARNING output includes list of untagged findings and `kb tag-gen` fix command
4. `kb add` with 11 positional args sets gpu_generation correctly
5. `kb add` with 9-10 positional args (existing usage) still works with gpu_generation=NULL
6. `kb add` with invalid generation in position 11 fails with actionable error
7. `kb` help output includes "Temporal commands" section
8. Help text lists supersede, unsupersede, tag-gen, freshness, migrate-temporal
9. Help text lists valid generation values
10. Help text lists search filter flags (--gen, --include-superseded)

## Technical Notes

- Reference: spec/TECH.md Sections 3.9-3.11 (Verify, Add, Help)
- Reference: spec/UX.md Sections 1.7-1.9 (Verify, Add, Help UX)
- Test: spec/QA.md Section 4.8 (V-1, A-1, H-1 tests)
- Verify check gated on pragma_table_info -- safe for pre-migration databases
- WARNING vs ERROR: controlled by single echo string; promote after backfill validation
- `kb add` uses positional args exclusively (no flag mixing per UX design decision)
- Help text "Temporal commands" section positioned for discoverability per UX Section 1.9
