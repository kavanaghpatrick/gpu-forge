---
id: backfill.BREAKDOWN
module: backfill
priority: 6
status: failing
version: 1
origin: spec-workflow
dependsOn: [cli-commands.BREAKDOWN, schema.BREAKDOWN]
tags: [gpu-forge, temporal-relevance]
testRequirements:
  unit:
    required: false
    pattern: "tests/unit/temporal.bats"
---

# BREAKDOWN: Backfill -- Auto-Tag Findings and Resolve Contradictions

## Context

After schema migration and CLI commands are implemented, the existing KB needs backfill: 203 findings already have generation-related tags in their `tags` field but no `gpu_generation` column value, and 7 confirmed cross-generation contradictions exist without supersession links. This module covers the one-time backfill execution and the manual contradiction resolution.

The backfill uses `kb tag-gen --auto` which infers gpu_generation from the tags field using case-insensitive substring matching. The contradiction resolution uses `kb freshness` to identify pairs and `kb supersede` to create links. Both workflows require human review (dry-run preview, manual detail inspection).

Reference: TECH.md Section 5 (Backfill Strategy), PM.md Section "Implementation Approach" Phase 1 Step 7, PM.md Q&A Q5 (tags-only backfill now)

## Tasks

### T-001: Run auto-tag dry-run preview

Execute `kb tag-gen --auto --dry-run` and review output:
1. Verify ~203 findings would be tagged (expected from PM scope analysis)
2. Check multi-gen findings are tagged with newest generation (PM Q&A Q2)
3. Check for false positives (e.g., `pre-m5` tagged as m5)
4. Verify findings with no generation indicators remain NULL
5. Verify findings with existing non-NULL gpu_generation are not overwritten

### T-002: Execute auto-tag backfill

Execute `kb tag-gen --auto`:
1. ~203 findings tagged with inferred gpu_generation
2. Run verification queries post-backfill:
   - `SELECT COUNT(*) FROM findings WHERE gpu_generation IS NOT NULL;` -- should jump from 0 to ~154-203
   - `SELECT gpu_generation, COUNT(*) FROM findings WHERE gpu_generation IS NOT NULL GROUP BY gpu_generation;`
   - `SELECT COUNT(*) FROM findings;` -- still 1,555 (no data loss)
3. Spot-check: M5 findings should include M5-specific tags
4. Check for misclassifications: findings where inferred gen doesn't appear in tags

### T-003: Identify contradictions via kb freshness

Execute `kb freshness` after backfill:
1. Review "Potential Contradictions" section
2. Should show ~7 contradiction pairs (known from PM analysis)
3. For each pair, run `kb detail` on both findings to assess

### T-004: Resolve contradictions via kb supersede

For each confirmed contradiction:
1. `kb detail <old_id>` and `kb detail <new_id>` to review
2. Determine if truly contradictory (same metric, newer replaces older)
3. If contradictory: `kb supersede <old_id> <new_id> "<reason>"`
4. If complementary (different metrics): tag each with correct generation, no supersession
5. Target: 0 unlinked contradictions after resolution

### T-005: Final verification

After all backfill and resolution:
1. `kb freshness` shows 0 contradictions
2. `kb verify` shows reduced or zero temporal warnings
3. Generation-tagged findings: ~13% (203/1555) minimum
4. All findings preserved: COUNT still 1,555+

## Acceptance Criteria

1. `kb tag-gen --auto --dry-run` shows reasonable preview before execution
2. `kb tag-gen --auto` tags ~154-203 findings with gpu_generation
3. No existing findings deleted or corrupted during backfill
4. Multi-gen findings tagged with newest generation per PM Q&A Q2
5. Findings with no generation indicators remain NULL (not force-tagged)
6. `kb freshness` contradiction count drops to 0 after resolution
7. All 7 known contradictions reviewed and either superseded or correctly tagged
8. Post-backfill `kb freshness` shows updated generation distribution
9. All 194 existing BATS tests still pass after backfill

## Technical Notes

- Reference: spec/TECH.md Section 5 (Backfill Strategy), Section 5.1-5.5 (Inference rules, verification queries)
- Reference: spec/PM.md Success Metrics (13% -> 80% gen-tagged target)
- Test: spec/QA.md Section 7 (Golden Query Test Suite) validates post-backfill search
- Backfill scope: tags-only (per PM Q&A Q5); claim-text scanning deferred
- False positive risk: `pre-m5` matches m5 -- caught by dry-run preview
- The 7 known contradictions are identified by `kb freshness` exact topic+skill match
- Human review required for each contradiction pair before supersession
