---
id: integration.BREAKDOWN
module: integration
priority: 999999
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN, schema.BREAKDOWN, cli-search.BREAKDOWN, cli-commands.BREAKDOWN, cli-verify.BREAKDOWN, agent-prompts.BREAKDOWN, backfill.BREAKDOWN, testing.BREAKDOWN]
tags: [gpu-forge, temporal-relevance]
testRequirements:
  unit:
    required: true
    pattern: "tests/unit/temporal.bats"
---

# BREAKDOWN: Integration -- End-to-End Validation

## Context

The final integration module validates the complete temporal relevance pipeline works end-to-end: schema migration, CLI commands, agent prompt updates, backfill, and search filtering all function together correctly. This module runs after all other modules are complete and serves as the final quality gate before the feature is considered done.

Key validation areas: backward compatibility (194 existing tests), golden query regression (50 existing + 4 new temporal queries), full BATS suite (248 total tests), agent workflow validation (manual), and production DB integrity.

Reference: TECH.md Section 8 (Implementation Order), QA.md Section 10 (Test Execution Runbook)

## Tasks

### T-001: Backward compatibility verification

Run all 194 existing BATS tests against the production DB after full migration and backfill:

1. `bats tests/unit/kb-cli.bats` -- 18 existing CLI tests pass
2. `bats tests/unit/db-integrity.bats` -- 9 existing DB integrity tests pass
3. `bats tests/unit/fts5-search.bats` -- 6 existing FTS5 tests pass
4. `bats tests/golden-queries.bats` -- 50 existing golden queries pass
5. `bats tests/integration/workflows.bats` -- existing workflow tests pass

Zero failures. Any failure is a P0 blocker.

### T-002: Full BATS suite regression

Run the complete test suite (existing + new):

```bash
bats tests/unit/ tests/integration/ tests/performance/ tests/golden-queries.bats tests/golden-temporal.bats
```

Expected: 248 tests (194 existing + 54 new), 0 failures, <80 seconds.

### T-003: Golden query suite validation

Verify search quality after full backfill:
1. Run all 50 existing golden queries -- same results as pre-migration
2. Run 4 new golden-temporal queries -- generation-filtered results correct
3. Verify superseded findings are excluded from default search
4. Verify `--gen` filter produces correct generation-specific results

### T-004: Agent workflow validation (manual)

Manual validation of agent prompt effectiveness:
1. **Knowledge retriever**: Query "What is the M4 GPU FP32 TFLOPS?" -- verify agent adds `--gen m4`
2. **Investigation agent**: Run mini-investigation -- verify agent tags gpu_generation on benchmarks
3. **Architecture advisor**: Ask hardware recommendation -- verify advisor checks gpu_generation on cited findings

Document results in PR description.

### T-005: Production DB integrity check

Post-migration, post-backfill verification on production DB:
1. `SELECT COUNT(*) FROM findings;` -- still 1,555+ (no data loss)
2. `SELECT COUNT(*) FROM findings WHERE gpu_generation IS NOT NULL;` -- ~154-203 tagged
3. `SELECT COUNT(*) FROM findings WHERE temporal_status = 'superseded';` -- 7 (contradictions resolved)
4. `kb freshness` shows 0 unlinked contradictions
5. `kb verify` shows reduced temporal warnings
6. FTS5 search works normally (no index corruption)

### T-006: Performance verification

Verify no performance regression:
1. `time kb search "GPU" --limit 10` -- <100ms (NFR-1)
2. `time kb search "GPU" --gen m4 --limit 10` -- <100ms (same latency target)
3. `time kb freshness` -- <2 seconds (dashboard with contradiction detection)

## Acceptance Criteria

1. All 194 existing BATS tests pass unchanged (P0 invariant)
2. All 54 new temporal tests pass
3. Full suite: 248 tests, 0 failures, <80 seconds
4. All 50 existing golden queries return same results as pre-migration
5. 4 new golden-temporal queries return correct generation-filtered results
6. Production DB: 1,555+ findings preserved, ~154-203 gen-tagged, 7 superseded
7. `kb freshness` shows 0 unlinked contradictions
8. Search latency <100ms with --gen filter (NFR-1)
9. Agent workflow manual validation documented
10. No FTS5 index corruption after migration + backfill

## Technical Notes

- Reference: spec/TECH.md Section 8 (Implementation Order with Dependencies)
- Reference: spec/QA.md Section 10 (Test Execution Runbook)
- Test: Full BATS suite via `bats tests/unit/ tests/integration/ tests/performance/ tests/golden-queries.bats tests/golden-temporal.bats`
- Agent workflow validation is manual per QA Section 6.2 (LLM behavior cannot be deterministically tested in BATS)
- Performance baseline: capture pre-migration search latency for comparison
- Pre-commit checklist from QA Section 10.5 should be completed before PR
