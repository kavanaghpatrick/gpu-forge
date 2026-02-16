---
id: testing.BREAKDOWN
module: testing
priority: 7
status: failing
version: 1
origin: spec-workflow
dependsOn: [cli-search.BREAKDOWN, cli-commands.BREAKDOWN, cli-verify.BREAKDOWN, agent-prompts.BREAKDOWN]
tags: [gpu-forge, temporal-relevance]
testRequirements:
  unit:
    required: true
    pattern: "tests/unit/temporal.bats"
---

# BREAKDOWN: Testing -- Comprehensive BATS Test Suite

## Context

The temporal relevance pipeline requires 54 new tests across 3 new BATS files and 1 modified file. The test strategy follows the existing project conventions: per-test `setup()` with fresh DB copy + migration, `teardown()` cleanup, `assert_success`/`assert_failure`/`assert_output --partial` from bats-assert. All tests run on REAL databases (no mocks per project rules).

The test pyramid: 42 unit tests (temporal.bats), 4 golden temporal queries (golden-temporal.bats), 4 integration workflows (temporal-workflows.bats), and 2 performance regression tests (added to existing benchmarks.bats). The existing 194 tests must pass unchanged.

Reference: TECH.md Section 6 (Testing Strategy), QA.md (full QA strategy)

## Tasks

### T-001: Create tests/unit/temporal.bats

42 unit tests organized by feature area:
1. **Migration** (7 tests): M-1 through M-7 (columns, defaults, idempotency, backup, data preservation, indexes)
2. **Schema Constraints** (5 tests): C-1 through C-5 (valid values, invalid values, NULL, defaults)
3. **Search --gen** (9 tests): S-1 through S-9 (backward compat, gen filtering, NULL inclusion, universal, invalid gen, short flag)
4. **Search --include-superseded** (3 tests): SS-1 through SS-3 (default exclusion, include flag, sort order)
5. **Supersede** (7 tests): SP-1 through SP-7 (basic, reason, invalid IDs, force, self-supersession guard)
6. **Unsupersede** (3 tests): US-1 through US-3 (restore, no-op on current, nonexistent)
7. **Tag-Gen** (6 tests): TG-1 through TG-6 (single, clear, invalid, dry-run, auto, no-overwrite)
8. **Freshness** (3 tests): F-1 through F-3 (dashboard, json, json keys)
9. **Verify/Detail/Add/Help** (5 tests): V-1, D-1, D-2, A-1, H-1
10. **Agent Prompts** (2 tests): AP-1, AP-2 (static content checks)

Setup pattern:
```bash
setup() {
  TEST_DB="${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$TEST_DB"
  export GPU_FORGE_DB="$TEST_DB"
  run "$KB" migrate-temporal
  assert_success
}
```

### T-002: Create tests/golden-temporal.bats

4 golden query tests on post-backfill DB:
1. GT-1: "M4 TFLOPS" --gen m4 returns gpu-silicon results
2. GT-2: "M5 neural accelerator" --gen m5 returns results
3. GT-3: "SIMD width" without --gen returns all generations
4. GT-4: "bandwidth" --gen m4 excludes m1 findings

Setup runs `kb migrate-temporal` + `kb tag-gen --auto` to simulate post-backfill state.

### T-003: Create tests/integration/temporal-workflows.bats

4 multi-step workflow tests:
1. Full temporal health check: freshness -> detail -> supersede -> freshness
2. Backfill workflow: tag-gen --auto --dry-run -> tag-gen --auto -> freshness
3. Supersession override: supersede -> supersede again (fail) -> --force (succeed)
4. Search + detail with temporal metadata visible

### T-004: Add performance tests to tests/performance/benchmarks.bats

2 new tests:
1. `kb search "GPU" --gen m4` completes under 1 second (NFR-1 validation)
2. `kb freshness` completes under 2 seconds (dashboard query performance)

### T-005: Create graceful degradation tests

4 tests for pre-migration DB behavior (in temporal-workflows.bats):
1. Search on pre-migration DB works without --gen
2. Search --gen on pre-migration DB degrades gracefully
3. freshness on pre-migration DB handles missing columns
4. supersede on pre-migration DB gives helpful error with migrate-temporal hint

### T-006: Verify backward compatibility

Run all 194 existing BATS tests against post-migration DB:
1. All existing tests pass (P0 blocker if any fail)
2. Golden queries (50 tests) return same results
3. Integration workflows unchanged
4. DB integrity checks pass

## Acceptance Criteria

1. `tests/unit/temporal.bats` contains 42 passing tests
2. `tests/golden-temporal.bats` contains 4 passing tests
3. `tests/integration/temporal-workflows.bats` contains 4 passing tests
4. `tests/performance/benchmarks.bats` has 2 new passing tests
5. All 194 existing BATS tests pass unchanged after migration
6. Total new test count: 54 (exceeds NFR-4 target of 15+ by 3.6x)
7. Per-test isolation: each test uses fresh DB copy in BATS_TEST_TMPDIR
8. Full suite runs in <80 seconds
9. Smoke test (`bats tests/unit/temporal.bats`) runs in <15 seconds

## Technical Notes

- Reference: spec/TECH.md Section 6 (Testing Strategy), Section 6.1 (temporal.bats), Section 6.2 (golden-temporal.bats)
- Reference: spec/QA.md (comprehensive test plan, all test IDs, acceptance criteria matrix)
- Test: spec/QA.md Section 9 (Acceptance Criteria Coverage Matrix) maps every FR/AC to test IDs
- Per-test migration in setup() per QA Q&A Q1
- Golden tests use production DB copy per QA Q&A Q3
- Self-supersession guard test (SP-8) per QA Q&A Q2
- Performance tests use `_now_ns` helper for timing (existing convention)
- Agent prompt tests are static content checks (grep for strings in .md files)
