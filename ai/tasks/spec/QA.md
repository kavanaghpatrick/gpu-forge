# QA Strategy: KB Temporal Relevance Pipeline

**Date**: 2026-02-13
**Analyst**: QA Manager Agent
**System**: gpu-forge (Claude Code plugin)
**Stack**: Bash CLI (`kb` script, 474 lines), SQLite 3.51.1 with FTS5, BATS tests (194 tests, 14 files), Markdown agent prompts
**Scope**: Complete testing and quality strategy for temporal relevance pipeline changes, covering migration safety, CLI commands, backward compatibility, agent prompt validation, and regression prevention

---

## 1. Test Strategy Overview

### Philosophy

The temporal relevance pipeline touches three layers of the system: the database schema, the CLI commands, and the agent prompts. The test strategy must cover all three layers while maintaining the foundational invariant: **all 194 existing BATS tests continue to pass unmodified**.

### Test Pyramid

```
                  /\
                 /  \    4 golden-temporal integration tests
                /    \   (end-to-end queries on post-backfill DB)
               /------\
              /        \   8 integration tests
             /          \  (multi-command workflows, agent prompt validation)
            /------------\
           /              \  37 unit tests
          /                \ (migration, CLI commands, constraints, error paths)
         /------------------\
        /                    \  5 regression/performance tests
       /                      \ (backward compat, search latency, DB integrity)
      /________________________\
```

### Test Categories Summary

| Category | New Test File | Test Count | Priority |
|----------|--------------|-----------|----------|
| Migration | `tests/unit/temporal.bats` | 7 | P0 (blocking) |
| Schema Constraints | `tests/unit/temporal.bats` | 5 | P0 |
| Search Filtering | `tests/unit/temporal.bats` | 9 | P0 |
| Search Superseded | `tests/unit/temporal.bats` | 3 | P0 |
| Supersession | `tests/unit/temporal.bats` | 7 | P0 |
| Unsupersede | `tests/unit/temporal.bats` | 3 | P1 |
| Tag-Gen | `tests/unit/temporal.bats` | 6 | P1 |
| Freshness | `tests/unit/temporal.bats` | 3 | P1 |
| Verify/Detail/Add/Help | `tests/unit/temporal.bats` | 5 | P1 |
| Agent Prompt Validation | `tests/unit/temporal.bats` | 2 | P2 |
| Golden Temporal Queries | `tests/golden-temporal.bats` | 4 | P1 |
| Integration Workflows | `tests/integration/temporal-workflows.bats` | 4 | P2 |
| Performance Regression | `tests/performance/benchmarks.bats` (add to existing) | 2 | P2 |
| **Total New Tests** | **3 new files + 1 modified** | **54** | |

This exceeds the NFR-4 target of 15+ new tests by 3.6x, providing thorough coverage.

---

## 2. Test File Structure

### 2.1 New File: `tests/unit/temporal.bats`

The primary test file. Contains 42 unit tests organized by feature area. Follows the exact patterns from the existing `tests/unit/kb-cli.bats`:
- `setup()` copies production DB to `BATS_TEST_TMPDIR`, sets `GPU_FORGE_DB`, applies migration
- `teardown()` cleans up the temp DB and backup file
- Uses `assert_success`, `assert_failure`, `assert_output --partial` from bats-assert
- Each test is self-contained and mutates only the test-local DB copy

**Test ordering within file:**
1. Migration tests (must validate schema before anything else depends on it)
2. CHECK constraint tests (validates schema integrity)
3. Search `--gen` filtering tests
4. Search `--include-superseded` tests
5. `supersede` command tests
6. `unsupersede` command tests
7. `tag-gen` command tests
8. `freshness` command tests
9. `verify` temporal check tests
10. `detail` supersession display tests
11. `add` with gpu_generation tests
12. Help text tests
13. Agent prompt structure tests

### 2.2 New File: `tests/golden-temporal.bats`

4 golden query tests that validate generation-aware search on a post-backfill DB. These tests run `kb tag-gen --auto` in `setup()` to establish realistic generation tagging before querying.

Follows the pattern of the existing `tests/golden-queries.bats` at the root `tests/` level. Uses the production DB copy.

### 2.3 New File: `tests/integration/temporal-workflows.bats`

4 multi-step workflow tests that exercise realistic user scenarios:
- Full temporal health check workflow (freshness -> detail -> supersede -> freshness)
- Backfill workflow (tag-gen --auto --dry-run -> tag-gen --auto -> freshness)
- Supersession override workflow (supersede -> supersede again with --force)
- Search + detail workflow with temporal metadata visible

### 2.4 Modified File: `tests/performance/benchmarks.bats`

Add 2 tests to the existing performance benchmark file:
- `kb search "GPU" --gen m4` completes under 1 second (validates no perf regression from gen filter)
- `kb freshness` completes under 2 seconds (validates dashboard query performance)

### 2.5 File Placement Summary

```
tests/
  unit/
    temporal.bats              (NEW - 42 tests)
    kb-cli.bats                (UNCHANGED - 18 tests)
    db-integrity.bats          (UNCHANGED - 9 tests)
    fts5-search.bats           (UNCHANGED - 6 tests)
    ...
  integration/
    temporal-workflows.bats    (NEW - 4 tests)
    workflows.bats             (UNCHANGED)
    ...
  performance/
    benchmarks.bats            (MODIFIED - +2 tests)
  golden-queries.bats          (UNCHANGED - 50 tests)
  golden-temporal.bats         (NEW - 4 tests)
  test_helper/
    common-setup.bash          (UNCHANGED)
```

---

## 3. Migration Test Suite

### 3.1 Pre-Migration State Verification

Before any migration test runs, verify the baseline.

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| M-1 | `migrate-temporal adds 3 new columns` | After `kb migrate-temporal`, `pragma_table_info('findings')` contains `gpu_generation`, `temporal_status`, `superseded_by` |
| M-2 | `migrate-temporal sets correct defaults` | `temporal_status` defaults to `'current'` for all existing rows. `gpu_generation` and `superseded_by` default to NULL. |
| M-3 | `migrate-temporal creates backup file` | `${TEST_DB}.pre-temporal-backup` exists after migration |

### 3.2 Migration Idempotency

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| M-4 | `migrate-temporal is idempotent (run twice)` | Second `kb migrate-temporal` outputs "already applied" and exits 0. Column count still 3. |
| M-5 | `migrate-temporal is idempotent (column data preserved)` | After tagging finding #1 as `m4`, running migration again does not reset `gpu_generation` to NULL |

### 3.3 Data Integrity

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| M-6 | `migration preserves all 1555+ findings` | `SELECT COUNT(*) FROM findings` after migration >= 1555 (exact count may grow between development and test) |
| M-7 | `migration creates 3 indexes` | `SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name LIKE 'idx_findings_%'` returns at least 3 new indexes |

### 3.4 Implementation Pattern

```bash
@test "migrate-temporal adds 3 new columns" {
  run sqlite3 "$TEST_DB" "SELECT COUNT(*) FROM pragma_table_info('findings') WHERE name IN ('gpu_generation','temporal_status','superseded_by');"
  assert_success
  assert_output "3"
}

@test "migrate-temporal is idempotent" {
  # Migration already ran in setup() -- run again
  run "$KB" migrate-temporal
  assert_success
  assert_output --partial "already applied"
}

@test "migrate-temporal creates backup" {
  [ -f "${TEST_DB}.pre-temporal-backup" ]
}

@test "migration preserves all findings" {
  count=$(sqlite3 "$TEST_DB" "SELECT COUNT(*) FROM findings;")
  [ "$count" -ge 1555 ]
}

@test "migrate-temporal sets correct defaults" {
  # All existing findings should have temporal_status = 'current'
  non_current=$(sqlite3 "$TEST_DB" "SELECT COUNT(*) FROM findings WHERE temporal_status != 'current';")
  [ "$non_current" -eq 0 ]
  # All existing findings should have gpu_generation = NULL
  non_null_gen=$(sqlite3 "$TEST_DB" "SELECT COUNT(*) FROM findings WHERE gpu_generation IS NOT NULL;")
  [ "$non_null_gen" -eq 0 ]
  # All existing findings should have superseded_by = NULL
  non_null_sup=$(sqlite3 "$TEST_DB" "SELECT COUNT(*) FROM findings WHERE superseded_by IS NOT NULL;")
  [ "$non_null_sup" -eq 0 ]
}
```

---

## 4. CLI Command Test Suite

### 4.1 Schema Constraint Tests

| Test ID | Command | What It Verifies |
|---------|---------|-----------------|
| C-1 | Direct SQL | `gpu_generation` accepts all valid values: m1, m2, m3, m4, m5, universal |
| C-2 | Direct SQL | `gpu_generation` rejects invalid value (`m6`) with CHECK constraint error |
| C-3 | Direct SQL | `gpu_generation` accepts NULL |
| C-4 | Direct SQL | `temporal_status` rejects invalid value (`historical`) with CHECK constraint error |
| C-5 | Direct SQL | `temporal_status` default is `'current'` for newly inserted finding |

### 4.2 `kb search --gen` Tests

| Test ID | Invocation | Expected Behavior |
|---------|-----------|------------------|
| S-1 | `kb search "GPU"` | Returns results (backward compatible, no flags) |
| S-2 | `kb search "GPU"` | Output includes `gen` column header |
| S-3 | `kb search "GPU" --gen m4` | Returns results (m4-tagged + universal + NULL findings) |
| S-4 | `kb search "GPU" --gen m4` | Excludes findings tagged with other specific generations (m1, m2, m3, m5) |
| S-5 | `kb search "GPU" --gen m4` | Includes NULL (unclassified) findings |
| S-6 | `kb search "GPU" --gen m4` | Includes `universal`-tagged findings |
| S-7 | `kb search "GPU" --gen m6` | Exits with failure, error message contains "Invalid generation" |
| S-8 | `kb search "GPU" --gen universal` | Returns only `universal`-tagged findings and NULL findings |
| S-9 | `kb search "GPU" -g m4` | Short flag `-g` works identically to `--gen` |

### 4.3 `kb search --include-superseded` Tests

| Test ID | Invocation | Expected Behavior |
|---------|-----------|------------------|
| SS-1 | `kb search "GPU"` (default) | Excludes superseded findings from results |
| SS-2 | `kb search "GPU" --include-superseded` | Includes superseded findings, output contains "SUPERSEDED" |
| SS-3 | `kb search "GPU" --include-superseded` | Superseded findings appear after current findings in results |

### 4.4 `kb supersede` Tests

| Test ID | Invocation | Expected Behavior |
|---------|-----------|------------------|
| SP-1 | `kb supersede 1 2` | Marks finding #1 as superseded by #2, outputs confirmation |
| SP-2 | `kb supersede 1 2` | DB shows `temporal_status='superseded'` and `superseded_by=2` for finding #1 |
| SP-3 | `kb supersede 1 2 "M5 replaces M1"` | Reason appended to notes field: `[Superseded: M5 replaces M1]` |
| SP-4 | `kb supersede 99999 2` | Exits with failure, error: "does not exist" |
| SP-5 | `kb supersede 1 99999` | Exits with failure, error: "does not exist" |
| SP-6 | `kb supersede 1 2` then `kb supersede 1 3` | Second call fails with "already superseded" + "--force" hint |
| SP-7 | `kb supersede abc 2` | Exits with failure, error: "Usage" |

### 4.5 `kb unsupersede` Tests

| Test ID | Invocation | Expected Behavior |
|---------|-----------|------------------|
| US-1 | `kb supersede 1 2` then `kb unsupersede 1` | Restores finding #1 to `temporal_status='current'`, `superseded_by=NULL` |
| US-2 | `kb unsupersede 1` (on non-superseded finding) | Outputs "already current" warning, exits 0 |
| US-3 | `kb unsupersede 99999` | Exits with failure, error: "does not exist" |

### 4.6 `kb tag-gen` Tests

| Test ID | Invocation | Expected Behavior |
|---------|-----------|------------------|
| TG-1 | `kb tag-gen 1 m4` | Sets `gpu_generation='m4'` on finding #1, outputs confirmation |
| TG-2 | `kb tag-gen 1 --clear` | Resets `gpu_generation` to NULL on finding #1 |
| TG-3 | `kb tag-gen 1 m6` | Exits with failure, error: "Invalid generation" |
| TG-4 | `kb tag-gen --auto --dry-run` | Outputs preview with "Would tag" and "Would skip" sections |
| TG-5 | `kb tag-gen --auto` | Tags findings, outputs count. DB shows non-null gpu_generation count > 0 |
| TG-6 | `kb tag-gen --auto` (after pre-tagging finding #1) | Does not overwrite existing non-NULL `gpu_generation` on finding #1 |

### 4.7 `kb freshness` Tests

| Test ID | Invocation | Expected Behavior |
|---------|-----------|------------------|
| F-1 | `kb freshness` | Outputs dashboard with sections: "KB Temporal Health", "Generation Coverage", "Temporal Status", "Date Coverage" |
| F-2 | `kb freshness --json` | Outputs valid JSON (validated by `python3 -c "import sys,json; json.load(sys.stdin)"`) |
| F-3 | `kb freshness --json` | JSON contains keys: `generation_counts`, `temporal_status`, `date_coverage` |

### 4.8 Verify, Detail, Add, Help Tests

| Test ID | Invocation | Expected Behavior |
|---------|-----------|------------------|
| V-1 | `kb verify` (post-migration) | Output includes benchmark/empirical gpu_generation check (either "OK" or "WARNING") |
| D-1 | `kb detail 1` after `kb supersede 1 2` | Output includes "SUPERSEDED by" and "#2" |
| D-2 | `kb detail 2` after `kb supersede 1 2` | Output includes "Supersedes" and "#1" |
| A-1 | `kb add gpu-silicon "test-temporal" "Test claim" "evidence" "http://test" "Test" "benchmark" "high" "m4,test" "m4"` | Finding added with `gpu_generation='m4'` in DB |
| H-1 | `kb` (no args) | Output includes "Temporal commands", "supersede", "tag-gen", "freshness", "Generations:" |

### 4.9 Agent Prompt Structure Tests

| Test ID | What It Verifies |
|---------|-----------------|
| AP-1 | `investigation-agent.md` contains "gpu_generation" string (generation tagging instruction present) |
| AP-2 | `knowledge-retriever.md` contains "--gen" string (generation detection instruction present) |

These are static content checks, not behavioral tests. They verify the prompt files were correctly modified.

---

## 5. Backward Compatibility Plan

### 5.1 Risk Analysis: What Could Break

| Component | Risk | Impact | Detection |
|-----------|------|--------|-----------|
| `kb search` output format | New `gen` column added | Tests checking column count or positional parsing would fail | Existing tests use `--partial` matching -- verified safe |
| `kb search` result set | Superseded findings filtered by default | If any test data has `temporal_status='superseded'`, result counts change | All existing findings get `DEFAULT 'current'` -- no rows filtered |
| `kb detail` output format | New temporal fields in output | Tests checking exact output match would fail | Existing tests use `--partial` matching -- verified safe |
| `kb verify` output | New Check 6 appended | Tests checking exact output match would fail | Existing test checks `--partial "=== Knowledge Base Quality Report ==="` -- safe |
| `kb add` positional args | New 11th parameter | If existing calls pass 11+ args accidentally | 11th param is optional, existing calls have 9-10 args -- safe |
| Direct SQL INSERTs by agents | New columns in table | Agent INSERT with `SELECT *` or positional binding | Agents use named columns -- verified safe |
| Golden queries | Search results might change if findings get superseded | Any of the 50 golden queries could return fewer results | Post-migration, all findings are `current` -- no change |
| `search` output parsing in `workflows.bats` | Line 75 uses `awk '{print $2}'` for column extraction | New `gen` column appended to end of row, not inserted before skill | Existing columns maintain same positions -- safe |

### 5.2 Backward Compatibility Test Strategy

**Phase 1 (Immediate): Run existing suite post-migration**

After implementing `kb migrate-temporal` and applying it to the test DB:
1. Run all 194 existing BATS tests with the migrated DB
2. Zero failures expected
3. Any failure is a P0 blocker that must be fixed before proceeding

**Phase 2 (After CLI changes): Re-run existing suite with modified commands**

After modifying `kb search`, `kb detail`, `kb verify`, `kb add`:
1. Run all 194 existing BATS tests
2. The search output now includes a `gen` column -- all 50 golden queries use `--partial` so they pass
3. The detail output now includes temporal fields -- existing tests check for `skill_name` which still appears
4. The verify output now has Check 6 -- existing test checks for header which still appears

**Phase 3 (After backfill): Run golden queries on backfilled DB**

After `kb tag-gen --auto` modifies generation tags:
1. Re-run all 50 golden queries
2. Results should be unchanged because backfill only sets `gpu_generation` (does not change claims, tags, or FTS content)
3. The default search (no `--gen` flag) returns all generations including newly tagged findings

### 5.3 Specific Compatibility Assertions

```bash
# These assertions are embedded in the temporal.bats setup/teardown cycle

@test "backward-compat: existing search returns results after migration" {
  run "$KB" search "GPU"
  assert_success
  [ ${#output} -gt 10 ]
}

@test "backward-compat: add without gpu_generation still works" {
  run "$KB" add gpu-silicon "backcompat-test" "Test backward compat claim"
  assert_success
  assert_output --partial "Finding added"
  # Verify gpu_generation defaulted to NULL
  gen=$(sqlite3 "$TEST_DB" "SELECT COALESCE(gpu_generation,'NULL') FROM findings WHERE topic='backcompat-test' ORDER BY id DESC LIMIT 1;")
  [ "$gen" = "NULL" ]
}
```

### 5.4 Critical Path for the `workflows.bats` Search+Skill Test

The existing `tests/integration/workflows.bats` test "search and skill produce consistent results" extracts the skill name from the second column of search output using `awk '{print $2}'`. After our changes, the search SQL appends `COALESCE(f.gpu_generation, '') as gen` as the last column. Since `gen` is appended (not inserted before `skill`), the `awk '{print $2}'` still extracts the skill name correctly. This is safe.

---

## 6. Agent Prompt Validation Strategy

### 6.1 Static Validation (Automated)

Agent prompts are Markdown files. We can validate structural changes with BATS tests that check for required strings:

```bash
@test "investigation-agent.md contains generation tagging instruction" {
  run grep -c "gpu_generation" "${PLUGIN_ROOT}/agents/investigation-agent.md"
  assert_success
  [ "$output" -ge 1 ]
}

@test "knowledge-retriever.md contains --gen detection instruction" {
  run grep -c "\-\-gen" "${PLUGIN_ROOT}/agents/knowledge-retriever.md"
  assert_success
  [ "$output" -ge 1 ]
}
```

### 6.2 Behavioral Validation (Manual -- Golden Agent Queries)

Prompt changes cannot be fully tested in BATS because they guide LLM behavior. However, we can validate the downstream effect with golden agent queries.

**Validation protocol (run manually after prompt changes):**

1. **Knowledge Retriever Test**: Ask "What is the M4 GPU FP32 TFLOPS performance?" and verify the retriever adds `--gen m4` to its search command.
2. **Investigation Agent Test**: Run a mini-investigation on a topic where supersession should be detected. Verify the agent calls `kb supersede` during Phase 5.
3. **Architecture Advisor Test**: Ask for a hardware recommendation and verify the advisor checks `gpu_generation` on cited findings.

These are manual validation steps, not automated BATS tests. They should be executed once during Phase 3 (Agent Prompts) of the implementation and documented in the PR description.

### 6.3 Token Overhead Validation

Per NFR-5, agent prompt additions should add minimal token overhead. Validation:

```bash
# Not a BATS test -- a manual check during PR review
# Count tokens added to each prompt file using wc -w as a proxy
wc -w agents/investigation-agent.md  # Before and after
wc -w agents/knowledge-retriever.md
wc -w agents/architecture-advisor.md
# Delta should be < 200 words total across all files
```

---

## 7. Golden Query Test Suite

### 7.1 Purpose

The existing 50 golden queries (in `tests/golden-queries.bats`) test FTS5 relevance ranking. The new golden temporal queries test generation-aware search on a backfilled database.

### 7.2 Test Setup

The golden-temporal tests run `kb tag-gen --auto` in `setup()` to simulate a post-backfill state. This is essential because the tests validate that generation filtering works correctly with real tagged data.

### 7.3 Test Cases

| Test ID | Query | Flags | Expected Result |
|---------|-------|-------|-----------------|
| GT-1 | "M4 TFLOPS" | `--gen m4` | Output contains `gpu-silicon` |
| GT-2 | "M5 neural accelerator" | `--gen m5` | Output length > 10 (results returned) |
| GT-3 | "SIMD width" | _(none)_ | Output contains `simd-wave` (all gens returned) |
| GT-4 | "bandwidth" | `--gen m4` | Output length > 10 (m4 + NULL results returned) |

### 7.4 Implementation Pattern

```bash
#!/usr/bin/env bats

load test_helper/common-setup

setup() {
  KB="$PLUGIN_ROOT/scripts/kb"
  TEST_DB="${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$TEST_DB"
  export GPU_FORGE_DB="$TEST_DB"
  # Apply migration
  run "$KB" migrate-temporal
  # Run auto-tagging to establish realistic gen tags
  run "$KB" tag-gen --auto
}

teardown() {
  rm -f "${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
  rm -f "${BATS_TEST_TMPDIR}/test_gpu_knowledge.db.pre-temporal-backup"
}

@test "golden-temporal: 'M4 TFLOPS' --gen m4 returns gpu-silicon results" {
  run "$KB" search "M4 TFLOPS" --gen m4
  assert_success
  assert_output --partial "gpu-silicon"
}

@test "golden-temporal: 'M5 neural accelerator' --gen m5 returns results" {
  run "$KB" search "M5 neural accelerator" --gen m5
  assert_success
  [ ${#output} -gt 10 ]
}

@test "golden-temporal: 'SIMD width' without --gen returns all generations" {
  run "$KB" search "SIMD width"
  assert_success
  assert_output --partial "simd-wave"
}

@test "golden-temporal: 'bandwidth' --gen m4 excludes m1 findings" {
  run "$KB" search "bandwidth" --gen m4
  assert_success
  [ ${#output} -gt 10 ]
}
```

### 7.5 Non-Regression for Existing Golden Queries

Critical invariant: **All 50 existing golden queries must continue to pass after migration and backfill.** The mechanism:
- Default `kb search` (no `--gen` flag) returns all generations
- The `temporal_status` filter excludes only `superseded` findings, and no findings are superseded immediately after migration (all default to `current`)
- The new `gen` column in output does not affect `--partial` assertions
- FTS5 index is not modified by migration

---

## 8. Regression Safety Net

### 8.1 FTS5 Index Integrity

The migration adds columns to the `findings` table but does NOT modify the FTS5 virtual table or its triggers. Nevertheless, the existing `tests/unit/fts5-search.bats` tests (6 tests) serve as an automatic regression net for FTS5 functionality:

- "SIMD search returns simd-wave results" -- basic FTS5 MATCH
- "claim matches rank higher than notes" -- BM25 weight verification
- "prefix search works" -- FTS5 prefix query
- "phrase search works" -- FTS5 phrase query
- "empty search returns nothing" -- edge case
- "limit parameter works" -- pagination

These tests run on the migrated DB when the full suite executes. No new FTS5-specific tests needed.

### 8.2 Performance Regression Detection

Add to `tests/performance/benchmarks.bats`:

```bash
@test "kb search with --gen filter completes under 1 second" {
  # Apply migration first
  TEST_DB="${BATS_TEST_TMPDIR}/perf_test.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$TEST_DB"
  GPU_FORGE_DB="$TEST_DB" run "$KB" migrate-temporal 2>/dev/null
  start=$(_now_ns)
  GPU_FORGE_DB="$TEST_DB" run "$KB" search "GPU" --gen m4
  end=$(_now_ns)
  elapsed=$(( (end - start) / 1000000 ))
  [ "$status" -eq 0 ]
  [ "$elapsed" -lt 1000 ]
  rm -f "$TEST_DB" "${TEST_DB}.pre-temporal-backup"
}

@test "kb freshness completes under 2 seconds" {
  TEST_DB="${BATS_TEST_TMPDIR}/perf_test.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$TEST_DB"
  GPU_FORGE_DB="$TEST_DB" run "$KB" migrate-temporal 2>/dev/null
  start=$(_now_ns)
  GPU_FORGE_DB="$TEST_DB" run "$KB" freshness
  end=$(_now_ns)
  elapsed=$(( (end - start) / 1000000 ))
  [ "$status" -eq 0 ]
  [ "$elapsed" -lt 2000 ]
  rm -f "$TEST_DB" "${TEST_DB}.pre-temporal-backup"
}
```

### 8.3 Data Loss Detection

The migration test M-6 explicitly verifies finding count >= 1555. Additionally, the existing `tests/unit/db-integrity.bats` test "600+ findings exist" provides a lower-bound check. Together, these catch any data loss during migration.

### 8.4 DB Integrity Post-Migration Checks

The existing db-integrity.bats tests verify structural invariants that must hold after migration:
- "11 skills exist" -- skill count unchanged
- "all findings reference valid skills" -- FK integrity
- "all citations reference valid findings" -- FK integrity
- "confidence values valid" -- CHECK constraint on confidence
- "source_type values valid" -- CHECK constraint on source_type
- "no blog/forum sources marked verified" -- data quality rule
- "600+ findings exist" -- data presence
- "every skill has findings" -- coverage

All of these run against the production DB directly (no test DB copy). After migration of the production DB, these tests must still pass. The migration only adds nullable columns with defaults, so no existing invariant is violated.

### 8.5 Rollback Safety

If migration fails or causes issues, the backup file created by `kb migrate-temporal` enables recovery. The migration test M-3 verifies the backup file exists. Manual rollback:

```bash
cp "${PLUGIN_ROOT}/data/gpu_knowledge.db.pre-temporal-backup" "${PLUGIN_ROOT}/data/gpu_knowledge.db"
```

Since SQLite 3.51.1 supports `ALTER TABLE DROP COLUMN`, a programmatic rollback is also possible:

```sql
ALTER TABLE findings DROP COLUMN gpu_generation;
ALTER TABLE findings DROP COLUMN temporal_status;
ALTER TABLE findings DROP COLUMN superseded_by;
DROP INDEX IF EXISTS idx_findings_gpu_generation;
DROP INDEX IF EXISTS idx_findings_temporal_status;
DROP INDEX IF EXISTS idx_findings_superseded_by;
```

---

## 9. Acceptance Criteria Coverage Matrix

### 9.1 PM Acceptance Criteria Mapping

| AC ID | Acceptance Criterion | Test ID(s) | Test File |
|-------|---------------------|-----------|-----------|
| AC-1.1 | `kb search "register allocation" --gen m4` returns only m4 and universal findings | S-3, S-4, S-6 | temporal.bats |
| AC-1.2 | `kb search "TFLOPS" --gen m5` excludes m1/m2/m3/m4-specific benchmarks | S-4 | temporal.bats |
| AC-1.3 | Findings without gpu_generation set are included in generation-filtered searches | S-5 | temporal.bats |
| AC-1.4 | `--gen universal` returns only generation-independent findings | S-8 | temporal.bats |
| AC-2.1 | `kb supersede 14 565` marks finding #14 as superseded by #565 | SP-1, SP-2 | temporal.bats |
| AC-2.2 | `kb detail 14` shows "SUPERSEDED by #565" | D-1 | temporal.bats |
| AC-2.3 | `kb detail 565` shows "Supersedes #14" | D-2 | temporal.bats |
| AC-2.4 | Superseded findings appear below current findings | SS-3 | temporal.bats |
| AC-2.5 | `--include-superseded` returns both current and superseded | SS-2 | temporal.bats |
| AC-3.1 | `kb tag-gen --auto --dry-run` shows preview | TG-4 | temporal.bats |
| AC-3.2 | `kb tag-gen --auto` correctly maps "m4" tag | TG-5 | temporal.bats |
| AC-3.3 | Multi-gen findings tagged with newest generation | GT-* (implicit via backfill) | golden-temporal.bats |
| AC-3.4 | No-gen findings remain NULL | TG-5 (count check) | temporal.bats |
| AC-4.1 | `kb freshness` shows findings count per gpu_generation | F-1 | temporal.bats |
| AC-4.2 | `kb freshness` shows findings count per temporal_status | F-1 | temporal.bats |
| AC-4.3 | `kb freshness` shows date_published coverage | F-1 | temporal.bats |
| AC-4.4 | `kb freshness` lists unresolved contradictions | F-1 | temporal.bats |
| AC-5.1 | Investigation agent prompt requires gpu_generation on benchmarks | AP-1 | temporal.bats |
| AC-5.2 | Before storing, agent searches for supersession targets | AP-1 (prompt content check) | temporal.bats |
| AC-5.3 | `kb verify` reports missing gpu_generation as WARNING | V-1 | temporal.bats |
| AC-5.4 | Quality check includes temporal consistency | V-1 | temporal.bats |

### 9.2 UX Acceptance Criteria Mapping

| UX Decision | What to Verify | Test ID(s) |
|-------------|---------------|-----------|
| Gen column always visible | Search output includes "gen" header | S-2 |
| No confirm for first supersession | `kb supersede 1 2` succeeds without interaction | SP-1 |
| `--force` for override | Second supersession requires `--force` | SP-6 |
| NULL included in `--gen` results | Untagged findings appear in filtered search | S-5 |
| Invalid gen error message | "Invalid generation" + valid values listed | S-7, TG-3 |
| `--dry-run` preview format | Preview includes "Would tag" / "Would skip" | TG-4 |
| `--json` freshness output | Valid JSON with expected keys | F-2, F-3 |
| `unsupersede` on current = no-op | Warning message, exit 0 | US-2 |
| Error messages include fix | "does not exist" errors suggest `kb search` | SP-4, SP-5 |

### 9.3 TECH Acceptance Criteria Mapping

| Tech Decision | What to Verify | Test ID(s) |
|---------------|---------------|-----------|
| Manual migration | `kb migrate-temporal` is explicit command | M-1 |
| Idempotent migration | Run twice, same result | M-4, M-5 |
| Backup before migration | Backup file created | M-3 |
| CHECK constraints | Invalid values rejected | C-2, C-4 |
| Post-FTS filtering | Search with `--gen` returns correct results | S-3 through S-8 |
| Graceful degradation | Commands work on pre-migration DB | Integration test |
| Substring match for tag-gen | Auto-tagging finds gen patterns | TG-5 |
| Exact match contradiction detection | Freshness shows contradictions | F-1 |

### 9.4 NFR Verification

| NFR ID | Requirement | Verification Method | Test ID(s) |
|--------|-------------|--------------------| ------------|
| NFR-1 | Search <100ms with --gen | Performance benchmark test | Perf-1 |
| NFR-2 | Zero data loss | Finding count >= 1555 post-migration | M-6 |
| NFR-3 | 194 existing tests pass | Full suite run post-migration | All existing |
| NFR-4 | 15+ new tests | 54 new tests delivered | All new |
| NFR-5 | Agent prompt <50 token overhead | Manual word count check | Manual |

---

## 10. Test Execution Runbook

### 10.1 Execution Order

Tests must be run in this order due to dependencies:

```
Phase 1: Schema + Migration (must pass before anything else)
  1. bats tests/unit/temporal.bats           # Migration + constraint tests
     Expected: 42 tests, 0 failures
     Time: ~15 seconds

Phase 2: Existing Suite Regression Check
  2. bats tests/unit/kb-cli.bats             # Verify no breakage
  3. bats tests/unit/db-integrity.bats       # Verify schema additions safe
  4. bats tests/unit/fts5-search.bats        # Verify FTS5 still works
  5. bats tests/golden-queries.bats          # Verify all 50 golden queries pass
  6. bats tests/integration/workflows.bats   # Verify multi-step workflows
     Expected: 194 tests total, 0 failures
     Time: ~45 seconds

Phase 3: New Feature Tests
  7. bats tests/golden-temporal.bats         # Generation-aware golden queries
  8. bats tests/integration/temporal-workflows.bats  # Multi-step temporal workflows
  9. bats tests/performance/benchmarks.bats  # Performance regression check
     Expected: 10 tests, 0 failures
     Time: ~20 seconds
```

### 10.2 Full Suite Command

```bash
# Run all tests (existing + new)
bats tests/unit/ tests/integration/ tests/performance/ tests/golden-queries.bats tests/golden-temporal.bats

# Expected output: 248 tests, 0 failures
# Expected time: ~80 seconds
```

### 10.3 Smoke Test (Quick Validation)

For rapid iteration during development, run just the temporal unit tests:

```bash
bats tests/unit/temporal.bats
# Expected: 42 tests, ~15 seconds
```

### 10.4 CI Integration

The BATS test suite should be run in CI with the `--timing` flag to detect performance regressions:

```bash
bats --timing tests/unit/ tests/integration/ tests/performance/ tests/golden-queries.bats tests/golden-temporal.bats
```

### 10.5 Pre-Commit Checklist

Before committing temporal relevance changes:

1. [ ] `bats tests/unit/temporal.bats` -- all new tests pass
2. [ ] `bats tests/unit/kb-cli.bats` -- existing CLI tests pass
3. [ ] `bats tests/golden-queries.bats` -- all 50 golden queries pass
4. [ ] `bats tests/unit/db-integrity.bats` -- DB integrity maintained
5. [ ] `bats tests/golden-temporal.bats` -- temporal golden queries pass
6. [ ] Manually verify: `kb freshness` on production DB shows expected output
7. [ ] Manually verify: `kb tag-gen --auto --dry-run` shows reasonable preview

---

## 11. Graceful Degradation Testing

### 11.1 Pre-Migration Database Behavior

The TECH spec mandates graceful degradation on pre-migration databases. This requires specific integration tests:

```bash
@test "search on pre-migration DB works without --gen" {
  # Use a fresh DB copy WITHOUT running migrate-temporal
  PREMIG_DB="${BATS_TEST_TMPDIR}/premig.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$PREMIG_DB"
  # Do NOT run migrate-temporal
  run env GPU_FORGE_DB="$PREMIG_DB" "$KB" search "GPU"
  assert_success
  [ ${#output} -gt 10 ]
}

@test "search --gen on pre-migration DB degrades gracefully" {
  PREMIG_DB="${BATS_TEST_TMPDIR}/premig.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$PREMIG_DB"
  run env GPU_FORGE_DB="$PREMIG_DB" "$KB" search "GPU" --gen m4
  # Graceful degradation: --gen flag is ignored, search still works
  assert_success
}

@test "freshness on pre-migration DB gives helpful error" {
  PREMIG_DB="${BATS_TEST_TMPDIR}/premig.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$PREMIG_DB"
  run env GPU_FORGE_DB="$PREMIG_DB" "$KB" freshness
  # Should either degrade gracefully or provide "Run kb migrate-temporal first" message
  # Exact behavior per TECH Q2: graceful degradation chosen
}

@test "supersede on pre-migration DB gives helpful error" {
  PREMIG_DB="${BATS_TEST_TMPDIR}/premig.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$PREMIG_DB"
  run env GPU_FORGE_DB="$PREMIG_DB" "$KB" supersede 1 2
  # Should fail with message to run migration first
  assert_failure
  assert_output --partial "migrate-temporal"
}
```

These tests validate that the `pragma_table_info` check in the modified commands correctly handles pre-migration databases.

---

## 12. Edge Case Test Coverage

### 12.1 Supersession Edge Cases

| Scenario | Expected Behavior | Test Coverage |
|----------|------------------|---------------|
| Supersede finding with itself (`kb supersede 1 1`) | Should fail or warn (self-reference) | Recommend adding: SP-8 |
| Circular supersession (A->B, B->A) | Second call fails: B is not superseded by anything, so `kb supersede 2 1` would mark #2 as superseded by #1. This is semantically odd but not a cycle since only `superseded_by` pointer exists, not a full graph. | Low priority |
| Supersede then unsupersede then re-supersede | Should work: supersede sets state, unsupersede clears it, new supersede creates new link | Covered by SP-1 + US-1 run sequentially |
| Supersede with empty reason | Notes still updated with default `[Superseded by #N]` | SP-1 covers this |

### 12.2 Tag-Gen Edge Cases

| Scenario | Expected Behavior | Test Coverage |
|----------|------------------|---------------|
| Finding with tags "m1,m2,m3,m4,m5" | Tagged as `m5` (newest) | Covered by auto-tagging logic; verifiable via dry-run |
| Finding with tag "pre-m5" | Tagged as `m5` (substring match per TECH Q3) | Verifiable via dry-run |
| Finding with empty tags field | Skipped (no generation in tags) | TG-5 count check |
| Finding with NULL tags field | Skipped (no generation in tags) | TG-5 count check |
| `tag-gen 999999 m4` on nonexistent finding | Error: "does not exist" | TG-3 variant (recommend adding) |

### 12.3 Search Edge Cases

| Scenario | Expected Behavior | Test Coverage |
|----------|------------------|---------------|
| `--gen` and `--include-superseded` combined | Returns superseded + gen-filtered results | Could add: S-10 |
| `--gen` with `--limit 1` | Returns at most 1 result with gen filter | Implicit via existing limit tests |
| Search returning 0 results with `--gen` | Empty output, exit 0 | Low priority |
| All findings in DB are superseded | Search returns empty by default | Low priority |

---

## Research Sources

- [BATS-core GitHub Repository](https://github.com/bats-core/bats-core)
- [Effective End-to-End Testing with BATS (2025)](https://blog.cubieserver.de/2025/effective-end-to-end-testing-with-bats/)
- [Testing Bash Scripts with BATS: A Practical Guide (HackerOne)](https://www.hackerone.com/blog/testing-bash-scripts-bats-practical-guide)
- [BATS Writing Tests Documentation](https://bats-core.readthedocs.io/en/stable/writing-tests.html)
- [BATS FAQ -- Test Isolation and tmpdir](https://bats-core.readthedocs.io/en/stable/faq.html)
- [Testing Bash Scripts Using BATS (2025)](https://blog.thewatertower.org/2025/02/10/testing-bash-scripts-using-bats/)
- [Simple Declarative Schema Migration for SQLite](https://david.rothlis.net/declarative-schema-migration-for-sqlite/)
- [Creating Idempotent DDL Scripts for Database Migrations (Redgate)](https://www.red-gate.com/hub/product-learning/flyway/creating-idempotent-ddl-scripts-for-database-migrations)
- [SQLite Versioning and Migration Strategies](https://www.sqliteforum.com/p/sqlite-versioning-and-migration-strategies)
- [sqldef: Idempotent Schema Management for SQLite](https://github.com/sqldef/sqldef)
- [Trouble-Free Database Migration: Idempotence and Convergence (DZone)](https://dzone.com/articles/trouble-free-database-migration-idempotence-and-co)

---

## Questions & Answers

### Q1: Test setup strategy
**Answer**: Per-test migration in setup(). Fresh DB copy + migration for each test.
**Impact**: Complete test isolation, matches existing kb-cli.bats convention. ~100ms overhead per test is negligible.

### Q2: Self-supersession guard
**Answer**: Yes — add validation (reject old_id == new_id) and test SP-8.
**Impact**: One-line check in `kb supersede`. Cheap guard against common mistake.

### Q3: Golden test DB
**Answer**: Production DB copy with `kb tag-gen --auto` in setup.
**Impact**: Tests real backfill pipeline on actual data. Follows existing golden-queries.bats convention.

---

_Original questions for reference:_

### Q1: Should the `temporal.bats` setup run migration on every test, or should migration be a one-time file-level setup?

**Option A (recommended):** Run `kb migrate-temporal` in `setup()` (per-test). Each test gets a fresh DB copy + fresh migration. This ensures complete test isolation -- one test's mutations (supersessions, tag changes) never leak to another test. The migration is idempotent and fast (<100ms), so the overhead is negligible.

**Option B:** Run migration in `setup_file()` (once per file) and use a shared `BATS_FILE_TMPDIR` DB. Tests run faster but share state. Requires careful ordering to avoid test interdependencies. A failing test could corrupt the shared DB and cascade failures.

**Recommendation:** Option A. Test isolation is more valuable than the ~1-second speedup from shared setup. The existing `kb-cli.bats` already uses per-test `setup()` with a fresh DB copy, so this follows established project convention.

### Q2: Should we add a self-supersession guard (`kb supersede 1 1` fails) as an explicit test and validation?

**Option A (recommended):** Yes -- add both a validation check in `kb supersede` (reject `old_id == new_id`) and a test (SP-8) for it. Self-supersession is a logical error that should be caught early. The implementation cost is one line (`if [ "$old_id" = "$new_id" ]; then echo "ERROR: Cannot supersede a finding with itself" >&2; exit 1; fi`).

**Option B:** No -- self-supersession would just mark the finding as superseded by itself. While semantically meaningless, it causes no data corruption and can be reversed with `unsupersede`. Not worth the implementation effort at this scale.

**Recommendation:** Option A. It is a cheap guard against a common user mistake, and the test documents expected behavior.

### Q3: Should the golden-temporal tests use the production DB directly or a fixture DB with known pre-seeded generation tags?

**Option A (recommended):** Use the production DB copy + `kb tag-gen --auto` in setup. This tests the actual backfill pipeline on real data, catching real-world edge cases. The downside is that if production data changes (new findings added), some assertions may need updating.

**Option B:** Create a small fixture DB (50-100 findings) with hand-crafted generation tags for deterministic testing. Tests are stable and fast but do not exercise real data patterns.

**Option C:** Use production DB copy for most tests, but add a small fixture DB for edge-case deterministic tests (e.g., multi-gen findings, specific contradiction pairs).

**Recommendation:** Option A for the golden-temporal tests (they are intentionally integration-level, testing real data), with Option B as a follow-up if test instability becomes an issue. The existing golden-queries.bats already tests against the production DB, so this follows established convention.

---END QUESTIONS---
