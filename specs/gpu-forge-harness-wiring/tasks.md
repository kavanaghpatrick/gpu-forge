---
spec: gpu-forge-harness-wiring
phase: tasks
total_tasks: 28
created: 2026-02-20
---

# Tasks: gpu-forge Harness Wiring

## Execution Context

| Question | Response |
|----------|----------|
| Testing depth | Comprehensive -- include E2E with all 16 new harness-wiring.bats tests |
| Execution priority | Quality first -- each gap fully implemented with tests before moving to next |
| Constraints | Must pass all existing BATS tests -- zero regressions |

**Quality commands** (from research.md):
- Full suite: `bats tests/ --recursive`
- Unit tests: `bats tests/unit/`
- Integration tests: `bats tests/integration/`
- KB verify: `scripts/kb verify`
- KB dedup: `scripts/kb dedup`
- No lint/typecheck/build (bash+markdown plugin)

**Local CI**: `bats tests/ --recursive && scripts/kb verify && scripts/kb dedup`

## Phase 1: Prerequisites

### Task 0.1: Fix hooks.bats for nested object format [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/tests/unit/hooks.bats`
  2. Fix test "hooks.json has hooks array" (line 10-13) -- change to check for `"object"` type instead of `"array"`:
     ```bash
     @test "hooks.json has hooks object" {
       run jq -e '.hooks | type' "${PLUGIN_ROOT}/hooks/hooks.json"
       assert_success
       assert_output '"object"'
     }
     ```
  3. Fix test "SessionStart event exists" (line 16-18) -- change from `.hooks[] | select(.event == "SessionStart")` to `.hooks.SessionStart`:
     ```bash
     @test "SessionStart event exists" {
       run jq -e '.hooks.SessionStart' "${PLUGIN_ROOT}/hooks/hooks.json"
       assert_success
     }
     ```
  4. Fix test "PostToolUse event exists" (line 21-23) -- change from `.hooks[] | select(.event == "PostToolUse")` to `.hooks.PostToolUse`:
     ```bash
     @test "PostToolUse event exists" {
       run jq -e '.hooks.PostToolUse' "${PLUGIN_ROOT}/hooks/hooks.json"
       assert_success
     }
     ```
  5. Fix test "hook timeouts under 30 seconds" (line 46-50) -- change jq from `.hooks[].hooks[]?` to `.hooks[][] | .hooks[]?`:
     ```bash
     @test "hook timeouts under 30 seconds" {
       local max_timeout
       max_timeout=$(jq '[.hooks[][] | .hooks[]? | .timeout // 0] | max' "${PLUGIN_ROOT}/hooks/hooks.json")
       [ "$max_timeout" -lt 30 ]
     }
     ```
- **Files**: `/Users/patrickkavanagh/gpu_kernel/tests/unit/hooks.bats`
- **Done when**: All 9 hooks.bats tests pass (4 fixed, 5 already passing)
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/unit/hooks.bats`
- **Commit**: `fix(hooks): update hooks.bats tests for nested object format`
- _Requirements: prerequisite for AC-2.1, AC-2.5_
- _Design: hooks.bats Fix (Prerequisite)_

## Phase 2: Core Infrastructure (Gap 8 + Gap 1)

### Task 1.1: Add `kb cleanup` subcommand [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/scripts/kb`
  2. Add `cleanup)` case between `fix-confidence)` (line ~431) and `*)` (line ~433). Use the exact implementation from design.md -- handles `--dry-run`, `--hours N` (default 1), marks stale running investigations as `status='failed'` with `completed_at=datetime('now')` and `summary='Auto-closed: agent did not complete'`
  3. Update help text in `*)` case -- add `cleanup` line under "Quality commands:" after the `fix-confidence` line:
     ```
     echo "  cleanup [--dry-run] [--hours N]  Close stale investigations (default: >1h)"
     ```
- **Files**: `/Users/patrickkavanagh/gpu_kernel/scripts/kb`
- **Done when**: `kb cleanup --dry-run` shows stale investigations; `kb cleanup --hours 1` closes them
- **Verify**: `GPU_FORGE_DB=/Users/patrickkavanagh/gpu_kernel/data/gpu_knowledge.db /Users/patrickkavanagh/gpu_kernel/scripts/kb cleanup --dry-run`
- **Commit**: `feat(kb): add cleanup subcommand for stale investigations`
- _Requirements: FR-1, FR-2, FR-3 (AC-1.1 through AC-1.5)_
- _Design: Gap 8: kb cleanup Subcommand_

### Task 1.2: Add stale investigation check to `kb verify` [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/scripts/kb`, locate the verify) case
  2. Add Check 6 after Check 5 (the "Findings without source URLs" block at line ~335-342), before the summary `echo` at line ~344. Use exact code from design.md:
     ```bash
     # Check 6: Stale investigations (running for >1 hour)
     count=$(run_sql "$DB" "SELECT COUNT(*) FROM investigations WHERE status='running' AND started_at < datetime('now', '-1 hours');")
     if [ "$count" -gt 0 ] 2>/dev/null; then
       echo "WARNING: $count investigations running for >1 hour (may be orphaned). Run: kb cleanup --dry-run"
       issues=$((issues + count))
     else
       echo "OK: No stale investigations"
     fi
     ```
- **Files**: `/Users/patrickkavanagh/gpu_kernel/scripts/kb`
- **Done when**: `kb verify` output includes stale investigation check line
- **Verify**: `/Users/patrickkavanagh/gpu_kernel/scripts/kb verify 2>&1 | grep -E "(stale investigations|No stale)"`
- **Commit**: `feat(kb): add stale investigation check to verify`
- _Requirements: FR-19 (AC-1.6)_
- _Design: Gap 8, verify) modification_

### Task 1.3: [VERIFY] Quality checkpoint after core kb changes [x]

- **Do**: Run hooks and kb-cli unit tests to ensure no regressions from tasks 0.1-1.2
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/unit/hooks.bats && bats /Users/patrickkavanagh/gpu_kernel/tests/unit/kb-cli.bats`
- **Done when**: All unit tests pass, zero regressions
- **Commit**: `chore(harness): pass quality checkpoint` (only if fixes needed)

### Task 2.1: Create investigation-cleanup.sh hook script [x]

- **Do**:
  1. Create `/Users/patrickkavanagh/gpu_kernel/hooks/scripts/investigation-cleanup.sh` with the exact content from design.md:
     - Reads stdin JSON
     - Resolves KB script via CLAUDE_PLUGIN_ROOT or dirname fallback
     - Calls `"$KB" cleanup --hours 0 2>/dev/null || true`
     - Always exits 0
  2. Make executable: `chmod +x /Users/patrickkavanagh/gpu_kernel/hooks/scripts/investigation-cleanup.sh`
- **Files**: `/Users/patrickkavanagh/gpu_kernel/hooks/scripts/investigation-cleanup.sh` (create)
- **Done when**: Script exists, is executable, exits 0 when given empty JSON input
- **Verify**: `echo '{}' | CLAUDE_PLUGIN_ROOT=/Users/patrickkavanagh/gpu_kernel GPU_FORGE_DB=/Users/patrickkavanagh/gpu_kernel/data/gpu_knowledge.db bash /Users/patrickkavanagh/gpu_kernel/hooks/scripts/investigation-cleanup.sh && echo "exit 0 OK"`
- **Commit**: `feat(hooks): add investigation-cleanup.sh for SubagentStop`
- _Requirements: FR-5 (AC-2.2, AC-2.3, AC-2.4)_
- _Design: Gap 1: SubagentStop Hook, investigation-cleanup.sh_

### Task 2.2: Add SubagentStop entry to hooks.json [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/hooks/hooks.json`
  2. Add `"SubagentStop"` key to the `.hooks` object, after `"PostToolUse"`. Content from design.md:
     ```json
     "SubagentStop": [
       {
         "matcher": "investigation-agent",
         "hooks": [
           {
             "type": "command",
             "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/investigation-cleanup.sh",
             "timeout": 5
           }
         ]
       }
     ]
     ```
- **Files**: `/Users/patrickkavanagh/gpu_kernel/hooks/hooks.json`
- **Done when**: `jq -e '.hooks.SubagentStop' hooks/hooks.json` succeeds; matcher is `investigation-agent`
- **Verify**: `jq -e '.hooks.SubagentStop[0].matcher' /Users/patrickkavanagh/gpu_kernel/hooks/hooks.json`
- **Commit**: `feat(hooks): add SubagentStop hook for investigation cleanup`
- _Requirements: FR-4 (AC-2.1)_
- _Design: Gap 1: SubagentStop Hook, hooks.json_

### Task 2.3: Add SubagentStop test to hooks.bats [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/tests/unit/hooks.bats`
  2. Add test after the `"PostToolUse event exists"` test:
     ```bash
     @test "SubagentStop event exists" {
       run jq -e '.hooks.SubagentStop' "${PLUGIN_ROOT}/hooks/hooks.json"
       assert_success
     }
     ```
- **Files**: `/Users/patrickkavanagh/gpu_kernel/tests/unit/hooks.bats`
- **Done when**: hooks.bats has SubagentStop test and all tests pass
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/unit/hooks.bats`
- **Commit**: `test(hooks): add SubagentStop existence test`
- _Requirements: AC-2.5_
- _Design: Gap 1, hooks.bats test_

### Task 2.4: [VERIFY] Quality checkpoint after Gap 8 + Gap 1 [x]

- **Do**: Run full unit test suite to verify no regressions after hooks.json changes and new script
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/unit/`
- **Done when**: All unit tests pass
- **Commit**: `chore(harness): pass quality checkpoint` (only if fixes needed)

## Phase 3: Independent Gaps (6, 4, 3, 5, 2, 7)

### Task 3.1: Create advise.md command (Gap 6) [x]

- **Do**:
  1. Create `/Users/patrickkavanagh/gpu_kernel/commands/advise.md` with the exact content from design.md:
     - Frontmatter: `name: advise`, `description`, `argument-hint`, `context: fork`, `agent: architecture-advisor`, `disable-model-invocation: true`
     - Body: skill areas list, example usage, how it works section
- **Files**: `/Users/patrickkavanagh/gpu_kernel/commands/advise.md` (create)
- **Done when**: File exists with correct frontmatter fields
- **Verify**: `head -8 /Users/patrickkavanagh/gpu_kernel/commands/advise.md | grep -c "advise\|context: fork\|architecture-advisor\|disable-model-invocation"`
- **Commit**: `feat(commands): add advise command for architecture-advisor`
- _Requirements: FR-14 (AC-7.1, AC-7.2, AC-7.5)_
- _Design: Gap 6: advise Command, commands/advise.md_

### Task 3.2: Update commands.bats for 7 commands (Gap 6) [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/tests/unit/commands.bats`
  2. Change test title at line 12: `@test "all 6 command files exist"` -> `@test "all 7 command files exist"`
  3. Update ALL 5 loops to add `advise` to the command list. Each loop currently reads:
     `for cmd in ask investigate knowledge scaffold template review; do`
     Change to:
     `for cmd in ask advise investigate knowledge scaffold template review; do`
     Lines: 13, 19, 25, 31, 65
  4. Add 2 new tests after the `investigate` tests (after line ~52):
     ```bash
     @test "advise command uses context: fork" {
       frontmatter=$(get_frontmatter "${COMMANDS_DIR}/advise.md")
       echo "$frontmatter" | grep -q "^context: fork"
     }

     @test "advise command uses agent: architecture-advisor" {
       frontmatter=$(get_frontmatter "${COMMANDS_DIR}/advise.md")
       echo "$frontmatter" | grep -q "^agent: architecture-advisor"
     }
     ```
- **Files**: `/Users/patrickkavanagh/gpu_kernel/tests/unit/commands.bats`
- **Done when**: All commands.bats tests pass including new advise tests
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/unit/commands.bats`
- **Commit**: `test(commands): update commands.bats for 7 commands including advise`
- _Requirements: FR-15 (AC-7.3, AC-7.4)_
- _Design: Gap 6, commands.bats updates_

### Task 3.3: [VERIFY] Quality checkpoint after Gap 6 [x]

- **Do**: Run commands and hooks unit tests to verify advise wiring
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/unit/commands.bats && bats /Users/patrickkavanagh/gpu_kernel/tests/unit/hooks.bats`
- **Done when**: All tests pass
- **Commit**: `chore(harness): pass quality checkpoint` (only if fixes needed)

### Task 4.1: Expand post-edit-validator.sh with 6 new checks (Gap 4) [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/hooks/scripts/post-edit-validator.sh`
  2. Replace the entire file content with the expanded version from design.md. The new version:
     - Keeps existing structure (stdin JSON, file_path extraction, .metal filter, WARNINGS array, hookSpecificOutput)
     - Refactors existing kernel-qualifier check into `check_missing_kernel_qualifier()` function
     - Adds 6 new check functions: `check_address_space_misuse()`, `check_threadgroup_size()`, `check_atomic_in_loop()`, `check_missing_namespace()`, `check_buffer_index_gaps()`, `check_barrier_near_tg_write()`
     - Calls all 7 check functions before JSON output
     - All checks append to WARNINGS array
     - Always exits 0
- **Files**: `/Users/patrickkavanagh/gpu_kernel/hooks/scripts/post-edit-validator.sh`
- **Done when**: Hook catches `device const`, non-32-multiple TG sizes, missing namespace, buffer gaps, atomics in loops, missing barriers
- **Verify**: `echo '{"tool_input":{"file_path":"/dev/null"}}' | bash /Users/patrickkavanagh/gpu_kernel/hooks/scripts/post-edit-validator.sh; echo "exit code: $?"`
- **Commit**: `feat(hooks): expand post-edit validator with 6 new MSL checks`
- _Requirements: FR-11 (AC-5.1 through AC-5.9)_
- _Design: Gap 4: Post-Edit Hook Expansion, post-edit-validator.sh_

### Task 4.2: Add Write matcher to hooks.json (Gap 4) [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/hooks/hooks.json`
  2. Add second PostToolUse entry for `Write` matcher after the existing `Edit` matcher:
     ```json
     {
       "matcher": "Write",
       "hooks": [
         {
           "type": "command",
           "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/post-edit-validator.sh",
           "timeout": 10
         }
       ]
     }
     ```
- **Files**: `/Users/patrickkavanagh/gpu_kernel/hooks/hooks.json`
- **Done when**: `jq '.hooks.PostToolUse | length' hooks/hooks.json` returns 2; both Edit and Write matchers present
- **Verify**: `jq '.hooks.PostToolUse | length' /Users/patrickkavanagh/gpu_kernel/hooks/hooks.json`
- **Commit**: `feat(hooks): add Write matcher for post-edit validator`
- _Requirements: FR-12 (AC-5.10)_
- _Design: Gap 4, hooks.json Write matcher_

### Task 4.3: [VERIFY] Quality checkpoint after Gap 4 [x]

- **Do**: Run hooks tests and validate post-edit hook catches common issues
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/unit/hooks.bats && bats /Users/patrickkavanagh/gpu_kernel/tests/unit/hooks-execution.bats`
- **Done when**: All hooks tests pass, hook exits 0 on test .metal files
- **Commit**: `chore(harness): pass quality checkpoint` (only if fixes needed)

### Task 5.1: Add `kb backfill-urls` subcommand (Gap 3) [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/scripts/kb`
  2. Add `backfill-urls)` case before `*)` (after `fix-confidence)` and `cleanup)` cases). Use exact implementation from design.md -- lists findings with NULL/empty source_url grouped by source_type
  3. Update help text in `*)` case -- add under "Quality commands:":
     ```
     echo "  backfill-urls                List findings missing source URLs"
     ```
- **Files**: `/Users/patrickkavanagh/gpu_kernel/scripts/kb`
- **Done when**: `kb backfill-urls` runs and shows grouped output
- **Verify**: `GPU_FORGE_DB=/Users/patrickkavanagh/gpu_kernel/data/gpu_knowledge.db /Users/patrickkavanagh/gpu_kernel/scripts/kb backfill-urls 2>&1 | head -5`
- **Commit**: `feat(kb): add backfill-urls subcommand`
- _Requirements: FR-9 (AC-4.2)_
- _Design: Gap 3: URL Backfill, backfill-urls case_

### Task 5.2: Update `kb verify` to distinguish web vs local missing URLs (Gap 3) [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/scripts/kb`, locate Check 5 in verify) (lines ~335-342)
  2. Replace Check 5 with split check from design.md:
     - Check 5a: Web-source findings without URLs (ERROR, increments issues)
     - Check 5b: Benchmark/empirical_test findings without URLs (INFO, does NOT increment issues)
- **Files**: `/Users/patrickkavanagh/gpu_kernel/scripts/kb`
- **Done when**: `kb verify` distinguishes web-source vs local-experiment missing URLs
- **Verify**: `/Users/patrickkavanagh/gpu_kernel/scripts/kb verify 2>&1 | grep -E "(web-source|benchmark/empirical_test|All web-source)"`
- **Commit**: `feat(kb): split verify URL check into web-source vs local`
- _Requirements: FR-10 (AC-4.3)_
- _Design: Gap 3, verify) Check 5 replacement_

### Task 5.3: Add local path convention to investigation-agent.md (Gap 3) [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/agents/investigation-agent.md`
  2. In Phase 3 "Rules for storing findings", after the `source_type` bullet (line ~86-87), add:
     ```markdown
     - For `benchmark` or `empirical_test` findings from local experiments, set `source_url` to the
       relative file path (e.g., `experiments/exp16_8bit.rs`, `metal-gpu-experiments/shaders/exp16_8bit.metal`).
       This makes the finding traceable to its source code.
     ```
- **Files**: `/Users/patrickkavanagh/gpu_kernel/agents/investigation-agent.md`
- **Done when**: investigation-agent.md contains local path convention for benchmark/empirical_test
- **Verify**: `grep -c "relative file path" /Users/patrickkavanagh/gpu_kernel/agents/investigation-agent.md`
- **Commit**: `docs(agents): add local path convention for benchmark findings`
- _Requirements: FR-8 (AC-4.1)_
- _Design: Gap 3, investigation-agent.md local path convention_

### Task 5.4: [VERIFY] Quality checkpoint after Gap 3 [x]

- **Do**: Run kb-cli unit tests and verify backfill-urls works
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/unit/kb-cli.bats && GPU_FORGE_DB=/Users/patrickkavanagh/gpu_kernel/data/gpu_knowledge.db /Users/patrickkavanagh/gpu_kernel/scripts/kb backfill-urls | head -3`
- **Done when**: All kb-cli tests pass, backfill-urls produces output
- **Commit**: `chore(harness): pass quality checkpoint` (only if fixes needed)

### Task 6.1: Add Step 5 to review.md (Gap 5) [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/commands/review.md`
  2. Add "Step 5: Store New Findings" section between "Step 4: Format Structured Review" (ends at line ~162) and "Severity Levels" (starts at line ~164). Use exact content from design.md:
     - Dedup check via `kb search`
     - Skip if found
     - Add if new via `kb add` with `source_type='empirical_test'`, `confidence='medium'`, tag `review-discovered`
     - Max 3 new findings per review
     - Only CRITICAL and WARNING issues
     - Report section at end
- **Files**: `/Users/patrickkavanagh/gpu_kernel/commands/review.md`
- **Done when**: review.md contains Step 5 with KB write-back instructions
- **Verify**: `grep -c "Step 5" /Users/patrickkavanagh/gpu_kernel/commands/review.md`
- **Commit**: `feat(commands): add review-to-KB feedback loop (Step 5)`
- _Requirements: FR-13 (AC-6.1 through AC-6.4)_
- _Design: Gap 5: Review-to-KB Feedback_

### Task 7.1: Restructure investigation-agent.md Phase 3+4 (Gap 2) [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/agents/investigation-agent.md`
  2. Replace the entire Phase 3 title with "Phase 3: Store Findings & Citations"
  3. Keep all existing Phase 3 content (rules, confidence table, dedup check, etc.)
  4. Add the "Inline Citation Protocol" subsection at the end of Phase 3, before Phase 5. Content from design.md:
     - After each academic_paper INSERT, immediately get `last_insert_rowid()` and INSERT citation
     - Author verification rules (extracted_from_page, from_metadata, unverified)
     - CRITICAL note: do NOT batch citations to end
     - Dedup check before citation
  5. Remove the standalone Phase 4 section entirely (lines ~120-147)
  6. Renumber Phase 5 to Phase 4
- **Files**: `/Users/patrickkavanagh/gpu_kernel/agents/investigation-agent.md`
- **Done when**: Phase 3 includes inline citations; no standalone Phase 4 exists; Phase 5 renumbered to 4
- **Verify**: `grep -c "Inline Citation Protocol" /Users/patrickkavanagh/gpu_kernel/agents/investigation-agent.md && ! grep -q "^### Phase 4: Citations" /Users/patrickkavanagh/gpu_kernel/agents/investigation-agent.md && echo "OK"`
- **Commit**: `refactor(agents): merge citation phase into Phase 3 (inline citations)`
- _Requirements: FR-6 (AC-3.1, AC-3.2)_
- _Design: Gap 2: Citation Enforcement, investigation-agent.md restructured Phase 3_

### Task 7.2: Add citation coverage check to `kb verify` (Gap 2) [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/scripts/kb`, locate verify) case
  2. Add Check 7 after Check 6 (stale investigations), before the summary echo. Use exact code from design.md:
     ```bash
     # Check 7: Academic findings without citations
     count=$(run_sql "$DB" \
       "SELECT COUNT(*) FROM findings f
        WHERE f.source_type='academic_paper'
          AND NOT EXISTS (SELECT 1 FROM citations c WHERE c.finding_id=f.id);")
     if [ "$count" -gt 0 ] 2>/dev/null; then
       total=$(run_sql "$DB" "SELECT COUNT(*) FROM findings WHERE source_type='academic_paper';")
       echo "WARNING: $count of $total academic paper findings have no citation record"
       issues=$((issues + count))
     else
       echo "OK: All academic paper findings have citations"
     fi
     ```
- **Files**: `/Users/patrickkavanagh/gpu_kernel/scripts/kb`
- **Done when**: `kb verify` reports citation coverage for academic findings
- **Verify**: `/Users/patrickkavanagh/gpu_kernel/scripts/kb verify 2>&1 | grep -E "(academic paper|All academic)"`
- **Commit**: `feat(kb): add citation coverage check to verify`
- _Requirements: FR-7 (AC-3.3)_
- _Design: Gap 2, verify Check 7_

### Task 7.3: [VERIFY] Quality checkpoint after Gap 5 + Gap 2 [x]

- **Do**: Run full unit test suite plus verify command
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/unit/ && /Users/patrickkavanagh/gpu_kernel/scripts/kb verify 2>&1 | tail -5`
- **Done when**: All unit tests pass, verify shows all 7 checks
- **Commit**: `chore(harness): pass quality checkpoint` (only if fixes needed)

### Task 8.1: Wire scaffold.md to JSON configs (Gap 7) [x]

- **Do**:
  1. Read `/Users/patrickkavanagh/gpu_kernel/commands/scaffold.md`
  2. Add the mapping table from design.md after Step 2 (line ~100), before Step 3:
     - Maps compute_pattern + language to JSON config file
     - Includes all 5 existing configs: reduction-kernel.json, matrix-multiply.json, scan-kernel.json, metal-compute-blank.json, mlx-custom-kernel.json
     - Fallback: unmapped combinations use metal-compute-blank.json
  3. Replace the entire Step 3 (lines ~109-121) with the rewritten version from design.md:
     - Step 3 now reads JSON config based on mapping
     - Extracts parameters, file mappings, knowledge areas
     - Falls back to metal-compute-blank.json
  4. Add "Suggested KB Queries" subsection to Step 9 (after "Next steps" block at ~297). From design.md:
     - Uses JSON `knowledge_applied` field to suggest relevant `/gpu-forge:ask` queries
- **Files**: `/Users/patrickkavanagh/gpu_kernel/commands/scaffold.md`
- **Done when**: scaffold.md references all 5 JSON configs, has mapping table, has fallback, and suggests KB queries
- **Verify**: `grep -c "reduction-kernel.json\|matrix-multiply.json\|scan-kernel.json\|metal-compute-blank.json\|mlx-custom-kernel.json" /Users/patrickkavanagh/gpu_kernel/commands/scaffold.md`
- **Commit**: `feat(commands): wire scaffold wizard to JSON template configs`
- _Requirements: FR-16, FR-17, FR-18 (AC-8.1 through AC-8.7)_
- _Design: Gap 7: Scaffold Wiring_

### Task 8.2: [VERIFY] Quality checkpoint after Gap 7 [x]

- **Do**: Run commands, templates, and full unit test suite
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/unit/commands.bats && bats /Users/patrickkavanagh/gpu_kernel/tests/unit/templates.bats`
- **Done when**: All commands and templates tests pass
- **Commit**: `chore(harness): pass quality checkpoint` (only if fixes needed)

## Phase 4: Integration Tests

### Task 9.1: Create harness-wiring.bats integration test file [x]

- **Do**:
  1. Create `/Users/patrickkavanagh/gpu_kernel/tests/integration/harness-wiring.bats` with the exact 16 tests from design.md:
     - Load `../test_helper/common-setup`
     - `setup()`: copy gpu_knowledge.db to BATS_TEST_TMPDIR, set GPU_FORGE_DB
     - `teardown()`: remove test DB
     - Gap 8 tests (3): cleanup --dry-run, cleanup closes stale, verify reports stale
     - Gap 1 tests (2): investigation-cleanup.sh exits 0, is executable
     - Gap 6 test (1): advise.md frontmatter
     - Gap 4 tests (3): catches device const, missing namespace, buffer index gaps
     - Gap 3 tests (2): backfill-urls runs, verify distinguishes URL types
     - Gap 5 test (1): review.md has Step 5
     - Gap 2 tests (2): investigation-agent has inline citation protocol, verify checks citation coverage
     - Gap 7 tests (2): scaffold references JSON configs, scaffold has fallback
  2. Use exact test implementations from design.md Integration Tests section
- **Files**: `/Users/patrickkavanagh/gpu_kernel/tests/integration/harness-wiring.bats` (create)
- **Done when**: All 16 integration tests pass
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/integration/harness-wiring.bats`
- **Commit**: `test(integration): add harness-wiring.bats with 16 gap tests`
- _Requirements: FR-20 (NFR-8)_
- _Design: Integration Tests section_

## Phase 5: Quality Gates

### Task 10.1: [VERIFY] Full local CI [x]

- **Do**: Run complete test suite to verify zero regressions and all new tests pass
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/ --recursive && /Users/patrickkavanagh/gpu_kernel/scripts/kb verify && /Users/patrickkavanagh/gpu_kernel/scripts/kb dedup`
- **Done when**: All tests pass (196 original + new tests), verify reports all checks, dedup clean
- **Commit**: `chore(harness): pass full local CI` (only if fixes needed)

### Task 10.2: Run `kb cleanup` on live DB to close orphaned investigations [x]

- **Do**:
  1. Run `kb cleanup --dry-run` to preview
  2. Run `kb cleanup --hours 1` to close all 98 orphaned investigations
  3. Verify: `kb verify` no longer shows stale investigation warning
- **Files**: `/Users/patrickkavanagh/gpu_kernel/data/gpu_knowledge.db` (data modification)
- **Done when**: 0 stale investigations remain in the live DB
- **Verify**: `GPU_FORGE_DB=/Users/patrickkavanagh/gpu_kernel/data/gpu_knowledge.db /Users/patrickkavanagh/gpu_kernel/scripts/kb cleanup --dry-run 2>&1 | grep "Would close: 0"`
- **Commit**: `chore(kb): close 98 orphaned investigations via kb cleanup`
- _Requirements: AC-1.5_
- _Design: Gap 8_

### Task 10.3: Create PR and verify CI

- **Do**:
  1. Verify current branch is a feature branch: `git branch --show-current`
  2. If on default branch, STOP and alert user
  3. Stage all changed and new files
  4. Push branch: `git push -u origin <branch-name>`
  5. Create PR using gh CLI:
     ```
     gh pr create --title "feat(gpu-forge): wire 8 harness gaps end-to-end" --body "..."
     ```
     Body includes: summary of all 8 gaps closed, list of files changed/created, test count increase
  6. If gh CLI unavailable, provide URL for manual PR creation
- **Verify**: `gh pr checks --watch` (wait for CI completion) -- all checks show passing
- **Done when**: All CI checks green, PR ready for review
- **If CI fails**: Read failure details with `gh pr checks`, fix locally, push fixes, re-verify
- **Commit**: None (PR creation, not code change)

## Phase 6: PR Lifecycle

### Task 11.1: [VERIFY] CI pipeline passes

- **Do**: Monitor CI after push, fix any failures
- **Verify**: `gh pr checks` shows all green
- **Done when**: CI pipeline passes
- **Commit**: None

### Task 11.2: [VERIFY] AC checklist

- **Do**: Programmatically verify each acceptance criterion is satisfied:
  1. AC-1.1-1.6: `kb cleanup` exists, --dry-run works, --hours works, verify reports stale
  2. AC-2.1-2.5: SubagentStop in hooks.json, script calls cleanup --hours 0, exits 0, <1s, tests pass
  3. AC-3.1-3.4: Inline citations in investigation-agent.md, verify checks citation coverage
  4. AC-4.1-4.4: Local path convention, backfill-urls exists, verify splits web/local, no schema change
  5. AC-5.1-5.10: All 7 checks in post-edit hook, Write matcher in hooks.json
  6. AC-6.1-6.4: Step 5 in review.md, dedup check, empirical_test + medium + review-discovered
  7. AC-7.1-7.5: advise.md exists, frontmatter correct, commands.bats updated
  8. AC-8.1-8.7: scaffold wired to JSON, mapping table, fallback, knowledge_applied
- **Verify**: `bats /Users/patrickkavanagh/gpu_kernel/tests/integration/harness-wiring.bats && bats /Users/patrickkavanagh/gpu_kernel/tests/unit/ && echo "All ACs verified"`
- **Done when**: All acceptance criteria confirmed via automated checks
- **Commit**: None

## Notes

### POC shortcuts taken
- None -- quality-first approach means each gap is fully implemented before moving to next

### Implementation order rationale
- Gap 8 (cleanup) first because Gap 1 (SubagentStop) depends on it
- Gap 6 (advise) early because it's lowest risk and highest test-regression risk (commands.bats loops)
- Gap 4 (post-edit hook) before Gap 3 because it modifies hooks.json (batch hooks.json changes)
- Gap 2 (citations) late because it depends on understanding Gap 3 changes to investigation-agent.md
- Gap 7 (scaffold) last because it's most complex and independent

### Test count expectations
- Before: 196 tests (with 7 pre-existing failures: 4 hooks format, 2 golden queries, 1 DB size)
- After: 196 original + 1 SubagentStop test + 2 advise tests + 16 integration tests = 215+ tests
- The 4 hooks.bats failures are fixed in Task 0.1
- The 2 golden query + 1 DB size failures are pre-existing and out of scope

### Files changed summary
| File | Action | Tasks |
|------|--------|-------|
| `tests/unit/hooks.bats` | Modify | 0.1, 2.3 |
| `scripts/kb` | Modify | 1.1, 1.2, 5.1, 5.2, 7.2 |
| `hooks/scripts/investigation-cleanup.sh` | Create | 2.1 |
| `hooks/hooks.json` | Modify | 2.2, 4.2 |
| `commands/advise.md` | Create | 3.1 |
| `tests/unit/commands.bats` | Modify | 3.2 |
| `hooks/scripts/post-edit-validator.sh` | Modify | 4.1 |
| `agents/investigation-agent.md` | Modify | 5.3, 7.1 |
| `commands/review.md` | Modify | 6.1 |
| `commands/scaffold.md` | Modify | 8.1 |
| `tests/integration/harness-wiring.bats` | Create | 9.1 |
| `data/gpu_knowledge.db` | Data modify | 10.2 |
