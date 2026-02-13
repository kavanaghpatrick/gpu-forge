---
spec: kb-temporal-relevance
phase: tasks
total_tasks: 32
created: 2026-02-13
---

# Tasks: KB Temporal Relevance Pipeline

Plugin root: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/`
Test command: `bats tests/ --recursive` (from plugin root)
Existing tests: 194 (must remain green throughout)

## Phase 1: Make It Work (POC)

Focus: Schema migration + core CLI commands working end-to-end. Skip edge cases, accept shortcuts.

- [x] 1.1 Update schema.sql with temporal columns and indexes
  - **Do**:
    1. Open `data/schema.sql`
    2. Add 3 columns after `investigation_session` in `CREATE TABLE findings`: `gpu_generation TEXT CHECK(...)`, `temporal_status TEXT CHECK(...)`, `superseded_by INTEGER REFERENCES findings(id)`
    3. Add 3 indexes after existing index definitions: `idx_findings_gpu_generation`, `idx_findings_temporal_status`, `idx_findings_superseded_by`
    4. Exact SQL from TECH.md Section 2.4
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/data/schema.sql`
  - **Done when**: `sqlite3 :memory: < data/schema.sql` exits 0; schema contains 3 new columns
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && sqlite3 :memory: < data/schema.sql && echo "OK"`
  - **Commit**: `feat(kb): add temporal columns to schema.sql`
  - _Requirements: FR-19_
  - _Design: Schema Migration, Section 2.4_

- [x] 1.2 Add `migrate-temporal` command to kb CLI
  - **Do**:
    1. Open `scripts/kb`
    2. Add `migrate-temporal)` case before the `*)` default case (before line 433)
    3. Implement idempotent migration: `pragma_table_info` check, backup creation, 3x ALTER TABLE ADD COLUMN, 3x CREATE INDEX, post-verify column count
    4. Exact bash from TECH.md Section 3.2
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/scripts/kb`
  - **Done when**: `kb migrate-temporal` succeeds on production DB; second run prints "already applied"; backup file created
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && cp data/gpu_knowledge.db /tmp/test_migrate.db && GPU_FORGE_DB=/tmp/test_migrate.db scripts/kb migrate-temporal && GPU_FORGE_DB=/tmp/test_migrate.db scripts/kb migrate-temporal && sqlite3 /tmp/test_migrate.db "SELECT COUNT(*) FROM pragma_table_info('findings') WHERE name IN ('gpu_generation','temporal_status','superseded_by');" | grep -q "3" && echo "OK" && rm -f /tmp/test_migrate.db /tmp/test_migrate.db.pre-temporal-backup`
  - **Commit**: `feat(kb): add migrate-temporal command`
  - _Requirements: FR-15, NFR-2_
  - _Design: Schema Migration, Section 3.2_

- [x] 1.3 Modify `search` command with --gen and --include-superseded flags
  - **Do**:
    1. Replace the existing `search)` case (lines 68-90 of `scripts/kb`)
    2. Add graceful degradation: `pragma_table_info` check for `gpu_generation` column
    3. If column exists: add `--gen|-g` flag parsing, `--include-superseded|-S` flag, `gen_clause`, `status_clause`, gen column in SELECT, CASE WHEN ordering for superseded
    4. If column missing: fall back to original query (ignore --gen, no status filter)
    5. Validate --gen values: m1|m2|m3|m4|m5|universal; error on invalid
    6. Exact SQL pattern from TECH.md Section 3.3
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/scripts/kb`
  - **Done when**: `kb search "GPU" --gen m4` returns results with gen column; `kb search "GPU"` (no flags) still works; invalid gen produces error
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && GPU_FORGE_DB=/tmp/test_search.db bash -c 'cp data/gpu_knowledge.db /tmp/test_search.db && scripts/kb migrate-temporal && scripts/kb search "GPU" --gen m4 | head -5 && scripts/kb search "GPU" | head -5 && scripts/kb search "GPU" --gen m6 2>&1 | grep -q "Invalid generation" && echo "OK" && rm -f /tmp/test_search.db /tmp/test_search.db.pre-temporal-backup'`
  - **Commit**: `feat(kb): add --gen and --include-superseded to search`
  - _Requirements: FR-4, FR-5, FR-18, AC-1.1, AC-1.2, AC-1.3, AC-1.4, AC-2.4, AC-2.5_
  - _Design: Search Filter Engine, Section 3.3_

- [x] 1.4 Add `supersede` command with self-guard
  - **Do**:
    1. Add `supersede)` case to `scripts/kb` before `*)`
    2. Parse: `old_id` (grep numeric), `new_id` (grep numeric), optional `reason`
    3. Validate both IDs exist via `SELECT COUNT(*)`
    4. Self-supersession guard: `old_id == new_id` -> error
    5. Check existing `superseded_by`; require `--force` to override
    6. UPDATE: set `temporal_status='superseded'`, `superseded_by=new_id`, append reason to notes
    7. Exact bash from TECH.md Section 3.4
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/scripts/kb`
  - **Done when**: `kb supersede 1 2` marks finding #1 superseded; `kb supersede 1 1` rejected; nonexistent IDs rejected
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && GPU_FORGE_DB=/tmp/test_sup.db bash -c 'cp data/gpu_knowledge.db /tmp/test_sup.db && scripts/kb migrate-temporal && scripts/kb supersede 1 2 "test" | grep -q "marked superseded" && scripts/kb supersede 99999 2 2>&1 | grep -q "does not exist" && echo "OK" && rm -f /tmp/test_sup.db /tmp/test_sup.db.pre-temporal-backup'`
  - **Commit**: `feat(kb): add supersede command`
  - _Requirements: FR-7, AC-2.1, AC-2.6_
  - _Design: Supersession Manager, Section 3.4_

- [x] 1.5 Add `unsupersede` command
  - **Do**:
    1. Add `unsupersede)` case to `scripts/kb` before `*)`
    2. Validate finding exists and is actually superseded
    3. Reset `temporal_status='current'`, clear `superseded_by=NULL`, append audit note
    4. No-op warning on already-current finding (exit 0)
    5. Exact bash from TECH.md Section 3.5
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/scripts/kb`
  - **Done when**: `kb unsupersede <id>` restores superseded finding to current; already-current produces warning
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && GPU_FORGE_DB=/tmp/test_unsup.db bash -c 'cp data/gpu_knowledge.db /tmp/test_unsup.db && scripts/kb migrate-temporal && scripts/kb supersede 1 2 && scripts/kb unsupersede 1 | grep -q "restored to current" && scripts/kb unsupersede 1 2>&1 | grep -q "already current" && echo "OK" && rm -f /tmp/test_unsup.db /tmp/test_unsup.db.pre-temporal-backup'`
  - **Commit**: `feat(kb): add unsupersede command`
  - _Requirements: FR-8_
  - _Design: Supersession Manager, Section 3.5_

- [x] V1 [VERIFY] Quality checkpoint: existing 194 BATS tests still pass
  - **Do**: Run full existing test suite to confirm no regressions from Phase 1 changes
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/ --recursive 2>&1 | tail -3`
  - **Done when**: All 194 tests pass (0 failures)
  - **Commit**: `chore(kb): pass quality checkpoint after core temporal commands` (only if fixes needed)
  - _Requirements: NFR-3_

- [x] 1.6 Modify `detail` command for supersession chain display
  - **Do**:
    1. Replace existing `detail)` case in `scripts/kb`
    2. After standard `SELECT f.*, s.name`, add supersession chain section
    3. Check `temporal_status` and `superseded_by` for upward link ("SUPERSEDED by #N")
    4. Query `WHERE superseded_by=<id>` for downward links ("Supersedes #N")
    5. Keep existing citations section
    6. Add graceful degradation: only show supersession section if temporal columns exist
    7. Exact bash from TECH.md Section 3.8
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/scripts/kb`
  - **Done when**: `kb detail 1` (after supersede 1->2) shows "SUPERSEDED by #2"; `kb detail 2` shows "Supersedes #1"
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && GPU_FORGE_DB=/tmp/test_det.db bash -c 'cp data/gpu_knowledge.db /tmp/test_det.db && scripts/kb migrate-temporal && scripts/kb supersede 1 2 "test" && scripts/kb detail 1 | grep -q "SUPERSEDED" && scripts/kb detail 2 | grep -q "Supersedes" && echo "OK" && rm -f /tmp/test_det.db /tmp/test_det.db.pre-temporal-backup'`
  - **Commit**: `feat(kb): show supersession chain in detail command`
  - _Requirements: FR-17, AC-2.2, AC-2.3_
  - _Design: Section 3.8_

- [x] 1.7 Modify `add` command for optional gpu_generation param
  - **Do**:
    1. Modify existing `add)` case in `scripts/kb`
    2. Accept optional 11th positional param `${11}` for gpu_generation
    3. Validate against allowed values: m1|m2|m3|m4|m5|universal
    4. Build conditional `gen_col` and `gen_val` for INSERT
    5. Existing 9-10 arg calls unchanged (column defaults to NULL)
    6. Exact bash from TECH.md Section 3.10
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/scripts/kb`
  - **Done when**: `kb add gpu-silicon "test" "claim" ... "m4"` inserts with gpu_generation=m4; `kb add` without 11th param still works
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && GPU_FORGE_DB=/tmp/test_add.db bash -c 'cp data/gpu_knowledge.db /tmp/test_add.db && scripts/kb migrate-temporal && scripts/kb add gpu-silicon "temporal-test" "Test claim" "evidence" "http://test" "Test" "benchmark" "high" "m4,test" "m4" && sqlite3 /tmp/test_add.db "SELECT gpu_generation FROM findings WHERE topic='"'"'temporal-test'"'"' ORDER BY id DESC LIMIT 1;" | grep -q "m4" && echo "OK" && rm -f /tmp/test_add.db /tmp/test_add.db.pre-temporal-backup'`
  - **Commit**: `feat(kb): add gpu_generation param to add command`
  - _Requirements: FR-16_
  - _Design: Section 3.10_

- [x] 1.8 POC Checkpoint: end-to-end temporal workflow
  - **Do**: Verify the complete write-search-supersede-detail flow works end-to-end on a test DB copy
    1. Migrate DB
    2. Add a finding with gpu_generation=m4
    3. Search with --gen m4 (should include new finding)
    4. Supersede old finding with new finding
    5. Detail shows supersession chain
    6. Search excludes superseded finding by default
    7. Search --include-superseded shows it
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && GPU_FORGE_DB=/tmp/test_poc.db bash -c 'cp data/gpu_knowledge.db /tmp/test_poc.db && scripts/kb migrate-temporal && scripts/kb add gpu-silicon "poc-test" "POC claim for temporal" "evidence" "http://poc" "POC" "benchmark" "high" "m4,poc" "m4" && scripts/kb search "POC claim" --gen m4 | grep -q "POC" && new_id=$(sqlite3 /tmp/test_poc.db "SELECT id FROM findings WHERE topic='"'"'poc-test'"'"' ORDER BY id DESC LIMIT 1;") && scripts/kb supersede 1 $new_id "POC test" && scripts/kb detail 1 | grep -q "SUPERSEDED" && echo "POC OK" && rm -f /tmp/test_poc.db /tmp/test_poc.db.pre-temporal-backup'`
  - **Done when**: Full temporal workflow verified end-to-end
  - **Commit**: `feat(kb): complete temporal POC`
  - _Requirements: FR-4, FR-5, FR-7, FR-15, FR-16, FR-17_

## Phase 2: Feature Complete

Focus: Remaining CLI commands (tag-gen, freshness, verify Check 6, help), agent prompts, and graceful degradation.

- [x] 2.1 Add `tag-gen` command (single, auto, clear modes)
  - **Do**:
    1. Add `tag-gen)` case to `scripts/kb` before `*)`
    2. Three modes: single (`<id> <gen>`), auto (`--auto [--dry-run] [--skill]`), clear (`<id> --clear`)
    3. Auto mode: read `tags` field, case-insensitive substring match for m1-m5, newest-gen wins (m5>m4>m3>m2>m1)
    4. Auto mode never overwrites existing non-NULL `gpu_generation`
    5. Dry-run shows preview with counts; without --dry-run executes UPDATE
    6. Validate generation values; validate finding exists for single mode
    7. Exact bash from TECH.md Section 3.6
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/scripts/kb`
  - **Done when**: `kb tag-gen 1 m4` tags finding; `kb tag-gen --auto --dry-run` shows preview; `kb tag-gen --auto` executes; `kb tag-gen 1 --clear` resets
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && GPU_FORGE_DB=/tmp/test_tg.db bash -c 'cp data/gpu_knowledge.db /tmp/test_tg.db && scripts/kb migrate-temporal && scripts/kb tag-gen 1 m4 | grep -q "tagged" && scripts/kb tag-gen 1 --clear | grep -q "cleared" && scripts/kb tag-gen --auto --dry-run | grep -q "Preview" && scripts/kb tag-gen --auto | grep -q "Auto-tagged" && echo "OK" && rm -f /tmp/test_tg.db /tmp/test_tg.db.pre-temporal-backup'`
  - **Commit**: `feat(kb): add tag-gen command for generation tagging`
  - _Requirements: FR-6, FR-14, AC-3.1, AC-3.2, AC-3.3, AC-3.4_
  - _Design: Generation Tagger, Section 3.6_

- [x] 2.2 Add `freshness` command (dashboard + JSON)
  - **Do**:
    1. Add `freshness)` case to `scripts/kb` before `*)`
    2. Parse `--skill <name>` and `--json` flags
    3. Human mode: show generation coverage, temporal status, date coverage %, contradiction pairs
    4. JSON mode: output JSON object with same data
    5. Contradiction SQL: exact `topic + skill_id` match between non-superseded findings with different gpu_generation
    6. Optional `--skill` filter for scoping
    7. Exact bash from TECH.md Section 3.7
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/scripts/kb`
  - **Done when**: `kb freshness` shows dashboard with 4 sections; `kb freshness --json` returns valid JSON; `kb freshness --skill gpu-silicon` filters
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && GPU_FORGE_DB=/tmp/test_fr.db bash -c 'cp data/gpu_knowledge.db /tmp/test_fr.db && scripts/kb migrate-temporal && scripts/kb freshness | grep -q "KB Temporal Health" && scripts/kb freshness --json | python3 -c "import sys,json;d=json.load(sys.stdin);print(\"JSON OK\")" && echo "OK" && rm -f /tmp/test_fr.db /tmp/test_fr.db.pre-temporal-backup'`
  - **Commit**: `feat(kb): add freshness dashboard command`
  - _Requirements: FR-13, AC-4.1, AC-4.2, AC-4.3, AC-4.4_
  - _Design: Freshness Dashboard, Section 3.7_

- [x] 2.3 Add Check 6 to `verify` command (temporal quality)
  - **Do**:
    1. In existing `verify)` case, add Check 6 after Check 5 (after line ~342, before the final summary block)
    2. Gate on `pragma_table_info` check (only run post-migration)
    3. Report benchmark/empirical_test findings without gpu_generation as WARNING
    4. Show up to 10 examples with truncated claim
    5. Exact bash from TECH.md Section 3.9
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/scripts/kb`
  - **Done when**: `kb verify` on post-migration DB includes temporal check; on pre-migration DB, Check 6 is skipped
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && GPU_FORGE_DB=/tmp/test_ver.db bash -c 'cp data/gpu_knowledge.db /tmp/test_ver.db && scripts/kb migrate-temporal && scripts/kb verify | grep -q "benchmark\|gpu_generation\|empirical" && echo "OK" && rm -f /tmp/test_ver.db /tmp/test_ver.db.pre-temporal-backup'`
  - **Commit**: `feat(kb): add temporal quality check to verify`
  - _Requirements: FR-11, AC-5.3_
  - _Design: Section 3.9_

- [x] V2 [VERIFY] Quality checkpoint: all 194 existing tests pass + new commands work
  - **Do**: Run full BATS suite to verify no regressions; spot-check new commands
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/ --recursive 2>&1 | tail -3`
  - **Done when**: All 194 tests pass (0 failures)
  - **Commit**: `chore(kb): pass quality checkpoint after feature-complete commands` (only if fixes needed)

- [x] 2.4 Update help text with temporal commands
  - **Do**:
    1. In the `*)` default/help case of `scripts/kb`, add "Temporal commands:" section after "Quality commands:"
    2. List: supersede, unsupersede, tag-gen, tag-gen --auto, freshness, migrate-temporal
    3. Add "Generations:" reference line and "Search filters:" section
    4. Exact text from TECH.md Section 3.11
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/scripts/kb`
  - **Done when**: `kb` (no args) output includes "Temporal commands", "Generations:", "Search filters:"
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && scripts/kb 2>&1 | grep -q "Temporal commands" && scripts/kb 2>&1 | grep -q "Generations:" && echo "OK"`
  - **Commit**: `feat(kb): add temporal commands to help text`
  - _Requirements: FR-20_
  - _Design: Section 3.11_

- [x] 2.5 Update investigation-agent.md with 3 temporal insertions
  - **Do**:
    1. Open `agents/investigation-agent.md`
    2. After Phase 2 step 3 ("Code Analysis", ~line 73): insert supersession check step
    3. After Phase 3 confidence-source table (~line 99): insert generation tagging requirement
    4. After Phase 5 `kb verify` / `kb dedup` step (~line 156): insert resolve supersessions step
    5. Exact markdown from TECH.md Section 4.1
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/agents/investigation-agent.md`
  - **Done when**: File contains "Supersession check", "Generation tagging", "Resolve supersessions" sections
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && grep -q "Supersession check" agents/investigation-agent.md && grep -q "Generation tagging" agents/investigation-agent.md && grep -q "Resolve supersessions" agents/investigation-agent.md && echo "OK"`
  - **Commit**: `feat(kb): add temporal awareness to investigation agent`
  - _Requirements: FR-9, FR-10, AC-5.1, AC-5.2_
  - _Design: Agent Prompt Changes, Section 4.1_

- [x] 2.6 Update knowledge-retriever.md with gen detection + formatting
  - **Do**:
    1. Open `agents/knowledge-retriever.md`
    2. After step 4 "Run the query" (~line 49): insert "Detect generation context" step
    3. After "Full Format" example (~line 68): insert "Generation-Aware Format" section
    4. Exact markdown from TECH.md Section 4.2
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/agents/knowledge-retriever.md`
  - **Done when**: File contains "Detect generation context" and "Generation-Aware Format" sections
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && grep -q "Detect generation context" agents/knowledge-retriever.md && grep -q "Generation-Aware Format" agents/knowledge-retriever.md && echo "OK"`
  - **Commit**: `feat(kb): add generation awareness to knowledge retriever`
  - _Requirements: FR-12_
  - _Design: Agent Prompt Changes, Section 4.2_

- [x] 2.7 Update architecture-advisor.md with gen consistency check
  - **Do**:
    1. Open `agents/architecture-advisor.md`
    2. After step 4 "Apply M4/M5 Hardware Constraints" (~line 55): insert "Check generation consistency" step
    3. Exact markdown from TECH.md Section 4.3
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/agents/architecture-advisor.md`
  - **Done when**: File contains "Check generation consistency" section
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && grep -q "Check generation consistency" agents/architecture-advisor.md && echo "OK"`
  - **Commit**: `feat(kb): add generation check to architecture advisor`
  - _Requirements: FR-12_
  - _Design: Agent Prompt Changes, Section 4.3_

## Phase 3: Testing

Focus: Comprehensive BATS test suite -- 54 new tests across 3 new files + 1 modified.

- [x] 3.1 Create `tests/unit/temporal.bats` with migration + constraint tests (12 tests)
  - **Do**:
    1. Create test file with setup/teardown matching existing kb-cli.bats pattern
    2. setup(): copy production DB, set GPU_FORGE_DB, run migrate-temporal
    3. Write 12 tests covering:
       - Migration adds 3 new columns (pragma_table_info check)
       - Migration is idempotent (second run prints "already applied")
       - Migration creates backup file
       - gpu_generation accepts valid values (m1-m5, universal)
       - gpu_generation rejects invalid value (m6)
       - gpu_generation accepts NULL
       - temporal_status default is 'current'
       - temporal_status rejects invalid value ('historical')
       - superseded_by accepts valid finding ID
       - superseded_by accepts NULL
       - CHECK constraint prevents invalid gpu_generation on INSERT
       - All existing rows have temporal_status='current' after migration
    4. Test patterns from TECH.md Section 6.1
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/tests/unit/temporal.bats`
  - **Done when**: `bats tests/unit/temporal.bats` passes all 12 tests
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/unit/temporal.bats`
  - **Commit**: `test(kb): add migration and constraint tests`
  - _Requirements: NFR-4, FR-15_
  - _Design: Test Strategy, Section 6.1_

- [x] 3.2 Add search filtering tests to temporal.bats (10 tests)
  - **Do**:
    1. Append to `tests/unit/temporal.bats`
    2. Write 10 tests covering:
       - Search without --gen returns results (backward compatible)
       - Search results include gen column header
       - Search --gen m4 returns results
       - Search --gen excludes other specific generations (tag #1 as m1, #2 as m4, search --gen m4)
       - Search --gen includes NULL findings
       - Search --gen includes universal findings
       - Search --gen rejects invalid generation
       - Search -g short flag works
       - Search excludes superseded findings by default
       - Search --include-superseded shows superseded with SUPERSEDED label
    3. Test patterns from TECH.md Section 6.1
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/tests/unit/temporal.bats`
  - **Done when**: All search filter tests pass
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/unit/temporal.bats`
  - **Commit**: `test(kb): add search filtering tests`
  - _Requirements: NFR-4, FR-4, FR-5_
  - _Design: Test Strategy, Section 6.1_

- [x] 3.3 Add supersession + unsupersede tests to temporal.bats (8 tests)
  - **Do**:
    1. Append to `tests/unit/temporal.bats`
    2. Write 8 tests covering:
       - Supersede marks finding as superseded (check DB state)
       - Supersede with reason appends to notes
       - Supersede validates old_id exists
       - Supersede validates new_id exists
       - Supersede self-guard: old_id == new_id rejected
       - Supersede requires --force for already-superseded
       - Unsupersede restores to current (check DB state)
       - Unsupersede on current finding is no-op warning
    3. Test patterns from TECH.md Section 6.1
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/tests/unit/temporal.bats`
  - **Done when**: All supersession tests pass
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/unit/temporal.bats`
  - **Commit**: `test(kb): add supersession tests`
  - _Requirements: NFR-4, FR-7, FR-8_
  - _Design: Test Strategy, Section 6.1_

- [x] V3 [VERIFY] Quality checkpoint: all tests pass including new temporal tests
  - **Do**: Run full BATS suite (existing 194 + new temporal unit tests)
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/ --recursive 2>&1 | tail -5`
  - **Done when**: All tests pass (0 failures)
  - **Commit**: `chore(kb): pass quality checkpoint after core test coverage` (only if fixes needed)

- [x] 3.4 Add tag-gen, freshness, verify, detail, add, help tests to temporal.bats (12 tests)
  - **Do**:
    1. Append to `tests/unit/temporal.bats`
    2. Write 12 tests covering:
       - tag-gen single: sets gpu_generation
       - tag-gen --clear: resets to NULL
       - tag-gen rejects invalid generation
       - tag-gen --auto --dry-run shows preview
       - tag-gen --auto executes tagging
       - tag-gen --auto does not overwrite existing
       - freshness shows dashboard
       - freshness --json returns valid JSON
       - verify includes temporal check after migration
       - detail shows supersession chain
       - add with gpu_generation parameter works
       - help text includes temporal commands
    3. Test patterns from TECH.md Section 6.1
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/tests/unit/temporal.bats`
  - **Done when**: All 12 tests pass; total temporal.bats count is ~42
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/unit/temporal.bats`
  - **Commit**: `test(kb): add tag-gen, freshness, verify, detail, add, help tests`
  - _Requirements: NFR-4, FR-6, FR-13, FR-11, FR-17, FR-16, FR-20_
  - _Design: Test Strategy, Section 6.1_

- [x] 3.5 Create `tests/golden-temporal.bats` with 4 golden query tests
  - **Do**:
    1. Create new test file at `tests/golden-temporal.bats`
    2. setup(): copy production DB, migrate, run tag-gen --auto
    3. Write 4 golden query tests:
       - `kb search "M4 TFLOPS" --gen m4` returns gpu-silicon results
       - `kb search "M5 neural accelerator" --gen m5` returns results
       - `kb search "SIMD width"` (no --gen) returns all generations
       - `kb search "bandwidth" --gen m4` excludes m1 findings
    4. Test patterns from TECH.md Section 6.2
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/tests/golden-temporal.bats`
  - **Done when**: `bats tests/golden-temporal.bats` passes all 4 tests
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/golden-temporal.bats`
  - **Commit**: `test(kb): add golden temporal query tests`
  - _Requirements: NFR-4_
  - _Design: Test Strategy, Section 6.2_

- [x] 3.6 Create `tests/integration/temporal-workflows.bats` with 4 workflow tests
  - **Do**:
    1. Create new test file at `tests/integration/temporal-workflows.bats`
    2. setup(): copy production DB, migrate
    3. Write 4 multi-step workflow tests:
       - Full supersession workflow: add new finding -> supersede old -> verify detail chain -> search excludes old
       - Auto-tag + freshness workflow: auto-tag -> freshness shows distribution
       - Tag-gen + search filter: tag finding m4 -> search --gen m4 includes it -> search --gen m1 excludes it
       - Unsupersede round-trip: supersede -> unsupersede -> verify restored
    4. Test patterns from TECH.md Section 6.1
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/tests/integration/temporal-workflows.bats`
  - **Done when**: `bats tests/integration/temporal-workflows.bats` passes all 4 tests
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/integration/temporal-workflows.bats`
  - **Commit**: `test(kb): add temporal workflow integration tests`
  - _Requirements: NFR-4_
  - _Design: Test Strategy_

- [x] 3.7 Add 2 performance regression tests to `tests/performance/benchmarks.bats`
  - **Do**:
    1. Append to existing `tests/performance/benchmarks.bats`
    2. Add setup for these tests: copy DB, migrate, auto-tag
    3. Write 2 tests:
       - `kb search "GPU" --gen m4` completes under 1 second
       - `kb freshness` completes under 2 seconds
    4. Follow existing timing pattern: `_now_ns()`, elapsed calculation, threshold check
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/tests/performance/benchmarks.bats`
  - **Done when**: Both performance tests pass with timing assertions
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/performance/benchmarks.bats`
  - **Commit**: `test(kb): add temporal performance benchmarks`
  - _Requirements: NFR-1, NFR-4_
  - _Design: Performance Considerations_

- [x] 3.8 Add agent prompt static verification tests (2 tests)
  - **Do**:
    1. Append to `tests/unit/temporal.bats`
    2. Write 2 tests:
       - investigation-agent.md contains "Supersession check" and "Generation tagging" and "Resolve supersessions"
       - knowledge-retriever.md contains "Detect generation context" and "Generation-Aware Format"
    3. Use `grep -c` pattern matching
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/tests/unit/temporal.bats`
  - **Done when**: Both agent prompt tests pass
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/unit/temporal.bats`
  - **Commit**: `test(kb): add agent prompt verification tests`
  - _Requirements: NFR-4, FR-9, FR-12_
  - _Design: Test Strategy_

## Phase 4: Quality Gates

Focus: Backfill, validation, backward compat, final quality.

- [x] 4.1 Run migration on production DB
  - **Do**:
    1. Run `kb migrate-temporal` on the actual production database
    2. Verify 3 new columns present via pragma_table_info
    3. Verify backup file created
    4. Verify all 1555+ findings preserved (row count unchanged)
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/data/gpu_knowledge.db`
  - **Done when**: Production DB has 3 temporal columns; backup exists; row count unchanged
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && scripts/kb migrate-temporal && sqlite3 data/gpu_knowledge.db "SELECT COUNT(*) FROM pragma_table_info('findings') WHERE name IN ('gpu_generation','temporal_status','superseded_by');" | grep -q "3" && sqlite3 data/gpu_knowledge.db "SELECT COUNT(*) FROM findings;" && echo "Migration verified"`
  - **Commit**: `feat(kb): apply temporal migration to production DB`
  - _Requirements: FR-15, NFR-2_
  - _Design: Schema Migration_

- [ ] 4.2 Run backfill: auto-tag existing findings from tags field
  - **Do**:
    1. Run `kb tag-gen --auto --dry-run` to preview
    2. Review output for false positives (especially "pre-m5" edge case)
    3. Run `kb tag-gen --auto` to execute backfill
    4. Verify tagged count increased
    5. Run verification queries from TECH.md Section 5.4
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/data/gpu_knowledge.db`
  - **Done when**: ~154 findings auto-tagged; no existing findings broken; total row count unchanged
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && scripts/kb tag-gen --auto --dry-run | tail -5 && scripts/kb tag-gen --auto && sqlite3 data/gpu_knowledge.db "SELECT COUNT(*) FROM findings WHERE gpu_generation IS NOT NULL;" && sqlite3 data/gpu_knowledge.db "SELECT gpu_generation, COUNT(*) FROM findings WHERE gpu_generation IS NOT NULL GROUP BY gpu_generation;" && echo "Backfill done"`
  - **Commit**: `feat(kb): backfill gpu_generation from existing tags`
  - _Requirements: FR-14, AC-3.2_
  - _Design: Backfill Strategy, Section 5_

- [ ] 4.3 Run freshness dashboard and identify contradictions
  - **Do**:
    1. Run `kb freshness` to see temporal health
    2. Check contradiction count (expected ~7 known contradictions)
    3. For each contradiction pair, use `kb detail <id>` on both findings
    4. Resolve with `kb supersede <old_id> <new_id> "<reason>"` where appropriate
    5. Re-run `kb freshness` to verify contradictions resolved
  - **Files**: `/Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev/data/gpu_knowledge.db`
  - **Done when**: `kb freshness` shows 0 unresolved contradictions (or documents why remaining are not true contradictions)
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && scripts/kb freshness`
  - **Commit**: `feat(kb): resolve temporal contradictions via supersession`
  - _Requirements: AC-4.4_
  - _Design: Backfill Strategy, Section 5.5_

- [ ] V4 [VERIFY] Full local CI: all tests pass (194 existing + ~54 new)
  - **Do**: Run complete BATS test suite, verify total count is ~248
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/ --recursive 2>&1 | tail -5`
  - **Done when**: All tests pass, 0 failures
  - **Commit**: `chore(kb): pass full local CI` (only if fixes needed)
  - _Requirements: NFR-3, NFR-4_

- [ ] 4.4 Create PR and verify CI
  - **Do**:
    1. Verify current branch is a feature branch: `git branch --show-current`
    2. If on default branch, STOP and alert user
    3. Stage all changed files in plugin root
    4. Push branch: `git push -u origin <branch-name>`
    5. Create PR using gh CLI with summary of changes
  - **Verify**: `gh pr checks --watch` (wait for CI) or `gh pr checks` (poll)
  - **Done when**: All CI checks green, PR ready for review
  - **Commit**: None (PR creation, not code change)

## Phase 5: PR Lifecycle

- [ ] 5.1 Monitor CI and fix failures
  - **Do**:
    1. Check CI status: `gh pr checks`
    2. If failures, read failure details and fix locally
    3. Push fixes and re-verify
  - **Verify**: `gh pr checks` shows all passing
  - **Done when**: CI pipeline green

- [ ] 5.2 Address review comments
  - **Do**:
    1. Check for review comments: `gh pr view --comments`
    2. Address each comment with code changes
    3. Push fixes and verify CI
  - **Verify**: `gh pr checks` still passing after fixes
  - **Done when**: All review comments addressed

- [ ] V5 [VERIFY] AC checklist: all 22 acceptance criteria verified
  - **Do**: Programmatically verify each acceptance criteria:
    1. AC-1.1: `kb search "register allocation" --gen m4` returns results
    2. AC-1.2: Search --gen m5 excludes m1-m4 specific benchmarks
    3. AC-1.3: NULL gpu_generation findings included in --gen searches
    4. AC-1.4: `--gen universal` returns universal + NULL
    5. AC-2.1: `kb supersede 14 565` (or equivalent valid IDs) works
    6. AC-2.2: `kb detail <superseded_id>` shows "SUPERSEDED by"
    7. AC-2.3: `kb detail <replacing_id>` shows "Supersedes"
    8. AC-2.4: Superseded findings not in default search
    9. AC-2.5: `--include-superseded` includes them
    10. AC-2.6: Self-supersession rejected
    11. AC-3.1: `kb tag-gen --auto --dry-run` previews
    12. AC-3.2: Auto maps "m4" in tags correctly
    13. AC-3.3: Multi-gen tagged with newest
    14. AC-3.4: No-gen findings remain NULL
    15. AC-4.1-4.4: `kb freshness` shows all 4 sections
    16. AC-5.1: investigation-agent.md has gen tagging requirement
    17. AC-5.2: investigation-agent.md has supersession check
    18. AC-5.3: `kb verify` reports temporal warnings
    19. NFR-1: Search --gen under 1s
    20. NFR-2: Row count unchanged
    21. NFR-3: 194 existing tests pass
    22. NFR-4: 54+ new tests
  - **Verify**: `cd /Users/patrickkavanagh/.claude/plugins/cache/gpu-forge-local/gpu-forge/1.0.0-dev && bats tests/ --recursive 2>&1 | tail -3 && scripts/kb freshness --json | python3 -c "import sys,json;d=json.load(sys.stdin);print('Generations:',d['generation_counts']);print('Status:',d['temporal_status'])" && grep -q "Supersession check" agents/investigation-agent.md && grep -q "Generation tagging" agents/investigation-agent.md && echo "AC CHECK COMPLETE"`
  - **Done when**: All acceptance criteria confirmed via automated checks
  - **Commit**: None

## Notes

- **POC shortcuts taken**: Phase 1 skips tag-gen, freshness, verify Check 6, help text, and agent prompts. These are added in Phase 2.
- **Production TODOs deferred to future specs**:
  - FTS-based contradiction detection (pairwise BM25 scoring)
  - Claim-text generation inference (beyond tags-only backfill)
  - date_published backfill from re-fetching sources
  - Agent-automated supersession (agents flag, human confirms)
- **Key file**: `scripts/kb` receives ~350 lines of additions (from 474 to ~824 lines)
- **Test count**: 44 unit + 4 golden + 4 integration + 2 performance = 54 new tests
- **Token budget**: ~105 (investigation) + ~90 (retriever) + ~35 (advisor) = ~230 tokens total across 3 agent prompts (slightly over NFR-5's ~200 target; accepted per TECH.md analysis)
- **Backward compatibility**: All existing `kb search` calls without flags produce identical results except for the new `gen` column appended to output. BATS tests use `--partial` matching and are unaffected.
- **Self-supersession guard**: `old_id != new_id` check added per QA-2, not in original PM spec
- **Graceful degradation**: Every temporal command checks `pragma_table_info` before executing; search falls back to original query on pre-migration DB
