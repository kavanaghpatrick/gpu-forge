---
spec: gpu-forge-harness-wiring
phase: requirements
created: 2026-02-20
---

# Requirements: gpu-forge Harness Wiring

## Goal

Close 8 workflow disconnects in the gpu-forge Claude Code plugin so every feature connects end-to-end, with integration tests proving each workflow works for external developers installing the plugin.

## User Decisions

| Question | Response |
|----------|----------|
| Problem | Plugin features exist but don't connect end-to-end -- workflows break at handoff points |
| Constraints | Must pass all 196 existing BATS tests -- zero regressions, only additive changes |
| Success criteria | All 8 audit gaps closed AND new e2e workflow integration tests |
| Primary users | Open-source community -- external developers install and use gpu-forge |
| Priority tradeoffs | Code quality and maintainability -- clean implementation, easy to extend |

## User Stories

### US-1: Investigation Lifecycle Cleanup
**As a** gpu-forge developer
**I want** stale investigations automatically marked as failed when agents terminate unexpectedly
**So that** the `kb investigations` report is accurate and doesn't show 98 phantom "running" sessions

**Acceptance Criteria:**
- [ ] AC-1.1: `kb cleanup` subcommand exists and marks `running` investigations older than threshold as `status='failed'`
- [ ] AC-1.2: `kb cleanup --dry-run` previews affected investigations without modifying them
- [ ] AC-1.3: Default threshold is 1 hour; `kb cleanup --hours N` overrides it
- [ ] AC-1.4: Cleaned investigations get `completed_at=datetime('now')` and `summary='Auto-closed: agent did not complete'`
- [ ] AC-1.5: Running `kb cleanup` on the current database closes all 98 orphaned investigations
- [ ] AC-1.6: `kb verify` reports stale investigations as a WARNING line

### US-2: SubagentStop Hook for Investigation Agents
**As a** gpu-forge developer
**I want** a SubagentStop hook that fires when investigation-agent terminates
**So that** future investigations are cleaned up automatically, regardless of how the agent exits

**Acceptance Criteria:**
- [ ] AC-2.1: `hooks.json` has a `SubagentStop` entry with matcher `investigation-agent`
- [ ] AC-2.2: Hook script calls `kb cleanup --hours 0` (close any running investigation for this session)
- [ ] AC-2.3: Hook script always exits 0 (informational, never blocks)
- [ ] AC-2.4: Hook script completes in <1 second
- [ ] AC-2.5: Existing 196 BATS tests still pass after hook addition

### US-3: Inline Citation Enforcement
**As a** gpu-forge user searching the KB
**I want** academic paper findings to always have citation records
**So that** I can trace findings back to their original source

**Acceptance Criteria:**
- [ ] AC-3.1: `investigation-agent.md` Phase 3 inserts citation immediately after each `source_type='academic_paper'` finding
- [ ] AC-3.2: Citations are no longer a separate Phase 4 -- they are inline with Phase 3
- [ ] AC-3.3: `kb verify` reports count of academic findings missing citations
- [ ] AC-3.4: New investigations produce >80% citation coverage on academic findings (vs current 41.4%)

### US-4: Local Path Convention for URL-less Findings
**As a** gpu-forge contributor running local experiments
**I want** benchmark/empirical_test findings to use local file paths as source_url
**So that** every finding has a traceable source location

**Acceptance Criteria:**
- [ ] AC-4.1: `investigation-agent.md` instructs: for benchmark/empirical_test, set `source_url` to relative path (e.g., `experiments/exp16_8bit.rs`)
- [ ] AC-4.2: `kb backfill-urls` subcommand lists findings with NULL/empty source_url grouped by source_type
- [ ] AC-4.3: `kb verify` distinguishes "missing URL on web source" (ERROR) from "missing URL on benchmark/empirical_test" (INFO)
- [ ] AC-4.4: No NOT NULL constraint added to source_url column -- schema unchanged

### US-5: Expanded Post-Edit Validator
**As a** developer editing Metal shaders in Claude Code
**I want** the post-edit hook to catch common MSL mistakes beyond missing kernel qualifiers
**So that** I get instant feedback on address space misuse, barrier omissions, and atomic contention

**Acceptance Criteria:**
- [ ] AC-5.1: Hook checks for `device const` that should be `constant` (address space misuse)
- [ ] AC-5.2: Hook checks for threadgroup sizes that are not multiples of 32
- [ ] AC-5.3: Hook checks for `atomic_fetch_*` inside loops (contention warning)
- [ ] AC-5.4: Hook checks for missing `using namespace metal;`
- [ ] AC-5.5: Hook checks for buffer index gaps (`[[buffer(0)]]` + `[[buffer(2)]]` with no `[[buffer(1)]]`)
- [ ] AC-5.6: Hook checks for missing `threadgroup_barrier` near threadgroup memory writes in loops
- [ ] AC-5.7: All checks are regex/grep-based -- no LLM calls, no KB queries
- [ ] AC-5.8: Hook completes in <200ms for a 500-line .metal file
- [ ] AC-5.9: Hook still always exits 0 (warnings only, never blocks)
- [ ] AC-5.10: Hook fires on both `Edit` and `Write` tool calls for .metal files

### US-6: Review-to-KB Feedback Loop
**As a** gpu-forge user running `/gpu-forge:review`
**I want** newly discovered anti-patterns to be stored in the KB automatically
**So that** review insights accumulate over time instead of being lost

**Acceptance Criteria:**
- [ ] AC-6.1: `review.md` has a Step 5 "Store New Findings" that calls `kb add` via Bash
- [ ] AC-6.2: New findings are tagged `source_type='empirical_test'`, `confidence='medium'`, includes tag `review-discovered`
- [ ] AC-6.3: Before adding, review searches KB for existing similar finding (dedup check)
- [ ] AC-6.4: Step 5 only stores genuinely new patterns -- not every review observation

### US-7: Architecture Advisor Slash Command
**As a** gpu-forge user
**I want** a `/gpu-forge:advise` command that invokes the architecture-advisor agent
**So that** I can get architectural recommendations through the standard command interface

**Acceptance Criteria:**
- [ ] AC-7.1: `commands/advise.md` exists with frontmatter: `name: advise`, `context: fork`, `agent: architecture-advisor`
- [ ] AC-7.2: Command body lists valid skill areas and example invocations (matching investigate.md pattern)
- [ ] AC-7.3: `commands.bats` updated from "all 6 command files exist" to "all 7 command files exist" including `advise`
- [ ] AC-7.4: All loop-based tests in `commands.bats` include `advise` in the command list
- [ ] AC-7.5: `advise.md` uses `disable-model-invocation: true` (same as investigate.md)

### US-8: Scaffold Wired to JSON Configs
**As a** developer scaffolding a new GPU compute project
**I want** the scaffold wizard to read parameters, file mappings, and knowledge hints from JSON configs
**So that** template configs are the single source of truth instead of being ignored

**Acceptance Criteria:**
- [ ] AC-8.1: `scaffold.md` Step 3 reads the appropriate JSON config based on pattern + language selection
- [ ] AC-8.2: Mapping table in scaffold.md maps wizard choices to JSON config files (5 existing configs)
- [ ] AC-8.3: JSON `parameters` field populates template substitution defaults
- [ ] AC-8.4: JSON `files` array determines which templates to read and output paths
- [ ] AC-8.5: JSON `knowledge_applied` field generates suggested `/gpu-forge:ask` queries after scaffolding
- [ ] AC-8.6: Unmapped pattern+language combinations fall back to `metal-compute-blank.json` as default
- [ ] AC-8.7: Existing template structure tests (`templates.bats`) still pass

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria | Gap |
|----|-------------|----------|---------------------|-----|
| FR-1 | `kb cleanup` subcommand: mark stale `running` investigations as `failed` | P1 | AC-1.1 through AC-1.5 | 8 |
| FR-2 | `kb cleanup --dry-run` mode for preview | P1 | AC-1.2 | 8 |
| FR-3 | `kb cleanup --hours N` configurable threshold (default 1) | P2 | AC-1.3 | 8 |
| FR-4 | `SubagentStop` hook in hooks.json for `investigation-agent` | P1 | AC-2.1 through AC-2.5 | 1 |
| FR-5 | SubagentStop cleanup script calls `kb cleanup --hours 0` | P1 | AC-2.2 | 1 |
| FR-6 | Inline citation insertion in investigation-agent Phase 3 | P2 | AC-3.1, AC-3.2 | 2 |
| FR-7 | `kb verify` check: academic findings without citations | P2 | AC-3.3 | 2 |
| FR-8 | Agent protocol: use local paths for benchmark/empirical_test source_url | P2 | AC-4.1 | 3 |
| FR-9 | `kb backfill-urls` subcommand listing URL-less findings | P3 | AC-4.2 | 3 |
| FR-10 | `kb verify` distinguish web-source vs local-experiment missing URLs | P2 | AC-4.3 | 3 |
| FR-11 | Post-edit hook: 6 new regex checks (address space, TG size, atomics, namespace, buffer gaps, barriers) | P1 | AC-5.1 through AC-5.6 | 4 |
| FR-12 | Post-edit hook fires on `Write` tool in addition to `Edit` | P2 | AC-5.10 | 4 |
| FR-13 | Review Step 5: store new findings via `kb add` | P2 | AC-6.1 through AC-6.4 | 5 |
| FR-14 | Create `commands/advise.md` command file | P1 | AC-7.1, AC-7.2, AC-7.5 | 6 |
| FR-15 | Update `commands.bats` for 7 commands (add `advise`) | P1 | AC-7.3, AC-7.4 | 6 |
| FR-16 | Scaffold Step 3 reads JSON configs based on pattern+language mapping | P1 | AC-8.1 through AC-8.4 | 7 |
| FR-17 | Scaffold fallback to `metal-compute-blank.json` for unmapped combinations | P2 | AC-8.6 | 7 |
| FR-18 | Scaffold suggests KB queries from JSON `knowledge_applied` | P3 | AC-8.5 | 7 |
| FR-19 | `kb verify` reports stale investigations warning | P2 | AC-1.6 | 8 |
| FR-20 | New integration test file `tests/integration/harness-wiring.bats` | P1 | All gaps tested e2e | All |

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Backwards compatibility | Existing BATS tests passing | 196/196 (zero regressions) |
| NFR-2 | Post-edit hook performance | Wall clock time on 500-line .metal file | <200ms |
| NFR-3 | SubagentStop hook performance | Wall clock time | <1 second |
| NFR-4 | `kb cleanup` data safety | Dry-run accuracy | 100% match between preview and actual |
| NFR-5 | Code quality | No hardcoded absolute paths in commands | grep `/Users/` returns 0 matches |
| NFR-6 | Maintainability | Each new check in post-edit hook is a separate function | Easy to add/remove individual checks |
| NFR-7 | Data quality | Citation coverage for new academic findings | >80% (up from 41.4%) |
| NFR-8 | Test coverage | New integration tests for wiring gaps | >= 8 tests (1 per gap minimum) |

## Out of Scope

- Schema changes to `data/schema.sql` (no NOT NULL constraints, no new tables)
- Creating new JSON configs for unmapped scaffold patterns (histogram-kernel, benchmark-harness)
- GPU-distributed skill content expansion
- Backfilling citations for the 302 existing academic findings without citations (separate effort)
- Backfilling source_url for the 95 existing URL-less findings (separate effort)
- LLM-powered analysis in post-edit hook (stays regex-only)
- Changing hook exit codes from 0 (hooks remain informational/non-blocking)
- Auto-running `kb cleanup` on SessionStart (manual invocation only for now)
- Review write-back requiring user confirmation (automatic for genuinely new patterns)

## Glossary

- **SubagentStop hook**: Claude Code hook event that fires when a forked subagent terminates (timeout, error, or normal exit)
- **Investigation**: A KB research session tracked in the `investigations` table with start/end times, status, and finding count
- **Orphaned investigation**: Investigation stuck in `status='running'` because the agent terminated without calling `log-end`
- **PostToolUse hook**: Claude Code hook event that fires after a tool (Edit, Write, etc.) executes; informational only, cannot block
- **hookSpecificOutput**: JSON payload returned by a hook script that Claude sees as additional context
- **BATS**: Bash Automated Testing System -- the test framework used by gpu-forge (196 tests)
- **Scaffold**: The `/gpu-forge:scaffold` command that generates project skeleton files from templates
- **KB**: Knowledge Base -- the SQLite FTS5 database at `data/gpu_knowledge.db` containing 3596 findings

## Dependencies

| From | To | Relationship |
|------|----|-------------|
| FR-4 (SubagentStop hook) | FR-1 (kb cleanup) | Hook script calls `kb cleanup`; cleanup must exist first |
| FR-6 (inline citations) | FR-4 (SubagentStop hook) | Reliable agent completion increases citation coverage |
| FR-15 (commands.bats update) | FR-14 (advise.md) | Test update depends on command file existing |
| FR-20 (integration tests) | FR-1 through FR-19 | Tests validate all other requirements |

Implementation order (from research): Gap 8 -> Gap 1 -> Gap 6 -> Gap 4 -> Gap 3 -> Gap 5 -> Gap 2 -> Gap 7

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| `commands.bats` hardcodes command names in multiple tests | Test breakage if `advise` not added to ALL loops | High | FR-15 explicitly requires updating every loop in commands.bats, not just the count test |
| Scaffold mapping doesn't cover all wizard combinations | Users hit unmapped pattern+language and get no JSON config | Medium | FR-17 requires fallback to `metal-compute-blank.json` for any unmapped combination |
| Post-edit hook false positives | Developers ignore all warnings due to noise | Medium | Each check targets specific, unambiguous patterns; `device const` is always wrong in MSL |
| SubagentStop hook not supported in user's Claude Code version | Hook silently ignored, investigations still orphan | Low | `kb cleanup` (FR-1) provides manual fallback regardless of hook support |
| Review write-back creates duplicate findings | KB noise, inflated finding count | Low | FR-13 requires dedup search before adding; `confidence='medium'` prevents inflating verified count |
| `kb cleanup` accidentally closes a legitimately long-running investigation | Data loss (investigation marked failed while still running) | Low | FR-2 provides `--dry-run`; default 1-hour threshold is generous for forked agents (maxTurns ~100) |

## Unresolved Questions

1. Should `kb cleanup` auto-run on SessionStart (via hook), or stay manual-only? (Decision: manual for now, per Out of Scope)
2. For unmapped scaffold combinations: create new JSON configs later, or permanently fall back to blank template?
3. Should review write-back be fully automatic or require user confirmation? (Decision: automatic with dedup, per AC-6.3)
4. What is the right stale investigation threshold -- 1 hour fits forked agent behavior, but should it be configurable? (Decision: 1h default, `--hours N` override per FR-3)

## Success Criteria

1. `bats tests/ --recursive` passes with 196+ tests (zero regressions, new tests added)
2. `kb cleanup` closes all 98 orphaned investigations
3. `kb verify` reports data quality metrics for citations, URLs, and stale investigations
4. `/gpu-forge:advise` command is invocable and delegates to architecture-advisor
5. Post-edit hook catches 7 MSL issue categories (up from 1) in <200ms
6. Scaffold reads JSON configs for all 5 existing template types
7. New `harness-wiring.bats` has >= 8 integration tests covering all gaps

## Next Steps

1. Review and approve these requirements
2. Design phase: file-level implementation plan for each gap
3. Task breakdown: ordered task list following Gap 8 -> 1 -> 6 -> 4 -> 3 -> 5 -> 2 -> 7
4. Implementation: additive changes only, run full BATS suite after each gap
