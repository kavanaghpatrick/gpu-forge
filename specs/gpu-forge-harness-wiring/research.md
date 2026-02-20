---
spec: gpu-forge-harness-wiring
phase: research
created: 2026-02-20
---

# Research: gpu-forge Harness Wiring

## Executive Summary

The gpu-forge Claude Code plugin has 8 workflow disconnects where individual components function correctly but fail to connect end-to-end. Root causes are: agent protocol gaps (investigations never close), missing enforcement (citations, URLs), too-basic hook (1 of 10+ possible checks), missing commands/subcommands (`advise`, `cleanup`), and scaffold command that ignores its own JSON configs. All 8 fixes are within the plugin's own code -- no external dependencies. 196 BATS tests currently pass (all green) and must continue to.

## Gap Analysis

### Gap 1: Orphaned Investigations (98 stuck in 'running')

**Current State**: 98 of 175 investigations (56%) have `status='running'` and will never complete. The oldest dates to 2026-02-12. 84 findings are linked to these orphaned sessions. The `investigation-agent.md` defines a 5-phase protocol where Phase 5 calls `kb log-end <id> <queries> <findings> "<summary>"`. The `scripts/kb` `log-end` subcommand works correctly -- it updates `completed_at`, `status='completed'`, `queries_run`, `findings_added`, and `summary`.

**Root Cause**: The investigation-agent runs in `context: fork` (isolated subagent) via `investigate.md`. Forked agents have a hard turn limit (`maxTurns: 100`). When the agent hits its turn limit, times out, or encounters an error during research, Phase 5 never executes. There is NO error handler, no `finally` block, no `SubagentStop` hook, and no cleanup mechanism. The `log-start` creates the investigation record with `status='running'`, but nothing guarantees `log-end` runs.

**Key Files**:
- `/Users/patrickkavanagh/gpu_kernel/agents/investigation-agent.md` (Phase 5 protocol)
- `/Users/patrickkavanagh/gpu_kernel/commands/investigate.md` (fork delegation)
- `/Users/patrickkavanagh/gpu_kernel/scripts/kb` (log-start/log-end subcommands)
- `/Users/patrickkavanagh/gpu_kernel/data/schema.sql` (investigations table)

**Proposed Fix Direction**: Two-pronged approach:
1. Add a `SubagentStop` hook in `hooks.json` that runs a cleanup script. The hook fires when any subagent finishes (including timeout/error). The script checks for `status='running'` investigations and marks them `status='failed'` with `completed_at=datetime('now')`. Matcher: `investigation-agent`.
2. Add a `kb cleanup` subcommand to `scripts/kb` that bulk-closes stale investigations (e.g., running for >24h). This handles the 98 existing orphans.

**Complexity**: Medium
**Risk**: Low. SubagentStop hook is informational (cannot block). The cleanup subcommand is additive. Neither changes existing test-covered behavior.

---

### Gap 2: Citation Coverage (6.6% apparent, actually 41% of academic papers)

**Current State**:
- 515 findings have `source_type='academic_paper'`
- 236 distinct findings have citations (239 total citation records)
- 213 of 515 academic papers have citations = **41.4% coverage**
- 302 academic paper findings have NO citation record
- Only 2 citations have `author_source='unverified'` (rest are properly sourced)

The 6.6% figure (236/3596) conflates all findings with academic citations. The real metric is: 41.4% of academic papers have citations, but 58.6% (302 findings) do not.

**Root Cause**: The investigation-agent.md Phase 4 (Citations) is documented but NOT enforced. It says "For academic papers, add a citation record" but there's no check that this happened. The agent frequently runs out of turns during Phase 2-3 (research + store) and never reaches Phase 4. The protocol is sequential -- citations come AFTER all findings are stored -- so if the agent runs long on research, citations are the first casualty.

**Key Files**:
- `/Users/patrickkavanagh/gpu_kernel/agents/investigation-agent.md` (Phase 4)
- `/Users/patrickkavanagh/gpu_kernel/scripts/kb` (cite subcommand, verify checks)
- `/Users/patrickkavanagh/gpu_kernel/data/schema.sql` (citations table)

**Proposed Fix Direction**:
1. Add a `kb verify` check: "academic_paper findings without citations" -- makes the gap visible.
2. Restructure investigation-agent Phase 3 to insert citations inline with each academic finding rather than as a separate phase. Citation immediately follows each `source_type='academic_paper'` INSERT.
3. Add a `kb backfill-citations` subcommand that lists academic findings missing citations for manual/automated followup.

**Complexity**: Medium
**Risk**: Low. Protocol change is in agent prompt (no code change). New verify check is additive. Backfill subcommand is read-only.

---

### Gap 3: Findings Missing Source URLs (95 findings)

**Current State**:
- 95 findings have `source_url IS NULL OR source_url=''`
- Breakdown by `source_type`:
  - `benchmark`: 14 findings
  - `empirical_test`: 81 findings
- These are ALL from benchmark/empirical_test types -- NOT from web research
- The `source_url` column has no NOT NULL constraint in schema.sql

**Root Cause**: Benchmark and empirical_test findings are generated from local experiments (radix sort experiments, GPU exploit experiments, etc.) -- they have no external URL because the "source" is local code. The investigation-agent protocol says "Always include the source URL. If there is no URL, use the source description" but this guidance is frequently ignored for local experiments where there genuinely IS no URL.

**Key Files**:
- `/Users/patrickkavanagh/gpu_kernel/data/schema.sql` (findings table, no NOT NULL on source_url)
- `/Users/patrickkavanagh/gpu_kernel/agents/investigation-agent.md` (Phase 3 rules)
- `/Users/patrickkavanagh/gpu_kernel/scripts/kb` (verify check already warns about missing URLs)

**Proposed Fix Direction**:
1. Do NOT add a NOT NULL constraint -- benchmark/empirical_test legitimately lack URLs.
2. Instead, add a convention: for local experiments, set `source_url` to a relative path like `experiments/exp16_8bit.rs` or `metal-gpu-experiments/shaders/exp16_8bit.metal`.
3. Add a `kb backfill-urls` subcommand that identifies URL-less findings and prompts for local path fill-in.
4. Update the `kb verify` warning to distinguish "missing URL on web source" (real problem) vs "missing URL on empirical_test/benchmark" (expected, but should have local path).

**Complexity**: Low
**Risk**: Low. Purely data quality improvement. No schema changes needed.

---

### Gap 4: Post-Edit Hook Too Basic

**Current State**: The `post-edit-validator.sh` hook fires on every `Edit` tool call for `.metal` files. It performs exactly ONE check: whether a `void functionName(` line is missing the `kernel` qualifier. It outputs a JSON `hookSpecificOutput` with warnings. It always exits 0 (never blocks edits). The hook has a 10-second timeout.

The `review.md` command checks 20+ issues across 6 categories: threadgroup sizing (4 checks), memory access patterns (4 checks), SIMD utilization (4 checks), atomic operations (3 checks), kernel attributes (3 checks), address spaces (4 checks). The hook does 1 of these.

**Root Cause**: The hook was implemented as a minimal proof-of-concept. The review command is a full agent invocation (model: sonnet, reads KB) -- it's designed for on-demand deep analysis, not per-edit validation. The gap is that the hook could catch more issues with simple regex/grep checks without needing an LLM.

**Key Files**:
- `/Users/patrickkavanagh/gpu_kernel/hooks/scripts/post-edit-validator.sh` (current hook)
- `/Users/patrickkavanagh/gpu_kernel/hooks/hooks.json` (hook config, timeout: 10)
- `/Users/patrickkavanagh/gpu_kernel/commands/review.md` (full review checklist)

**Proposed Fix Direction**: Add fast, grep-based checks (must stay under 1s total, exit 0 always):
1. `device const` that should be `constant` (address space misuse)
2. Threadgroup size hardcoded to non-multiple-of-32
3. Missing `threadgroup_barrier` near threadgroup memory writes in loops
4. `atomic_fetch_*` inside inner loops (high contention warning)
5. Missing `using namespace metal;`
6. Buffer index gaps (e.g., `[[buffer(0)]]` and `[[buffer(2)]]` but no `[[buffer(1)]]`)

Each check: grep for pattern -> if found, append to WARNINGS array. Same output format. Hook also fires on `Write` tool for `.metal` files (currently only `Edit`).

**Complexity**: Low
**Risk**: Low. Hook always exits 0, so even false positives don't block edits. Just additional warnings.

---

### Gap 5: No Review -> KB Feedback Loop

**Current State**: The `review.md` command reads from the KB (search anti-patterns, query gpu-perf/msl-kernels/simd-wave) but NEVER writes back. When review discovers a new anti-pattern or performance insight not already in the KB, that knowledge is lost -- it appears in the review output and then disappears.

There is no mechanism for any command except `investigate` (via investigation-agent) and direct `kb add` to write findings to the KB.

**Root Cause**: The review command was designed as read-only (allowed-tools: `[Read, Bash, Grep]`). It has no Write tool and no instruction to store new discoveries. The `kb add` subcommand exists but review doesn't know about it.

**Key Files**:
- `/Users/patrickkavanagh/gpu_kernel/commands/review.md` (read-only KB access)
- `/Users/patrickkavanagh/gpu_kernel/scripts/kb` (add subcommand)

**Proposed Fix Direction**:
1. Add a "Step 5: Store New Findings" section to `review.md` that uses `Bash` (already allowed) to call `kb add` for any anti-pattern or performance finding discovered during review that doesn't match an existing KB finding.
2. New findings from review get tagged `source_type='empirical_test'`, `confidence='medium'`, `tags='review-discovered'`.
3. Add dedup check: before adding, search KB for similar claim to avoid duplicates.
4. Keep it optional/lightweight -- only store genuinely new patterns, not every review observation.

**Complexity**: Low
**Risk**: Low. The review command already has Bash tool access. Adding KB writes is a prompt change only. New findings get `medium` confidence so they don't inflate the verified count.

---

### Gap 6: Architecture Advisor Has No Command

**Current State**: The `architecture-advisor.md` agent exists with full capabilities (5 preloaded skills, KB querying, structured output format). But there is NO slash command to invoke it. The 6 existing commands are: `ask`, `investigate`, `knowledge`, `review`, `scaffold`, `template`. No `advise` command exists.

The `investigate` command delegates to `investigation-agent` via `context: fork` with `agent: investigation-agent`. This pattern is the exact blueprint needed.

**Root Cause**: The agent was created but the corresponding command file was never written. Simple omission.

**Key Files**:
- `/Users/patrickkavanagh/gpu_kernel/agents/architecture-advisor.md` (agent definition)
- `/Users/patrickkavanagh/gpu_kernel/commands/investigate.md` (pattern to follow)
- `/Users/patrickkavanagh/gpu_kernel/commands/` (missing advise.md)

**Proposed Fix Direction**: Create `commands/advise.md` following the `investigate.md` pattern:
```yaml
---
name: advise
description: "Get architectural recommendations for GPU compute designs, backed by knowledge database findings."
argument-hint: "<description-of-requirements>"
context: fork
agent: architecture-advisor
disable-model-invocation: true
---
```
Add body text explaining usage, valid skill areas, and example invocations.

**Complexity**: Low (copy pattern from investigate.md, adjust for architecture-advisor)
**Risk**: Low. Adding a new command file. Existing `commands.bats` test checks for 6 commands by name -- test must be updated to check for 7. The `all 6 command files exist` test will need to become `all 7 command files exist`.

---

### Gap 7: Scaffold Disconnected from Templates

**Current State**: The `scaffold.md` command defines a 9-step wizard that asks users 5 questions (name, chip, pattern, language, complexity), determines a template type, reads templates from the templates directory, and generates project files.

Five JSON config files exist in `templates/project/`: `reduction-kernel.json`, `matrix-multiply.json`, `scan-kernel.json`, `metal-compute-blank.json`, `mlx-custom-kernel.json`. Each JSON defines: `name`, `description`, `parameters` (with types, defaults, options), `files` (template -> output mappings), `knowledge_applied` (skills).

The scaffold command NEVER references these JSON configs. Step 3 says "Read scaffold specification files from the templates directory" and `ls ${CLAUDE_PLUGIN_ROOT}/templates/project/` but never uses the JSON data. The wizard hardcodes all parameter defaults, file mappings, and template selections in the markdown prose. The JSON configs are completely orphaned.

**Root Cause**: The JSON configs and the scaffold command were developed independently. The scaffold command was written as a self-contained wizard. The JSON configs were written as machine-readable specs but never wired into the wizard. The `templates.bats` tests verify the JSON files exist and are valid, and that their file refs point to existing templates -- but nothing tests that scaffold actually USES them.

**Key Files**:
- `/Users/patrickkavanagh/gpu_kernel/commands/scaffold.md` (wizard, ignores JSON)
- `/Users/patrickkavanagh/gpu_kernel/templates/project/*.json` (5 orphaned configs)
- `/Users/patrickkavanagh/gpu_kernel/tests/unit/templates.bats` (validates JSON structure)

**Proposed Fix Direction**:
1. Rewrite scaffold.md Step 3 to: read the appropriate JSON config based on compute pattern + language selection (e.g., pattern=reduction + language=swift-metal -> `reduction-kernel.json`).
2. Extract parameters from the JSON to populate template substitutions (use the `parameters` field's defaults and options).
3. Use the JSON `files` array to determine which templates to read and where to output them.
4. Use the JSON `knowledge_applied` field to suggest relevant `/gpu-forge:ask` queries after scaffolding.
5. Add a mapping table in scaffold.md: pattern+language -> JSON config file.

**Complexity**: Medium
**Risk**: Low-Medium. The scaffold command has `disable-model-invocation: true` so it's deterministic. Changes are to the prompt, not to executable code. But the mapping between wizard choices and JSON configs needs careful design -- not all combinations have a JSON config (e.g., `histogram` pattern has a metal template but no JSON config for `histogram-kernel.json`; `benchmark` complexity level has no JSON config).

---

### Gap 8: Investigation Tracking (Stale Sessions)

**Current State**: 98 investigations stuck in `status='running'`, dating from 2026-02-12 to 2026-02-20. The `scripts/kb` has `investigations` (list) and `log-end` (close) subcommands but NO cleanup/maintenance subcommand. There is no automated way to:
- Close stale investigations in bulk
- Mark timed-out investigations as `failed`
- Identify which running investigations have findings vs which are truly empty

The `kb verify` command checks for: inflated confidence, unverified citations, near-duplicates, suspicious categorization, missing URLs. It does NOT check for stale investigations.

**Root Cause**: Same as Gap 1 -- the protocol assumes `log-end` always runs, but forked agents can die without executing it. No maintenance tooling was built to handle the inevitable cleanup.

**Key Files**:
- `/Users/patrickkavanagh/gpu_kernel/scripts/kb` (missing cleanup subcommand)
- `/Users/patrickkavanagh/gpu_kernel/data/schema.sql` (investigations table)

**Proposed Fix Direction**:
1. Add `kb cleanup` subcommand: marks `running` investigations older than N hours as `status='failed'`, sets `completed_at=datetime('now')`, `summary='Auto-closed: agent did not complete'`. Default threshold: 1 hour.
2. Add `kb cleanup --dry-run` mode to preview what would be closed.
3. Add stale investigation check to `kb verify` output: "WARNING: N investigations running for >1 hour".
4. The 98 existing orphans: run `kb cleanup` once to clear them.

**Complexity**: Low
**Risk**: Low. New subcommand, additive only. `verify` change is a new warning (existing checks unchanged).

---

## Cross-Cutting Concerns

### Agent Protocol Reliability
Gaps 1 and 8 share the same root cause: forked agents can terminate without running cleanup code. The SubagentStop hook is the systematic fix -- it fires regardless of how the subagent ends. This also applies to any future forked agents.

### KB Data Quality
Gaps 2, 3, and 5 are all about KB data completeness. A unified approach: enhance `kb verify` to report ALL data quality metrics in one report (missing citations, missing URLs, stale investigations, orphaned findings). This becomes the single "health check" command.

### Command Coverage
Gaps 6 and 7 are both about commands that exist on paper but don't work in practice. Gap 6 is a missing file; Gap 7 is a disconnected workflow. Both are prompt-only changes.

### Hook Enhancement
Gap 4 is the only one touching executable code (bash script). Keep it simple: regex-based, fast, always exit 0.

### Test Impact
- Gap 6 requires updating `commands.bats` test (`all 6 command files exist` -> 7)
- Gap 8 requires new test for `kb cleanup` in `kb-cli.bats`
- All gaps need new integration tests in `workflows.bats`
- Current: 196 BATS tests, all passing

## Test Coverage Analysis

**Current Test Distribution** (196 tests):
| File | Count | Coverage |
|------|-------|----------|
| golden-queries.bats | 54 | FTS5 search quality |
| kb-cli.bats | 20 | KB subcommands |
| plugin-structure.bats | 19 | File existence, structure |
| skills.bats | 16 | Skill file validation |
| templates.bats | 16 | Template files, compilation |
| agents.bats | 12 | Agent file structure |
| commands.bats | 11 | Command file structure |
| benchmarks.bats | 11 | Performance baselines |
| db-integrity.bats | 9 | Schema constraints |
| hooks.bats | 9 | Hook config, scripts |
| fts5-search.bats | 6 | FTS5 search features |
| hooks-execution.bats | 5 | Hook execution |
| workflows.bats | 4 | End-to-end workflows |
| security.bats | 4 | SQL injection, concurrency |

**Missing Test Coverage**:
- No test for `kb log-start` -> `kb log-end` workflow
- No test for investigation lifecycle (start -> add findings -> close)
- No test that scaffold uses JSON configs
- No test for post-edit-validator catching specific patterns
- No e2e test for review -> KB write-back
- No test for advise command (doesn't exist yet)
- Only 4 workflow integration tests

## Dependencies Between Gaps

```
Gap 8 (cleanup subcommand) --enables--> Gap 1 (SubagentStop hook uses cleanup logic)
Gap 1 (SubagentStop hook)  --enables--> Gap 2 (investigation completes, citations run)
Gap 3 (URL backfill)       --independent
Gap 4 (hook expansion)     --independent
Gap 5 (review feedback)    --independent
Gap 6 (advise command)     --independent
Gap 7 (scaffold wiring)    --independent
```

Recommended implementation order:
1. Gap 8 (cleanup subcommand) -- foundation, fixes existing data
2. Gap 1 (SubagentStop hook) -- prevents future orphans
3. Gap 6 (advise command) -- simplest, one file
4. Gap 4 (hook expansion) -- one file, fast
5. Gap 3 (URL backfill) -- data quality
6. Gap 5 (review feedback) -- prompt change
7. Gap 2 (citation enforcement) -- protocol change
8. Gap 7 (scaffold wiring) -- most complex mapping

## Quality Commands

| Type | Command | Source |
|------|---------|--------|
| Test (all) | `bats tests/ --recursive` | BATS test framework |
| Test (unit) | `bats tests/unit/` | tests/unit/*.bats |
| Test (integration) | `bats tests/integration/` | tests/integration/*.bats |
| Test (golden) | `bats tests/golden-queries.bats` | tests/golden-queries.bats |
| Test (perf) | `bats tests/performance/benchmarks.bats` | tests/performance/ |
| KB Verify | `scripts/kb verify` | scripts/kb |
| KB Dedup | `scripts/kb dedup` | scripts/kb |
| Lint | Not found | N/A (no linter configured) |
| TypeCheck | Not found | N/A (bash/markdown plugin) |
| Build | Not found | N/A (no build step) |

**Local CI**: `bats tests/ --recursive && scripts/kb verify && scripts/kb dedup`

## Related Specs

| Spec | Relevance | Relationship | mayNeedUpdate |
|------|-----------|-------------|---------------|
| kb-temporal-relevance | Medium | Adds temporal scoring to KB -- new findings from review (Gap 5) would benefit from temporal metadata | false |

No other specs directly overlap with this plugin-internal wiring work.

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | High | All fixes are within the plugin's own files (markdown, bash, JSON). No external deps. |
| Effort Estimate | M | ~8 files to create/modify. Largest: scaffold rewrite (Gap 7), hook expansion (Gap 4). |
| Risk Level | Low | All changes are additive. Existing tests are structural (file existence, JSON validity). Zero schema changes. Hook always exits 0. |

## Recommendations for Requirements

1. **Implement Gap 8 first** -- the `kb cleanup` subcommand unblocks Gap 1 and fixes existing data immediately.
2. **Use SubagentStop hook** for Gap 1 -- it's the only reliable mechanism since forked agents can die without running cleanup code. The hook API is well-documented and supports the `investigation-agent` matcher.
3. **Do NOT add NOT NULL constraint on source_url** (Gap 3) -- benchmark/empirical_test findings legitimately lack URLs. Use convention (local path) instead.
4. **Keep post-edit hook fast** (Gap 4) -- regex checks only, no KB queries, no LLM calls. Target <200ms total.
5. **Scaffold JSON wiring (Gap 7) needs a mapping table** -- not all wizard combinations have JSON configs. Define fallback behavior for unmapped combinations.
6. **Add an e2e workflow test file** -- `tests/integration/harness-wiring.bats` that tests each gap's fix as an end-to-end workflow.
7. **Citation enforcement (Gap 2)** should be inline (with each finding INSERT) not a separate phase -- the current sequential approach is why citations get skipped.
8. **commands.bats hardcodes "all 6 command files"** -- must update to 7 when adding advise.md.

## Open Questions

1. Should `kb cleanup` auto-run at session start (via SessionStart hook), or only be manual?
2. For Gap 7: should the missing JSON configs (histogram-kernel, benchmark-harness) be created, or should scaffold fall back to pure template-based generation for unmapped patterns?
3. For Gap 5: should review write-back be automatic (always store new patterns) or user-confirmed (ask before storing)?
4. What's the right threshold for stale investigation cleanup -- 1 hour? 24 hours?

## Sources

### Internal Files Read
- `/Users/patrickkavanagh/gpu_kernel/agents/investigation-agent.md`
- `/Users/patrickkavanagh/gpu_kernel/agents/architecture-advisor.md`
- `/Users/patrickkavanagh/gpu_kernel/commands/review.md`
- `/Users/patrickkavanagh/gpu_kernel/commands/investigate.md`
- `/Users/patrickkavanagh/gpu_kernel/commands/scaffold.md`
- `/Users/patrickkavanagh/gpu_kernel/commands/knowledge.md`
- `/Users/patrickkavanagh/gpu_kernel/commands/ask.md`
- `/Users/patrickkavanagh/gpu_kernel/commands/template.md`
- `/Users/patrickkavanagh/gpu_kernel/hooks/hooks.json`
- `/Users/patrickkavanagh/gpu_kernel/hooks/scripts/post-edit-validator.sh`
- `/Users/patrickkavanagh/gpu_kernel/hooks/scripts/context-loader.sh`
- `/Users/patrickkavanagh/gpu_kernel/hooks/scripts/kb-wrapper.sh`
- `/Users/patrickkavanagh/gpu_kernel/scripts/kb`
- `/Users/patrickkavanagh/gpu_kernel/data/schema.sql`
- `/Users/patrickkavanagh/gpu_kernel/templates/project/*.json` (all 5)
- `/Users/patrickkavanagh/gpu_kernel/tests/unit/*.bats` (all 9)
- `/Users/patrickkavanagh/gpu_kernel/tests/integration/*.bats` (all 3)

### External Sources
- [Claude Code Hooks Reference](https://code.claude.com/docs/en/hooks) -- PostToolUse, SubagentStop hook specifications, matcher patterns, hookSpecificOutput format
- [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills) -- Command delegation, context: fork, agent field
- [Claude Code Slash Commands](https://platform.claude.com/docs/en/agent-sdk/slash-commands) -- Command file format, frontmatter fields

### Database Queries Run
- `SELECT COUNT(*) FROM investigations WHERE status='running'` -> 98
- `SELECT COUNT(*) FROM findings WHERE source_type='academic_paper'` -> 515
- `SELECT COUNT(DISTINCT finding_id) FROM citations` -> 236
- `SELECT COUNT(*) FROM findings WHERE source_url IS NULL OR source_url=''` -> 95
- `SELECT source_type, COUNT(*) FROM findings WHERE source_url IS NULL OR source_url='' GROUP BY source_type` -> benchmark:14, empirical_test:81
- `SELECT status, COUNT(*) FROM investigations GROUP BY status` -> completed:77, running:98
- `SELECT COUNT(*) FROM findings` -> 3596
- `SELECT COUNT(*) FROM citations` -> 239
- `SELECT COUNT(*) FROM investigations` -> 175
- `SELECT COUNT(*) FROM citations WHERE author_source='unverified' OR author_source IS NULL` -> 2
- Academic findings without citations: 302
- Findings linked to completed investigations: 2059
- Findings linked to running investigations: 84
- `bats tests/ --recursive --count` -> 196 (all passing)
