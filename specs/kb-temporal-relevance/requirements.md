# Requirements: KB Temporal Relevance Pipeline

## Goal

Eliminate stale benchmark pollution from gpu-forge agent search results by adding temporal awareness (generation tagging, supersession tracking, generation-filtered search) to the KB schema, CLI, and investigation agent pipeline. Targets ~30 genuinely stale findings among 1,555 total, with 82% generation-agnostic findings untouched.

## User Stories

### US-1: Generation-Filtered Search
**As an** architecture-advisor agent
**I want to** search findings filtered by GPU generation
**So that** my hardware recommendations use data from the correct Apple Silicon generation

**Acceptance Criteria:**
- [ ] AC-1.1: `kb search "register allocation" --gen m4` returns only m4 and universal findings
- [ ] AC-1.2: `kb search "TFLOPS" --gen m5` excludes m1/m2/m3/m4-specific benchmarks
- [ ] AC-1.3: Findings with NULL gpu_generation included in all `--gen` filtered searches (backward compatible)
- [ ] AC-1.4: `--gen universal` returns only universal-tagged and NULL findings

### US-2: Supersession Tracking
**As an** investigation agent
**I want to** mark old findings as superseded when I discover newer data
**So that** the KB maintains a clean evolution chain and stale data is deprioritized

**Acceptance Criteria:**
- [ ] AC-2.1: `kb supersede 14 565` marks finding #14 as superseded by #565
- [ ] AC-2.2: `kb detail 14` shows "SUPERSEDED by #565" in output
- [ ] AC-2.3: `kb detail 565` shows "Supersedes #14" in output
- [ ] AC-2.4: Superseded findings appear below current findings in search results
- [ ] AC-2.5: `kb search "TFLOPS" --include-superseded` returns both current and superseded
- [ ] AC-2.6: `kb supersede 1 1` rejected (self-supersession guard)

### US-3: Automatic Generation Inference
**As a** human developer maintaining the KB
**I want to** batch-tag findings with generation metadata inferred from existing tags
**So that** the 203 already-tagged findings get proper gpu_generation values without manual review

**Acceptance Criteria:**
- [ ] AC-3.1: `kb tag-gen --auto --dry-run` shows preview without modifying DB
- [ ] AC-3.2: `kb tag-gen --auto` correctly maps "m4" in tags to gpu_generation='m4'
- [ ] AC-3.3: Multi-generation tags (e.g., "m1,m2") tagged with newest generation mentioned
- [ ] AC-3.4: Findings with no generation indicators remain NULL (not force-tagged)

### US-4: Temporal Health Dashboard
**As a** human developer
**I want to** see a summary of the KB's temporal health
**So that** I know which areas need generation tagging or supersession review

**Acceptance Criteria:**
- [ ] AC-4.1: `kb freshness` shows findings count per gpu_generation
- [ ] AC-4.2: `kb freshness` shows findings count per temporal_status
- [ ] AC-4.3: `kb freshness` shows date_published coverage percentage
- [ ] AC-4.4: `kb freshness` lists unresolved contradictions (same topic+skill, different generation, no supersession link)

### US-5: Investigation Agent Temporal Discipline
**As the** investigation agent pipeline
**I want to** enforce generation tagging and supersession checks during investigation
**So that** new findings automatically maintain temporal quality

**Acceptance Criteria:**
- [ ] AC-5.1: Investigation agent prompt requires gpu_generation on benchmark/empirical_test findings
- [ ] AC-5.2: Before storing a generation-specific finding, agent searches for existing findings on same skill+topic that may be superseded
- [ ] AC-5.3: `kb verify` reports generation-specific findings without gpu_generation as WARNING
- [ ] AC-5.4: Quality check at Phase 5 includes temporal consistency check

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | Add `gpu_generation` column to findings table | High | Column accepts: `m1`, `m2`, `m3`, `m4`, `m5`, `universal`, NULL. CHECK constraint enforced. Existing data preserved via migration. |
| FR-2 | Add `temporal_status` column to findings table | High | Values: `current`, `superseded`. Default `current`. CHECK constraint enforced. No `historical` status. |
| FR-3 | Add `superseded_by` nullable FK column to findings | High | Points to replacing finding ID. `kb detail` shows supersession chain. |
| FR-4 | `kb search --gen <generation>` filter | High | Filters via post-FTS JOIN WHERE clause. Includes matching gen + `universal` + NULL. `-g` short flag. Invalid gen values produce actionable error. |
| FR-5 | Deprioritize superseded findings in search | High | Superseded findings excluded by default. `--include-superseded` / `-S` flag to override. Superseded findings sort after current when included. |
| FR-6 | `kb tag-gen` command for generation tagging | Medium | Single mode: `kb tag-gen <id> <gen>`. Auto mode: `kb tag-gen --auto [--dry-run] [--skill <name>]`. Clear mode: `kb tag-gen <id> --clear`. Auto mode uses case-insensitive substring match, picks newest gen. |
| FR-7 | `kb supersede <old_id> <new_id> [reason]` command | Medium | Validates both IDs exist. Sets temporal_status='superseded', superseded_by=new_id. Appends reason to notes. Self-supersession guard (old_id != new_id). `--force` required to override existing supersession. |
| FR-8 | `kb unsupersede <finding_id>` command | Medium | Restores to current, clears superseded_by. No-op warning on already-current finding. |
| FR-9 | Update investigation agent prompt for generation tagging | Medium | Phase 3: require gpu_generation on benchmark/empirical_test INSERTs. ~45 tokens added. |
| FR-10 | Update investigation agent prompt for supersession checks | Medium | Phase 2: search for existing findings on same skill+topic before storing. Phase 5: resolve supersessions. ~60 tokens added. |
| FR-11 | `kb verify` temporal quality check (Check 6) | Medium | Reports benchmark/empirical_test findings without gpu_generation as WARNING. Gated on migration (checks column existence). |
| FR-12 | Update knowledge retriever prompt for `--gen` auto-detection | Low | Detect generation from query text (e.g., "M4" -> `--gen m4`). Generation-aware result formatting. ~90 tokens added. |
| FR-13 | `kb freshness [--skill <name>] [--json]` dashboard | Low | Shows: generation distribution, temporal status counts, date_published coverage %, unresolved contradictions (exact topic+skill match). JSON mode for programmatic consumption. |
| FR-14 | Backfill gpu_generation for existing tagged findings | Medium | One-time `kb tag-gen --auto` on 203 generation-tagged findings. Tags-only backfill now; claim-text inference deferred. |
| FR-15 | `kb migrate-temporal` manual migration command | High | Idempotent. Creates backup before ALTER TABLE. Adds 3 columns + 3 indexes. Verifies column presence post-migration. |
| FR-16 | Modify `kb add` for optional gpu_generation parameter | Medium | 11th positional parameter. Validates against valid generations. Existing 9-10 arg calls unchanged. |
| FR-17 | Modify `kb detail` for supersession chain display | Medium | Shows "SUPERSEDED by #N" for superseded findings. Shows "Supersedes #N" for replacing findings. |
| FR-18 | Graceful degradation on pre-migration databases | Medium | `pragma_table_info` check on temporal columns. Fall back to original queries if not migrated. Modified `kb search` works on both pre- and post-migration DBs. |
| FR-19 | Update `data/schema.sql` for fresh DB creation | High | Add 3 columns to CREATE TABLE findings. Add 3 indexes. FTS5 virtual table and triggers unchanged. |
| FR-20 | Update help text with temporal commands | Low | Add "Temporal commands" section. Add "Generations" reference. Add search filter docs. |

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Search performance with --gen filter | Query latency | <100ms (same as unfiltered). Post-FTS WHERE on indexed columns adds <1ms. |
| NFR-2 | Migration safety | Data preservation | Zero data loss. All 1,555+ findings preserved. Backup created before ALTER TABLE. |
| NFR-3 | Backward compatibility | Existing test suite | All 194 existing BATS tests pass unchanged. |
| NFR-4 | Test coverage | New BATS tests | 54 new tests across 3 new files + 1 modified. |
| NFR-5 | Agent prompt token overhead | Total tokens added across 3 prompts | <200 tokens total (~105 investigation + ~90 retriever + ~35 advisor). |

## Design Constraints

1. **FTS5 virtual table unchanged** -- generation filtering via post-FTS JOIN WHERE, not FTS MATCH
2. **Existing `kb search` without `--gen` behaves identically** -- generation awareness is opt-in
3. **Investigation agents INSERT with named columns** -- new column defaults (NULL, 'current', NULL) ensure existing INSERTs work
4. **SQLite 3.51.1** -- supports ADD COLUMN with CHECK and DROP COLUMN for rollback
5. **No FTS5 custom ranking** -- post-filter approach preferred over BM25 modification

## Key Decisions (from Q&A Sessions)

| # | Decision | Impact |
|---|----------|--------|
| 1 | NULL = unclassified; explicit `universal` for gen-agnostic | NULL findings included in `--gen` searches for backward compat; `kb verify` flags NULL on benchmarks |
| 2 | Multi-gen findings: tag with newest generation | No `multi-gen` category; simplifies tagging logic |
| 3 | Two-status model: `current` and `superseded` only | No `historical` status; reduces cognitive overhead |
| 4 | `kb verify`: WARNING initially, ERROR after backfill | Non-blocking during ramp-up |
| 5 | Backfill: tags-only (203 findings) now, claim-text later | Phase 1 scope bounded; text inference deferred |
| 6 | Gen column always visible in search output | Passive discovery; generation context for agents |
| 7 | Manual migration via `kb migrate-temporal` | Explicit, safe, idempotent; no per-command latency |
| 8 | Graceful degradation on pre-migration DBs | `pragma_table_info` check; fallback to original queries |
| 9 | Per-test migration in BATS setup() | Complete isolation; matches existing kb-cli.bats convention |
| 10 | Self-supersession guard (reject old_id == new_id) | One-line check; cheap mistake prevention |
| 11 | Exact-match contradiction detection for Phase 1 | Zero false positives; FTS tier deferred |
| 12 | Case-insensitive substring match for tag auto-inference | 99%+ accuracy; simple bash implementation |

## Glossary

- **Generation**: Apple Silicon chip family (M1, M2, M3, M4, M5). Determines GPU core architecture, memory bandwidth, feature support.
- **Universal finding**: Knowledge that applies across all GPU generations (Metal API semantics, MSL syntax, algorithm theory).
- **Supersession**: Newer finding replaces older one on same topic. Old finding marked `superseded` and linked to replacement, never deleted.
- **Temporal status**: Lifecycle state: `current` (active) or `superseded` (replaced by newer finding).
- **Benchmark finding**: Finding of source_type `benchmark` or `empirical_test` -- most temporally sensitive types.
- **Contradiction**: Two findings in same skill+topic with incompatible claims about different GPU generations, no supersession link.
- **Backfill**: One-time inference of gpu_generation from existing tags field for already-tagged findings.
- **Post-FTS filtering**: Applying WHERE clauses on the JOIN result after FTS5 narrows candidates. Documented SQLite best practice.

## Out of Scope

- **Automated freshness decay scoring** -- overkill for ~30 stale findings; manual supersession sufficient
- **Finding deletion** -- stale findings superseded, never deleted; historical data has research value
- **FTS5 custom ranking function** -- BM25 temporal weighting adds complexity without proportional benefit
- **Agent-automated supersession** -- agents flag potential supersessions; human confirms
- **date_published backfill** -- inferring dates for 92% missing requires re-fetching sources; separate effort
- **Cross-KB deduplication** -- merging near-duplicate findings across generations is a separate quality task
- **NVIDIA/CUDA temporal tracking** -- KB is Apple Silicon focused; ~20 CUDA comparison findings excluded
- **FTS-based contradiction detection** -- pairwise BM25 scoring deferred; exact match only for Phase 1
- **`historical` temporal status** -- simplify to two states; old findings stay `current` with generation tag

## Dependencies

- **SQLite 3.51.1** -- supports ADD COLUMN with CHECK and DROP COLUMN for rollback
- **BATS + bats-assert** -- already installed via Homebrew; needed for 54 new tests
- **Agent prompt files** -- `investigation-agent.md`, `knowledge-retriever.md`, `architecture-advisor.md` must exist and follow current structure
- **7 known contradictions** -- must be identified and resolved via supersession as part of backfill validation
- **Production DB backup** -- migration creates automatic backup; manual backup recommended before first run

## Success Criteria

- Generation-tagged findings increase from 13% (203/1555) to 80%+ of gen-specific findings (`SELECT COUNT(*) FROM findings WHERE gpu_generation IS NOT NULL`)
- Stale data in top-5 search results for gen-specific queries drops from ~30% to <5% (golden query test suite)
- Unlinked contradictions drop from 7 to 0 (`kb freshness` contradiction report)
- All 194 existing BATS tests pass unchanged after migration
- 54 new BATS tests pass covering full temporal feature surface
- `kb search "TFLOPS" --gen m4` returns generation-appropriate results excluding stale benchmarks

## Unresolved Questions

- None -- all 15 questions resolved in Q&A sessions (PM Q1-Q4, TECH Q1-Q4, QA Q1-Q3, UX decisions). See Key Decisions table above.

## Next Steps

1. Design phase: translate FR/NFR into implementation plan with exact SQL, bash code, file changes
2. Task phase: break into ordered implementation tasks with verification commands
3. Implementation: Phase 1 (schema+migration), Phase 2 (CLI), Phase 3 (agent prompts), Phase 4 (backfill+validation)
