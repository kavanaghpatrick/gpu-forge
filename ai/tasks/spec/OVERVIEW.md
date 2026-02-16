---
id: spec.OVERVIEW
module: spec
status: complete
version: 1
origin: spec-workflow
tags: [gpu-forge, temporal-relevance, kb-pipeline]
---

# KB Temporal Relevance Pipeline -- Specification Overview

## Executive Summary

The gpu-forge knowledge base (1,555 findings across 11 GPU domain skills) suffers from temporal relevance pollution: stale M1/M2-era benchmark findings contaminate agent search results alongside current M4/M5 data, with zero mechanism to distinguish generation applicability. This pipeline adds three columns to the findings table (`gpu_generation`, `temporal_status`, `superseded_by`), five new CLI commands (`supersede`, `unsupersede`, `tag-gen`, `freshness`, `migrate-temporal`), generation-aware search filtering (`--gen`, `--include-superseded`), surgical agent prompt updates (<200 tokens total across 3 prompts), and a backfill of 203 already-tagged findings. The result: agents get generation-appropriate search results, contradictions are tracked and resolved via supersession chains, and KB temporal health is continuously monitored. All 194 existing BATS tests pass unchanged; 54 new tests cover the full feature surface.

---

## Product Summary (PM)

### Problem
- FTS5 BM25 ranking has zero temporal signal -- M1-era findings rank equally with M4/M5 data
- 7 confirmed cross-generation contradictions exist with no linking mechanism
- 82% of findings are generation-agnostic (Metal API, MSL syntax, algorithms) and do not decay
- Only ~30 findings are genuinely stale benchmarks, but they pollute critical agent queries

### Key Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | `gpu_generation` column (m1-m5, universal, NULL) | High |
| FR-2 | `temporal_status` column (current, superseded) | High |
| FR-3 | `superseded_by` FK column | High |
| FR-4 | `kb search --gen <generation>` filter | High |
| FR-5 | Deprioritize superseded findings in search | High |
| FR-6 | `kb tag-gen` batch generation tagging | Medium |
| FR-7 | `kb supersede <old_id> <new_id>` | Medium |
| FR-8 | Investigation agent generation tagging | Medium |
| FR-9 | Investigation agent supersession checks | Medium |
| FR-10 | `kb verify` generation contradiction check | Medium |
| FR-11 | Knowledge retriever `--gen` auto-detection | Low |
| FR-12 | `kb freshness` temporal health dashboard | Low |
| FR-13 | Backfill 203 generation-tagged findings | Medium |

### Success Metrics
| Metric | Current | Target |
|--------|---------|--------|
| Generation-tagged findings | 13% (203/1555) | 80%+ of gen-specific |
| Stale data in top-5 results | ~30% | <5% |
| Unlinked contradictions | 7 | 0 |
| `date_published` coverage | 8% | 30%+ |

---

## UX Summary (UX)

### CLI Command Ergonomics
- **`kb search --gen <gen> --include-superseded`**: Additive flags, backward compatible. `-g` short flag. Invalid gen values produce actionable error messages.
- **`kb supersede <old> <new> [reason]`**: Positional args (git-style). No confirm for first supersession; `--force` for override. Reversible via `kb unsupersede`.
- **`kb tag-gen <id> <gen>` / `kb tag-gen --auto [--dry-run]`**: Single and batch modes. Auto-inference from tags field with preview.
- **`kb freshness [--skill] [--json]`**: Dashboard with generation distribution, temporal status, date coverage, contradiction list.
- **Gen column always visible** in search output for passive discovery.

### Agent Interaction
- **Investigation agent**: 3 surgical prompt additions (~105 tokens) for generation tagging, supersession check, and resolution
- **Knowledge retriever**: Generation detection heuristic from query text + generation-aware result formatting (~90 tokens)
- **Architecture advisor**: Cross-generation consistency check when citing findings (~35 tokens)

### Progressive Disclosure
- Layer 0: Existing commands unchanged
- Layer 1: Gen column + verify warnings surface passively
- Layer 2: `kb freshness` for active exploration
- Layer 3: `kb supersede/tag-gen/unsupersede` for curation
- Layer 4: Agent prompts maintain temporal quality automatically

---

## Architecture Summary (TECH)

### Schema Changes
- 3 new columns on `findings` table: `gpu_generation TEXT`, `temporal_status TEXT DEFAULT 'current'`, `superseded_by INTEGER`
- CHECK constraints enforce valid values; NULL permitted for backward compatibility
- 3 new indexes for query performance
- FTS5 virtual table NOT modified -- generation filtering via post-FTS JOIN WHERE clauses

### Migration Strategy
- Manual trigger: `kb migrate-temporal` (idempotent, backup-before-alter)
- Graceful degradation: `pragma_table_info` check, fallback to original queries on pre-migration DB
- SQLite 3.51.1 supports both ADD COLUMN with CHECK and DROP COLUMN for rollback

### Implementation Order
| Phase | Scope | Duration |
|-------|-------|----------|
| Phase 1 | Schema + Migration | Day 1 |
| Phase 2 | CLI Commands (search, supersede, unsupersede, tag-gen, freshness, detail, verify, add, help) | Days 2-3 |
| Phase 3 | Agent Prompt Updates | Day 4 |
| Phase 4 | Backfill + Validation | Day 5 |

### Key Design Decisions
1. Post-FTS filtering (no FTS5 schema changes)
2. NULL-inclusive `--gen` filtering (87% of KB stays visible during backfill)
3. Two-status model only (`current`, `superseded`)
4. Exact-match contradiction detection (FTS tier deferred)
5. Case-insensitive substring match for tag auto-inference

---

## Quality Summary (QA)

### Test Strategy
- **54 new tests** across 3 new files + 1 modified (exceeds NFR-4 target of 15+ by 3.6x)
- **194 existing tests** must pass unchanged (foundational invariant)
- **Test pyramid**: 37 unit + 8 integration + 4 golden-temporal + 5 regression/performance

### Test Coverage
| Category | File | Count |
|----------|------|-------|
| Migration + Schema | `tests/unit/temporal.bats` | 12 |
| Search Filtering | `tests/unit/temporal.bats` | 12 |
| Supersession | `tests/unit/temporal.bats` | 10 |
| Tag-Gen | `tests/unit/temporal.bats` | 6 |
| Freshness/Verify/Detail/Add/Help | `tests/unit/temporal.bats` | 7 |
| Agent Prompts | `tests/unit/temporal.bats` | 2 |
| Golden Temporal Queries | `tests/golden-temporal.bats` | 4 |
| Integration Workflows | `tests/integration/temporal-workflows.bats` | 4 |
| Performance Regression | `tests/performance/benchmarks.bats` | 2 |

### Backward Compatibility
- All existing BATS tests use `--partial` matching -- new `gen` column in output does not break assertions
- `kb add` new 11th parameter is optional -- existing 9-10 arg calls unchanged
- All new columns have defaults -- existing agent INSERTs work without modification
- FTS5 index unchanged -- golden queries pass

---

## Consolidated Q&A Decisions

| # | Domain | Question | Decision | Impact |
|---|--------|----------|----------|--------|
| PM-1 | Scope | NULL vs universal semantics | NULL = unclassified; explicit `universal` for confirmed gen-agnostic | Eliminates ambiguity; NULL included in filtered searches for backward compat |
| PM-2 | Scope | Multi-gen findings tagging | Tag with newest generation mentioned | Simplifies tagging logic; no `multi-gen` category |
| PM-3 | Scope | Supersession vs historical | Just `current` and `superseded` (no `historical`) | Reduced cognitive overhead and filter logic |
| PM-4 | Scope | Enforcement strictness | WARNING initially, promote to ERROR after backfill | Non-blocking initially; upgrade after system proven |
| PM-5 | Scope | Backfill scope | Tags-only (203 findings) now, claim-text scanning later | Phase 1 targets existing tagged findings; text inference deferred |
| UX-1 | UX | Gen column visibility | Always show gen column in search output | Passive discovery for developers; generation context for agents |
| UX-2 | UX | Supersession confirmation | No confirm for first; `--force` for override | Minimal friction; reversible via `kb unsupersede` |
| UX-3 | UX | Contradiction matching | Two-tier: exact only for Phase 1, FTS deferred | High precision first; FTS tier as follow-up |
| UX-4 | UX | NULL in --gen filter | Include NULL findings in --gen filtered results | 87% of KB stays visible during backfill ramp |
| TECH-1 | Tech | Migration trigger | Manual command (`kb migrate-temporal`) | Explicit, safe, idempotent |
| TECH-2 | Tech | Graceful degradation | Yes, check column existence, fallback | Modified commands work on pre- and post-migration DBs |
| TECH-3 | Tech | Tag pattern matching | Case-insensitive substring match on m1-m5 | 99%+ accuracy; simple bash implementation |
| TECH-4 | Tech | Contradiction detection scope | Exact match only for Phase 1 | Zero false positives; catches 7 known contradictions |
| QA-1 | Quality | Test setup strategy | Per-test migration in setup() | Complete isolation; matches existing convention |
| QA-2 | Quality | Self-supersession guard | Yes, add validation + test | One-line check; cheap guard against common mistake |
| QA-3 | Quality | Golden test DB | Production DB copy with `kb tag-gen --auto` in setup | Tests real backfill pipeline on actual data |

---

## Module Roadmap

| Priority | Module | Description | Key Deliverables | Dependencies |
|----------|--------|-------------|-----------------|--------------|
| 0 | **devops** | Schema migration, DB backup, migration script | `kb migrate-temporal` command, backup/rollback procedure, idempotent wrapper | None |
| 1 | **schema** | ALTER TABLE statements, indexes, schema.sql updates | 3 new columns, 3 indexes, CHECK constraints, `data/schema.sql` update | devops |
| 2 | **cli-search** | Generation-filtered search | `--gen`, `--include-superseded` flags, gen column display, post-FTS filtering SQL | schema |
| 3 | **cli-commands** | Temporal management commands | `kb supersede`, `unsupersede`, `tag-gen`, `freshness` commands | schema |
| 4 | **cli-verify** | Verify and add command updates | `kb verify` Check 6, `kb add` position 11, help text update | schema, cli-search |
| 5 | **agent-prompts** | Agent prompt temporal awareness | Investigation agent, knowledge retriever, architecture advisor prompt updates | cli-search, cli-commands |
| 6 | **backfill** | Auto-tag existing findings, resolve contradictions | Auto-tag 203 findings, resolve 7 contradictions | cli-commands, schema |
| 7 | **testing** | Comprehensive BATS test suite | `temporal.bats` (42 tests), `golden-temporal.bats` (4 tests), integration tests, performance tests | All CLI modules |
| 999999 | **integration** | End-to-end validation | Backward compat verification, golden query suite, full BATS regression, agent workflow validation | All modules |

---

## Key Architecture Decisions

1. **Post-FTS filtering**: FTS5 narrows to <100 rows; WHERE on indexed columns adds <1ms. No FTS5 schema changes.
2. **NULL-inclusive filtering**: `--gen m4` includes NULL findings (87% of KB) for backward compatibility during backfill ramp.
3. **Two-status model**: Only `current` and `superseded`. No `historical` status.
4. **Idempotent migration**: Column-existence checks make migration safe to run multiple times.
5. **Additive schema**: All new columns have sensible defaults; existing agent INSERTs unchanged.
6. **Manual migration trigger**: `kb migrate-temporal` -- explicit is better than implicit for schema changes.
7. **Exact-match contradiction detection**: High precision, zero false positives for Phase 1.
8. **Substring tag matching**: Simple, 99%+ accurate generation inference from tags field.

---

## Sources

All spec files with full references:
- **PM.md**: 13 FRs, 5 NFRs, 5 user stories, 4 risks, competitive analysis, success metrics
- **UX.md**: 9 CLI command specifications, agent interaction design, developer workflows, output formatting, progressive disclosure
- **TECH.md**: Schema migration, CLI implementation (350 LOC), agent prompt changes, backfill strategy, testing plan (41 tests), 5-day implementation order
- **QA.md**: 54 tests across 4 files, backward compatibility plan, golden query strategy, regression safety net, acceptance criteria matrix
