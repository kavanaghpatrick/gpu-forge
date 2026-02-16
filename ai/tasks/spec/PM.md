# Product Manager Analysis: KB Temporal Relevance Pipeline

**Date**: 2026-02-13
**Analyst**: PM Agent
**System**: gpu-forge (Claude Code plugin)
**Stack**: Bash scripts (kb CLI), SQLite FTS5, BATS tests (194), Markdown agent prompts
**Knowledge Base**: 1,555 findings across 11 GPU domain skills, 63 investigations
**Goal**: Eliminate stale benchmark pollution from agent-produced analyses by adding temporal awareness to the investigation and retrieval pipeline

---

## Problem Statement

### The Core Problem

When AI agents query the gpu-forge KB for current-gen GPU analysis, stale findings from earlier Apple Silicon generations (M1/M2-era) pollute results alongside M4/M5 data. The KB's FTS5 search ranks by text relevance (BM25) with zero temporal signal. An agent asking "GPU register allocation behavior" gets M1-era findings mixed with M4 findings that directly contradict them -- and has no mechanism to distinguish which is current.

### Quantified Scope

| Metric | Value |
|--------|-------|
| Total findings | 1,555 |
| Generation-agnostic (architecture/API/algorithm) | ~1,278 (82%) |
| Generation-tagged in tags field | ~203 (13%) |
| Have `date_published` populated | 126 (8%) |
| Benchmark-type findings | 41 |
| Confirmed cross-generation contradictions | 7 |
| Genuinely stale findings needing action | ~30 |

### Why This Matters to Agent Users

1. **Architecture advisor** cites Finding #14 (M1 TFLOPS/bandwidth specs) alongside Finding #565 (M5 neural accelerator benchmarks) without distinguishing generation applicability. Agent produces a recommendation mixing M1-era constraints with M5 capabilities.

2. **Knowledge retriever** returns 10 results for "FP16 throughput" -- 3 are M1/M2-era benchmarks, 4 are M4/M5-era, 3 are generation-agnostic theory. The agent cannot rank these by temporal relevance.

3. **Investigation agent** adds new M5 findings but has no mechanism to mark older M2 benchmarks on the same topic as superseded. The stale findings persist indefinitely.

### What Is NOT the Problem

- 82% of findings are generation-independent (Metal API, MSL syntax, algorithm theory). These do not decay.
- The DB is not broadly polluted. Only ~30 findings are genuinely stale benchmarks.
- Agent prompts already say "Prioritize 2024-2026" -- but this is unenforced guidance.

---

## User Impact

### Primary Users (AI Agents)

| Agent | Current Pain | Impact |
|-------|-------------|--------|
| architecture-advisor | Cites contradictory findings from different generations without flagging | Produces incorrect hardware constraint assumptions |
| knowledge-retriever | Returns stale benchmarks ranked equally with current data | User gets mixed-generation answers |
| investigation-agent | No mechanism to link new findings to older ones they supersede | KB accumulates contradictions over time |

### Secondary User (Human Developer)

- Must manually review agent outputs for generation accuracy
- Cannot trust agent recommendations about hardware-specific behavior without cross-checking
- KB quality degrades with each investigation as new findings accumulate without superseding old ones

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Generation-tagged findings | 13% (203/1555) | 80%+ of gen-specific findings tagged | `SELECT COUNT(*) FROM findings WHERE gpu_generation IS NOT NULL` |
| Search results with stale data in top-5 | ~30% for gen-specific queries | <5% | Golden query test suite |
| Confirmed contradictions without supersession links | 7 | 0 | `kb verify` reports zero unlinked contradictions |
| `date_published` coverage | 8% (126/1555) | 30%+ | SQL count on non-null date_published |
| Agent confidence in gen-specific answers | Unquantified | Retriever returns generation-appropriate results for 95% of gen-specific queries | Golden query tests |

---

## Requirements

### Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | Add `gpu_generation` column to findings table | High | Column accepts: `m1`, `m2`, `m3`, `m4`, `m5`, `universal`, NULL. Existing data preserved via migration. |
| FR-2 | Add `temporal_status` column to findings table | High | Values: `current`, `superseded`, `historical`. Default `current`. Superseded findings link to replacement via `superseded_by` FK. |
| FR-3 | Add `superseded_by` nullable FK column to findings | High | When set, points to the finding ID that replaces this one. `kb detail` shows supersession chain. |
| FR-4 | Modify `kb search` to accept `--gen <generation>` filter | High | `kb search "register allocation" --gen m4` excludes non-m4 findings unless tagged `universal`. |
| FR-5 | Modify `kb search` to deprioritize `superseded` findings | High | Superseded findings appear after current findings in results. `--include-superseded` flag to override. |
| FR-6 | Add `kb tag-gen` command for batch generation tagging | Medium | `kb tag-gen <finding_id> <generation>` sets gpu_generation. `kb tag-gen --auto` infers from tags/claim text. |
| FR-7 | Add `kb supersede <old_id> <new_id>` command | Medium | Sets old finding's temporal_status to `superseded`, sets superseded_by to new_id. Validates both IDs exist. |
| FR-8 | Update investigation agent prompt to require generation tagging | Medium | Agent prompt includes generation tagging in Phase 3 (Store Findings). Agent must set gpu_generation on every benchmark/empirical_test finding. |
| FR-9 | Update investigation agent prompt to check for supersession | Medium | Phase 2 includes: before storing a finding, search for existing findings on same topic+skill with different generation. If found, create supersession link. |
| FR-10 | Add `kb verify` check for generation contradictions | Medium | Verify detects findings in same skill+topic with overlapping claims but different generations and no supersession link. Reports as WARNING. |
| FR-11 | Update knowledge retriever prompt to pass generation context | Low | When query mentions specific hardware (M4, M5), retriever auto-adds `--gen` filter. |
| FR-12 | Add `kb freshness` command showing temporal health | Low | Reports: findings by generation, findings by temporal_status, date_published coverage, contradictions without supersession. |
| FR-13 | Backfill gpu_generation for existing tagged findings | Medium | One-time migration script infers gpu_generation from tags field for existing 203 generation-tagged findings. |

### Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Search performance | Query latency with --gen filter | <100ms (same as current unfiltered) |
| NFR-2 | Migration safety | Zero data loss during schema migration | All 1,555 findings preserved, spot-check 10 |
| NFR-3 | Backward compatibility | Existing kb CLI commands unchanged | All 194 existing BATS tests pass |
| NFR-4 | Test coverage | New BATS tests for temporal features | 15+ new tests covering FR-1 through FR-13 |
| NFR-5 | Agent prompt changes | No increase in agent token overhead | Generation tagging adds <50 tokens to prompts |

---

## User Stories

### US-1: Generation-Filtered Search
**As an** architecture-advisor agent
**I want to** search findings filtered by GPU generation
**So that** my hardware recommendations use data from the correct Apple Silicon generation

**Acceptance Criteria:**
- [ ] AC-1.1: `kb search "register allocation" --gen m4` returns only m4 and universal findings
- [ ] AC-1.2: `kb search "TFLOPS" --gen m5` excludes m1/m2/m3/m4-specific benchmarks
- [ ] AC-1.3: Findings without gpu_generation set are included in all generation-filtered searches (backward compatible)
- [ ] AC-1.4: `--gen universal` returns only generation-independent findings

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

### US-3: Automatic Generation Inference
**As a** human developer maintaining the KB
**I want to** batch-tag findings with generation metadata inferred from existing tags/claims
**So that** the 203 already-tagged findings get proper gpu_generation values without manual review

**Acceptance Criteria:**
- [ ] AC-3.1: `kb tag-gen --auto --dry-run` shows what would be tagged without modifying DB
- [ ] AC-3.2: `kb tag-gen --auto` correctly maps "m4" in tags to gpu_generation='m4'
- [ ] AC-3.3: Findings with multiple generation tags (e.g., "m1,m2") get tagged as the newest generation mentioned
- [ ] AC-3.4: Findings with no generation indicators remain NULL (not force-tagged)

### US-4: Temporal Health Dashboard
**As a** human developer
**I want to** see a summary of the KB's temporal health
**So that** I know which areas need generation tagging or supersession review

**Acceptance Criteria:**
- [ ] AC-4.1: `kb freshness` shows findings count per gpu_generation
- [ ] AC-4.2: `kb freshness` shows findings count per temporal_status
- [ ] AC-4.3: `kb freshness` shows date_published coverage percentage
- [ ] AC-4.4: `kb freshness` lists unresolved contradictions (same topic, different generation, no supersession link)

### US-5: Investigation Agent Temporal Discipline
**As the** investigation agent pipeline
**I want to** enforce generation tagging and supersession checks during investigation
**So that** new findings automatically maintain temporal quality

**Acceptance Criteria:**
- [ ] AC-5.1: Investigation agent prompt requires gpu_generation on benchmark/empirical_test findings
- [ ] AC-5.2: Before storing a generation-specific finding, agent searches for existing findings on same skill+topic that may be superseded
- [ ] AC-5.3: `kb verify` reports generation-specific findings without gpu_generation as WARNING
- [ ] AC-5.4: Quality check at Phase 5 includes temporal consistency check

---

## Design Constraints

### Must Preserve

1. **All 194 existing BATS tests pass unchanged** -- schema migration must be additive
2. **FTS5 index continues to work** -- generation filtering is applied post-FTS or via JOIN, not by modifying FTS content
3. **Existing `kb search` without `--gen` behaves identically** -- generation awareness is opt-in
4. **Investigation agent can still INSERT directly via sqlite3** -- schema additions must have sensible defaults

### SQLite FTS5 Limitation

FTS5 does not support column-level filtering natively. The `--gen` filter must be applied as a WHERE clause on the joined findings table, not within the FTS MATCH expression. This means:

```sql
-- Correct approach: filter after FTS match
SELECT f.* FROM findings_fts fts
JOIN findings f ON f.id = fts.rowid
WHERE findings_fts MATCH 'query'
  AND (f.gpu_generation = 'm4' OR f.gpu_generation = 'universal' OR f.gpu_generation IS NULL)
  AND f.temporal_status != 'superseded'
ORDER BY bm25(findings_fts, 10.0, 5.0, 2.0, 1.0);
```

This approach has no performance penalty since the FTS narrows results first, then the WHERE filters the small result set.

---

## Glossary

- **Generation**: Apple Silicon chip family (M1, M2, M3, M4, M5). Determines GPU core architecture, memory bandwidth, and feature support.
- **Universal finding**: Knowledge that applies across all GPU generations (e.g., Metal API semantics, MSL syntax, algorithm theory).
- **Supersession**: When a newer finding replaces an older one on the same topic. The old finding is not deleted but marked `superseded` and linked to its replacement.
- **Temporal status**: Lifecycle state of a finding: `current` (active), `superseded` (replaced by newer finding), `historical` (kept for reference but not returned by default).
- **Benchmark finding**: A finding of source_type `benchmark` or `empirical_test` -- these are the most temporally sensitive types.
- **Contradiction**: Two findings in the same skill+topic that make incompatible claims about the same metric but apply to different GPU generations without a supersession link.
- **Generation-agnostic**: Findings about architecture concepts, API semantics, or algorithms that do not change between GPU generations.

---

## Out of Scope

- **Automated freshness decay scoring** -- Industry practice (time-based relevance decay) is overkill for ~30 stale findings. Manual supersession is sufficient at this scale.
- **Finding deletion** -- Stale findings are superseded, never deleted. Historical data has research value.
- **FTS5 custom ranking function** -- Modifying BM25 weights to incorporate temporal signals adds complexity without proportional benefit. Post-filter approach is simpler and equally effective.
- **Agent-automated supersession** -- Agents flag potential supersessions; human confirms. Fully automated supersession risks incorrect links.
- **date_published backfill** -- Inferring publication dates for the 92% of findings without them requires re-fetching sources. Separate effort.
- **Cross-KB deduplication** -- Merging near-duplicate findings across generations is a separate quality task.
- **NVIDIA/CUDA temporal tracking** -- KB is Apple Silicon focused. The ~20 CUDA comparison findings do not need generation tracking.

---

## Dependencies

- **SQLite schema migration**: Must run `ALTER TABLE` on production DB. Needs backup/rollback plan.
- **BATS test framework**: New tests require BATS + bats-assert (already installed via Homebrew).
- **Agent prompt updates**: Changes to investigation-agent.md and knowledge-retriever.md require re-testing agent workflows.
- **Existing 7 contradictions**: Must be resolved (supersession links created) as part of initial backfill to validate the system works.

---

## Risk Assessment

### High Risk

**R1: Schema migration breaks existing agent workflows**
Investigation agents INSERT directly via sqlite3 with explicit column lists. Adding new columns with defaults ensures existing INSERTs work, but any agent that uses `SELECT *` may break if column order assumptions exist.
*Mitigation*: All new columns have defaults (gpu_generation=NULL, temporal_status='current', superseded_by=NULL). Agents use named columns in INSERT, not positional. Run all 194 BATS tests post-migration.

### Medium Risk

**R2: Auto-tagging inference accuracy**
Inferring gpu_generation from tags like "m4,performance" is straightforward. But findings tagged "m1,m2,m3,m4" (comparison findings) need special handling -- they are not M4-specific.
*Mitigation*: `--dry-run` flag for review before committing. Conservative rules: only tag when exactly one generation appears in tags. Multi-gen findings stay NULL or get tagged `universal`.

**R3: Agent adoption of new workflow**
Investigation agent prompt changes are guidance, not enforcement. Agents may skip generation tagging on non-benchmark findings.
*Mitigation*: `kb verify` catches missing generation tags on benchmark/empirical_test findings as WARNINGs. Human reviews verify output periodically.

### Low Risk

**R4: Performance impact of generation-filtered search**
Adding a WHERE clause to FTS results adds negligible overhead (filtering a small result set).
*Mitigation*: Benchmark before/after. FTS narrows to <100 rows; WHERE on indexed column is <1ms.

**R5: Supersession chain complexity**
Long supersession chains (A -> B -> C -> D) could make `kb detail` output confusing.
*Mitigation*: In practice, most findings will have at most one supersession. Show only immediate predecessor/successor in detail view.

---

## Implementation Approach (Recommended)

### Phase 1: Schema + CLI (Lowest risk, highest value)
1. Schema migration: add 3 columns to findings table
2. `kb search --gen` filter
3. `kb supersede` command
4. `kb tag-gen` with `--auto` and `--dry-run`
5. `kb freshness` dashboard
6. BATS tests for all new commands
7. Backfill 203 already-tagged findings

### Phase 2: Agent Prompts
8. Update investigation-agent.md Phase 3 (generation tagging)
9. Update investigation-agent.md Phase 2 (supersession check)
10. Update knowledge-retriever.md (auto --gen from query)
11. Update `kb verify` with temporal checks

### Phase 3: Validation
12. Resolve 7 known contradictions using new supersession system
13. Run golden query tests confirming stale data is deprioritized
14. Run full BATS suite confirming backward compatibility

---

## Unresolved Questions

1. **Multi-generation findings**: How should findings comparing generations (e.g., "M4 has 2x bandwidth vs M2") be tagged? Options: (a) tag with newest gen mentioned, (b) new value `multi-gen`, (c) tag NULL and rely on text search. Recommendation: tag with newest gen.

2. **Universal vs NULL**: Should generation-agnostic findings be explicitly tagged `universal`, or should NULL mean "applies to all"? NULL-means-universal is simpler but ambiguous (could also mean "not yet tagged"). Recommendation: use `universal` explicitly for confirmed gen-agnostic findings; NULL means "not yet classified."

3. **Supersession granularity**: If Finding #14 covers M1/M1 Pro/M1 Max/M1 Ultra specs and Finding #565 covers M5, should the supersession link be created even though they cover different generations? The M1 finding is not "wrong" -- it is just about older hardware. Recommendation: create supersession links only when findings contradict on the same metric. Non-contradicting old-gen findings stay `current` with their generation tag.

4. **Verify strictness**: Should `kb verify` treat missing gpu_generation on benchmark findings as ERROR (blocks investigation close) or WARNING (advisory)? Recommendation: WARNING initially, upgrade to ERROR after backfill proves the system works.

---

## Next Steps

1. User answers unresolved questions above
2. Technical design for schema migration and CLI changes
3. Implement Phase 1 (schema + CLI + tests)
4. Backfill existing findings with generation tags
5. Update agent prompts (Phase 2)
6. Validate with golden query tests (Phase 3)

---

## Sources

- [SQLite FTS5 Extension Documentation](https://sqlite.org/fts5.html)
- [Content Freshness: Best Practices for Automating Updates and Deletions](https://cobbai.com/blog/knowledge-freshness-automation)
- [Data Quality Monitoring at Scale with Agentic AI (Databricks)](https://www.databricks.com/blog/data-quality-monitoring-scale-agentic-ai)
- [Agentic Data Quality Monitoring: Integrity and Performance (Acceldata)](https://www.acceldata.io/blog/how-agentic-ai-data-quality-monitoring-reduces-downtime)
- [Knowledge Base Article Lifecycle Process (ServiceNow)](https://www.servicenow.com/community/sysadmin-forum/knowledge-base-article-lifecycle-process/m-p/2585542)
- [Document Lifecycle with Version Control (ClickHelp)](https://clickhelp.com/clickhelp-technical-writing-blog/document-lifecycle-with-version-control-from-creation-to-archiving/)
- [Create and Manage Article Versions (Microsoft)](https://learn.microsoft.com/en-us/dynamics365/customer-service/use/ka-versions)
- [RAG Temporal Relevance Challenges (arXiv survey)](https://arxiv.org/html/2506.00054v1)

---

## Questions & Answers

### Q1: NULL vs universal semantics
**Answer**: NULL = unclassified. Explicit `universal` for confirmed gen-agnostic findings.
**Impact**: Eliminates ambiguity permanently. Backfill must include a step to tag confirmed gen-agnostic findings as `universal`. NULL findings are treated as "included in all generation-filtered searches" for backward compatibility, but flagged by `kb verify` as needing classification.

### Q2: Multi-generation comparison findings
**Answer**: Tag with newest generation mentioned.
**Impact**: Comparison findings (e.g., "M4 has 2x bandwidth vs M2") get tagged `m4`. This is the context in which they're most useful. Simplifies tagging logic — no need for a `multi-gen` category.

### Q3: Supersession vs historical
**Answer**: Simplify to just `current` and `superseded`. No `historical` status.
**Impact**: `temporal_status` CHECK constraint is `('current', 'superseded')`. Reduces cognitive overhead and filter logic. Old findings that aren't contradicted stay `current` with their generation tag.

### Q4: Enforcement strictness
**Answer**: WARNING initially, promote to ERROR after backfill proves system works.
**Impact**: `kb verify` reports missing gpu_generation on benchmark/empirical_test findings as WARNING. After the initial backfill and one month of agent usage, this can be promoted to ERROR to prevent regression.

### Q5: Backfill scope
**Answer**: Tags-only backfill now. Claim-text scanning as separate follow-up.
**Impact**: Phase 1 backfill targets the 203 findings with generation tags. Claim-text inference is noisier and deferred to a follow-up task to push coverage from 13% toward 30%+.
