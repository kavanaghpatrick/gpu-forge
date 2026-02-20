# UX/Developer Experience Analysis: KB Temporal Relevance Pipeline

**Date**: 2026-02-13
**Analyst**: UX/DX Designer Agent
**System**: gpu-forge (Claude Code plugin)
**Stack**: Bash CLI (`kb` script), SQLite FTS5, Agent prompts (Markdown), BATS tests
**Scope**: CLI command ergonomics, agent prompt integration, developer workflows, and output formatting for the new temporal relevance features

---

## Design Principles

Before specifying commands, these principles guide every decision:

1. **Additive, not disruptive**: Every existing `kb` invocation must work identically. Temporal features are opt-in via new flags and commands.
2. **Progressive disclosure**: `kb search` returns clean results by default. Temporal metadata appears only when relevant (superseded findings, generation context). Developers discover temporal features naturally through output hints, not documentation mandates.
3. **Agents are primary users**: The CLI is consumed by AI agents more than humans. Command output must be parseable, unambiguous, and token-efficient. Favor structured single-line metadata over decorative formatting.
4. **One concept per flag**: `--gen` filters by generation. `--include-superseded` controls visibility. These never combine into compound flags.
5. **Fail informatively**: Every error includes the fix. Every warning includes the next action.

---

## 1. CLI Command Ergonomics

### 1.1 Modified Command: `kb search`

**Current syntax (preserved):**
```
kb search <query> [--limit N]
```

**New syntax (additive flags):**
```
kb search <query> [--limit N] [--gen <generation>] [--include-superseded]
```

**Flag design:**

| Flag | Short | Values | Default | Behavior |
|------|-------|--------|---------|----------|
| `--gen` | `-g` | `m1`, `m2`, `m3`, `m4`, `m5`, `universal` | _(none)_ | Include only findings matching this generation, plus `universal`, plus `NULL` (unclassified). Excludes findings tagged to other specific generations. |
| `--include-superseded` | `-S` | _(boolean flag)_ | off | Include `superseded` findings in results. Without this flag, superseded findings are excluded from results. |
| `--limit` | `-l` | integer | 10 | _(unchanged)_ |

**Why `-g` not `--generation`**: The flag will be typed thousands of times by agents. Two fewer characters per invocation. `-g` has no conflicting convention in the existing CLI.

**Why `-S` (uppercase) for superseded**: Lowercase `-s` is conventionally `--silent` or `--sort`. Capital `-S` signals an unusual/power-user option, which is appropriate since most queries should exclude superseded results.

**Interaction rules:**
- `kb search "TFLOPS"` -- returns current findings only (superseded excluded), all generations (backward compatible)
- `kb search "TFLOPS" --gen m4` -- returns current findings tagged `m4`, `universal`, or `NULL`
- `kb search "TFLOPS" --gen m4 --include-superseded` -- includes superseded findings too, marked visually
- `kb search "TFLOPS" --gen universal` -- returns only `universal`-tagged findings (confirmed gen-agnostic)
- Invalid generation value: `ERROR: Invalid generation 'mx'. Valid: m1, m2, m3, m4, m5, universal`

**SQL implementation pattern:**
```sql
-- When --gen m4 is specified:
AND (f.gpu_generation = 'm4' OR f.gpu_generation = 'universal' OR f.gpu_generation IS NULL)

-- When --include-superseded is NOT specified (default):
AND (f.temporal_status = 'current' OR f.temporal_status IS NULL)
```

### 1.2 New Command: `kb supersede`

**Syntax:**
```
kb supersede <old_id> <new_id> [reason]
```

**Examples:**
```bash
kb supersede 14 565                              # Mark #14 superseded by #565
kb supersede 14 565 "M5 benchmarks replace M1"   # With reason
```

**Behavior:**
1. Validate both IDs exist; exit with `ERROR: Finding #999 does not exist` if not
2. Validate old_id is not already superseded; if so: `WARNING: Finding #14 is already superseded by #301. Override? Use --force to confirm.`
3. Set `temporal_status = 'superseded'` and `superseded_by = <new_id>` on old finding
4. Append reason to old finding's `notes` field: `[Superseded: <reason>]`
5. Output: `Finding #14 marked superseded by #565`

**Why positional args, not flags**: This mirrors `git rebase <upstream> <branch>` -- the relationship direction is clear from position. `<old> <new>` reads naturally as "old is replaced by new."

**Safety:**
- No `--force` needed for first supersession (the common case)
- Reversible via: `kb unsupersede <id>` (restores `temporal_status = 'current'`, clears `superseded_by`)

### 1.3 New Command: `kb unsupersede`

**Syntax:**
```
kb unsupersede <finding_id>
```

**Behavior:**
1. Validate finding exists and is currently superseded
2. Set `temporal_status = 'current'`, `superseded_by = NULL`
3. Append to notes: `[Unsuperseded: restored to current]`
4. Output: `Finding #14 restored to current status`

**Why a separate command**: Undo operations should be explicit commands, not flags on the forward operation. This follows `git stash` / `git stash pop` convention.

### 1.4 New Command: `kb tag-gen`

**Syntax:**
```
kb tag-gen <finding_id> <generation>              # Tag single finding
kb tag-gen --auto [--dry-run] [--skill <name>]    # Auto-infer from tags field
```

**Single finding mode:**
```bash
kb tag-gen 14 m1           # Set gpu_generation = 'm1' on finding #14
kb tag-gen 14 universal    # Mark as generation-agnostic
kb tag-gen 14 --clear      # Reset to NULL (unclassified)
```

**Auto-inference mode:**
```bash
kb tag-gen --auto --dry-run                 # Preview all auto-tagging decisions
kb tag-gen --auto --dry-run --skill gpu-perf  # Preview for one skill only
kb tag-gen --auto                           # Execute auto-tagging
```

**Auto-inference rules (documented in help text):**
1. Scan `tags` field for generation keywords: `m1`, `m2`, `m3`, `m4`, `m5`
2. If exactly one generation found: tag with that generation
3. If multiple generations found: tag with the newest (per PM Q&A decision Q2)
4. If no generation found: leave as NULL (do not force-tag)
5. Never overwrite an existing non-NULL `gpu_generation` value

**Dry-run output format:**
```
=== Auto-Generation Tagging Preview ===

  Would tag:
    #14   m1       (tags: "m1,performance,architecture")
    #87   m4       (tags: "m4,bandwidth,benchmark")
    #203  m4       (tags: "m2,m4,comparison")  ← newest: m4

  Would skip (no generation in tags):
    542 findings

  Would skip (already tagged):
    12 findings

  Total: 203 would be tagged, 554 unchanged
  Run without --dry-run to apply.
```

### 1.5 New Command: `kb freshness`

**Syntax:**
```
kb freshness [--skill <name>] [--json]
```

**Default output (human-readable dashboard):**
```
=== KB Temporal Health ===

Generation Coverage:
  m1          12   (0.8%)
  m2          34   (2.2%)
  m3          28   (1.8%)
  m4          98   (6.3%)
  m5          31   (2.0%)
  universal    0   (0.0%)
  unclassified 1352 (86.9%)   ← NULL
  ---------------------------------
  total       1555

Temporal Status:
  current      1548  (99.6%)
  superseded      7  (0.4%)

Date Coverage:
  date_published set: 126/1555 (8.1%)

Potential Contradictions:
  3 finding pairs share skill+topic with different generations and no supersession link:
    #14 (m1, gpu-silicon/specs) <-> #565 (m5, gpu-silicon/specs)
    #88 (m2, gpu-perf/bandwidth) <-> #401 (m4, gpu-perf/bandwidth)
    #92 (m1, unified-memory/slc-behavior) <-> #499 (m4, unified-memory/slc-behavior)

  Use `kb supersede <old_id> <new_id>` to resolve.
```

**Why a dashboard, not raw counts**: The developer needs actionable context, not just numbers. The contradiction list is the highest-value section -- it directly tells you what to fix. The percentages contextualize whether tagging coverage is improving over time.

**`--json` output** for programmatic consumption:
```json
{
  "generation_counts": {"m1": 12, "m2": 34, "m3": 28, "m4": 98, "m5": 31, "universal": 0, "null": 1352},
  "temporal_status": {"current": 1548, "superseded": 7},
  "date_coverage": {"with_date": 126, "total": 1555, "percentage": 8.1},
  "contradictions": [
    {"finding_a": 14, "gen_a": "m1", "finding_b": 565, "gen_b": "m5", "skill": "gpu-silicon", "topic": "specs"}
  ]
}
```

### 1.6 Modified Command: `kb detail`

**Current behavior (preserved):** Shows finding fields and any citations.

**New behavior (additive):** Shows temporal metadata in a distinct section.

```
$ kb detail 14

=== Finding #14 ===
skill_name   gpu-silicon
topic        specs
claim        Apple M1 GPU provides 2.6 TFLOPS FP32 compute performance
evidence     ...
confidence   verified
source_type  apple_docs
source_url   https://...
tags         m1,performance,architecture
date_found   2025-04-12
gpu_generation  m1
temporal_status SUPERSEDED by #565
notes        [Superseded: M5 benchmarks replace M1 specs]

--- Supersession Chain ---
  #14 (m1, 2025-04-12) --superseded-by--> #565 (m5, 2025-11-03)

--- Citations ---
  (none)
```

For the new finding:
```
$ kb detail 565

=== Finding #565 ===
...
gpu_generation  m5
temporal_status current
notes        ...

--- Supersession Chain ---
  Supersedes #14 (m1, 2025-04-12)

--- Citations ---
  ...
```

**Design decision: show only immediate links.** Per PM risk assessment R5, long chains are unlikely. Showing only direct predecessor/successor keeps output clean. If chains grow, a future `kb chain <id>` command can walk the full graph.

### 1.7 Modified Command: `kb verify`

**New check added (Check 6):**
```
# Check 6: Generation tagging on benchmark/empirical findings
count=$(run_sql "$DB" "SELECT COUNT(*) FROM findings
  WHERE source_type IN ('benchmark','empirical_test')
    AND gpu_generation IS NULL;")
if [ "$count" -gt 0 ] 2>/dev/null; then
  echo "WARNING: $count benchmark/empirical findings have no gpu_generation tag:"
  ...
fi
```

**Output format:**
```
WARNING: 8 benchmark/empirical findings have no gpu_generation tag:
  #201  gpu-perf    "FP16 throughput on GPU reaches..."    benchmark
  #245  gpu-silicon "ALU utilization measured at..."        empirical_test
  ...
  Tag with: kb tag-gen <id> <generation>
```

**Enforcement escalation (documented in help):**
- Phase 1 (initial): WARNING severity. Does not block investigation close.
- Phase 2 (after backfill): Promote to ERROR. Investigation agents must tag benchmarks.
- The transition is a one-line change: `echo "WARNING:"` to `echo "ERROR:"`.

### 1.8 Modified Command: `kb add`

**New optional parameter:**
```
kb add <skill> <topic> <claim> [evidence] [url] [title] [type] [confidence] [tags] [gpu_generation]
```

Position 11 (index `${11}`) accepts the generation value. This is backward compatible -- existing calls without position 11 leave `gpu_generation` as NULL (the column default).

**Alternative considered:** A `--gen` flag on `kb add`. Rejected because `kb add` already uses positional args exclusively. Mixing positional and flag-based args in bash is fragile and inconsistent with the existing design.

### 1.9 Help Text Update

The help output (default/unknown command) gains a new section:

```
Temporal commands:
  supersede <old_id> <new_id> [reason]   Mark finding as replaced by newer one
  unsupersede <finding_id>               Restore superseded finding to current
  tag-gen <id> <gen>                     Set GPU generation on a finding
  tag-gen --auto [--dry-run]             Auto-infer generation from tags field
  freshness [--skill <name>]             Temporal health dashboard

Generations: m1, m2, m3, m4, m5, universal (NULL = unclassified)

Search filters:
  --gen <gen>            Filter by GPU generation (includes universal + NULL)
  --include-superseded   Include superseded findings in results
```

This section appears after "Quality commands" and before "Investigation tracking" -- positioned to be discoverable without disrupting the familiar layout.

---

## 2. Agent Interaction Design

### 2.1 Investigation Agent: Minimal Prompt Addition

The investigation agent prompt (`investigation-agent.md`) needs two surgical additions, not a rewrite. Token overhead target: <50 tokens.

**Addition 1: Phase 3 (Store Findings) -- Generation tagging requirement**

Insert after the "Rules for storing findings" section:

```markdown
**Generation tagging (required for benchmarks):**
- For `benchmark` or `empirical_test` findings, always set `gpu_generation` to the
  relevant Apple Silicon generation: m1, m2, m3, m4, m5, or universal.
- Add gpu_generation to your INSERT: `..., gpu_generation)` VALUES `..., '<gen>')`.
- If generation is unclear, leave NULL — `kb verify` will flag it for review.
```

**Addition 2: Phase 2 (Research) -- Supersession check**

Insert at the end of Phase 2:

```markdown
4. **Supersession check**: Before storing a generation-specific finding, check if the KB
   already has findings on the same skill+topic with a different generation:
   ```bash
   ${CLAUDE_PLUGIN_ROOT}/scripts/kb search "<topic keywords>" --limit 5
   ```
   If an older-generation finding covers the same metric, note it for supersession in Phase 5.
```

**Addition 3: Phase 5 (Quality Check) -- Resolve supersessions**

Insert after the `kb verify` / `kb dedup` step:

```markdown
3. **Resolve supersessions** if you identified older findings during Phase 2:
   ```bash
   ${CLAUDE_PLUGIN_ROOT}/scripts/kb supersede <old_id> <new_id> "<reason>"
   ```
```

**Total token overhead**: ~40 tokens in the system prompt. Well within the NFR-5 target of <50 tokens.

### 2.2 Knowledge Retriever Agent: Auto-Generation Detection

The knowledge-retriever agent (`knowledge-retriever.md`) needs a lightweight heuristic to detect generation context from user queries.

**Addition to "Parse keywords and domain hints" section:**

```markdown
5. **Detect generation context** from the query:
   - If query mentions "M4", "M4 Pro", "M4 Max": add `--gen m4` to search
   - If query mentions "M5": add `--gen m5` to search
   - If query mentions "M1"/"M2"/"M3": add the corresponding `--gen` flag
   - If query asks about "current" or "latest": add `--gen m4` (or m5 when available)
   - If no generation mentioned: do NOT add `--gen` (return all generations)
```

**Addition to "Result Formatting" section:**

```markdown
### Generation-Aware Format:
When --gen was used, note it in the response:
```
[verified] Finding #123 (gpu-silicon, m4): Apple M4 GPU has 10 cores...
                                    ^^^^
```
When a finding is from a different generation than requested, flag it:
```
[verified] Finding #14 (gpu-silicon, m1) ⚠ DIFFERENT GEN: M1 specs, not M4
```
```

**Why heuristic, not enforcement**: The retriever is a Haiku model with 10 maxTurns. It needs a simple rule, not a decision tree. False negatives (missing a gen hint) are harmless -- the search still works. False positives are also harmless -- the agent just gets more focused results.

### 2.3 Architecture Advisor Agent: Cross-Generation Awareness

The architecture-advisor agent (`architecture-advisor.md`) needs awareness that findings may span generations.

**Addition to "Cross-Reference Findings Across Skills" section:**

```markdown
5. **Check generation consistency**: When citing findings, verify they apply to the
   target hardware generation. Use `kb detail <id>` to check gpu_generation.
   Flag findings from earlier generations that may not apply to the target hardware.
```

This is a soft guideline, not a hard constraint. The architecture advisor (Sonnet model) is capable enough to exercise judgment about when generation differences matter.

---

## 3. Developer Workflows

### 3.1 Temporal Health Check Workflow

**When**: After an investigation session, before a batch of agent runs, or on a weekly cadence.

```bash
# Step 1: Get the dashboard
kb freshness

# Step 2: If contradictions exist, review each pair
kb detail 14    # Old finding
kb detail 565   # New finding
# Human judgment: are they truly contradictory? Is the old one obsolete?

# Step 3: Resolve contradictions
kb supersede 14 565 "M5 replaces M1 TFLOPS spec"

# Step 4: Check for untagged benchmarks
kb verify
# Follow up on any WARNING about missing gpu_generation

# Step 5: Verify resolution
kb freshness
# Contradiction count should be lower
```

### 3.2 Contradiction Resolution Workflow

**When**: `kb freshness` reports potential contradictions.

The key insight is that not all "contradictions" need supersession. The workflow has three outcomes:

```
                  kb freshness reports contradiction
                            |
                    kb detail <both ids>
                            |
              +---------+---+---------+
              |                       |
    Truly contradictory        Not contradictory
    (same metric, newer        (different metrics,
     replaces older)            or complementary)
              |                       |
    kb supersede old new       kb tag-gen <id> <gen>
    "reason"                   (tag correctly, no
                               supersession needed)
              |                       |
              +-------+-------+-------+
                      |
            kb freshness  (verify)
```

**Decision criteria for human review:**
1. Do both findings claim a specific numeric value for the same metric? -> Supersede the older one
2. Do they describe the same concept but for different hardware? -> Tag each with its generation, no supersession
3. Is one a comparison finding ("M4 has 2x bandwidth vs M2")? -> Tag with newest gen (m4), no supersession

### 3.3 Backfill Workflow (One-Time)

**When**: Phase 1 implementation is complete. Run once to backfill existing 203 tagged findings.

```bash
# Step 1: Preview what would be tagged
kb tag-gen --auto --dry-run

# Step 2: Review the preview. Check for anything surprising.
# Especially check multi-gen findings (tagged with newest gen per Q2 decision).

# Step 3: Execute
kb tag-gen --auto

# Step 4: Verify
kb freshness
# gpu_generation coverage should jump from ~0% to ~13%

# Step 5: Resolve the 7 known contradictions
kb supersede 14 565 "M5 replaces M1 specs"
# ... repeat for each pair

# Step 6: Final check
kb verify
kb freshness
```

### 3.4 Onboarding Experience

A developer encountering temporal features for the first time will discover them through one of these paths:

**Path 1: Search results hint**
When a search returns results and some findings have generation tags, the output now shows them:
```
id   skill         claim                              confidence  source_title      gen
---  -----------   --------------------------------   ----------  ---------------   ---
565  gpu-silicon   M5 GPU provides 12.8 TFLOPS FP32   verified   Apple Tech Note   m5
402  gpu-perf      FP32 throughput scales linearly...  high       Metal Best Prac   m4
```
The `gen` column in search output is a natural discovery mechanism. "What's this gen column?" leads to `kb --help` which documents temporal commands.

**Path 2: Verify output hint**
Running `kb verify` (already part of the investigation agent workflow) now produces temporal warnings:
```
WARNING: 8 benchmark/empirical findings have no gpu_generation tag
  Tag with: kb tag-gen <id> <generation>
```
The warning includes the exact command to fix it. Zero documentation required.

**Path 3: Help text**
The new "Temporal commands" section in `kb --help` is positioned prominently.

**Path 4: Freshness dashboard**
`kb freshness` is a self-contained onboarding experience. It shows the current state, explains what the numbers mean, and lists actionable items.

---

## 4. Output Formatting Specifications

### 4.1 Search Results: Generation Column

**Current output columns:**
```
id | skill | claim | confidence | source_title
```

**New output columns:**
```
id | skill | claim | confidence | source_title | gen | status
```

The `gen` column shows `gpu_generation` (or empty for NULL). The `status` column only appears when `--include-superseded` is used, to distinguish current vs superseded findings.

**Without `--include-superseded` (default):**
```
id    skill          claim                                 confidence  source_title          gen
----  -------------  ------------------------------------  ----------  --------------------  ----
565   gpu-silicon    M5 GPU provides 12.8 TFLOPS FP32     verified    Apple Tech Note       m5
402   gpu-perf       FP32 throughput scales linearly       high        Metal Best Practices  m4
1201  msl-kernels    Register spilling behavior unchanged  verified    WWDC25                universal
89    gpu-silicon    GPU core counts across M-series       high        Anandtech Review
```

Notes:
- `gen` column is blank for NULL findings (unclassified), not "NULL" -- reduces visual noise
- Superseded findings are silently excluded (they don't appear at all)
- Existing column order is preserved; `gen` is appended

**With `--include-superseded`:**
```
id    skill          claim                                 confidence  source_title          gen   status
----  -------------  ------------------------------------  ----------  --------------------  ----  ----------
565   gpu-silicon    M5 GPU provides 12.8 TFLOPS FP32     verified    Apple Tech Note       m5    current
402   gpu-perf       FP32 throughput scales linearly       high        Metal Best Practices  m4    current
14    gpu-silicon    M1 GPU provides 2.6 TFLOPS FP32      verified    Apple Spec Sheet      m1    SUPERSEDED
```

The `SUPERSEDED` status is uppercase to draw attention. Current findings show `current` in lowercase since it is the expected default state.

### 4.2 Detail View: Supersession Chain

The `kb detail` output is column-aligned (sqlite3 `-header -column` mode). The new fields integrate naturally:

```
=== Finding #14 ===
id               14
skill_name       gpu-silicon
topic            specs
claim            Apple M1 GPU provides 2.6 TFLOPS FP32 compute performance
evidence         Official Apple specification sheet for M1 chip
source_url       https://support.apple.com/...
source_title     Apple M1 Chip Specifications
source_type      apple_docs
confidence       verified
date_found       2025-04-12
date_published   2020-11-10
tags             m1,performance,architecture
gpu_generation   m1
temporal_status  superseded
superseded_by    565
notes            [Superseded: M5 benchmarks replace M1 specs]

--- Supersession ---
  SUPERSEDED by Finding #565 (m5, gpu-silicon/specs, 2025-11-03)

--- Citations ---
  (none)
```

The supersession section is clearly separated with a `---` divider. It uses the same visual language as the existing `--- Citations ---` section.

For the superseding finding:
```
--- Supersession ---
  Supersedes Finding #14 (m1, gpu-silicon/specs, 2025-04-12)
```

### 4.3 Freshness Dashboard: Visual Hierarchy

The dashboard (Section 1.5 above) uses three visual levels:

1. **Section headers** (`===`): Top-level categories
2. **Labeled counts with percentages**: Quick scannable metrics
3. **Actionable lists**: Specific items requiring attention (contradictions)

The dashboard intentionally avoids:
- Color codes (agents cannot process ANSI colors)
- Progress bars (unnecessary for counts)
- Sparklines or trends (no historical data to trend against)

### 4.4 Verify Output: Temporal Check Integration

The new temporal check follows the existing verify output pattern exactly:

```
=== Knowledge Base Quality Report ===

OK: No confidence inflation found
OK: All citation authors verified
OK: No near-duplicate findings detected
OK: No suspicious source categorization
OK: All findings have source URLs
WARNING: 8 benchmark/empirical findings have no gpu_generation tag:
  #201  gpu-perf    "FP16 throughput on GPU reaches..."    benchmark
  #245  gpu-silicon "ALU utilization measured at..."        empirical_test
  Tag with: kb tag-gen <id> <generation>

TOTAL: 8 issue(s) found
```

The temporal check is appended after existing checks (Check 6). This preserves the existing check numbering and ensures all 194 existing BATS tests pass without modification.

---

## 5. Accessibility and Discoverability

### 5.1 Error Messages

Every error message follows the pattern: **what happened + how to fix it**.

| Scenario | Message |
|----------|---------|
| Invalid gen value | `ERROR: Invalid generation 'mx'. Valid: m1, m2, m3, m4, m5, universal` |
| Supersede nonexistent ID | `ERROR: Finding #999 does not exist. Check ID with: kb search "<query>"` |
| Supersede already-superseded | `WARNING: Finding #14 is already superseded by #301. Use --force to override.` |
| tag-gen invalid gen | `ERROR: Invalid generation 'foo'. Valid: m1, m2, m3, m4, m5, universal` |
| tag-gen on nonexistent ID | `ERROR: Finding #999 does not exist` |
| unsupersede on current finding | `WARNING: Finding #14 is already current. Nothing to do.` |

### 5.2 Progressive Disclosure Layers

**Layer 0 (zero effort):** Existing commands work identically. No temporal features visible unless sought.

**Layer 1 (passive discovery):** Search results now include `gen` column. `kb verify` produces temporal warnings. These surface naturally during normal workflow.

**Layer 2 (active exploration):** Developer runs `kb freshness` to understand temporal state. Sees contradictions, runs `kb detail` on flagged findings.

**Layer 3 (active management):** Developer uses `kb supersede`, `kb tag-gen`, `kb unsupersede` to actively curate temporal metadata.

**Layer 4 (automated discipline):** Investigation agent prompt changes ensure new findings get generation tags automatically. The system maintains itself.

### 5.3 Agent Discoverability

Agents discover temporal features through two mechanisms:

1. **Prompt instructions** (explicit): The investigation agent prompt tells it to tag generations and check for supersession. This is authoritative.

2. **Output feedback** (implicit): When `kb verify` reports temporal warnings, the agent sees them during Phase 5. The warning text includes the exact fix command. An agent that follows the existing instruction "Fix any issues reported before completing the investigation" will naturally address temporal issues.

### 5.4 Tab Completion Consideration

The `kb` script uses a simple `case` statement for command dispatch. If tab completion is ever added (via a zsh completion file), the temporal commands follow predictable naming:
- All temporal commands are single-word or hyphenated: `supersede`, `unsupersede`, `tag-gen`, `freshness`
- Flag names follow standard conventions: `--gen`, `--dry-run`, `--include-superseded`, `--force`
- No ambiguous prefixes: no existing command starts with `s` (supersede), `u` (unsupersede), `t` (tag-gen), or `f` (freshness) -- wait, `fix-confidence` starts with `f`. Rename consideration: `freshness` vs `temporal-health` vs `health`. Recommendation: keep `freshness` -- it is more specific than `health` and the `f` prefix ambiguity only matters with tab completion, which does not exist yet.

---

## 6. Implementation Notes for CLI Developers

### 6.1 Argument Parsing in `kb search`

The current `kb search` uses a `while/shift` loop to parse `--limit`. The same pattern extends cleanly:

```bash
search)
    query=$(escape_sql "$2")
    limit=10
    gen=""
    include_superseded=0
    shift 2
    while [ $# -gt 0 ]; do
      case "$1" in
        --limit|-l) shift; limit="$1" ;;
        --gen|-g) shift; gen="$1" ;;
        --include-superseded|-S) include_superseded=1 ;;
      esac
      shift
    done

    # Build WHERE clauses
    gen_clause=""
    if [ -n "$gen" ]; then
      # Validate generation value
      case "$gen" in
        m1|m2|m3|m4|m5|universal) ;;
        *) echo "ERROR: Invalid generation '$gen'. Valid: m1, m2, m3, m4, m5, universal" >&2; exit 1 ;;
      esac
      gen_clause="AND (f.gpu_generation = '$gen' OR f.gpu_generation = 'universal' OR f.gpu_generation IS NULL)"
    fi

    status_clause=""
    if [ "$include_superseded" -eq 0 ]; then
      status_clause="AND (f.temporal_status = 'current' OR f.temporal_status IS NULL)"
    fi

    run_sql -header -column "$DB" \
      "SELECT f.id, s.name as skill, f.claim, f.confidence, f.source_title,
              COALESCE(f.gpu_generation, '') as gen
       FROM findings_fts fts
       JOIN findings f ON f.id = fts.rowid
       JOIN skills s ON s.id = f.skill_id
       WHERE findings_fts MATCH '$query'
         $gen_clause
         $status_clause
       ORDER BY bm25(findings_fts, 10.0, 5.0, 2.0, 1.0)
       LIMIT $limit;"
    ;;
```

### 6.2 Supersession Ordering in Search Results

When `--include-superseded` is used, superseded findings should sort after current findings within the same BM25 relevance tier:

```sql
ORDER BY
  CASE WHEN f.temporal_status = 'superseded' THEN 1 ELSE 0 END,
  bm25(findings_fts, 10.0, 5.0, 2.0, 1.0)
```

This ensures that even with `--include-superseded`, current findings always appear first.

### 6.3 Column Width for `gen` in Output

The `gen` column maximum width is 9 characters (`universal`). Using sqlite3 `-column` mode, the column auto-sizes. If manual formatting is needed: `printf "%-9s"`.

---

## Summary of All Changes

| Component | Change | Effort |
|-----------|--------|--------|
| `kb search` | Add `--gen`, `--include-superseded` flags, add `gen` column to output | Medium |
| `kb detail` | Show `gpu_generation`, `temporal_status`, `superseded_by`, supersession chain section | Low |
| `kb supersede` | New command | Medium |
| `kb unsupersede` | New command | Low |
| `kb tag-gen` | New command (single + auto modes) | High |
| `kb freshness` | New command (dashboard) | Medium |
| `kb verify` | Add Check 6 (temporal) | Low |
| `kb add` | Accept position 11 for gpu_generation | Low |
| `kb` help text | Add temporal commands section | Low |
| investigation-agent.md | 3 additions (~40 tokens) | Low |
| knowledge-retriever.md | 2 additions (~30 tokens) | Low |
| architecture-advisor.md | 1 addition (~20 tokens) | Low |

---

## Questions & Answers

### Q1: Gen column visibility
**Answer**: Always show gen column by default.
**Impact**: Search results always include generation metadata. Passive discovery for developers, generation context for agents.

### Q2: Supersession confirmation
**Answer**: No confirmation for first supersession. `--force` only for overriding existing link.
**Impact**: Minimal friction. Reversible via `kb unsupersede`.

### Q3: Contradiction matching
**Answer**: Two-tier: exact topic+skill as "definite", FTS as "potential".
**Impact**: `kb freshness` shows both sections — precision and recall.

### Q4: NULL in --gen filter
**Answer**: Include NULL findings in --gen filtered results.
**Impact**: Per PM FR-4. 87% of KB stays visible during backfill ramp.

---

_Original questions for reference:_

### Q1: Should `kb search` show the `gen` column by default, or only when `--gen` is used?

**Option A (recommended):** Always show `gen` column. Developers passively learn about generation tagging. Agents always see generation context. The column is narrow (max 9 chars) and adds minimal noise.

**Option B:** Show `gen` column only when `--gen` flag is used. Keeps existing output identical. But developers never discover the feature passively, and agents lose generation context on unfiltered queries.

**Recommendation:** Option A. Passive discovery outweighs the minimal column-width cost. The existing BATS tests that check search output format will need updating regardless (schema migration adds the column to SELECT), so this is the natural time to expose it.

### Q2: Should `kb supersede` require confirmation for the first supersession, or only for overriding an existing supersession?

**Option A (recommended):** No confirmation for first supersession. Only require `--force` when overriding an existing supersession link. Minimizes friction for the common case. Agents can supersede without interactive prompts.

**Option B:** Always require `--force` or `--yes` to confirm. Safer but adds friction. Agents would need to always pass `--force`, which defeats the safety purpose.

**Option C:** Interactive confirmation for humans (detect TTY), auto-confirm for agents (non-TTY). Clever but adds complexity for minimal benefit at the current KB scale (~30 findings needing supersession).

**Recommendation:** Option A. The operation is reversible (`kb unsupersede`), the scale is small, and agent compatibility matters more than human safety guards at this scale.

### Q3: Should the `kb freshness` contradiction detection query use exact topic matching or fuzzy/FTS matching to find potential contradictions?

**Option A:** Exact topic match (`WHERE a.topic = b.topic AND a.skill_id = b.skill_id`). Simple, fast, no false positives. But misses contradictions where topics are named slightly differently (e.g., "specs" vs "hardware specs").

**Option B (recommended):** Exact topic + same skill match, with a secondary FTS pass for cross-topic contradictions flagged separately. Two-tier approach: exact matches are "definite contradictions," FTS matches are "potential contradictions."

**Option C:** Pure FTS matching on claim text. Most comprehensive but noisy -- many false positives at 1,555 findings.

**Recommendation:** Option B. Start with exact matching for the primary contradiction list (high signal). Add a "Possible contradictions (topic similarity)" section below for FTS-based matches. This gives both precision and recall without overwhelming the developer.

### Q4: Should `--gen` filter behavior include NULL findings (unclassified) or exclude them?

**Option A (recommended):** Include NULL findings in `--gen` filtered results. Rationale: NULL means "not yet classified," not "wrong generation." Excluding them would hide 87% of the KB from every filtered search, making `--gen` useless until backfill reaches high coverage. This is the PM.md specification (FR-4, AC-1.3).

**Option B:** Exclude NULL findings from `--gen` results. Pure filter -- only shows findings explicitly tagged. Would incentivize faster backfill but makes the feature impractical until coverage is high.

**Option C:** Include NULL by default, add `--strict` flag to exclude NULL. Maximum flexibility but adds yet another flag.

**Recommendation:** Option A, per the PM spec. After backfill pushes NULL below 50%, revisit whether `--strict` exclusion mode adds value.

---END QUESTIONS---
