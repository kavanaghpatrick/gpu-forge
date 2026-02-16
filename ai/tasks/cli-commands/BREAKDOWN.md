---
id: cli-commands.BREAKDOWN
module: cli-commands
priority: 3
status: failing
version: 1
origin: spec-workflow
dependsOn: [schema.BREAKDOWN]
tags: [gpu-forge, temporal-relevance]
testRequirements:
  unit:
    required: true
    pattern: "tests/unit/temporal.bats"
---

# BREAKDOWN: CLI Commands -- Temporal Management Commands

## Context

This module implements the four new temporal management commands: `kb supersede`, `kb unsupersede`, `kb tag-gen`, and `kb freshness`. These commands provide the core temporal curation workflow: marking findings as superseded, reversing supersession, tagging findings with GPU generation metadata, and monitoring KB temporal health.

All commands follow existing `kb` script conventions: positional args for required params, `case`/`shift` for optional flags, `run_sql` for SQL execution, `escape_sql` for input sanitization, and plain text output (no colors/emojis).

Reference: TECH.md Sections 3.4-3.7 (Supersede, Unsupersede, Tag-Gen, Freshness), UX.md Sections 1.2-1.5

## Tasks

### T-001: Implement `kb supersede <old_id> <new_id> [reason]`

Full implementation per TECH.md Section 3.4:
1. Validate both IDs are numeric (`grep -E '^[0-9]+$'`)
2. Validate both IDs exist in findings table
3. Self-supersession guard: reject `old_id == new_id` (per QA Q&A Q2)
4. Check if old finding is already superseded; require `--force` to override
5. Append reason to notes field: `[Superseded: <reason>]` or `[Superseded by #<new_id>]`
6. UPDATE findings SET temporal_status='superseded', superseded_by=<new_id>
7. Output: "Finding #<old_id> marked superseded by #<new_id>"

### T-002: Implement `kb unsupersede <finding_id>`

Full implementation per TECH.md Section 3.5:
1. Validate ID is numeric and finding exists
2. Check finding is actually superseded; if current, warn and exit 0
3. UPDATE findings SET temporal_status='current', superseded_by=NULL
4. Append to notes: `[Unsuperseded: restored to current]`
5. Output: "Finding #<id> restored to current status"

### T-003: Implement `kb tag-gen` (single finding mode)

Syntax: `kb tag-gen <finding_id> <generation>`
1. Validate ID is numeric, generation is valid (m1-m5, universal)
2. Validate finding exists
3. UPDATE findings SET gpu_generation='<gen>'
4. Output: "Finding #<id> tagged with gpu_generation=<gen>"

Also: `kb tag-gen <finding_id> --clear` to reset gpu_generation to NULL.

### T-004: Implement `kb tag-gen --auto` (batch auto-inference mode)

Syntax: `kb tag-gen --auto [--dry-run] [--skill <name>]`
1. Query findings with NULL gpu_generation and non-empty tags
2. For each finding, parse tags for generation patterns (case-insensitive):
   - Check m5 first (newest), then m4, m3, m2, m1
   - If multiple found, tag with newest (per PM Q&A Q2)
3. Never overwrite existing non-NULL gpu_generation
4. `--dry-run`: Show preview with "Would tag" / "Would skip" sections
5. Without `--dry-run`: Execute UPDATE statements, output count
6. `--skill <name>`: Filter to one skill only

### T-005: Implement `kb freshness` (temporal health dashboard)

Syntax: `kb freshness [--skill <name>] [--json]`

Human-readable mode:
1. Generation Coverage: count per gpu_generation value with percentages
2. Temporal Status: current vs superseded counts
3. Date Coverage: date_published non-null count and percentage
4. Potential Contradictions: exact topic+skill match, different generations, no supersession link

JSON mode (`--json`):
- Output single-line JSON with generation_counts, temporal_status, date_coverage keys
- Validate with `python3 -c "import sys,json; json.load(sys.stdin)"`

### T-006: Modify `kb detail` for supersession display

Per TECH.md Section 3.8:
1. After existing detail output, check temporal_status and superseded_by
2. If superseded: show "--- Supersession ---" section with "SUPERSEDED by Finding #N"
3. Check reverse: show "Supersedes Finding #N" for findings that this one supersedes
4. Show only immediate links (not full chain) per PM risk R5

## Acceptance Criteria

1. `kb supersede 1 2` marks finding #1 superseded by #2 with correct DB state
2. `kb supersede 1 2 "reason"` appends reason to notes field
3. `kb supersede 99999 2` fails with "does not exist" error
4. `kb supersede 1 1` fails with self-supersession guard
5. Double supersession requires `--force` flag
6. `kb unsupersede` restores finding to current status
7. `kb unsupersede` on current finding is a no-op with warning
8. `kb tag-gen 1 m4` sets gpu_generation correctly
9. `kb tag-gen 1 --clear` resets gpu_generation to NULL
10. `kb tag-gen --auto --dry-run` shows preview without modifying DB
11. `kb tag-gen --auto` tags findings and does not overwrite existing tags
12. `kb freshness` shows all 4 dashboard sections
13. `kb freshness --json` outputs valid JSON
14. `kb detail` shows supersession chain for superseded findings

## Technical Notes

- Reference: spec/TECH.md Sections 3.4-3.8 (CLI Command Implementations)
- Reference: spec/UX.md Sections 1.2-1.6 (Command UX Design)
- Test: spec/QA.md Sections 4.4-4.8 (Command Tests SP-1 through H-1)
- Contradiction detection uses exact topic+skill_id matching (Phase 1 per TECH Q&A Q4)
- Tag pattern matching: case-insensitive substring on m1-m5 (per TECH Q&A Q3)
- Self-supersession guard: `old_id == new_id` check (per QA Q&A Q2)
- All error messages include fix suggestion (per UX Section 5.1)
