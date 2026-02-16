---
id: schema.BREAKDOWN
module: schema
priority: 1
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN]
tags: [gpu-forge, temporal-relevance]
testRequirements:
  unit:
    required: true
    pattern: "tests/unit/temporal.bats"
---

# BREAKDOWN: Schema -- Database Column and Index Definitions

## Context

The temporal relevance pipeline adds 3 new columns and 3 indexes to the `findings` table. This module defines the exact SQL schema, CHECK constraints, defaults, and index specifications. It also updates `data/schema.sql` (used for fresh DB creation) to include the new columns inline in the CREATE TABLE statement.

The FTS5 virtual table (`findings_fts`) and its triggers are NOT modified -- generation filtering is applied post-FTS via JOIN WHERE clauses. The new columns are categorical/relational data, not free text, so they do not belong in the FTS5 index.

Reference: TECH.md Section 2 (Schema Migration), Section 2.4 (Schema File Update), Section 2.5 (FTS5 Impact)

## Tasks

### T-001: Define gpu_generation column

```sql
ALTER TABLE findings ADD COLUMN gpu_generation TEXT
  CHECK(gpu_generation IS NULL OR gpu_generation IN ('m1','m2','m3','m4','m5','universal'))
  DEFAULT NULL;
```

- Valid values: `m1`, `m2`, `m3`, `m4`, `m5`, `universal`, NULL
- NULL = unclassified (not yet tagged), distinct from `universal` (confirmed gen-agnostic)
- Per PM Q&A Q1: NULL included in `--gen` filtered searches for backward compatibility
- Per PM Q&A Q2: Multi-gen findings tagged with newest generation mentioned

### T-002: Define temporal_status column

```sql
ALTER TABLE findings ADD COLUMN temporal_status TEXT
  CHECK(temporal_status IS NULL OR temporal_status IN ('current','superseded'))
  DEFAULT 'current';
```

- Two-status model only: `current` and `superseded` (no `historical` per PM Q&A Q3)
- DEFAULT 'current' means all existing findings start as current
- CHECK allows NULL defensively, but default ensures existing rows get 'current'

### T-003: Define superseded_by column

```sql
ALTER TABLE findings ADD COLUMN superseded_by INTEGER
  REFERENCES findings(id)
  DEFAULT NULL;
```

- FK to findings(id) -- points to the finding that replaces this one
- NULL = not superseded (the default/expected state)
- REFERENCES declared but enforcement depends on PRAGMA foreign_keys

### T-004: Create performance indexes

```sql
CREATE INDEX IF NOT EXISTS idx_findings_gpu_generation ON findings(gpu_generation);
CREATE INDEX IF NOT EXISTS idx_findings_temporal_status ON findings(temporal_status);
CREATE INDEX IF NOT EXISTS idx_findings_superseded_by ON findings(superseded_by);
```

- Generation filter: O(1) per row on indexed column during post-FTS filtering
- Status filter: O(1) per row for superseded exclusion
- Superseded_by: Enables reverse lookup ("what does this finding supersede?")

### T-005: Update data/schema.sql

Add the 3 new columns to the existing `CREATE TABLE findings` statement (after `investigation_session`). Add the 3 indexes after existing index definitions. This ensures fresh DB creation includes temporal columns.

## Acceptance Criteria

1. `gpu_generation` CHECK constraint accepts m1, m2, m3, m4, m5, universal, NULL
2. `gpu_generation` CHECK constraint rejects invalid values (e.g., 'm6', 'historical')
3. `temporal_status` CHECK constraint accepts 'current', 'superseded', NULL
4. `temporal_status` CHECK constraint rejects 'historical' (removed per PM Q&A Q3)
5. `temporal_status` default is 'current' for newly inserted findings
6. `superseded_by` accepts valid finding IDs and NULL
7. All 3 indexes created with `IF NOT EXISTS` (idempotent)
8. `data/schema.sql` includes new columns in CREATE TABLE statement
9. FTS5 virtual table and triggers remain unchanged

## Technical Notes

- Reference: spec/TECH.md Section 2.1-2.5 (Schema Migration details)
- Test: spec/QA.md Section 4.1 (Schema Constraint Tests) -- tests C-1 through C-5
- SQLite 3.37.0+ tests CHECK against existing rows on ALTER TABLE ADD COLUMN
- Existing FTS5 triggers sync `claim, evidence, tags, notes` -- new columns excluded
- No column reordering: new columns appended after `investigation_session`
