---
id: devops.BREAKDOWN
module: devops
priority: 0
status: failing
version: 1
origin: spec-workflow
dependsOn: []
tags: [gpu-forge, temporal-relevance]
testRequirements:
  unit:
    required: false
    pattern: "tests/unit/temporal.bats"
---

# BREAKDOWN: DevOps -- Schema Migration Infrastructure

## Context

The KB temporal relevance pipeline requires adding 3 new columns to the SQLite `findings` table (1,555 rows). This module provides the migration infrastructure: the `kb migrate-temporal` command with idempotent execution, automatic backup before ALTER TABLE, column-existence verification, and rollback documentation. The migration must be safe to run multiple times, preserve all existing data, and create a backup for recovery.

The production DB is at `data/gpu_knowledge.db`. SQLite version is 3.51.1, which supports CHECK constraints on ADD COLUMN (3.37.0+) and DROP COLUMN for rollback (3.35.0+). The migration is triggered manually via CLI command, not automatically.

Reference: TECH.md Section 2 (Schema Migration), Section 3.2 (Migration Command)

## Tasks

### T-001: Implement `kb migrate-temporal` command

Add the `migrate-temporal` case to the `kb` script's command dispatch. The command must:

1. Check if migration is already applied via `pragma_table_info('findings')` for `gpu_generation` column
2. If already applied, print "Migration already applied" and exit 0
3. If not applied, create backup: `cp "$DB" "${DB}.pre-temporal-backup"`
4. Execute 3 ALTER TABLE ADD COLUMN statements (gpu_generation, temporal_status, superseded_by)
5. Execute 3 CREATE INDEX IF NOT EXISTS statements
6. Verify all 3 columns exist via pragma_table_info
7. Print success message with column/index counts

### T-002: Document rollback procedure

Document two rollback paths:
- **Option 1**: Restore from backup file (`${DB}.pre-temporal-backup`)
- **Option 2**: SQLite 3.35+ DROP COLUMN (3 ALTER TABLE DROP COLUMN + 3 DROP INDEX)

### T-003: Add migrate-temporal to help text

Add `migrate-temporal` to the "Temporal commands" section of `kb` help output.

## Acceptance Criteria

1. `kb migrate-temporal` adds 3 columns and 3 indexes to a fresh (pre-migration) DB copy
2. `kb migrate-temporal` on an already-migrated DB prints "already applied" and exits 0 (idempotent)
3. Backup file `${DB}.pre-temporal-backup` exists after first migration run
4. All 1,555+ existing findings are preserved (COUNT unchanged)
5. `temporal_status` defaults to `'current'` for all existing rows
6. `gpu_generation` and `superseded_by` default to NULL for all existing rows
7. All 194 existing BATS tests pass after migration

## Technical Notes

- Reference: spec/TECH.md Section 2 (Schema Migration), Section 2.3 (Idempotent Wrapper)
- Test: spec/QA.md Section 3 (Migration Test Suite) -- tests M-1 through M-7
- SQLite constraint: ALTER TABLE ADD COLUMN CHECK must permit NULL for existing rows (CHECK includes `IS NULL`)
- `temporal_status DEFAULT 'current'` means existing rows get 'current', not NULL
- Migration uses `run_sql` helper from existing kb script for consistency
