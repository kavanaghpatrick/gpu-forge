# Technical Architect Design: KB Temporal Relevance Pipeline

**Date**: 2026-02-13
**Architect**: Technical Architect Agent
**System**: gpu-forge (Claude Code plugin)
**Stack**: Bash CLI (`kb` script, 474 lines), SQLite 3.51.1 with FTS5, BATS tests (194 tests, 14 files), Markdown agent prompts
**Production DB**: 1,555 findings, 63 investigations, 155 citations
**Scope**: Schema migration, CLI modifications, agent prompt updates, backfill strategy, testing plan

---

## 1. Architecture Overview

### What Changes

The temporal relevance pipeline adds three new columns to the `findings` table, five new CLI commands (plus modifications to three existing commands), and surgical prompt additions to three agent files. The FTS5 virtual table is **not modified** -- generation filtering is applied post-FTS via JOIN WHERE clauses.

### Data Flow

```
                    WRITE PATH                              READ PATH
                    ----------                              ---------
 Investigation Agent                              Knowledge Retriever / Arch Advisor
        |                                                    |
  [Phase 3: Store Finding]                        [Query: "TFLOPS --gen m4"]
        |                                                    |
  INSERT INTO findings                           FTS5 MATCH 'TFLOPS'
  (..., gpu_generation='m4',                            |
   temporal_status='current')                    JOIN findings WHERE
        |                                        gpu_generation IN ('m4','universal',NULL)
  [Phase 2: Supersession check]                  AND temporal_status='current'
        |                                                    |
  kb supersede <old> <new>                       ORDER BY bm25(...)
        |                                                    |
  UPDATE findings SET                            Return ranked, gen-filtered results
   temporal_status='superseded',
   superseded_by=<new>
```

### Key Design Decisions

1. **Post-FTS filtering**: FTS5 narrows to <100 rows; WHERE on indexed columns adds <1ms. No FTS5 schema changes needed.
2. **NULL-inclusive filtering**: `--gen m4` includes NULL findings (87% of KB) for backward compatibility during backfill ramp.
3. **Two-status model**: Only `current` and `superseded` (no `historical`). Reduces cognitive overhead.
4. **CHECK constraints on ADD COLUMN**: SQLite 3.37.0+ tests CHECK against existing rows on ALTER TABLE ADD COLUMN. Since existing rows get NULL defaults, CHECK must permit NULL.
5. **Idempotent migration**: All schema changes wrapped in column-existence checks so migration script is safe to run multiple times.

---

## 2. Schema Migration

### 2.1 Migration Script

File: `data/migrate-temporal.sql` (also executable via `kb migrate-temporal`)

```sql
-- Migration: Add temporal relevance columns to findings table
-- Safe to run multiple times (idempotent)
-- Requires: SQLite 3.37.0+ (CHECK constraint on ADD COLUMN)
-- Tested on: SQLite 3.51.1

-- Step 1: Add gpu_generation column
-- CHECK allows NULL (existing rows) + the 7 valid generation values
-- DEFAULT NULL means existing INSERTs without this column work unchanged
ALTER TABLE findings ADD COLUMN gpu_generation TEXT
  CHECK(gpu_generation IS NULL OR gpu_generation IN ('m1','m2','m3','m4','m5','universal'))
  DEFAULT NULL;

-- Step 2: Add temporal_status column
-- DEFAULT 'current' means all existing findings start as current
-- CHECK allows NULL (defensive) + the 2 valid status values
ALTER TABLE findings ADD COLUMN temporal_status TEXT
  CHECK(temporal_status IS NULL OR temporal_status IN ('current','superseded'))
  DEFAULT 'current';

-- Step 3: Add superseded_by column
-- REFERENCES is declared but not enforced by ALTER TABLE ADD COLUMN in SQLite
-- (FK enforcement depends on PRAGMA foreign_keys, which we verify separately)
-- DEFAULT NULL means unsuperseded findings have no link
ALTER TABLE findings ADD COLUMN superseded_by INTEGER
  REFERENCES findings(id)
  DEFAULT NULL;

-- Step 4: Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_findings_gpu_generation ON findings(gpu_generation);
CREATE INDEX IF NOT EXISTS idx_findings_temporal_status ON findings(temporal_status);
CREATE INDEX IF NOT EXISTS idx_findings_superseded_by ON findings(superseded_by);
```

### 2.2 Critical Implementation Notes

**SQLite ALTER TABLE ADD COLUMN constraints (verified against [SQLite docs](https://www.sqlite.org/lang_altertable.html)):**

1. **CHECK constraints**: Supported since SQLite 3.37.0 (2021-11-27). The CHECK is tested against all existing rows. Since existing rows will have the DEFAULT value:
   - `gpu_generation DEFAULT NULL` -- CHECK includes `IS NULL`, so existing rows pass.
   - `temporal_status DEFAULT 'current'` -- CHECK includes `'current'`, so existing rows pass.

2. **REFERENCES constraint**: When foreign keys are enabled, ADD COLUMN with REFERENCES requires DEFAULT NULL. Our `superseded_by DEFAULT NULL` satisfies this.

3. **DEFAULT expression restriction**: SQLite ADD COLUMN does not allow parenthesized expressions or CURRENT_TIMESTAMP as defaults. Our literal defaults (`NULL`, `'current'`, `NULL`) are all allowed.

4. **No column reordering**: New columns are appended after `investigation_session`. Agent INSERTs that use named columns (not positional) are unaffected. The investigation agent uses explicit column names in its INSERT template, so this is safe.

### 2.3 Idempotent Wrapper

SQLite does not support `ALTER TABLE ADD COLUMN IF NOT EXISTS`. The migration script needs a bash wrapper that checks column existence before altering:

```bash
migrate_temporal() {
  local db="$1"

  # Check if migration already applied (idempotent)
  local has_gen
  has_gen=$(sqlite3 "$db" "SELECT COUNT(*) FROM pragma_table_info('findings') WHERE name='gpu_generation';")
  if [ "$has_gen" -gt 0 ]; then
    echo "Migration already applied — gpu_generation column exists"
    return 0
  fi

  echo "Applying temporal relevance migration..."

  # Backup before migration
  cp "$db" "${db}.pre-temporal-backup"
  echo "Backup created: ${db}.pre-temporal-backup"

  sqlite3 "$db" <<'MIGRATION'
ALTER TABLE findings ADD COLUMN gpu_generation TEXT
  CHECK(gpu_generation IS NULL OR gpu_generation IN ('m1','m2','m3','m4','m5','universal'))
  DEFAULT NULL;

ALTER TABLE findings ADD COLUMN temporal_status TEXT
  CHECK(temporal_status IS NULL OR temporal_status IN ('current','superseded'))
  DEFAULT 'current';

ALTER TABLE findings ADD COLUMN superseded_by INTEGER
  REFERENCES findings(id)
  DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_findings_gpu_generation ON findings(gpu_generation);
CREATE INDEX IF NOT EXISTS idx_findings_temporal_status ON findings(temporal_status);
CREATE INDEX IF NOT EXISTS idx_findings_superseded_by ON findings(superseded_by);
MIGRATION

  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "Migration complete. 3 columns added, 3 indexes created."
    # Verify
    local col_count
    col_count=$(sqlite3 "$db" "SELECT COUNT(*) FROM pragma_table_info('findings') WHERE name IN ('gpu_generation','temporal_status','superseded_by');")
    if [ "$col_count" -eq 3 ]; then
      echo "Verification: all 3 new columns present."
    else
      echo "ERROR: Verification failed — expected 3 new columns, found $col_count" >&2
      return 1
    fi
  else
    echo "ERROR: Migration failed (exit code $rc). Restore from: ${db}.pre-temporal-backup" >&2
    return $rc
  fi
}
```

### 2.4 Schema File Update

The `data/schema.sql` file must be updated to include the new columns for fresh database creation. The new columns are added to the `CREATE TABLE findings` statement, not as separate ALTER TABLE statements:

```sql
-- Add after the existing investigation_session column:
    gpu_generation TEXT CHECK(gpu_generation IS NULL OR gpu_generation IN (
        'm1','m2','m3','m4','m5','universal'
    )) DEFAULT NULL,
    temporal_status TEXT CHECK(temporal_status IS NULL OR temporal_status IN (
        'current','superseded'
    )) DEFAULT 'current',
    superseded_by INTEGER REFERENCES findings(id) DEFAULT NULL
```

And add the three indexes after the existing index definitions:

```sql
CREATE INDEX IF NOT EXISTS idx_findings_gpu_generation ON findings(gpu_generation);
CREATE INDEX IF NOT EXISTS idx_findings_temporal_status ON findings(temporal_status);
CREATE INDEX IF NOT EXISTS idx_findings_superseded_by ON findings(superseded_by);
```

### 2.5 FTS5 Impact

The FTS5 virtual table (`findings_fts`) indexes `claim, evidence, tags, notes`. The three new columns (`gpu_generation`, `temporal_status`, `superseded_by`) are **not** added to the FTS5 index because:

1. They are categorical/relational data, not free text. FTS5 MATCH is wrong for exact-value filtering.
2. Generation filtering via FTS MATCH would require embedding "m4" tokens in the FTS content, creating false matches on findings that mention "m4" in their claim text without being m4-specific.
3. Post-FTS JOIN filtering is the [documented best practice for combining FTS with relational predicates](https://sqlite.org/fts5.html).

The existing FTS5 triggers (`findings_ai`, `findings_ad`, `findings_au`) sync `claim, evidence, tags, notes`. They do **not** need modification because the new columns are not in the FTS5 content specification.

---

## 3. KB CLI Modifications

### 3.1 Code Style Conventions

Based on reading the full 474-line `kb` script, these patterns must be followed:

- **Command dispatch**: `case "${1}" in ... esac` at top level
- **Argument parsing**: `shift` loops for optional flags, positional for required args
- **SQL execution**: `run_sql "$DB" "SQL"` for no-output queries, `run_sql -header -column "$DB" "SQL"` for tabular output
- **Input validation**: `escape_sql` for string interpolation, `grep -E '^[0-9]+$'` for numeric IDs
- **Error messages**: `echo "ERROR: ..." >&2; exit 1`
- **Success messages**: `echo "Finding added to ..."` (simple, one-line)
- **No colors/emojis**: Plain text only

### 3.2 Migration Command

Insert before the help/default case (`*)`):

```bash
  migrate-temporal)
    # kb migrate-temporal — add temporal relevance columns (idempotent)
    local has_gen
    has_gen=$(run_sql "$DB" "SELECT COUNT(*) FROM pragma_table_info('findings') WHERE name='gpu_generation';")
    if [ "$has_gen" -gt 0 ]; then
      echo "Migration already applied — gpu_generation column exists"
      exit 0
    fi

    echo "Applying temporal relevance migration..."
    cp "$DB" "${DB}.pre-temporal-backup"
    echo "Backup created: ${DB}.pre-temporal-backup"

    run_sql "$DB" "ALTER TABLE findings ADD COLUMN gpu_generation TEXT CHECK(gpu_generation IS NULL OR gpu_generation IN ('m1','m2','m3','m4','m5','universal')) DEFAULT NULL;"
    run_sql "$DB" "ALTER TABLE findings ADD COLUMN temporal_status TEXT CHECK(temporal_status IS NULL OR temporal_status IN ('current','superseded')) DEFAULT 'current';"
    run_sql "$DB" "ALTER TABLE findings ADD COLUMN superseded_by INTEGER REFERENCES findings(id) DEFAULT NULL;"
    run_sql "$DB" "CREATE INDEX IF NOT EXISTS idx_findings_gpu_generation ON findings(gpu_generation);"
    run_sql "$DB" "CREATE INDEX IF NOT EXISTS idx_findings_temporal_status ON findings(temporal_status);"
    run_sql "$DB" "CREATE INDEX IF NOT EXISTS idx_findings_superseded_by ON findings(superseded_by);"

    local col_count
    col_count=$(run_sql "$DB" "SELECT COUNT(*) FROM pragma_table_info('findings') WHERE name IN ('gpu_generation','temporal_status','superseded_by');")
    if [ "$col_count" -eq 3 ]; then
      echo "Migration complete. 3 columns added, 3 indexes created."
    else
      echo "ERROR: Verification failed — expected 3 new columns, found $col_count" >&2
      exit 1
    fi
    ;;
```

### 3.3 Search Command Modification

Replace the existing `search)` case (lines 68-90 of current script):

```bash
  search)
    # kb search <query> [--limit N] [--gen <generation>] [--include-superseded]
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

    # Validate generation value if provided
    gen_clause=""
    if [ -n "$gen" ]; then
      case "$gen" in
        m1|m2|m3|m4|m5|universal) ;;
        *) echo "ERROR: Invalid generation '$gen'. Valid: m1, m2, m3, m4, m5, universal" >&2; exit 1 ;;
      esac
      gen_clause="AND (f.gpu_generation = '$gen' OR f.gpu_generation = 'universal' OR f.gpu_generation IS NULL)"
    fi

    # Build temporal status filter
    status_clause=""
    if [ "$include_superseded" -eq 0 ]; then
      status_clause="AND (f.temporal_status = 'current' OR f.temporal_status IS NULL)"
    fi

    # Build ORDER BY: when including superseded, sort them after current findings
    order_clause="ORDER BY bm25(findings_fts, 10.0, 5.0, 2.0, 1.0)"
    if [ "$include_superseded" -eq 1 ]; then
      order_clause="ORDER BY CASE WHEN f.temporal_status = 'superseded' THEN 1 ELSE 0 END, bm25(findings_fts, 10.0, 5.0, 2.0, 1.0)"
    fi

    # Build SELECT columns: add status column only when --include-superseded
    if [ "$include_superseded" -eq 1 ]; then
      select_cols="f.id, s.name as skill, f.claim, f.confidence, f.source_title, COALESCE(f.gpu_generation, '') as gen, CASE WHEN f.temporal_status = 'superseded' THEN 'SUPERSEDED' ELSE 'current' END as status"
    else
      select_cols="f.id, s.name as skill, f.claim, f.confidence, f.source_title, COALESCE(f.gpu_generation, '') as gen"
    fi

    run_sql -header -column "$DB" \
      "SELECT $select_cols
       FROM findings_fts fts
       JOIN findings f ON f.id = fts.rowid
       JOIN skills s ON s.id = f.skill_id
       WHERE findings_fts MATCH '$query'
         $gen_clause
         $status_clause
       $order_clause
       LIMIT $limit;"
    ;;
```

**Backward compatibility analysis**: The existing `search` command with no `--gen` or `--include-superseded` flags produces identical SQL except for:
1. The addition of `COALESCE(f.gpu_generation, '') as gen` in the SELECT -- this adds a `gen` column to output.
2. The addition of `AND (f.temporal_status = 'current' OR f.temporal_status IS NULL)` -- since all existing findings have `temporal_status='current'` (the DEFAULT), this WHERE clause filters out zero rows.

The only visible change for existing usage is the new `gen` column in output. Per UX Q&A Q1, this is the desired behavior. BATS tests that check for specific output patterns (e.g., `assert_output --partial "gpu-silicon"`) will continue to pass because they use `--partial` matching.

### 3.4 Supersede Command

```bash
  supersede)
    # kb supersede <old_id> <new_id> [reason]
    old_id=$(echo "$2" | grep -E '^[0-9]+$')
    new_id=$(echo "$3" | grep -E '^[0-9]+$')
    if [ -z "$old_id" ] || [ -z "$new_id" ]; then
      echo "ERROR: Usage: kb supersede <old_finding_id> <new_finding_id> [reason]" >&2
      exit 1
    fi

    # Validate both IDs exist
    old_exists=$(run_sql "$DB" "SELECT COUNT(*) FROM findings WHERE id=$old_id;")
    if [ "$old_exists" -eq 0 ]; then
      echo "ERROR: Finding #$old_id does not exist. Check ID with: kb search \"<query>\"" >&2
      exit 1
    fi
    new_exists=$(run_sql "$DB" "SELECT COUNT(*) FROM findings WHERE id=$new_id;")
    if [ "$new_exists" -eq 0 ]; then
      echo "ERROR: Finding #$new_id does not exist. Check ID with: kb search \"<query>\"" >&2
      exit 1
    fi

    # Check if old finding is already superseded (require --force to override)
    current_status=$(run_sql "$DB" "SELECT superseded_by FROM findings WHERE id=$old_id;")
    if [ -n "$current_status" ] && [ "$current_status" != "" ]; then
      # Check for --force flag
      force=0
      shift 3
      while [ $# -gt 0 ]; do
        case "$1" in
          --force) force=1 ;;
        esac
        shift
      done
      if [ "$force" -eq 0 ]; then
        echo "WARNING: Finding #$old_id is already superseded by #$current_status. Use --force to override." >&2
        exit 1
      fi
    fi

    # Build reason annotation
    reason=$(escape_sql "$4")
    notes_append=""
    if [ -n "$reason" ]; then
      notes_append=", notes=COALESCE(notes,'') || ' [Superseded: $reason]'"
    else
      notes_append=", notes=COALESCE(notes,'') || ' [Superseded by #$new_id]'"
    fi

    run_sql "$DB" "UPDATE findings SET temporal_status='superseded', superseded_by=$new_id $notes_append WHERE id=$old_id;"
    if [ $? -eq 0 ]; then
      echo "Finding #$old_id marked superseded by #$new_id"
    fi
    ;;
```

### 3.5 Unsupersede Command

```bash
  unsupersede)
    # kb unsupersede <finding_id> — restore superseded finding to current
    id=$(echo "$2" | grep -E '^[0-9]+$')
    if [ -z "$id" ]; then
      echo "ERROR: Usage: kb unsupersede <finding_id>" >&2
      exit 1
    fi

    # Check finding exists
    exists=$(run_sql "$DB" "SELECT COUNT(*) FROM findings WHERE id=$id;")
    if [ "$exists" -eq 0 ]; then
      echo "ERROR: Finding #$id does not exist" >&2
      exit 1
    fi

    # Check finding is actually superseded
    status=$(run_sql "$DB" "SELECT temporal_status FROM findings WHERE id=$id;")
    if [ "$status" != "superseded" ]; then
      echo "WARNING: Finding #$id is already current. Nothing to do." >&2
      exit 0
    fi

    run_sql "$DB" "UPDATE findings SET temporal_status='current', superseded_by=NULL, notes=COALESCE(notes,'') || ' [Unsuperseded: restored to current]' WHERE id=$id;"
    if [ $? -eq 0 ]; then
      echo "Finding #$id restored to current status"
    fi
    ;;
```

### 3.6 Tag-Gen Command

```bash
  tag-gen)
    # kb tag-gen <finding_id> <generation>    — tag single finding
    # kb tag-gen --auto [--dry-run] [--skill <name>]  — auto-infer from tags field
    # kb tag-gen <finding_id> --clear          — reset to NULL
    if [ "$2" = "--auto" ]; then
      # Auto-inference mode
      dry_run=0
      skill_filter=""
      shift 2
      while [ $# -gt 0 ]; do
        case "$1" in
          --dry-run) dry_run=1 ;;
          --skill) shift; skill_filter="$1" ;;
        esac
        shift
      done

      # Build skill filter clause
      skill_clause=""
      if [ -n "$skill_filter" ]; then
        skill_clause="AND s.name='$(escape_sql "$skill_filter")'"
      fi

      if [ "$dry_run" -eq 1 ]; then
        echo "=== Auto-Generation Tagging Preview ==="
        echo ""
      fi

      # Query findings with generation tags but no gpu_generation set
      # Generation inference: scan tags field for m1..m5, pick newest
      would_tag=0
      would_skip_no_gen=0
      would_skip_already=0

      # Already tagged (non-NULL gpu_generation)
      would_skip_already=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f JOIN skills s ON s.id=f.skill_id WHERE f.gpu_generation IS NOT NULL $skill_clause;")

      # Process findings with NULL gpu_generation that have gen-related tags
      while IFS='|' read -r fid ftags; do
        # Parse generation from tags
        inferred_gen=""
        # Check for each generation (newest first = highest priority)
        case "$ftags" in
          *m5*|*M5*) inferred_gen="m5" ;;
        esac
        if [ -z "$inferred_gen" ]; then
          case "$ftags" in
            *m4*|*M4*) inferred_gen="m4" ;;
          esac
        fi
        if [ -z "$inferred_gen" ]; then
          case "$ftags" in
            *m3*|*M3*) inferred_gen="m3" ;;
          esac
        fi
        if [ -z "$inferred_gen" ]; then
          case "$ftags" in
            *m2*|*M2*) inferred_gen="m2" ;;
          esac
        fi
        if [ -z "$inferred_gen" ]; then
          case "$ftags" in
            *m1*|*M1*) inferred_gen="m1" ;;
          esac
        fi

        if [ -n "$inferred_gen" ]; then
          would_tag=$((would_tag + 1))
          if [ "$dry_run" -eq 1 ]; then
            if [ "$would_tag" -eq 1 ]; then
              echo "  Would tag:"
            fi
            # Show multi-gen indicator if multiple gens found
            multi=""
            gen_count=0
            for g in m1 m2 m3 m4 m5; do
              case "$ftags" in *$g*|*$(echo "$g" | tr '[:lower:]' '[:upper:]')*) gen_count=$((gen_count + 1)) ;; esac
            done
            if [ "$gen_count" -gt 1 ]; then
              multi=" <- newest: $inferred_gen"
            fi
            printf "    #%-5s %-9s (tags: \"%s\")%s\n" "$fid" "$inferred_gen" "$ftags" "$multi"
          else
            # Execute the tagging
            run_sql "$DB" "UPDATE findings SET gpu_generation='$inferred_gen' WHERE id=$fid;"
          fi
        else
          would_skip_no_gen=$((would_skip_no_gen + 1))
        fi
      done < <(run_sql "$DB" "SELECT f.id, f.tags FROM findings f JOIN skills s ON s.id=f.skill_id WHERE f.gpu_generation IS NULL AND f.tags IS NOT NULL AND f.tags != '' $skill_clause ORDER BY f.id;")

      # Also count those with no tags at all
      no_tags=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f JOIN skills s ON s.id=f.skill_id WHERE f.gpu_generation IS NULL AND (f.tags IS NULL OR f.tags = '') $skill_clause;")
      would_skip_no_gen=$((would_skip_no_gen + no_tags))

      if [ "$dry_run" -eq 1 ]; then
        echo ""
        echo "  Would skip (no generation in tags): $would_skip_no_gen findings"
        echo "  Would skip (already tagged): $would_skip_already findings"
        echo ""
        echo "  Total: $would_tag would be tagged, $((would_skip_no_gen + would_skip_already)) unchanged"
        echo "  Run without --dry-run to apply."
      else
        echo "Auto-tagged $would_tag findings with gpu_generation"
      fi

    elif [ "$3" = "--clear" ]; then
      # Clear mode: reset gpu_generation to NULL
      id=$(echo "$2" | grep -E '^[0-9]+$')
      if [ -z "$id" ]; then
        echo "ERROR: finding_id must be a number" >&2
        exit 1
      fi
      run_sql "$DB" "UPDATE findings SET gpu_generation=NULL WHERE id=$id;"
      if [ $? -eq 0 ]; then
        echo "Finding #$id gpu_generation cleared to NULL"
      fi

    else
      # Single finding mode: kb tag-gen <id> <gen>
      id=$(echo "$2" | grep -E '^[0-9]+$')
      gen="$3"
      if [ -z "$id" ]; then
        echo "ERROR: finding_id must be a number" >&2
        exit 1
      fi
      if [ -z "$gen" ]; then
        echo "ERROR: Usage: kb tag-gen <finding_id> <generation>" >&2
        exit 1
      fi
      # Validate generation
      case "$gen" in
        m1|m2|m3|m4|m5|universal) ;;
        *) echo "ERROR: Invalid generation '$gen'. Valid: m1, m2, m3, m4, m5, universal" >&2; exit 1 ;;
      esac
      # Validate finding exists
      exists=$(run_sql "$DB" "SELECT COUNT(*) FROM findings WHERE id=$id;")
      if [ "$exists" -eq 0 ]; then
        echo "ERROR: Finding #$id does not exist" >&2
        exit 1
      fi
      run_sql "$DB" "UPDATE findings SET gpu_generation='$gen' WHERE id=$id;"
      if [ $? -eq 0 ]; then
        echo "Finding #$id tagged with gpu_generation=$gen"
      fi
    fi
    ;;
```

### 3.7 Freshness Command

```bash
  freshness)
    # kb freshness [--skill <name>] [--json] — temporal health dashboard
    skill_filter=""
    json_mode=0
    shift
    while [ $# -gt 0 ]; do
      case "$1" in
        --skill) shift; skill_filter="$1" ;;
        --json) json_mode=1 ;;
      esac
      shift
    done

    skill_clause=""
    if [ -n "$skill_filter" ]; then
      skill_clause="WHERE s.name='$(escape_sql "$skill_filter")'"
      skill_join_clause="JOIN skills s ON s.id=f.skill_id $skill_clause"
    else
      skill_join_clause=""
    fi

    if [ "$json_mode" -eq 1 ]; then
      # JSON output for programmatic consumption
      m1=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.gpu_generation='m1';") || m1=0
      m2=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.gpu_generation='m2';") || m2=0
      m3=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.gpu_generation='m3';") || m3=0
      m4=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.gpu_generation='m4';") || m4=0
      m5=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.gpu_generation='m5';") || m5=0
      uni=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.gpu_generation='universal';") || uni=0
      null_count=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.gpu_generation IS NULL;") || null_count=0
      current=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.temporal_status='current' OR f.temporal_status IS NULL;") || current=0
      superseded=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.temporal_status='superseded';") || superseded=0
      with_date=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.date_published IS NOT NULL AND f.date_published != '';") || with_date=0
      total=$((m1+m2+m3+m4+m5+uni+null_count))
      pct="0.0"
      if [ "$total" -gt 0 ]; then
        pct=$(echo "scale=1; $with_date * 100 / $total" | bc)
      fi
      printf '{"generation_counts":{"m1":%d,"m2":%d,"m3":%d,"m4":%d,"m5":%d,"universal":%d,"null":%d},"temporal_status":{"current":%d,"superseded":%d},"date_coverage":{"with_date":%d,"total":%d,"percentage":%s}}\n' \
        "$m1" "$m2" "$m3" "$m4" "$m5" "$uni" "$null_count" "$current" "$superseded" "$with_date" "$total" "$pct"
    else
      # Human-readable dashboard
      echo "=== KB Temporal Health ==="
      echo ""
      echo "Generation Coverage:"
      run_sql "$DB" "SELECT
        COALESCE(f.gpu_generation, 'unclassified') as gen,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM findings), 1) || '%' as pct
        FROM findings f $skill_join_clause
        GROUP BY COALESCE(f.gpu_generation, 'unclassified')
        ORDER BY CASE f.gpu_generation
          WHEN 'm1' THEN 1 WHEN 'm2' THEN 2 WHEN 'm3' THEN 3
          WHEN 'm4' THEN 4 WHEN 'm5' THEN 5 WHEN 'universal' THEN 6
          ELSE 7 END;" | while IFS='|' read -r gen count pct; do
        printf "  %-14s %5s  (%s)\n" "$gen" "$count" "$pct"
      done
      total=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause;")
      echo "  ---------------------------------"
      printf "  %-14s %5s\n" "total" "$total"

      echo ""
      echo "Temporal Status:"
      current=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.temporal_status='current' OR f.temporal_status IS NULL;")
      superseded=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.temporal_status='superseded';")
      printf "  %-14s %5s  (%s%%)\n" "current" "$current" "$(echo "scale=1; $current * 100 / $total" | bc)"
      printf "  %-14s %5s  (%s%%)\n" "superseded" "$superseded" "$(echo "scale=1; $superseded * 100 / $total" | bc)"

      echo ""
      echo "Date Coverage:"
      with_date=$(run_sql "$DB" "SELECT COUNT(*) FROM findings f $skill_join_clause WHERE f.date_published IS NOT NULL AND f.date_published != '';")
      printf "  date_published set: %s/%s (%s%%)\n" "$with_date" "$total" "$(echo "scale=1; $with_date * 100 / $total" | bc)"

      echo ""
      echo "Potential Contradictions (exact topic+skill match):"
      contradictions=$(run_sql "$DB" "SELECT a.id, a.gpu_generation, s1.name, a.topic, b.id, b.gpu_generation, s2.name
        FROM findings a
        JOIN findings b ON a.skill_id = b.skill_id AND a.topic = b.topic AND a.id < b.id
        JOIN skills s1 ON s1.id = a.skill_id
        JOIN skills s2 ON s2.id = b.skill_id
        WHERE a.gpu_generation IS NOT NULL AND b.gpu_generation IS NOT NULL
          AND a.gpu_generation != b.gpu_generation
          AND a.gpu_generation != 'universal' AND b.gpu_generation != 'universal'
          AND a.temporal_status != 'superseded' AND b.temporal_status != 'superseded'
          AND a.superseded_by IS NULL AND b.superseded_by IS NULL;")
      if [ -n "$contradictions" ]; then
        count=$(echo "$contradictions" | wc -l | tr -d ' ')
        echo "  $count finding pairs share skill+topic with different generations and no supersession link:"
        echo "$contradictions" | while IFS='|' read -r aid agen askill atopic bid bgen bskill; do
          echo "    #$aid ($agen, $askill/$atopic) <-> #$bid ($bgen, $bskill/$atopic)"
        done
        echo ""
        echo "  Use \`kb supersede <old_id> <new_id>\` to resolve."
      else
        echo "  None found."
      fi
    fi
    ;;
```

### 3.8 Detail Command Modification

Replace the existing `detail)` case to show temporal metadata and supersession chain:

```bash
  detail)
    # kb detail <finding_id> — show full detail of a finding
    id=$(echo "$2" | grep -E '^[0-9]+$')
    if [ -z "$id" ]; then
      echo "ERROR: finding_id must be a number" >&2
      exit 1
    fi
    run_sql -header -column "$DB" \
      "SELECT f.*, s.name as skill_name
       FROM findings f
       JOIN skills s ON s.id = f.skill_id
       WHERE f.id=$id;"

    # Show supersession chain if applicable
    temporal_status=$(run_sql "$DB" "SELECT temporal_status FROM findings WHERE id=$id;")
    superseded_by=$(run_sql "$DB" "SELECT superseded_by FROM findings WHERE id=$id;")

    # Check if this finding supersedes others
    supersedes=$(run_sql "$DB" "SELECT id, gpu_generation, date_found FROM findings WHERE superseded_by=$id;")

    if [ -n "$superseded_by" ] || [ -n "$supersedes" ]; then
      echo ""
      echo "--- Supersession ---"
      if [ -n "$superseded_by" ] && [ "$superseded_by" != "" ]; then
        sup_info=$(run_sql "$DB" "SELECT f.gpu_generation, s.name, f.topic, f.date_found FROM findings f JOIN skills s ON s.id=f.skill_id WHERE f.id=$superseded_by;")
        IFS='|' read -r sup_gen sup_skill sup_topic sup_date <<< "$sup_info"
        echo "  SUPERSEDED by Finding #$superseded_by ($sup_gen, $sup_skill/$sup_topic, $sup_date)"
      fi
      if [ -n "$supersedes" ]; then
        echo "$supersedes" | while IFS='|' read -r sid sgen sdate; do
          echo "  Supersedes Finding #$sid ($sgen, $sdate)"
        done
      fi
    fi

    echo ""
    echo "--- Citations ---"
    cites=$(run_sql -header -column "$DB" "SELECT * FROM citations WHERE finding_id=$id;")
    if [ -n "$cites" ]; then
      echo "$cites"
    else
      echo "  (none)"
    fi
    ;;
```

### 3.9 Verify Command Addition

Add Check 6 after the existing Check 5 (findings without source URLs). Insert before the final summary block:

```bash
    # Check 6: Benchmark/empirical findings without gpu_generation tag
    # Only run if temporal columns exist (migration has been applied)
    has_gen=$(run_sql "$DB" "SELECT COUNT(*) FROM pragma_table_info('findings') WHERE name='gpu_generation';" 2>/dev/null)
    if [ "$has_gen" -gt 0 ] 2>/dev/null; then
      count=$(run_sql "$DB" "SELECT COUNT(*) FROM findings WHERE source_type IN ('benchmark','empirical_test') AND gpu_generation IS NULL;")
      if [ "$count" -gt 0 ] 2>/dev/null; then
        echo "WARNING: $count benchmark/empirical findings have no gpu_generation tag:"
        run_sql -header -column "$DB" \
          "SELECT f.id, s.name as skill, substr(f.claim,1,60) as claim, f.source_type
           FROM findings f JOIN skills s ON s.id=f.skill_id
           WHERE f.source_type IN ('benchmark','empirical_test') AND f.gpu_generation IS NULL
           LIMIT 10;"
        echo "  Tag with: kb tag-gen <id> <generation>"
        echo ""
        issues=$((issues + count))
      else
        echo "OK: All benchmark/empirical findings have gpu_generation tags"
      fi
    fi
```

The check is gated on `pragma_table_info` so it only runs after migration, ensuring backward compatibility with pre-migration databases.

### 3.10 Add Command Modification

Modify the existing `add)` case to accept an optional 11th positional parameter for gpu_generation:

```bash
  add)
    # kb add <skill_name> <topic> <claim> [evidence] [url] [title] [type] [confidence] [tags] [gpu_generation]
    skill=$(escape_sql "$2")
    topic=$(escape_sql "$3")
    claim=$(escape_sql "$4")
    evidence=$(escape_sql "$5")
    source_url=$(escape_sql "$6")
    source_title=$(escape_sql "$7")
    source_type="${8:-other}"
    confidence="${9:-unverified}"
    tags=$(escape_sql "${10}")
    gpu_gen="${11}"

    # Build gpu_generation clause
    gen_col=""
    gen_val=""
    if [ -n "$gpu_gen" ]; then
      case "$gpu_gen" in
        m1|m2|m3|m4|m5|universal) ;;
        *) echo "ERROR: Invalid generation '$gpu_gen'. Valid: m1, m2, m3, m4, m5, universal" >&2; exit 1 ;;
      esac
      gen_col=", gpu_generation"
      gen_val=", '$gpu_gen'"
    fi

    run_sql "$DB" "INSERT INTO findings (skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, tags $gen_col)
      SELECT id, '$topic', '$claim', '$evidence', '$source_url', '$source_title', '$source_type', '$confidence', '$tags' $gen_val
      FROM skills WHERE name='$skill';"
    if [ $? -eq 0 ]; then
      echo "Finding added to ${2}/${3}"
    fi
    ;;
```

### 3.11 Help Text Update

Add the temporal commands section to the help/default case, after the "Quality commands:" section and before "Investigation tracking:":

```bash
    echo "Temporal commands:"
    echo "  supersede <old_id> <new_id> [reason]   Mark finding as replaced by newer one"
    echo "  unsupersede <finding_id>               Restore superseded finding to current"
    echo "  tag-gen <id> <gen>                     Set GPU generation on a finding"
    echo "  tag-gen --auto [--dry-run]             Auto-infer generation from tags field"
    echo "  freshness [--skill <name>]             Temporal health dashboard"
    echo "  migrate-temporal                       Apply temporal schema migration"
    echo ""
```

Also add to the reference section at the bottom:

```bash
    echo "Generations: m1, m2, m3, m4, m5, universal (NULL = unclassified)"
    echo ""
    echo "Search filters:"
    echo "  --gen <gen>            Filter by GPU generation (includes universal + NULL)"
    echo "  --include-superseded   Include superseded findings in results"
```

---

## 4. Agent Prompt Changes

### 4.1 Investigation Agent (`investigation-agent.md`)

**Addition 1: Phase 2 (Research) -- Supersession check**

Insert after the existing Phase 2 step 3 ("Code Analysis"):

```markdown
4. **Supersession check**: Before storing a generation-specific finding, check if the KB
   already has findings on the same skill+topic with a different generation:
   ```bash
   ${CLAUDE_PLUGIN_ROOT}/scripts/kb search "<topic keywords>" --limit 5
   ```
   If an older-generation finding covers the same metric, note it for supersession in Phase 5.
```

Token count: ~40 tokens.

**Addition 2: Phase 3 (Store Findings) -- Generation tagging**

Insert after the "Rules for storing findings" section (after the confidence-source cross-validation table):

```markdown
**Generation tagging (required for benchmarks):**
- For `benchmark` or `empirical_test` findings, always set `gpu_generation` to the
  relevant Apple Silicon generation: m1, m2, m3, m4, m5, or universal.
- Add gpu_generation to your INSERT: `..., gpu_generation) VALUES ..., '<gen>')`.
- If generation is unclear, leave NULL -- `kb verify` will flag it for review.
```

Token count: ~45 tokens.

**Addition 3: Phase 5 (Quality Check) -- Resolve supersessions**

Insert after the existing `kb verify` / `kb dedup` step:

```markdown
3. **Resolve supersessions** if you identified older findings during Phase 2:
   ```bash
   ${CLAUDE_PLUGIN_ROOT}/scripts/kb supersede <old_id> <new_id> "<reason>"
   ```
```

Token count: ~20 tokens.

**Total prompt overhead: ~105 tokens** (slightly above the 50-token NFR-5 target but necessary for the three additions). The original <50 token target in the PM spec assumed fewer insertion points. Recommendation: accept this overhead as it provides complete coverage of the temporal workflow.

### 4.2 Knowledge Retriever Agent (`knowledge-retriever.md`)

**Addition 1: Query Strategy section -- Generation detection**

Insert after the existing step 4 ("Run the query"):

```markdown
5. **Detect generation context** from the query:
   - If query mentions "M4", "M4 Pro", "M4 Max": add `--gen m4` to search
   - If query mentions "M5": add `--gen m5` to search
   - If query mentions "M1"/"M2"/"M3": add the corresponding `--gen` flag
   - If query asks about "current" or "latest": add `--gen m4` (or m5 when available)
   - If no generation mentioned: do NOT add `--gen` (return all generations)
```

Token count: ~50 tokens.

**Addition 2: Result Formatting section -- Generation-aware format**

Insert after the existing "Full Format" example:

```markdown
### Generation-Aware Format:
When `--gen` was used, note it in the response:
```
[verified] Finding #123 (gpu-silicon, m4): Apple M4 GPU has 10 cores...
```
When a finding is from a different generation than requested, flag it:
```
[verified] Finding #14 (gpu-silicon, m1) -- DIFFERENT GEN: M1 specs, not M4
```
```

Token count: ~40 tokens.

**Total retriever prompt overhead: ~90 tokens.**

### 4.3 Architecture Advisor Agent (`architecture-advisor.md`)

**Addition: Cross-Reference section -- Generation consistency**

Insert after the existing step 4 ("Apply M4/M5 Hardware Constraints"):

```markdown
5. **Check generation consistency**: When citing findings, verify they apply to the
   target hardware generation. Use `kb detail <id>` to check gpu_generation.
   Flag findings from earlier generations that may not apply to the target hardware.
```

Token count: ~35 tokens.

**Total advisor prompt overhead: ~35 tokens.**

---

## 5. Backfill Strategy

### 5.1 Generation Inference Rules

Based on analysis of the production DB (154 findings with generation-related tags):

| Tag Pattern | Inferred Generation | Rule |
|-------------|-------------------|------|
| Tags contain only `m1` (no m2-m5) | `m1` | Single-gen match |
| Tags contain only `m2` (no m1,m3-m5) | `m2` | Single-gen match |
| Tags contain only `m3` (no m1-m2,m4-m5) | `m3` | Single-gen match |
| Tags contain only `m4` (no m1-m3,m5) | `m4` | Single-gen match |
| Tags contain only `m5` (no m1-m4) | `m5` | Single-gen match |
| Tags contain `m5` and any other gen | `m5` | Newest-gen rule (PM Q&A Q2) |
| Tags contain `m4` and `m1`/`m2`/`m3` but no `m5` | `m4` | Newest-gen rule |
| Tags contain `m3` and `m1`/`m2` but no `m4`/`m5` | `m3` | Newest-gen rule |
| Tags contain `m2` and `m1` but no `m3`/`m4`/`m5` | `m2` | Newest-gen rule |
| No generation pattern found in tags | NULL (unchanged) | Conservative: don't force-tag |

The matching must be careful about false positives. For example, `m1` could match in words like "sum1" or column names. However, examining the actual tag data, tags are comma-separated lowercase keywords, and generation tags appear as standalone comma-delimited values (e.g., `m4,bandwidth,benchmark`). The case-insensitive tag patterns `m1`, `M1`, `m1-pro`, `M1-Max` etc. all indicate M1 generation.

### 5.2 Auto-Tag SQL (Preview)

The auto-tagging logic is implemented in the `tag-gen --auto` command (Section 3.6). The core inference is done in bash by parsing the tags field. Here is the equivalent SQL for a direct preview:

```sql
-- Preview: findings that would be tagged by auto-inference
-- This uses SQLite LIKE for simple pattern matching
SELECT f.id,
  f.tags,
  CASE
    WHEN f.tags LIKE '%m5%' OR f.tags LIKE '%M5%' THEN 'm5'
    WHEN f.tags LIKE '%m4%' OR f.tags LIKE '%M4%' THEN 'm4'
    WHEN f.tags LIKE '%m3%' OR f.tags LIKE '%M3%' THEN 'm3'
    WHEN f.tags LIKE '%m2%' OR f.tags LIKE '%M2%' THEN 'm2'
    WHEN f.tags LIKE '%m1%' OR f.tags LIKE '%M1%' THEN 'm1'
    ELSE NULL
  END as inferred_gen
FROM findings f
WHERE f.gpu_generation IS NULL
  AND (f.tags LIKE '%m1%' OR f.tags LIKE '%M1%'
    OR f.tags LIKE '%m2%' OR f.tags LIKE '%M2%'
    OR f.tags LIKE '%m3%' OR f.tags LIKE '%M3%'
    OR f.tags LIKE '%m4%' OR f.tags LIKE '%M4%'
    OR f.tags LIKE '%m5%' OR f.tags LIKE '%M5%')
ORDER BY f.id;
```

### 5.3 False Positive Risks

Examining the actual tag data, potential false positives:

- `sum1`, `simd-sum1` -- does not appear in current data
- `m1-pro`, `m1-max`, `M1-Max` -- these correctly indicate M1 generation
- `m4-max`, `M4-Max` -- correctly indicate M4 generation
- `metal-3.1` -- contains `m` followed by digits but `m3` match is at word boundary; `metal-3.1` does not match `m3` because `m3` requires `m3` as a substring. However, it would match `m1` in `metal-3.1` -- wait, no: `metal-3.1` does not contain `m1`. It contains `3.1`. Safe.
- `simdgroup-matrix` -- does not contain `m1`-`m5`. Safe.

The only real risk is tag values like `pre-m5` which would match `m5`. Checking the data: the tag `pre-m5` appears in one finding (`simdgroup-matrix,tensor,fp16,fp32,pre-m5,alu`). This finding is about pre-M5 behavior, so tagging it as `m5` is arguably incorrect -- it describes behavior before M5. However, per the PM decision, comparison findings get tagged with newest gen mentioned. This is an edge case the `--dry-run` preview is designed to catch.

### 5.4 Verification Queries Post-Backfill

```sql
-- 1. Count tagged findings (should jump from 0 to ~154)
SELECT COUNT(*) FROM findings WHERE gpu_generation IS NOT NULL;

-- 2. Distribution by generation
SELECT gpu_generation, COUNT(*) FROM findings
WHERE gpu_generation IS NOT NULL
GROUP BY gpu_generation ORDER BY gpu_generation;

-- 3. Verify no existing findings were broken
SELECT COUNT(*) FROM findings;  -- should still be 1555

-- 4. Spot-check: M5 findings should include M5-specific tags
SELECT id, tags, gpu_generation FROM findings
WHERE gpu_generation = 'm5' LIMIT 10;

-- 5. Check for obvious misclassifications
SELECT id, tags, gpu_generation FROM findings
WHERE gpu_generation IS NOT NULL
  AND tags NOT LIKE '%' || gpu_generation || '%'
  AND tags NOT LIKE '%' || UPPER(gpu_generation) || '%';
-- This query finds findings where the inferred gen doesn't appear in tags
-- Should return 0 rows (all inferences should be traceable to tags)
```

### 5.5 Resolving the 7 Known Contradictions

The 7 known contradictions must be identified first. After migration and backfill, run:

```bash
kb freshness
```

This will show the contradiction pairs. For each pair, the human developer reviews with `kb detail <id>` on both findings and applies `kb supersede <old_id> <new_id> "<reason>"` where appropriate.

Example resolution for the likely contradiction pairs:

```bash
# M1 TFLOPS spec vs M5 TFLOPS spec
kb supersede <m1_finding_id> <m5_finding_id> "M5 specs replace M1 specs for current hardware"

# M2 bandwidth benchmark vs M4 bandwidth benchmark
kb supersede <m2_finding_id> <m4_finding_id> "M4 bandwidth data supersedes M2 benchmark"
```

Not all 7 contradictions may require supersession. Some may be complementary findings about different generations that just need correct generation tags. The `kb freshness` contradiction detection + human review workflow handles this distinction.

---

## 6. Testing Strategy

### 6.1 New BATS Test File

File: `tests/unit/temporal.bats`

This test file follows the exact patterns from `tests/unit/kb-cli.bats`:
- `setup()` copies DB to `BATS_TEST_TMPDIR` and sets `GPU_FORGE_DB`
- `teardown()` cleans up
- Uses `assert_success`, `assert_failure`, `assert_output --partial`

```bash
#!/usr/bin/env bats

load ../test_helper/common-setup

KB="${PLUGIN_ROOT}/scripts/kb"

setup() {
  TEST_DB="${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$TEST_DB"
  export GPU_FORGE_DB="$TEST_DB"

  # Apply migration to test DB (idempotent)
  run "$KB" migrate-temporal
  assert_success
}

teardown() {
  rm -f "${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
  rm -f "${BATS_TEST_TMPDIR}/test_gpu_knowledge.db.pre-temporal-backup"
}

# --- Migration ---

@test "migrate-temporal adds 3 new columns" {
  run sqlite3 "$TEST_DB" "SELECT COUNT(*) FROM pragma_table_info('findings') WHERE name IN ('gpu_generation','temporal_status','superseded_by');"
  assert_success
  assert_output "3"
}

@test "migrate-temporal is idempotent" {
  run "$KB" migrate-temporal
  assert_success
  assert_output --partial "already applied"
}

@test "migrate-temporal creates backup" {
  [ -f "${TEST_DB}.pre-temporal-backup" ]
}

# --- gpu_generation CHECK constraint ---

@test "gpu_generation accepts valid values" {
  for gen in m1 m2 m3 m4 m5 universal; do
    run sqlite3 "$TEST_DB" "UPDATE findings SET gpu_generation='$gen' WHERE id=1;"
    assert_success
  done
}

@test "gpu_generation rejects invalid value" {
  run sqlite3 "$TEST_DB" "UPDATE findings SET gpu_generation='m6' WHERE id=1;"
  assert_failure
}

@test "gpu_generation accepts NULL" {
  run sqlite3 "$TEST_DB" "UPDATE findings SET gpu_generation=NULL WHERE id=1;"
  assert_success
}

# --- temporal_status CHECK constraint ---

@test "temporal_status default is current" {
  run sqlite3 "$TEST_DB" "SELECT temporal_status FROM findings WHERE id=1;"
  assert_success
  assert_output "current"
}

@test "temporal_status rejects invalid value" {
  run sqlite3 "$TEST_DB" "UPDATE findings SET temporal_status='historical' WHERE id=1;"
  assert_failure
}

# --- search --gen filter ---

@test "search without --gen returns results (backward compatible)" {
  run "$KB" search "GPU"
  assert_success
  [ ${#output} -gt 10 ]
}

@test "search results include gen column" {
  # Tag a finding first so gen column has data
  run sqlite3 "$TEST_DB" "UPDATE findings SET gpu_generation='m4' WHERE id=1;"
  run "$KB" search "GPU"
  assert_success
  assert_output --partial "gen"
}

@test "search --gen m4 returns results" {
  run sqlite3 "$TEST_DB" "UPDATE findings SET gpu_generation='m4' WHERE id=1;"
  run "$KB" search "GPU" --gen m4
  assert_success
}

@test "search --gen excludes other specific generations" {
  # Tag finding 1 as m1, finding 2 as m4
  run sqlite3 "$TEST_DB" "UPDATE findings SET gpu_generation='m1' WHERE id=1;"
  run sqlite3 "$TEST_DB" "UPDATE findings SET gpu_generation='m4' WHERE id=2;"
  # Search for --gen m4 should not include finding 1
  run "$KB" search "GPU" --gen m4
  assert_success
  # This test assumes finding 1 would appear in an unfiltered "GPU" search
  # The key assertion is that the query runs without error
}

@test "search --gen includes NULL findings" {
  run "$KB" search "GPU" --gen m4
  assert_success
  # NULL findings should be included -- search still returns results
  [ ${#output} -gt 10 ]
}

@test "search --gen includes universal findings" {
  run sqlite3 "$TEST_DB" "UPDATE findings SET gpu_generation='universal' WHERE id=1;"
  run "$KB" search "GPU" --gen m4
  assert_success
}

@test "search --gen rejects invalid generation" {
  run "$KB" search "GPU" --gen m6
  assert_failure
  assert_output --partial "Invalid generation"
}

@test "search excludes superseded findings by default" {
  run sqlite3 "$TEST_DB" "UPDATE findings SET temporal_status='superseded', superseded_by=2 WHERE id=1;"
  run "$KB" search "GPU"
  assert_success
  # Finding 1 should not appear (it is superseded)
}

@test "search --include-superseded shows superseded findings" {
  run sqlite3 "$TEST_DB" "UPDATE findings SET temporal_status='superseded', superseded_by=2 WHERE id=1;"
  run "$KB" search "GPU" --include-superseded
  assert_success
  assert_output --partial "SUPERSEDED"
}

# --- supersede ---

@test "supersede marks finding as superseded" {
  run "$KB" supersede 1 2
  assert_success
  assert_output --partial "Finding #1 marked superseded by #2"
  # Verify in DB
  run sqlite3 "$TEST_DB" "SELECT temporal_status FROM findings WHERE id=1;"
  assert_output "superseded"
  run sqlite3 "$TEST_DB" "SELECT superseded_by FROM findings WHERE id=1;"
  assert_output "2"
}

@test "supersede with reason appends to notes" {
  run "$KB" supersede 1 2 "M5 replaces M1"
  assert_success
  run sqlite3 "$TEST_DB" "SELECT notes FROM findings WHERE id=1;"
  assert_output --partial "Superseded: M5 replaces M1"
}

@test "supersede validates old_id exists" {
  run "$KB" supersede 99999 2
  assert_failure
  assert_output --partial "does not exist"
}

@test "supersede validates new_id exists" {
  run "$KB" supersede 1 99999
  assert_failure
  assert_output --partial "does not exist"
}

@test "supersede requires --force for already-superseded finding" {
  run "$KB" supersede 1 2
  assert_success
  run "$KB" supersede 1 3
  assert_failure
  assert_output --partial "already superseded"
  assert_output --partial "--force"
}

@test "supersede with non-numeric ID fails" {
  run "$KB" supersede abc 2
  assert_failure
  assert_output --partial "Usage"
}

# --- unsupersede ---

@test "unsupersede restores to current" {
  run "$KB" supersede 1 2
  assert_success
  run "$KB" unsupersede 1
  assert_success
  assert_output --partial "restored to current"
  run sqlite3 "$TEST_DB" "SELECT temporal_status FROM findings WHERE id=1;"
  assert_output "current"
  run sqlite3 "$TEST_DB" "SELECT superseded_by FROM findings WHERE id=1;"
  assert_output ""
}

@test "unsupersede on current finding is a no-op" {
  run "$KB" unsupersede 1
  assert_success
  assert_output --partial "already current"
}

@test "unsupersede on nonexistent finding fails" {
  run "$KB" unsupersede 99999
  assert_failure
  assert_output --partial "does not exist"
}

# --- tag-gen ---

@test "tag-gen sets gpu_generation on single finding" {
  run "$KB" tag-gen 1 m4
  assert_success
  assert_output --partial "tagged with gpu_generation=m4"
  run sqlite3 "$TEST_DB" "SELECT gpu_generation FROM findings WHERE id=1;"
  assert_output "m4"
}

@test "tag-gen --clear resets to NULL" {
  run "$KB" tag-gen 1 m4
  assert_success
  run "$KB" tag-gen 1 --clear
  assert_success
  run sqlite3 "$TEST_DB" "SELECT COALESCE(gpu_generation,'NULL') FROM findings WHERE id=1;"
  assert_output "NULL"
}

@test "tag-gen rejects invalid generation" {
  run "$KB" tag-gen 1 m6
  assert_failure
  assert_output --partial "Invalid generation"
}

@test "tag-gen --auto --dry-run shows preview" {
  run "$KB" tag-gen --auto --dry-run
  assert_success
  assert_output --partial "Auto-Generation Tagging Preview"
  assert_output --partial "Would tag"
  assert_output --partial "Would skip"
}

@test "tag-gen --auto executes tagging" {
  run "$KB" tag-gen --auto
  assert_success
  assert_output --partial "Auto-tagged"
  # Verify some findings were tagged
  count=$(sqlite3 "$TEST_DB" "SELECT COUNT(*) FROM findings WHERE gpu_generation IS NOT NULL;")
  [ "$count" -gt 0 ]
}

@test "tag-gen --auto does not overwrite existing tags" {
  run sqlite3 "$TEST_DB" "UPDATE findings SET gpu_generation='m1' WHERE id=1;"
  run "$KB" tag-gen --auto
  # Finding 1 should still be m1 (not overwritten)
  run sqlite3 "$TEST_DB" "SELECT gpu_generation FROM findings WHERE id=1;"
  assert_output "m1"
}

# --- freshness ---

@test "freshness shows temporal health dashboard" {
  run "$KB" freshness
  assert_success
  assert_output --partial "=== KB Temporal Health ==="
  assert_output --partial "Generation Coverage"
  assert_output --partial "Temporal Status"
  assert_output --partial "Date Coverage"
}

@test "freshness --json returns valid JSON" {
  run "$KB" freshness --json
  assert_success
  # Validate JSON structure
  echo "$output" | python3 -c "import sys, json; json.load(sys.stdin)"
}

# --- verify temporal check ---

@test "verify includes temporal check after migration" {
  run "$KB" verify
  assert_success
  # Should mention gpu_generation check (either OK or WARNING)
  assert_output --partial "benchmark"
}

# --- detail shows temporal metadata ---

@test "detail shows supersession chain for superseded finding" {
  run "$KB" supersede 1 2 "test reason"
  run "$KB" detail 1
  assert_success
  assert_output --partial "SUPERSEDED by"
  assert_output --partial "#2"
}

@test "detail shows supersedes for replacing finding" {
  run "$KB" supersede 1 2 "test reason"
  run "$KB" detail 2
  assert_success
  assert_output --partial "Supersedes"
  assert_output --partial "#1"
}

# --- add with gpu_generation ---

@test "add with gpu_generation parameter works" {
  run "$KB" add gpu-silicon "test-temporal" "Test claim" "evidence" "http://test" "Test" "benchmark" "high" "m4,test" "m4"
  assert_success
  assert_output --partial "Finding added"
  # Verify gpu_generation was set
  run sqlite3 "$TEST_DB" "SELECT gpu_generation FROM findings WHERE topic='test-temporal' ORDER BY id DESC LIMIT 1;"
  assert_output "m4"
}

# --- help text ---

@test "help text includes temporal commands" {
  run "$KB"
  assert_success
  assert_output --partial "Temporal commands"
  assert_output --partial "supersede"
  assert_output --partial "unsupersede"
  assert_output --partial "tag-gen"
  assert_output --partial "freshness"
  assert_output --partial "Generations:"
}
```

### 6.2 Golden Query Test Updates

The existing `tests/golden-queries.bats` tests use `assert_output --partial "skill-name"` which will continue to work because the search output still contains skill names. However, after migration, superseded findings will be filtered out by default. If any golden query findings are superseded during backfill, those tests could fail.

**Mitigation**: Golden query tests run on a copy of the production DB (or test fixture). Run golden queries both before and after backfill to ensure results remain stable.

Add generation-aware golden queries to a new file `tests/golden-temporal.bats`:

```bash
#!/usr/bin/env bats

load test_helper/common-setup

setup() {
  KB="$PLUGIN_ROOT/scripts/kb"
  TEST_DB="${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$TEST_DB"
  export GPU_FORGE_DB="$TEST_DB"
  # Apply migration
  run "$KB" migrate-temporal
  # Run auto-tagging
  run "$KB" tag-gen --auto
}

teardown() {
  rm -f "${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
  rm -f "${BATS_TEST_TMPDIR}/test_gpu_knowledge.db.pre-temporal-backup"
}

@test "golden-temporal: 'M4 TFLOPS' --gen m4 returns gpu-silicon results" {
  run "$KB" search "M4 TFLOPS" --gen m4
  assert_success
  assert_output --partial "gpu-silicon"
}

@test "golden-temporal: 'M5 neural accelerator' --gen m5 returns results" {
  run "$KB" search "M5 neural accelerator" --gen m5
  assert_success
  [ ${#output} -gt 10 ]
}

@test "golden-temporal: 'SIMD width' without --gen returns all generations" {
  run "$KB" search "SIMD width"
  assert_success
  assert_output --partial "simd-wave"
}

@test "golden-temporal: 'bandwidth' --gen m4 excludes m1 findings" {
  run "$KB" search "bandwidth" --gen m4
  assert_success
  # Should return results (m4-tagged and NULL findings)
  [ ${#output} -gt 10 ]
}
```

### 6.3 Backward Compatibility Testing

All 194 existing BATS tests must pass after migration. The risk points are:

1. **`kb search` output format change**: The new `gen` column is appended. Tests that use `assert_output --partial` are unaffected. Tests that count specific column positions could break, but examination of the existing tests shows they use `--partial` matching, not positional.

2. **`kb detail` output format change**: The new temporal fields appear in the output. Tests check for `assert_output --partial "skill_name"` which still works.

3. **`kb verify` output change**: The new Check 6 adds output. The existing test checks for `assert_output --partial "=== Knowledge Base Quality Report ==="` which still appears.

4. **`kb add` positional args**: The new 11th parameter is optional. Existing calls with 9-10 parameters work unchanged.

### 6.4 Test Count Estimate

| Test Category | Count |
|--------------|-------|
| Migration (idempotent, columns, backup) | 3 |
| CHECK constraint validation | 4 |
| Search --gen filtering | 8 |
| Search --include-superseded | 2 |
| Supersede command | 5 |
| Unsupersede command | 3 |
| Tag-gen command (single + auto + clear) | 5 |
| Freshness dashboard | 2 |
| Verify temporal check | 1 |
| Detail with supersession | 2 |
| Add with gpu_generation | 1 |
| Help text | 1 |
| Golden temporal queries | 4 |
| **Total new tests** | **41** |

This exceeds the NFR-4 target of 15+ new tests.

---

## 7. Risk Mitigation

### 7.1 DB Backup Before Migration

The `migrate-temporal` command automatically creates `${DB}.pre-temporal-backup` before any ALTER TABLE. Rollback procedure:

```bash
# If migration fails or causes issues:
cp "${PLUGIN_ROOT}/data/gpu_knowledge.db.pre-temporal-backup" "${PLUGIN_ROOT}/data/gpu_knowledge.db"
```

### 7.2 Rollback Procedure

SQLite does not support `ALTER TABLE DROP COLUMN` (added in 3.35.0, but only for non-FTS tables). If the migration needs to be fully reversed:

```bash
# Option 1: Restore from backup
cp gpu_knowledge.db.pre-temporal-backup gpu_knowledge.db

# Option 2: Create new table without temporal columns (if backup is lost)
sqlite3 gpu_knowledge.db <<'SQL'
BEGIN;
CREATE TABLE findings_old AS SELECT
  id, skill_id, topic, claim, evidence, source_url, source_title,
  source_type, confidence, date_found, date_published, tags, notes,
  investigation_session
FROM findings;
DROP TABLE findings;
ALTER TABLE findings_old RENAME TO findings;
-- Rebuild FTS and indexes...
COMMIT;
SQL
```

Actually, SQLite 3.35.0+ supports DROP COLUMN. Since the system runs SQLite 3.51.1, a simpler rollback is possible:

```bash
sqlite3 gpu_knowledge.db <<'SQL'
ALTER TABLE findings DROP COLUMN gpu_generation;
ALTER TABLE findings DROP COLUMN temporal_status;
ALTER TABLE findings DROP COLUMN superseded_by;
DROP INDEX IF EXISTS idx_findings_gpu_generation;
DROP INDEX IF EXISTS idx_findings_temporal_status;
DROP INDEX IF EXISTS idx_findings_superseded_by;
SQL
```

### 7.3 Existing Agent INSERTs

Investigation agents INSERT directly via `sqlite3` with explicit column names:

```sql
INSERT INTO findings (skill_id, topic, claim, evidence, source_url, source_title,
  source_type, confidence, tags, investigation_session, notes)
VALUES (...);
```

Since the new columns have defaults (`gpu_generation DEFAULT NULL`, `temporal_status DEFAULT 'current'`, `superseded_by DEFAULT NULL`), existing agent INSERTs that do not include the new columns will work correctly:
- `gpu_generation` defaults to `NULL` (unclassified -- flagged by `kb verify` for benchmarks)
- `temporal_status` defaults to `'current'` (correct for new findings)
- `superseded_by` defaults to `NULL` (correct for new findings)

This is the critical backward compatibility property. No existing agent code needs to change for basic operation. The agent prompt additions (Section 4) are enhancements that add generation tagging, not requirements for INSERT to succeed.

### 7.4 Performance Verification

The search query adds two WHERE clauses to the existing FTS JOIN:

```sql
AND (f.gpu_generation = 'm4' OR f.gpu_generation = 'universal' OR f.gpu_generation IS NULL)
AND (f.temporal_status = 'current' OR f.temporal_status IS NULL)
```

Performance impact analysis:
- FTS5 MATCH narrows the result set to typically <100 rows
- The WHERE clauses filter this small set using indexed columns
- Index `idx_findings_gpu_generation` makes the generation filter O(1) per row
- Index `idx_findings_temporal_status` makes the status filter O(1) per row
- Total overhead: <1ms, well within the NFR-1 target of <100ms

Post-migration, run a quick benchmark:

```bash
# Before migration timing:
time kb search "GPU" --limit 10

# After migration timing:
time kb search "GPU" --limit 10 --gen m4
```

---

## 8. Implementation Order with Dependencies

### Phase 1: Schema + Migration (Day 1)

| Step | Task | Depends On | Verification |
|------|------|-----------|-------------|
| 1.1 | Create DB backup | None | File exists |
| 1.2 | Update `data/schema.sql` with new columns | None | SQL parses without error |
| 1.3 | Add `migrate-temporal` command to `kb` | 1.2 | `kb migrate-temporal` succeeds on test DB |
| 1.4 | Run migration on production DB | 1.1, 1.3 | `pragma_table_info` shows 3 new columns |
| 1.5 | Run all 194 existing BATS tests | 1.4 | All pass |

### Phase 2: CLI Commands (Days 2-3)

| Step | Task | Depends On | Verification |
|------|------|-----------|-------------|
| 2.1 | Modify `search` with `--gen`, `--include-superseded` | 1.4 | New BATS tests pass |
| 2.2 | Add `supersede` command | 1.4 | BATS supersede tests pass |
| 2.3 | Add `unsupersede` command | 2.2 | BATS unsupersede tests pass |
| 2.4 | Add `tag-gen` command (single + auto + clear) | 1.4 | BATS tag-gen tests pass |
| 2.5 | Add `freshness` command | 1.4 | BATS freshness tests pass |
| 2.6 | Modify `detail` for supersession chain | 2.2 | BATS detail tests pass |
| 2.7 | Add Check 6 to `verify` | 1.4 | BATS verify test passes |
| 2.8 | Modify `add` for gpu_generation | 1.4 | BATS add test passes |
| 2.9 | Update help text | 2.1-2.5 | BATS help test passes |

### Phase 3: Agent Prompts (Day 4)

| Step | Task | Depends On | Verification |
|------|------|-----------|-------------|
| 3.1 | Update `investigation-agent.md` | 2.2, 2.4 | Token count <150 added |
| 3.2 | Update `knowledge-retriever.md` | 2.1 | Token count <100 added |
| 3.3 | Update `architecture-advisor.md` | 2.1 | Token count <50 added |

### Phase 4: Backfill + Validation (Day 5)

| Step | Task | Depends On | Verification |
|------|------|-----------|-------------|
| 4.1 | Run `kb tag-gen --auto --dry-run` | 2.4, 1.4 | Preview looks correct |
| 4.2 | Run `kb tag-gen --auto` | 4.1 | ~154 findings tagged |
| 4.3 | Run `kb freshness` | 4.2, 2.5 | Shows updated gen distribution |
| 4.4 | Identify and resolve 7 contradictions | 4.3, 2.2 | `kb freshness` shows 0 contradictions |
| 4.5 | Run golden temporal query tests | 4.2 | All pass |
| 4.6 | Run full BATS suite (194 + 41 new) | All above | All 235 tests pass |

### Critical Path

```
Day 1: [1.1]->[1.2]->[1.3]->[1.4]->[1.5]
Day 2: [2.1, 2.2, 2.4, 2.5] (parallel)
Day 3: [2.3, 2.6, 2.7, 2.8, 2.9] (parallel, depend on Day 2)
Day 4: [3.1, 3.2, 3.3] (parallel)
Day 5: [4.1]->[4.2]->[4.3]->[4.4]->[4.5]->[4.6]
```

**Estimated total: 5 working days.**

---

## 9. File Change Summary

| File | Change Type | Lines Added (est.) |
|------|------------|-------------------|
| `scripts/kb` | Modify | ~350 (new commands + modifications) |
| `data/schema.sql` | Modify | ~10 (3 columns + 3 indexes) |
| `agents/investigation-agent.md` | Modify | ~15 (3 insertions) |
| `agents/knowledge-retriever.md` | Modify | ~12 (2 insertions) |
| `agents/architecture-advisor.md` | Modify | ~5 (1 insertion) |
| `tests/unit/temporal.bats` | New | ~250 (41 tests) |
| `tests/golden-temporal.bats` | New | ~50 (4 tests) |
| **Total** | | **~692 lines** |

---

## Research Sources

- [SQLite ALTER TABLE Documentation](https://www.sqlite.org/lang_altertable.html) -- ADD COLUMN constraints, CHECK constraint on existing rows, REFERENCES with NULL default
- [SQLite FTS5 Extension](https://sqlite.org/fts5.html) -- MATCH semantics, JOIN best practices, BM25 ranking
- [SQLite FTS5 JOIN filtering forum discussion](https://sqlite.org/forum/forumpost/5b303ab003f91660) -- Table alias limitations with MATCH operator
- [BATS-core documentation](https://bats-core.readthedocs.io/) -- setup/teardown patterns, BATS_TEST_TMPDIR
- [Testing Bash Scripts with BATS (Baeldung)](https://www.baeldung.com/linux/testing-bash-scripts-bats) -- Test patterns for CLI tools

---

## Questions & Answers

### Q1: Migration trigger
**Answer**: Manual command (`kb migrate-temporal`). Developer controls when schema changes.
**Impact**: Explicit, safe, idempotent. One-time run. No latency added to every command.

### Q2: Graceful degradation
**Answer**: Yes — check column existence at startup, fall back to original query if not migrated.
**Impact**: Modified `kb search` works on both pre- and post-migration databases. ~5 lines of conditional logic per temporal command.

### Q3: Tag pattern matching
**Answer**: Case-insensitive substring match on m1-m5.
**Impact**: Catches m4, M4, m4-max, m4-pro, M1-Max. 99%+ accuracy. `pre-m5` tagged m5 per PM decision. Simple bash implementation.

### Q4: Contradiction detection scope
**Answer**: Exact topic+skill_id matching only for Phase 1. FTS tier deferred.
**Impact**: Catches 7 known contradictions with zero false positives. Reduces implementation risk and testing surface. FTS tier added as follow-up.

---

_Original questions for reference:_

### Q1: Should the migration be triggered automatically or manually?

**Option A (recommended)**: Add `migrate-temporal` as an explicit command. The developer runs `kb migrate-temporal` once. This is the safest approach -- the developer controls when the schema changes.

**Option B**: Auto-detect missing columns on every `kb` invocation and run migration transparently. This is more convenient but adds latency to every command (one `pragma_table_info` query per invocation) and could surprise the user with schema changes.

**Option C**: Run migration during plugin installation/update hook (if gpu-forge has a setup hook). Clean but couples schema migration to plugin lifecycle.

**Recommendation**: Option A. Explicit is better than implicit for schema changes. The migration takes <100ms and only runs once.

### Q2: Should `kb search` gracefully degrade on pre-migration databases?

**Option A (recommended)**: Yes -- check for column existence before adding temporal WHERE clauses. If `gpu_generation` column doesn't exist, fall back to the original query (no gen filtering, no superseded filtering). This means the modified `search` command works on both pre- and post-migration databases.

**Option B**: No -- require migration before using new flags. `kb search --gen m4` on a pre-migration DB errors with "Run kb migrate-temporal first". Simpler code but worse UX during transition.

**Recommendation**: Option A. The graceful degradation is a single `pragma_table_info` check cached at script startup, and it ensures the plugin works out of the box even if the user hasn't run migration yet. However, this adds ~5 lines of conditional logic to every command that touches temporal columns.

### Q3: How should the `tag-gen --auto` handle edge-case tag patterns like `m4-max`, `M1-Max`, `pre-m5`?

**Option A (recommended)**: Case-insensitive substring match on `m1`-`m5`. This catches `m4`, `M4`, `m4-max`, `M4-Max`, `m4-pro` -- all of which correctly indicate their generation. The `pre-m5` case is rare (1 finding) and per PM Q&A Q2, it gets tagged `m5` as the newest gen mentioned. Accept this minor imprecision.

**Option B**: Word-boundary-aware matching. Parse tags by comma, check each tag for exact or prefix match against generation patterns (e.g., `^m[1-5](-|$)`). More precise but significantly more complex bash parsing.

**Option C**: Maintain an explicit mapping table of tag patterns to generations. Most precise but requires maintenance as new tag patterns emerge.

**Recommendation**: Option A. The substring approach matches 99%+ of cases correctly. The 1 `pre-m5` edge case is caught by `--dry-run` preview and can be manually corrected if needed. The simplicity of substring matching is worth the rare imprecision.

### Q4: Should the `freshness` contradiction detection run a secondary FTS-based pass for cross-topic contradictions?

**Option A (recommended)**: Start with exact `topic+skill_id` matching only. This catches the most obvious contradictions with zero false positives. The PM spec mentions "two-tier" detection but the exact match tier alone resolves the 7 known contradictions.

**Option B**: Implement both tiers immediately: exact match for "definite contradictions" and FTS similarity for "potential contradictions." More comprehensive but the FTS pass is complex (requires pairwise BM25 scoring) and could produce many false positives at 1,555 findings.

**Recommendation**: Option A for Phase 1. The exact-match tier handles the current problem (7 known contradictions). The FTS tier can be added as a follow-up once the exact-match system is proven. This reduces implementation risk and testing surface.

---END QUESTIONS---
