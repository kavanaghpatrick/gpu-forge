#!/bin/bash
# Bootstrap the GPU Knowledge Database
# Creates an empty DB from schema.sql if one doesn't exist, seeds all 11 skills.
# Usage: bootstrap-db.sh

set -euo pipefail

# 3-level DB path resolution:
# 1. GPU_FORGE_DB env var (highest priority)
# 2. ${CLAUDE_PLUGIN_ROOT}/data/gpu_knowledge.db (plugin runtime)
# 3. $(dirname "$0")/../data/gpu_knowledge.db (script-relative fallback)
if [ -n "${GPU_FORGE_DB:-}" ]; then
  DB="$GPU_FORGE_DB"
elif [ -n "${CLAUDE_PLUGIN_ROOT:-}" ]; then
  DB="${CLAUDE_PLUGIN_ROOT}/data/gpu_knowledge.db"
else
  DB="$(dirname "$0")/../data/gpu_knowledge.db"
fi

# Resolve schema.sql using same base directory as DB
SCHEMA_DIR="$(dirname "$DB")"
SCHEMA="${SCHEMA_DIR}/schema.sql"

# Check if DB already exists
if [ -f "$DB" ]; then
  SKILL_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM skills;" 2>/dev/null || echo "0")
  FINDING_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM findings;" 2>/dev/null || echo "0")
  echo "Database already exists: $DB"
  echo "  Skills: $SKILL_COUNT"
  echo "  Findings: $FINDING_COUNT"
  exit 0
fi

# Verify schema file exists
if [ ! -f "$SCHEMA" ]; then
  echo "ERROR: schema.sql not found at $SCHEMA" >&2
  exit 1
fi

# Ensure the data directory exists
mkdir -p "$(dirname "$DB")"

# Create DB from schema (includes CREATE TABLE and skill seed INSERT statements)
echo "Creating database: $DB"
sqlite3 "$DB" < "$SCHEMA"

# Verify creation
SKILL_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM skills;")
if [ "$SKILL_COUNT" -eq 11 ]; then
  echo "  Created with $SKILL_COUNT skills"
else
  echo "WARNING: Expected 11 skills, got $SKILL_COUNT" >&2
fi

echo "Bootstrap complete."
