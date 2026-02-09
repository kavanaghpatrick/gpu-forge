#!/usr/bin/env bash
# SessionStart hook: Load GPU knowledge context
# Verifies knowledge DB exists and outputs finding count

set -euo pipefail

DB_PATH="${CLAUDE_PLUGIN_ROOT}/data/gpu_knowledge.db"

if [[ ! -f "$DB_PATH" ]]; then
  echo "gpu-forge: Knowledge DB not found at $DB_PATH" >&2
  exit 0
fi

# Count findings in knowledge base
FINDING_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM findings;" 2>/dev/null || echo "0")

echo "gpu-forge: $FINDING_COUNT findings loaded"
exit 0
