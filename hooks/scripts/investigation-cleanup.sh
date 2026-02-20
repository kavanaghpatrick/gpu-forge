#!/usr/bin/env bash
# SubagentStop hook: Clean up orphaned investigations when investigation-agent exits.
# Always exits 0 — informational only, never blocks.

set -uo pipefail

# Read stdin JSON (contains agent name, exit reason, etc.)
INPUT=$(cat)

# Resolve KB script
if [ -n "${CLAUDE_PLUGIN_ROOT:-}" ]; then
  KB="${CLAUDE_PLUGIN_ROOT}/scripts/kb"
else
  KB="$(dirname "$0")/../../scripts/kb"
fi

# Close any investigations that are still 'running' — agent just stopped
"$KB" cleanup --hours 0 2>/dev/null || true

exit 0
