#!/usr/bin/env bash
# KB wrapper: sets GPU_FORGE_DB and delegates to the KB CLI
# Used by hooks that need knowledge base access.

set -euo pipefail

# Set GPU_FORGE_DB from CLAUDE_PLUGIN_ROOT if not already set
if [[ -z "${GPU_FORGE_DB:-}" ]] && [[ -n "${CLAUDE_PLUGIN_ROOT:-}" ]]; then
  export GPU_FORGE_DB="${CLAUDE_PLUGIN_ROOT}/data/gpu_knowledge.db"
fi

# Delegate all arguments to the KB CLI
exec "${CLAUDE_PLUGIN_ROOT}/scripts/kb" "$@"
