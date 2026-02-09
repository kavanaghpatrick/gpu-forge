#!/usr/bin/env bash
# PostToolUse hook: Validate .metal file edits
# Reads tool input JSON from stdin, checks for common MSL issues.
# Always exits 0 — never blocks edits, only warns.

set -uo pipefail

# Read JSON from stdin
INPUT=$(cat)

# Extract file_path from tool_input JSON
if command -v jq &>/dev/null; then
  FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
else
  # Fallback: extract file_path with grep/sed
  FILE_PATH=$(echo "$INPUT" | grep -o '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*"file_path"[[:space:]]*:[[:space:]]*"//;s/"$//')
fi

# Only act on .metal files
if [[ -z "$FILE_PATH" ]] || [[ "$FILE_PATH" != *.metal ]]; then
  exit 0
fi

# Check if file exists (it should after an edit)
if [[ ! -f "$FILE_PATH" ]]; then
  exit 0
fi

WARNINGS=()

# Check for kernel functions missing [[kernel]] attribute
# Look for `void <name>(` patterns that aren't preceded by [[kernel]]
while IFS= read -r line; do
  # Skip lines that have [[kernel]] attribute
  if [[ "$line" == *"[[kernel]]"* ]]; then
    continue
  fi
  # Check for function definitions that look like kernel entry points
  # Pattern: kernel void <name>( — this is correct MSL
  # Pattern: void <name>( without kernel keyword — might be missing [[kernel]]
  if echo "$line" | grep -qE '^\s*void\s+\w+\s*\(' 2>/dev/null; then
    # Check if the previous context has [[kernel]] (best effort with single-line check)
    func_name=$(echo "$line" | grep -oE 'void\s+(\w+)' | awk '{print $2}')
    if [[ -n "$func_name" ]]; then
      # Verify it's not already declared with kernel keyword on same line
      if ! echo "$line" | grep -qE '^\s*kernel\s+void' 2>/dev/null; then
        WARNINGS+=("Function '${func_name}' may be missing 'kernel' qualifier or [[kernel]] attribute")
      fi
    fi
  fi
done < "$FILE_PATH"

# Output hookSpecificOutput JSON
if [[ ${#WARNINGS[@]} -gt 0 ]]; then
  # Build JSON warnings array
  WARN_JSON="["
  for i in "${!WARNINGS[@]}"; do
    if [[ $i -gt 0 ]]; then
      WARN_JSON+=","
    fi
    # Escape quotes in warning message
    escaped=$(echo "${WARNINGS[$i]}" | sed 's/"/\\"/g')
    WARN_JSON+="\"${escaped}\""
  done
  WARN_JSON+="]"
  echo "{\"hookSpecificOutput\":{\"metal_warnings\":${WARN_JSON},\"file\":\"${FILE_PATH}\"}}"
else
  echo "{\"hookSpecificOutput\":{\"metal_warnings\":[],\"file\":\"${FILE_PATH}\",\"status\":\"ok\"}}"
fi

exit 0
