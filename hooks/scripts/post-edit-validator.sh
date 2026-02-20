#!/usr/bin/env bash
# PostToolUse hook: Validate .metal file edits
# Reads tool input JSON from stdin, checks for common MSL issues.
# Always exits 0 — never blocks edits, only warns.

set -uo pipefail

INPUT=$(cat)

# Extract file_path from tool_input JSON
if command -v jq &>/dev/null; then
  FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
else
  FILE_PATH=$(echo "$INPUT" | grep -o '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*"file_path"[[:space:]]*:[[:space:]]*"//;s/"$//')
fi

# Only act on .metal files
if [[ -z "$FILE_PATH" ]] || [[ "$FILE_PATH" != *.metal ]]; then
  exit 0
fi

if [[ ! -f "$FILE_PATH" ]]; then
  exit 0
fi

WARNINGS=()

# --- Check functions (each appends to WARNINGS) ---

check_missing_kernel_qualifier() {
  while IFS= read -r line; do
    if [[ "$line" == *"[[kernel]]"* ]]; then continue; fi
    if echo "$line" | grep -qE '^\s*void\s+\w+\s*\(' 2>/dev/null; then
      if ! echo "$line" | grep -qE '^\s*kernel\s+void' 2>/dev/null; then
        func_name=$(echo "$line" | grep -oE 'void\s+(\w+)' | awk '{print $2}')
        if [[ -n "$func_name" ]]; then
          WARNINGS+=("Function '${func_name}' may be missing 'kernel' qualifier or [[kernel]] attribute")
        fi
      fi
    fi
  done < "$FILE_PATH"
}

check_address_space_misuse() {
  # 'device const' should be 'constant' (misses constant cache)
  if grep -nE '\bdevice\s+const\b' "$FILE_PATH" >/dev/null 2>&1; then
    local lines
    lines=$(grep -nE '\bdevice\s+const\b' "$FILE_PATH" | head -3)
    WARNINGS+=("Address space misuse: 'device const' should likely be 'constant' for read-only data (constant cache). Lines: $(echo "$lines" | cut -d: -f1 | tr '\n' ',' | sed 's/,$//')")
  fi
}

check_threadgroup_size() {
  # Threadgroup sizes hardcoded to non-multiples of 32
  local tg_sizes
  tg_sizes=$(grep -oE '\[\[max_total_threads_per_threadgroup\(([0-9]+)\)\]\]' "$FILE_PATH" 2>/dev/null | grep -oE '[0-9]+')
  for size in $tg_sizes; do
    if [ $((size % 32)) -ne 0 ]; then
      WARNINGS+=("Threadgroup size ${size} is not a multiple of 32 (SIMD width). Wastes lanes.")
    fi
  done
}

check_atomic_in_loop() {
  # atomic_fetch_* inside for/while loops (contention risk)
  # Simple heuristic: look for atomic within indented blocks after for/while
  if grep -nE '^\s+(for|while)\b' "$FILE_PATH" >/dev/null 2>&1; then
    # Check if any atomic ops appear in the file at all
    if grep -nE 'atomic_fetch_(add|sub|min|max|or|and|xor)' "$FILE_PATH" >/dev/null 2>&1; then
      # Rough check: are atomics inside loop bodies?
      local in_loop=0
      local loop_depth=0
      local line_num=0
      while IFS= read -r line; do
        line_num=$((line_num + 1))
        if echo "$line" | grep -qE '^\s+(for|while)\b' 2>/dev/null; then
          in_loop=1
        fi
        if [[ "$in_loop" -eq 1 ]] && echo "$line" | grep -qE 'atomic_fetch_' 2>/dev/null; then
          WARNINGS+=("Potential high-contention atomic at line ${line_num} inside loop. Consider simdgroup reduction first.")
          break  # One warning is enough
        fi
        # Reset after seeing a closing brace at base indent (heuristic)
      done < "$FILE_PATH"
    fi
  fi
}

check_missing_namespace() {
  # Missing 'using namespace metal;' -- required in .metal files
  if ! grep -qE '^\s*using\s+namespace\s+metal\s*;' "$FILE_PATH" 2>/dev/null; then
    # Only warn if file has kernel functions (not a header)
    if grep -qE '\b(kernel|vertex|fragment)\b' "$FILE_PATH" 2>/dev/null; then
      WARNINGS+=("Missing 'using namespace metal;' -- required for MSL types and functions")
    fi
  fi
}

check_buffer_index_gaps() {
  # Extract buffer indices and check for gaps
  local indices
  indices=$(grep -oE '\[\[buffer\(([0-9]+)\)\]\]' "$FILE_PATH" 2>/dev/null | grep -oE '[0-9]+' | sort -n | uniq)
  if [[ -n "$indices" ]]; then
    local prev=-1
    for idx in $indices; do
      if [[ $prev -ge 0 ]] && [[ $((idx - prev)) -gt 1 ]]; then
        WARNINGS+=("Buffer index gap: [[buffer(${prev})]] to [[buffer(${idx})]] -- missing [[buffer($((prev + 1)))]]")
      fi
      prev=$idx
    done
  fi
}

check_barrier_near_tg_write() {
  # Threadgroup memory writes in loops without threadgroup_barrier
  # Heuristic: if file has threadgroup writes AND for/while loops, check for barriers
  if grep -qE '\bthreadgroup\b.*\[' "$FILE_PATH" 2>/dev/null; then
    if grep -qE '^\s+(for|while)\b' "$FILE_PATH" 2>/dev/null; then
      if ! grep -qE 'threadgroup_barrier' "$FILE_PATH" 2>/dev/null; then
        WARNINGS+=("Threadgroup memory used in file with loops but no threadgroup_barrier found. Ensure proper synchronization.")
      fi
    fi
  fi
}

# --- Run all checks ---
check_missing_kernel_qualifier
check_address_space_misuse
check_threadgroup_size
check_atomic_in_loop
check_missing_namespace
check_buffer_index_gaps
check_barrier_near_tg_write

# --- Output hookSpecificOutput JSON ---
if [[ ${#WARNINGS[@]} -gt 0 ]]; then
  WARN_JSON="["
  for i in "${!WARNINGS[@]}"; do
    if [[ $i -gt 0 ]]; then WARN_JSON+=","; fi
    escaped=$(echo "${WARNINGS[$i]}" | sed 's/"/\\"/g')
    WARN_JSON+="\"${escaped}\""
  done
  WARN_JSON+="]"
  echo "{\"hookSpecificOutput\":{\"metal_warnings\":${WARN_JSON},\"file\":\"${FILE_PATH}\"}}"
else
  echo "{\"hookSpecificOutput\":{\"metal_warnings\":[],\"file\":\"${FILE_PATH}\",\"status\":\"ok\"}}"
fi

exit 0
