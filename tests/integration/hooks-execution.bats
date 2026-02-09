#!/usr/bin/env bats

load ../test_helper/common-setup

setup() {
  TEST_DB="${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$TEST_DB"
  export GPU_FORGE_DB="$TEST_DB"
}

teardown() {
  rm -f "${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
}

# --- context-loader ---

@test "context-loader succeeds with valid DB" {
  run env CLAUDE_PLUGIN_ROOT="${PLUGIN_ROOT}" bash "${PLUGIN_ROOT}/hooks/scripts/context-loader.sh"
  assert_success
  # Should report finding count
  assert_output --partial "findings loaded"
}

@test "context-loader handles missing DB gracefully" {
  # Use an empty temp dir as CLAUDE_PLUGIN_ROOT so DB won't exist
  run env CLAUDE_PLUGIN_ROOT="${BATS_TEST_TMPDIR}/nonexistent" bash "${PLUGIN_ROOT}/hooks/scripts/context-loader.sh"
  # Should still exit 0 (graceful degradation)
  assert_success
  assert_output --partial "Knowledge DB not found"
}

# --- post-edit-validator ---

@test "post-edit-validator ignores .swift files" {
  run bash -c 'echo "{\"tool_input\":{\"file_path\":\"/tmp/test.swift\"}}" | CLAUDE_PLUGIN_ROOT="${PLUGIN_ROOT}" bash "${PLUGIN_ROOT}/hooks/scripts/post-edit-validator.sh"'
  assert_success
  # Should produce no metal warnings output (file is ignored)
  refute_output --partial "metal_warnings"
}

@test "post-edit-validator checks .metal files" {
  # Create a temp .metal file with a suspicious void function
  local metal_file="${BATS_TEST_TMPDIR}/test_shader.metal"
  cat > "$metal_file" <<'MSL'
#include <metal_stdlib>
using namespace metal;

void my_kernel(device float* buf [[buffer(0)]]) {
  buf[0] = 1.0;
}
MSL

  run bash -c "echo '{\"tool_input\":{\"file_path\":\"${metal_file}\"}}' | CLAUDE_PLUGIN_ROOT=\"${PLUGIN_ROOT}\" bash \"${PLUGIN_ROOT}/hooks/scripts/post-edit-validator.sh\""
  assert_success
  # Should produce metal_warnings output (it's a .metal file)
  assert_output --partial "metal_warnings"
}

# --- kb-wrapper ---

@test "kb-wrapper resolves correct DB path" {
  # kb-wrapper should delegate to kb and use the DB
  run env CLAUDE_PLUGIN_ROOT="${PLUGIN_ROOT}" GPU_FORGE_DB="$TEST_DB" \
    bash "${PLUGIN_ROOT}/hooks/scripts/kb-wrapper.sh" stats
  assert_success
  assert_output --partial "=== Knowledge Base Stats ==="
}
