#!/usr/bin/env bats

load ../test_helper/common-setup

KB="${PLUGIN_ROOT}/scripts/kb"

setup() {
  TEST_DB="${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$TEST_DB"
  export GPU_FORGE_DB="$TEST_DB"
}

teardown() {
  rm -f "${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
}

# --- Gap 8: kb cleanup ---

@test "gap8: kb cleanup --dry-run shows stale investigations" {
  # Create a stale investigation (started 2 hours ago)
  sqlite3 "$TEST_DB" "INSERT INTO investigations (skill_id, topic, status, started_at)
    VALUES (1, 'test-stale', 'running', datetime('now', '-2 hours'));"
  run "$KB" cleanup --dry-run --hours 1
  assert_success
  assert_output --partial "Would close:"
}

@test "gap8: kb cleanup closes stale investigations" {
  sqlite3 "$TEST_DB" "INSERT INTO investigations (skill_id, topic, status, started_at)
    VALUES (1, 'test-stale', 'running', datetime('now', '-2 hours'));"
  run "$KB" cleanup --hours 1
  assert_success
  # Verify it was closed
  local status
  status=$(sqlite3 "$TEST_DB" "SELECT status FROM investigations WHERE topic='test-stale';")
  [ "$status" = "failed" ]
}

@test "gap8: kb verify reports stale investigations" {
  sqlite3 "$TEST_DB" "INSERT INTO investigations (skill_id, topic, status, started_at)
    VALUES (1, 'test-stale', 'running', datetime('now', '-2 hours'));"
  run "$KB" verify
  assert_success
  assert_output --partial "running for >1 hour"
}

# --- Gap 1: SubagentStop hook ---

@test "gap1: investigation-cleanup.sh exits 0" {
  run bash -c 'echo "{}" | CLAUDE_PLUGIN_ROOT="${PLUGIN_ROOT}" GPU_FORGE_DB="${GPU_FORGE_DB}" bash "${PLUGIN_ROOT}/hooks/scripts/investigation-cleanup.sh"'
  assert_success
}

@test "gap1: investigation-cleanup.sh is executable" {
  [ -x "${PLUGIN_ROOT}/hooks/scripts/investigation-cleanup.sh" ]
}

# --- Gap 6: advise command ---

@test "gap6: advise.md exists with correct frontmatter" {
  [ -f "${PLUGIN_ROOT}/commands/advise.md" ]
  local fm
  fm=$(awk 'BEGIN{n=0} /^---$/{n++; if(n==2) exit; next} n==1{print}' "${PLUGIN_ROOT}/commands/advise.md")
  echo "$fm" | grep -q "^name: advise"
  echo "$fm" | grep -q "^context: fork"
  echo "$fm" | grep -q "^agent: architecture-advisor"
  echo "$fm" | grep -q "^disable-model-invocation: true"
}

# --- Gap 4: post-edit hook expansion ---

@test "gap4: post-edit-validator catches device const" {
  local tmpfile="${BATS_TEST_TMPDIR}/test_check.metal"
  cat > "$tmpfile" <<'EOF'
using namespace metal;
kernel void test_kernel(device const float* input [[buffer(0)]]) {}
EOF
  run bash -c "echo '{\"tool_input\":{\"file_path\":\"${tmpfile}\"}}' | bash '${PLUGIN_ROOT}/hooks/scripts/post-edit-validator.sh'"
  assert_success
  assert_output --partial "device const"
}

@test "gap4: post-edit-validator catches missing namespace" {
  local tmpfile="${BATS_TEST_TMPDIR}/test_check.metal"
  cat > "$tmpfile" <<'EOF'
kernel void test_kernel(device float* input [[buffer(0)]]) {}
EOF
  run bash -c "echo '{\"tool_input\":{\"file_path\":\"${tmpfile}\"}}' | bash '${PLUGIN_ROOT}/hooks/scripts/post-edit-validator.sh'"
  assert_success
  assert_output --partial "using namespace metal"
}

@test "gap4: post-edit-validator catches buffer index gaps" {
  local tmpfile="${BATS_TEST_TMPDIR}/test_check.metal"
  cat > "$tmpfile" <<'EOF'
using namespace metal;
kernel void test_kernel(
    device float* a [[buffer(0)]],
    device float* b [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {}
EOF
  run bash -c "echo '{\"tool_input\":{\"file_path\":\"${tmpfile}\"}}' | bash '${PLUGIN_ROOT}/hooks/scripts/post-edit-validator.sh'"
  assert_success
  assert_output --partial "Buffer index gap"
}

# --- Gap 3: URL backfill ---

@test "gap3: kb backfill-urls runs without error" {
  run "$KB" backfill-urls
  assert_success
  assert_output --partial "Findings Missing Source URLs"
}

@test "gap3: kb verify distinguishes web vs local missing URLs" {
  run "$KB" verify
  assert_success
  # Should NOT say just "findings have no source URL" -- should distinguish
  # Either "web-source" or "benchmark/empirical_test"
  if echo "$output" | grep -q "no source URL"; then
    # Old undifferentiated message -- test should fail
    fail "verify still uses undifferentiated URL check"
  fi
}

# --- Gap 5: review write-back ---

@test "gap5: review.md contains Step 5 for KB storage" {
  run grep -c "Step 5" "${PLUGIN_ROOT}/commands/review.md"
  assert_success
  [ "$output" -ge 1 ]
}

# --- Gap 2: citation enforcement ---

@test "gap2: investigation-agent.md has inline citation protocol" {
  run grep -c "Inline Citation Protocol" "${PLUGIN_ROOT}/agents/investigation-agent.md"
  assert_success
  [ "$output" -ge 1 ]
}

@test "gap2: kb verify checks academic findings without citations" {
  run "$KB" verify
  assert_success
  # Should have either "academic paper findings have no citation" or "All academic paper findings have citations"
  echo "$output" | grep -qE "(academic paper findings|All academic paper findings have citations)"
}

# --- Gap 7: scaffold wiring ---

@test "gap7: scaffold.md references JSON config files" {
  run grep -c "reduction-kernel.json\|matrix-multiply.json\|scan-kernel.json\|metal-compute-blank.json\|mlx-custom-kernel.json" "${PLUGIN_ROOT}/commands/scaffold.md"
  assert_success
  [ "$output" -ge 3 ]
}

@test "gap7: scaffold.md has fallback to metal-compute-blank" {
  run grep -c "metal-compute-blank.json" "${PLUGIN_ROOT}/commands/scaffold.md"
  assert_success
  [ "$output" -ge 1 ]
}
