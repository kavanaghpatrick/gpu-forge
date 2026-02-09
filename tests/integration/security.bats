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

# --- SQL injection ---

@test "search rejects SQL injection" {
  run "$KB" search '"; DROP TABLE findings; --'
  # Command may fail due to FTS5 parse error — that's acceptable
  # The critical check: findings table must still be intact
  run sqlite3 "$TEST_DB" "SELECT COUNT(*) FROM findings;"
  assert_success
  [ "$output" -gt 0 ]
}

@test "detail rejects SQL injection" {
  run "$KB" detail '1; DROP TABLE findings'
  assert_failure
  assert_output --partial "finding_id must be a number"
  # Verify findings table is intact
  run sqlite3 "$TEST_DB" "SELECT COUNT(*) FROM findings;"
  assert_success
  [ "$output" -gt 0 ]
}

# --- Special characters ---

@test "add escapes special characters" {
  # Add a finding with quotes and special chars in the claim
  run "$KB" add gpu-silicon "test-topic" "It's a \"quoted\" claim with 'apostrophes'"
  assert_success
  assert_output --partial "Finding added to gpu-silicon/test-topic"

  # Verify it was stored correctly
  run "$KB" search "apostrophes"
  assert_success
  assert_output --partial "apostrophes"
}

# --- Concurrent reads ---

@test "concurrent DB reads don't fail" {
  # Launch 5 parallel searches and collect exit codes
  local pids=()
  for i in 1 2 3 4 5; do
    "$KB" search "GPU" > /dev/null 2>&1 &
    pids+=($!)
  done

  # Wait for all and check that none failed
  local failures=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done

  [ "$failures" -eq 0 ]
}
