#!/usr/bin/env bats

load ../test_helper/common-setup

KB="${PLUGIN_ROOT}/scripts/kb"

setup() {
  # Create an isolated copy of the DB for test safety
  TEST_DB="${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
  cp "${PLUGIN_ROOT}/data/gpu_knowledge.db" "$TEST_DB"
  export GPU_FORGE_DB="$TEST_DB"
}

teardown() {
  rm -f "${BATS_TEST_TMPDIR}/test_gpu_knowledge.db"
}

# --- No arguments: help/usage ---

@test "kb with no arguments shows usage help" {
  run "$KB"
  assert_success
  assert_output --partial "GPU Knowledge Base CLI"
  assert_output --partial "Usage: kb <command>"
}

@test "kb help lists all 11 skills" {
  run "$KB"
  assert_success
  assert_output --partial "gpu-silicon"
  assert_output --partial "unified-memory"
  assert_output --partial "metal-compute"
  assert_output --partial "msl-kernels"
  assert_output --partial "gpu-io"
  assert_output --partial "gpu-perf"
  assert_output --partial "simd-wave"
  assert_output --partial "mlx-compute"
  assert_output --partial "metal4-api"
  assert_output --partial "gpu-distributed"
  assert_output --partial "gpu-centric-arch"
}

# --- stats ---

@test "stats returns output with Knowledge Base Stats header" {
  run "$KB" stats
  assert_success
  assert_output --partial "=== Knowledge Base Stats ==="
}

@test "stats lists all 11 skills" {
  run "$KB" stats
  assert_success
  assert_output --partial "gpu-silicon"
  assert_output --partial "gpu-centric-arch"
  assert_output --partial "Total findings:"
  assert_output --partial "Total citations:"
}

# --- search ---

@test "search SIMD returns results" {
  run "$KB" search "SIMD"
  assert_success
  assert_output --partial "simd"
}

@test "search nonexistent term returns empty" {
  run "$KB" search "xyznonexistent123qqq"
  assert_success
  # No results — output should be empty or very short (no table rows)
  [ ${#output} -lt 10 ]
}

@test "search with --limit restricts results" {
  run "$KB" search "GPU" --limit 2
  assert_success
  # Count non-header lines (data rows) — should be at most 2
  local data_lines
  data_lines=$(echo "$output" | tail -n +3 | grep -c '[0-9]' || true)
  [ "$data_lines" -le 2 ]
}

# --- skill ---

@test "skill gpu-silicon returns findings" {
  run "$KB" skill gpu-silicon
  assert_success
  assert_output --partial "gpu-silicon" || assert_output --partial "architecture"
  # Should have actual table content
  [ ${#output} -gt 10 ]
}

@test "skill invalid-name returns empty" {
  run "$KB" skill invalid-name
  assert_success
  # No findings for nonexistent skill — output is empty
  [ ${#output} -lt 10 ]
}

# --- detail ---

@test "detail 1 returns finding details" {
  run "$KB" detail 1
  assert_success
  assert_output --partial "skill_name"
  [ ${#output} -gt 20 ]
}

@test "detail with non-numeric ID rejects input" {
  run "$KB" detail abc
  assert_failure
  assert_output --partial "finding_id must be a number"
}

@test "detail rejects SQL injection attempt" {
  run "$KB" detail "1; DROP TABLE findings"
  assert_failure
  assert_output --partial "finding_id must be a number"
}

@test "detail with negative number rejects input" {
  run "$KB" detail "-1"
  assert_failure
  assert_output --partial "finding_id must be a number"
}

# --- unverified ---

@test "unverified runs without error" {
  run "$KB" unverified
  assert_success
}

# --- export ---

@test "export produces markdown output" {
  run "$KB" export
  assert_success
  # Markdown headers expected
  assert_output --partial "##"
  [ ${#output} -gt 100 ]
}

@test "export with skill name produces filtered output" {
  run "$KB" export gpu-silicon
  assert_success
  assert_output --partial "gpu-silicon"
  [ ${#output} -gt 50 ]
}

# --- investigations ---

@test "investigations runs without error" {
  run "$KB" investigations
  assert_success
}

# --- verify ---

@test "verify runs quality checks" {
  run "$KB" verify
  assert_success
  assert_output --partial "=== Knowledge Base Quality Report ==="
}

# --- dedup ---

@test "dedup runs duplicate detection" {
  run "$KB" dedup
  assert_success
  assert_output --partial "=== Duplicate Detection Report ==="
}

# --- add (write to test DB) ---

@test "add inserts a finding into test DB" {
  run "$KB" add gpu-silicon "test-topic" "Test claim for BATS"
  assert_success
  assert_output --partial "Finding added to gpu-silicon/test-topic"
  # Verify it's in the DB
  run "$KB" search "BATS"
  assert_success
  assert_output --partial "Test claim for BATS"
}
