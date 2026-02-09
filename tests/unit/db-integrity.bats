#!/usr/bin/env bats

load ../test_helper/common-setup

setup() {
  DB_PATH="${PLUGIN_ROOT}/data/gpu_knowledge.db"
}

@test "11 skills exist" {
  run sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM skills;"
  assert_success
  assert_output "11"
}

@test "skill names match expected" {
  run sqlite3 "$DB_PATH" "SELECT name FROM skills ORDER BY id;"
  assert_success
  assert_line --index 0 "gpu-silicon"
  assert_line --index 1 "unified-memory"
  assert_line --index 2 "metal-compute"
  assert_line --index 3 "msl-kernels"
  assert_line --index 4 "gpu-io"
  assert_line --index 5 "gpu-perf"
  assert_line --index 6 "simd-wave"
  assert_line --index 7 "mlx-compute"
  assert_line --index 8 "metal4-api"
  assert_line --index 9 "gpu-distributed"
  assert_line --index 10 "gpu-centric-arch"
}

@test "all findings reference valid skills" {
  run sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM findings WHERE skill_id NOT IN (SELECT id FROM skills);"
  assert_success
  assert_output "0"
}

@test "all citations reference valid findings" {
  run sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM citations WHERE finding_id NOT IN (SELECT id FROM findings);"
  assert_success
  assert_output "0"
}

@test "confidence values valid" {
  run sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM findings WHERE confidence NOT IN ('verified','high','medium','low','unverified');"
  assert_success
  assert_output "0"
}

@test "source_type values valid" {
  run sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM findings WHERE source_type NOT IN ('academic_paper','apple_docs','wwdc_session','github_repo','blog_post','reverse_engineering','benchmark','empirical_test','patent','forum_post','book','other');"
  assert_success
  assert_output "0"
}

@test "no blog/forum sources marked verified" {
  run sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM findings WHERE source_type IN ('blog_post','forum_post','other') AND confidence='verified';"
  assert_success
  assert_output "0"
}

@test "600+ findings exist" {
  result=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM findings;")
  [ "$result" -ge 600 ]
}

@test "every skill has findings" {
  run sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM skills WHERE id NOT IN (SELECT DISTINCT skill_id FROM findings);"
  assert_success
  assert_output "0"
}
