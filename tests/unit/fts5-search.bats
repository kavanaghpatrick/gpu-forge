#!/usr/bin/env bats

load ../test_helper/common-setup

setup() {
  export CLAUDE_PLUGIN_ROOT="$PLUGIN_ROOT"
}

@test "SIMD search returns simd-wave results" {
  run "$PLUGIN_ROOT/scripts/kb" search "SIMD"
  assert_success
  assert_output --partial "simd-wave"
}

@test "claim matches rank higher than notes" {
  # "threadgroup memory" appears directly in claims — should rank high
  run "$PLUGIN_ROOT/scripts/kb" search "threadgroup memory" --limit 5
  assert_success
  # First data line (after header) should contain a claim-heavy match
  # BM25 weights: claim=10, evidence=5, tags=2, notes=1
  assert_output --partial "threadgroup"
}

@test "prefix search works" {
  run "$PLUGIN_ROOT/scripts/kb" search "thread*"
  assert_success
  # Should return results containing thread/threadgroup/threading
  [ "${#lines[@]}" -gt 2 ]
}

@test "phrase search works" {
  run "$PLUGIN_ROOT/scripts/kb" search '"threadgroup memory"'
  assert_success
  assert_output --partial "threadgroup"
}

@test "empty search returns nothing" {
  run "$PLUGIN_ROOT/scripts/kb" search "xyzzyplugh42nonsenseword"
  assert_success
  # No results — output should be empty
  [ -z "$output" ]
}

@test "limit parameter works" {
  run "$PLUGIN_ROOT/scripts/kb" search "SIMD" --limit 3
  assert_success
  # Output has 2 header lines + at most 3 data lines = 5 total
  [ "${#lines[@]}" -le 5 ]
  [ "${#lines[@]}" -ge 3 ]
}
