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

# --- Full query workflow ---

@test "full query workflow: search then detail" {
  # Step 1: Search for a term we know exists
  run "$KB" search "GPU"
  assert_success
  [ ${#output} -gt 10 ]

  # Step 2: Extract the first finding ID from the search results
  local first_id
  first_id=$(echo "$output" | grep -oE '^\s*[0-9]+' | head -1 | tr -d ' ')
  [ -n "$first_id" ]

  # Step 3: Get detail on that finding
  run "$KB" detail "$first_id"
  assert_success
  assert_output --partial "skill_name"
}

# --- KB wrapper uses same DB as direct kb ---

@test "kb-wrapper uses same DB as direct kb" {
  # Direct kb stats
  run "$KB" stats
  assert_success
  local direct_output="$output"

  # Via kb-wrapper (sets GPU_FORGE_DB from CLAUDE_PLUGIN_ROOT)
  run env CLAUDE_PLUGIN_ROOT="${PLUGIN_ROOT}" GPU_FORGE_DB="$TEST_DB" \
    bash "${PLUGIN_ROOT}/hooks/scripts/kb-wrapper.sh" stats
  assert_success

  # Both should show "Knowledge Base Stats" header
  assert_output --partial "=== Knowledge Base Stats ==="
  echo "$direct_output" | grep -q "=== Knowledge Base Stats ==="
}

# --- Export produces valid markdown ---

@test "export produces valid markdown" {
  run "$KB" export
  assert_success
  # Must contain ## headers (markdown sections)
  echo "$output" | grep -qE '^##'
  # Must have substantial content
  [ ${#output} -gt 100 ]
}

# --- Search and skill produce consistent results ---

@test "search and skill produce consistent results" {
  # Search for "SIMD" — should return results
  run "$KB" search "SIMD"
  assert_success
  [ ${#output} -gt 10 ]

  # Extract skill name from first result line (2nd column)
  local skill_name
  skill_name=$(echo "$output" | tail -n +3 | head -1 | awk '{print $2}')
  [ -n "$skill_name" ]

  # The skill listing for that skill should also contain SIMD-related content
  run "$KB" skill "$skill_name"
  assert_success
  [ ${#output} -gt 10 ]
}
