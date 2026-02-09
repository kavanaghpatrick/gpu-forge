#!/usr/bin/env bats

load ../test_helper/common-setup

@test "hooks.json is valid JSON" {
  run jq . "${PLUGIN_ROOT}/hooks/hooks.json"
  assert_success
}

@test "hooks.json has hooks array" {
  run jq -e '.hooks | type' "${PLUGIN_ROOT}/hooks/hooks.json"
  assert_success
  assert_output '"array"'
}

@test "SessionStart event exists" {
  run jq -e '.hooks[] | select(.event == "SessionStart")' "${PLUGIN_ROOT}/hooks/hooks.json"
  assert_success
}

@test "PostToolUse event exists" {
  run jq -e '.hooks[] | select(.event == "PostToolUse")' "${PLUGIN_ROOT}/hooks/hooks.json"
  assert_success
}

@test "all hook scripts are executable" {
  for script in "${PLUGIN_ROOT}"/hooks/scripts/*.sh; do
    [ -x "$script" ] || fail "Not executable: $script"
  done
}

@test "context-loader exits 0 with valid DB" {
  run env CLAUDE_PLUGIN_ROOT="${PLUGIN_ROOT}" bash "${PLUGIN_ROOT}/hooks/scripts/context-loader.sh"
  assert_success
}

@test "post-edit-validator ignores non-.metal files" {
  run bash -c 'echo "{\"tool_input\":{\"file_path\":\"/tmp/test.swift\"}}" | CLAUDE_PLUGIN_ROOT="${PLUGIN_ROOT}" bash "${PLUGIN_ROOT}/hooks/scripts/post-edit-validator.sh"'
  assert_success
}

@test "kb-wrapper is executable" {
  [ -x "${PLUGIN_ROOT}/hooks/scripts/kb-wrapper.sh" ]
}

@test "hook timeouts under 30 seconds" {
  # Check any explicit timeout values in hooks.json are under 30
  local max_timeout
  max_timeout=$(jq '[.hooks[].hooks[]? | .timeout // 0] | max' "${PLUGIN_ROOT}/hooks/hooks.json")
  [ "$max_timeout" -lt 30 ]
}
