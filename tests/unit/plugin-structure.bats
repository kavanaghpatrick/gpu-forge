#!/usr/bin/env bats

load ../test_helper/common-setup

@test "plugin.json exists and is valid JSON" {
  [ -f "${PLUGIN_ROOT}/.claude-plugin/plugin.json" ]
  run jq . "${PLUGIN_ROOT}/.claude-plugin/plugin.json"
  assert_success
}

@test "plugin.json name field is 'gpu-forge'" {
  run jq -r .name "${PLUGIN_ROOT}/.claude-plugin/plugin.json"
  assert_output "gpu-forge"
}

@test "skills directory exists" {
  [ -d "${PLUGIN_ROOT}/skills" ]
}

@test "agents directory exists" {
  [ -d "${PLUGIN_ROOT}/agents" ]
}

@test "commands directory exists" {
  [ -d "${PLUGIN_ROOT}/commands" ]
}

@test "hooks directory exists" {
  [ -d "${PLUGIN_ROOT}/hooks" ]
}

@test "data directory exists" {
  [ -d "${PLUGIN_ROOT}/data" ]
}

@test "scripts directory exists" {
  [ -d "${PLUGIN_ROOT}/scripts" ]
}

@test "templates directory exists" {
  [ -d "${PLUGIN_ROOT}/templates" ]
}

@test "tests directory exists" {
  [ -d "${PLUGIN_ROOT}/tests" ]
}

@test "scripts/kb is executable" {
  [ -x "${PLUGIN_ROOT}/scripts/kb" ]
}

@test "no hardcoded /Users/ paths in scripts" {
  run grep -r '/Users/' "${PLUGIN_ROOT}/scripts/"
  assert_failure
}

@test "no hardcoded /Users/ paths in data directory" {
  # Only check .sql files, skip .db binary
  if compgen -G "${PLUGIN_ROOT}/data/*.sql" > /dev/null; then
    run grep -r '/Users/' "${PLUGIN_ROOT}/data/"*.sql
    assert_failure
  else
    skip "No SQL files to check"
  fi
}

@test "hooks/hooks.json is valid JSON" {
  if [ -f "${PLUGIN_ROOT}/hooks/hooks.json" ]; then
    run jq . "${PLUGIN_ROOT}/hooks/hooks.json"
    assert_success
  else
    skip "hooks.json not yet created"
  fi
}

@test "at least 3 SKILL.md files exist" {
  count=$(find "${PLUGIN_ROOT}/skills" -name "SKILL.md" -type f | wc -l | tr -d ' ')
  [ "$count" -ge 3 ]
}

@test "plugin.json contains required name field" {
  run jq -e '.name' "${PLUGIN_ROOT}/.claude-plugin/plugin.json"
  assert_success
}

@test "LICENSE file exists" {
  [ -f "${PLUGIN_ROOT}/LICENSE" ]
}

@test "CHANGELOG.md exists" {
  [ -f "${PLUGIN_ROOT}/CHANGELOG.md" ]
}

@test ".gitignore exists" {
  [ -f "${PLUGIN_ROOT}/.gitignore" ]
}
