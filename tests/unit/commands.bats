#!/usr/bin/env bats

load ../test_helper/common-setup

COMMANDS_DIR="${PLUGIN_ROOT}/commands"

# Helper: extract YAML frontmatter (lines between first --- and second ---)
get_frontmatter() {
  awk 'BEGIN{n=0} /^---$/{n++; if(n==2) exit; next} n==1{print}' "$1"
}

@test "all 6 command files exist" {
  for cmd in ask investigate knowledge scaffold template review; do
    [ -f "${COMMANDS_DIR}/${cmd}.md" ]
  done
}

@test "all commands have YAML frontmatter" {
  for cmd in ask investigate knowledge scaffold template review; do
    run head -1 "${COMMANDS_DIR}/${cmd}.md"
    assert_output "---"
  done
}

@test "all commands have name field" {
  for cmd in ask investigate knowledge scaffold template review; do
    frontmatter=$(get_frontmatter "${COMMANDS_DIR}/${cmd}.md")
    echo "$frontmatter" | grep -q "^name:"
  done
}

@test "all commands have description field" {
  for cmd in ask investigate knowledge scaffold template review; do
    frontmatter=$(get_frontmatter "${COMMANDS_DIR}/${cmd}.md")
    echo "$frontmatter" | grep -q "^description:"
  done
}

@test "ask command uses model: sonnet" {
  frontmatter=$(get_frontmatter "${COMMANDS_DIR}/ask.md")
  echo "$frontmatter" | grep -q "^model: sonnet"
}

@test "investigate command uses context: fork" {
  frontmatter=$(get_frontmatter "${COMMANDS_DIR}/investigate.md")
  echo "$frontmatter" | grep -q "^context: fork"
}

@test "investigate command uses agent: investigation-agent" {
  frontmatter=$(get_frontmatter "${COMMANDS_DIR}/investigate.md")
  echo "$frontmatter" | grep -q "^agent: investigation-agent"
}

@test "knowledge command uses model: haiku" {
  frontmatter=$(get_frontmatter "${COMMANDS_DIR}/knowledge.md")
  echo "$frontmatter" | grep -q "^model: haiku"
}

@test "scaffold has disable-model-invocation: true" {
  frontmatter=$(get_frontmatter "${COMMANDS_DIR}/scaffold.md")
  echo "$frontmatter" | grep -q "^disable-model-invocation: true"
}

@test "all commands reference ARGUMENTS or CLAUDE_PLUGIN_ROOT" {
  for cmd in ask investigate knowledge scaffold template review; do
    content=$(cat "${COMMANDS_DIR}/${cmd}.md")
    if ! echo "$content" | grep -q 'ARGUMENTS' && ! echo "$content" | grep -q 'CLAUDE_PLUGIN_ROOT'; then
      echo "FAIL: ${cmd}.md references neither ARGUMENTS nor CLAUDE_PLUGIN_ROOT"
      return 1
    fi
  done
}

@test "no hardcoded paths in commands" {
  run grep -r '/Users/' "${COMMANDS_DIR}/"
  assert_failure
}
