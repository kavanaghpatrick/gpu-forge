#!/usr/bin/env bats

load ../test_helper/common-setup

AGENTS_DIR="${PLUGIN_ROOT}/agents"

# Helper: extract YAML frontmatter (lines between first --- and second ---)
get_frontmatter() {
  awk 'BEGIN{n=0} /^---$/{n++; if(n==2) exit; next} n==1{print}' "$1"
}

@test "all 3 agent files exist" {
  for agent in knowledge-retriever investigation-agent architecture-advisor; do
    [ -f "${AGENTS_DIR}/${agent}.md" ]
  done
}

@test "knowledge-retriever uses model: haiku" {
  frontmatter=$(get_frontmatter "${AGENTS_DIR}/knowledge-retriever.md")
  echo "$frontmatter" | grep -q "^model: haiku"
}

@test "knowledge-retriever has maxTurns: 10" {
  frontmatter=$(get_frontmatter "${AGENTS_DIR}/knowledge-retriever.md")
  echo "$frontmatter" | grep -q "^maxTurns: 10"
}

@test "knowledge-retriever has Bash tool" {
  frontmatter=$(get_frontmatter "${AGENTS_DIR}/knowledge-retriever.md")
  echo "$frontmatter" | grep -q "Bash"
}

@test "investigation-agent uses model: opus" {
  frontmatter=$(get_frontmatter "${AGENTS_DIR}/investigation-agent.md")
  echo "$frontmatter" | grep -q "^model: opus"
}

@test "investigation-agent has maxTurns: 100" {
  frontmatter=$(get_frontmatter "${AGENTS_DIR}/investigation-agent.md")
  echo "$frontmatter" | grep -q "^maxTurns: 100"
}

@test "investigation-agent has WebSearch tool" {
  frontmatter=$(get_frontmatter "${AGENTS_DIR}/investigation-agent.md")
  echo "$frontmatter" | grep -q "WebSearch"
}

@test "investigation-agent has WebFetch tool" {
  frontmatter=$(get_frontmatter "${AGENTS_DIR}/investigation-agent.md")
  echo "$frontmatter" | grep -q "WebFetch"
}

@test "architecture-advisor uses model: sonnet" {
  frontmatter=$(get_frontmatter "${AGENTS_DIR}/architecture-advisor.md")
  echo "$frontmatter" | grep -q "^model: sonnet"
}

@test "architecture-advisor preloads gpu-silicon skill" {
  frontmatter=$(get_frontmatter "${AGENTS_DIR}/architecture-advisor.md")
  echo "$frontmatter" | grep -q "gpu-silicon"
}

@test "architecture-advisor preloads gpu-perf skill" {
  frontmatter=$(get_frontmatter "${AGENTS_DIR}/architecture-advisor.md")
  echo "$frontmatter" | grep -q "gpu-perf"
}

@test "all agents use portable paths" {
  run grep -r '/Users/' "${AGENTS_DIR}/"
  assert_failure
}
