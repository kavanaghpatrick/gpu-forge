#!/usr/bin/env bats

load ../test_helper/common-setup

SKILLS=(gpu-silicon unified-memory metal-compute msl-kernels gpu-io gpu-perf simd-wave mlx-compute metal4-api gpu-distributed gpu-centric-arch)

@test "all 11 skills directories exist" {
  for skill in "${SKILLS[@]}"; do
    [ -d "${PLUGIN_ROOT}/skills/${skill}" ]
  done
}

@test "all 11 SKILL.md files exist" {
  for skill in "${SKILLS[@]}"; do
    [ -f "${PLUGIN_ROOT}/skills/${skill}/SKILL.md" ]
  done
}

@test "all SKILL.md have YAML frontmatter" {
  for skill in "${SKILLS[@]}"; do
    local first_line
    first_line=$(head -1 "${PLUGIN_ROOT}/skills/${skill}/SKILL.md")
    [ "$first_line" = "---" ]
  done
}

@test "all SKILL.md have name field in frontmatter" {
  for skill in "${SKILLS[@]}"; do
    run grep -m1 '^name:' "${PLUGIN_ROOT}/skills/${skill}/SKILL.md"
    assert_success
  done
}

@test "all SKILL.md have description field" {
  for skill in "${SKILLS[@]}"; do
    run grep -m1 '^description:' "${PLUGIN_ROOT}/skills/${skill}/SKILL.md"
    assert_success
  done
}

@test "all SKILL.md under 500 lines" {
  for skill in "${SKILLS[@]}"; do
    local lines
    lines=$(wc -l < "${PLUGIN_ROOT}/skills/${skill}/SKILL.md")
    [ "$lines" -lt 500 ]
  done
}

@test "all SKILL.md descriptions under 1200 characters" {
  for skill in "${SKILLS[@]}"; do
    local desc_len
    desc_len=$(awk '
      /^---/ { n++; next }
      n==1 && /^description:/ { found=1; sub(/^description: *>? */, ""); buf=$0; next }
      found && /^  / { sub(/^  +/, ""); buf=buf " " $0; next }
      found && !/^  / { found=0 }
      END { print length(buf) }
    ' "${PLUGIN_ROOT}/skills/${skill}/SKILL.md")
    [ "$desc_len" -lt 1200 ]
  done
}

@test "all SKILL.md contain KB query instructions" {
  for skill in "${SKILLS[@]}"; do
    run grep 'CLAUDE_PLUGIN_ROOT.*/scripts/kb\|scripts/kb' "${PLUGIN_ROOT}/skills/${skill}/SKILL.md"
    assert_success
  done
}

@test "all SKILL.md use portable paths" {
  for skill in "${SKILLS[@]}"; do
    run grep 'CLAUDE_PLUGIN_ROOT' "${PLUGIN_ROOT}/skills/${skill}/SKILL.md"
    assert_success
  done
}

@test "all skills have references/ directory" {
  for skill in "${SKILLS[@]}"; do
    [ -d "${PLUGIN_ROOT}/skills/${skill}/references" ]
  done
}

@test "all skills have references/all-findings.md" {
  for skill in "${SKILLS[@]}"; do
    [ -f "${PLUGIN_ROOT}/skills/${skill}/references/all-findings.md" ]
  done
}

@test "all SKILL.md have cross-reference section" {
  for skill in "${SKILLS[@]}"; do
    run grep -i 'cross.ref\|cross.skill\|related.skill' "${PLUGIN_ROOT}/skills/${skill}/SKILL.md"
    assert_success
  done
}

@test "gpu-silicon has SIMD trigger keyword" {
  run grep 'SIMD' "${PLUGIN_ROOT}/skills/gpu-silicon/SKILL.md"
  assert_success
}

@test "metal-compute has MTLComputePipelineState trigger" {
  run grep 'MTLComputePipelineState' "${PLUGIN_ROOT}/skills/metal-compute/SKILL.md"
  assert_success
}

@test "gpu-perf has occupancy trigger keyword" {
  run grep 'occupancy' "${PLUGIN_ROOT}/skills/gpu-perf/SKILL.md"
  assert_success
}

@test "total description length under 16000 chars" {
  local total=0
  for skill in "${SKILLS[@]}"; do
    local desc_len
    desc_len=$(awk '
      /^---/ { n++; next }
      n==1 && /^description:/ { found=1; sub(/^description: *>? */, ""); buf=$0; next }
      found && /^  / { sub(/^  +/, ""); buf=buf " " $0; next }
      found && !/^  / { found=0 }
      END { print length(buf) }
    ' "${PLUGIN_ROOT}/skills/${skill}/SKILL.md")
    total=$((total + desc_len))
  done
  [ "$total" -lt 16000 ]
}
