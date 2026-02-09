#!/usr/bin/env bats

load ../test_helper/common-setup

# --- Metal templates ---

@test "all 5 Metal templates exist" {
  for t in reduction gemm scan histogram blank; do
    [ -f "${PLUGIN_ROOT}/templates/metal/${t}.metal.tmpl" ] || fail "Missing: ${t}.metal.tmpl"
  done
}

# --- Swift templates ---

@test "all 3 Swift templates exist" {
  for t in Package.swift main.swift MetalCompute.swift; do
    [ -f "${PLUGIN_ROOT}/templates/swift/${t}.tmpl" ] || fail "Missing: ${t}.tmpl"
  done
}

# --- MLX template ---

@test "MLX template exists" {
  [ -f "${PLUGIN_ROOT}/templates/mlx/custom-kernel.py.tmpl" ]
}

# --- Scaffold specs ---

@test "all 5 scaffold specs are valid JSON" {
  for s in reduction-kernel matrix-multiply scan-kernel metal-compute-blank mlx-custom-kernel; do
    run jq . "${PLUGIN_ROOT}/templates/project/${s}.json"
    assert_success
  done
}

@test "scaffold specs have name field" {
  for s in reduction-kernel matrix-multiply scan-kernel metal-compute-blank mlx-custom-kernel; do
    run jq -e '.name' "${PLUGIN_ROOT}/templates/project/${s}.json"
    assert_success
  done
}

@test "scaffold specs have files array" {
  for s in reduction-kernel matrix-multiply scan-kernel metal-compute-blank mlx-custom-kernel; do
    run jq -e '.files | type' "${PLUGIN_ROOT}/templates/project/${s}.json"
    assert_success
    assert_output '"array"'
  done
}

@test "scaffold file refs point to existing templates" {
  for s in reduction-kernel matrix-multiply scan-kernel metal-compute-blank mlx-custom-kernel; do
    while IFS= read -r tmpl; do
      [ -f "${PLUGIN_ROOT}/templates/${tmpl}" ] || fail "Missing template: ${tmpl} (referenced by ${s}.json)"
    done < <(jq -r '.files[].template' "${PLUGIN_ROOT}/templates/project/${s}.json")
  done
}

# --- Metal template content ---

@test "Metal templates have metal_stdlib include" {
  for t in reduction gemm scan histogram blank; do
    grep -q '#include <metal_stdlib>' "${PLUGIN_ROOT}/templates/metal/${t}.metal.tmpl" \
      || fail "${t}.metal.tmpl missing metal_stdlib include"
  done
}

@test "Metal templates have kernel void" {
  for t in reduction gemm scan histogram blank; do
    grep -q 'kernel void' "${PLUGIN_ROOT}/templates/metal/${t}.metal.tmpl" \
      || fail "${t}.metal.tmpl missing kernel void"
  done
}

# --- Template parameter syntax ---

@test "templates use {{PARAM}} syntax" {
  local found=0
  for tmpl in "${PLUGIN_ROOT}"/templates/metal/*.tmpl \
              "${PLUGIN_ROOT}"/templates/swift/*.tmpl \
              "${PLUGIN_ROOT}"/templates/mlx/*.tmpl; do
    if grep -qE '\{\{[A-Z_]+\}\}' "$tmpl"; then
      found=$((found + 1))
    fi
  done
  # All 9 templates should use parameter syntax
  [ "$found" -ge 9 ] || fail "Only ${found}/9 templates use {{PARAM}} syntax"
}

# --- Template rendering ---

@test "reduction template renders with defaults" {
  run bash -c "sed -e 's/{{TYPE}}/float/g' -e 's/{{OP}}/sum/g' -e 's/{{IDENTITY}}/0/g' '${PLUGIN_ROOT}/templates/metal/reduction.metal.tmpl'"
  assert_success
  # Verify no unsubstituted parameters remain
  refute_output --partial '{{TYPE}}'
  refute_output --partial '{{OP}}'
  refute_output --partial '{{IDENTITY}}'
  # Verify valid Metal content
  assert_output --partial 'kernel void'
  assert_output --partial '#include <metal_stdlib>'
}

# --- Metal compilation tests ---

@test "reduction template compiles with Metal compiler" {
  if ! command -v xcrun &>/dev/null; then
    skip "xcrun not available"
  fi
  local rendered
  rendered=$(mktemp /tmp/test_reduction_XXXXXX.metal)
  sed -e 's/{{TYPE}}/float/g' -e 's/{{OP}}/sum/g' -e 's/{{IDENTITY}}/0/g' \
    "$PLUGIN_ROOT/templates/metal/reduction.metal.tmpl" > "$rendered"
  run xcrun -sdk macosx metal -c "$rendered" -o /dev/null
  rm -f "$rendered"
  [ "$status" -eq 0 ]
}

@test "gemm template compiles with Metal compiler" {
  if ! command -v xcrun &>/dev/null; then
    skip "xcrun not available"
  fi
  local rendered
  rendered=$(mktemp /tmp/test_gemm_XXXXXX.metal)
  sed -e 's/{{TYPE}}/float/g' -e 's/{{TILE_SIZE}}/16/g' \
    "$PLUGIN_ROOT/templates/metal/gemm.metal.tmpl" > "$rendered"
  run xcrun -sdk macosx metal -c "$rendered" -o /dev/null
  rm -f "$rendered"
  [ "$status" -eq 0 ]
}

@test "scan template compiles with Metal compiler" {
  if ! command -v xcrun &>/dev/null; then
    skip "xcrun not available"
  fi
  local rendered
  rendered=$(mktemp /tmp/test_scan_XXXXXX.metal)
  sed -e 's/{{TYPE}}/float/g' -e 's/{{OP}}/sum/g' -e 's/{{IDENTITY}}/0/g' \
    "$PLUGIN_ROOT/templates/metal/scan.metal.tmpl" > "$rendered"
  run xcrun -sdk macosx metal -c "$rendered" -o /dev/null
  rm -f "$rendered"
  [ "$status" -eq 0 ]
}

@test "histogram template compiles with Metal compiler" {
  if ! command -v xcrun &>/dev/null; then
    skip "xcrun not available"
  fi
  local rendered
  rendered=$(mktemp /tmp/test_histogram_XXXXXX.metal)
  sed -e 's/{{TYPE}}/uint/g' -e 's/{{BINS}}/256/g' \
    "$PLUGIN_ROOT/templates/metal/histogram.metal.tmpl" > "$rendered"
  run xcrun -sdk macosx metal -c "$rendered" -o /dev/null
  rm -f "$rendered"
  [ "$status" -eq 0 ]
}

@test "blank template compiles with Metal compiler" {
  if ! command -v xcrun &>/dev/null; then
    skip "xcrun not available"
  fi
  local rendered
  rendered=$(mktemp /tmp/test_blank_XXXXXX.metal)
  sed -e 's/{{KERNEL_NAME}}/my_kernel/g' -e 's/{{TYPE}}/float/g' \
    "$PLUGIN_ROOT/templates/metal/blank.metal.tmpl" > "$rendered"
  run xcrun -sdk macosx metal -c "$rendered" -o /dev/null
  rm -f "$rendered"
  [ "$status" -eq 0 ]
}
