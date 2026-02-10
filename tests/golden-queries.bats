#!/usr/bin/env bats
#
# Golden Queries — 72 FTS5 search queries that verify the knowledge DB
# returns relevant results for every skill domain.
#
# Each test runs `kb search "<query>"` and asserts the output contains
# the expected skill name and/or keyword, proving FTS5 relevance ranking
# surfaces the right findings.

load test_helper/common-setup

setup() {
  export CLAUDE_PLUGIN_ROOT="$PLUGIN_ROOT"
  KB="$PLUGIN_ROOT/scripts/kb"
}

# ---------------------------------------------------------------------------
# gpu-silicon (9 queries)
# ---------------------------------------------------------------------------

@test "golden: 'M4 GPU cores' returns gpu-silicon results" {
  run "$KB" search "M4 GPU cores"
  assert_success
  assert_output --partial "gpu-silicon"
}

@test "golden: 'SIMD width 32' returns simd-wave results" {
  run "$KB" search "SIMD width 32"
  assert_success
  assert_output --partial "simd-wave"
}

@test "golden: 'TBDR' returns gpu-silicon results" {
  run "$KB" search "TBDR"
  assert_success
  assert_output --partial "gpu-silicon"
}

@test "golden: 'ALU pipeline' returns gpu-silicon results" {
  run "$KB" search "ALU pipeline"
  assert_success
  assert_output --partial "gpu-silicon"
}

@test "golden: 'Apple GPU clock' returns gpu-silicon results" {
  run "$KB" search "Apple GPU clock"
  assert_success
  assert_output --partial "gpu-silicon"
}

@test "golden: 'register occupancy' returns gpu-silicon results" {
  run "$KB" search "register occupancy"
  assert_success
  assert_output --partial "gpu-silicon"
}

@test "golden: 'register spill' returns gpu-silicon results" {
  run "$KB" search "register spill"
  assert_success
  assert_output --partial "gpu-silicon"
}

@test "golden: 'uniform registers' returns gpu-silicon results" {
  run "$KB" search "uniform registers"
  assert_success
  assert_output --partial "gpu-silicon"
}

@test "golden: 'dynamic register allocation' returns gpu-silicon results" {
  run "$KB" search "dynamic register allocation"
  assert_success
  assert_output --partial "gpu-silicon"
}

# ---------------------------------------------------------------------------
# unified-memory (8 queries)
# ---------------------------------------------------------------------------

@test "golden: 'SLC cache' returns unified-memory results" {
  run "$KB" search "SLC cache"
  assert_success
  assert_output --partial "unified-memory"
}

@test "golden: 'storage mode' returns unified-memory results" {
  run "$KB" search "storage mode"
  assert_success
  assert_output --partial "unified-memory"
}

@test "golden: 'zero copy' returns unified-memory results" {
  run "$KB" search "zero copy"
  assert_success
  assert_output --partial "unified-memory"
}

@test "golden: 'unified memory bandwidth' returns unified-memory results" {
  run "$KB" search "unified memory bandwidth"
  assert_success
  assert_output --partial "unified-memory"
}

@test "golden: 'GPU weakly ordered memory' returns unified-memory results" {
  run "$KB" search "GPU weakly ordered memory"
  assert_success
  assert_output --partial "unified-memory"
}

@test "golden: 'UAT page table' returns unified-memory results" {
  run "$KB" search "UAT page table"
  assert_success
  assert_output --partial "unified-memory"
}

@test "golden: 'DART IOMMU' returns unified-memory results" {
  run "$KB" search "DART IOMMU"
  assert_success
  assert_output --partial "unified-memory"
}

@test "golden: 'GPU TLB entries' returns unified-memory results" {
  run "$KB" search "GPU TLB entries"
  assert_success
  assert_output --partial "unified-memory"
}

# ---------------------------------------------------------------------------
# metal-compute (10 queries)
# ---------------------------------------------------------------------------

@test "golden: 'MTLCommandQueue' returns metal-compute results" {
  run "$KB" search "MTLCommandQueue"
  assert_success
  assert_output --partial "metal-compute"
}

@test "golden: 'compute pipeline' returns metal-compute results" {
  run "$KB" search "compute pipeline"
  assert_success
  assert_output --partial "metal-compute"
}

@test "golden: 'command encoder' returns metal-compute results" {
  run "$KB" search "command encoder"
  assert_success
  assert_output --partial "metal-compute"
}

@test "golden: 'indirect dispatch' returns metal-compute results" {
  run "$KB" search "indirect dispatch"
  assert_success
  assert_output --partial "metal-compute"
}

@test "golden: 'command allocator' returns metal-compute results" {
  run "$KB" search "command allocator"
  assert_success
  assert_output --partial "metal-compute"
}

@test "golden: 'binary archive' returns metal-compute results" {
  run "$KB" search "binary archive"
  assert_success
  assert_output --partial "metal-compute"
}

@test "golden: 'ICB compute' returns metal-compute results" {
  run "$KB" search "ICB compute"
  assert_success
  assert_output --partial "metal-compute"
}

@test "golden: 'Metal GPU driven compute' returns metal-compute results" {
  run "$KB" search "Metal GPU driven compute"
  assert_success
  assert_output --partial "metal-compute"
}

@test "golden: 'shader validation' returns metal-compute results" {
  run "$KB" search "shader validation"
  assert_success
  assert_output --partial "metal-compute"
}

@test "golden: 'MTLSharedEvent' returns metal-compute results" {
  run "$KB" search "MTLSharedEvent"
  assert_success
  assert_output --partial "metal-compute"
}

# ---------------------------------------------------------------------------
# msl-kernels (10 queries)
# ---------------------------------------------------------------------------

@test "golden: 'address space device' returns msl-kernels results" {
  run "$KB" search "address space device"
  assert_success
  assert_output --partial "msl-kernels"
}

@test "golden: 'function constant' returns msl-kernels results" {
  run "$KB" search "function constant"
  assert_success
  assert_output --partial "msl-kernels"
}

@test "golden: 'threadgroup memory' returns msl-kernels results" {
  run "$KB" search "threadgroup memory"
  assert_success
  assert_output --partial "msl-kernels"
}

@test "golden: 'atomic operations' returns msl-kernels results" {
  run "$KB" search "atomic operations"
  assert_success
  assert_output --partial "msl-kernels"
}

@test "golden: 'half precision' returns msl-kernels results" {
  run "$KB" search "half precision"
  assert_success
  assert_output --partial "msl-kernels"
}

@test "golden: 'bank conflict' returns msl-kernels results" {
  run "$KB" search "bank conflict"
  assert_success
  assert_output --partial "msl-kernels"
}

@test "golden: 'float atomic CAS' returns msl-kernels results" {
  run "$KB" search "float atomic CAS"
  assert_success
  assert_output --partial "msl-kernels"
}

@test "golden: 'simd_ballot' returns msl-kernels results" {
  run "$KB" search "simd_ballot"
  assert_success
  assert_output --partial "msl-kernels"
}

@test "golden: 'packed vector' returns msl-kernels results" {
  run "$KB" search "packed vector"
  assert_success
  assert_output --partial "msl-kernels"
}

@test "golden: 'tensor_ops Metal 4' returns msl-kernels results" {
  run "$KB" search "tensor_ops Metal 4"
  assert_success
  assert_output --partial "msl-kernels"
}

# ---------------------------------------------------------------------------
# gpu-io (5 queries)
# ---------------------------------------------------------------------------

@test "golden: 'MTLIOCommandQueue' returns gpu-io results" {
  run "$KB" search "MTLIOCommandQueue"
  assert_success
  assert_output --partial "gpu-io"
}

@test "golden: 'mmap GPU' returns gpu-io results" {
  run "$KB" search "mmap GPU"
  assert_success
  assert_output --partial "gpu-io"
}

@test "golden: 'fast resource loading' returns gpu-io results" {
  run "$KB" search "fast resource loading"
  assert_success
  assert_output --partial "gpu-io"
}

@test "golden: 'SSD streaming' returns gpu-io results" {
  run "$KB" search "SSD streaming"
  assert_success
  assert_output --partial "gpu-io"
}

@test "golden: 'makeBuffer bytesNoCopy' returns gpu-io results" {
  run "$KB" search "makeBuffer bytesNoCopy"
  assert_success
  assert_output --partial "gpu-io"
}

# ---------------------------------------------------------------------------
# gpu-perf (5 queries)
# ---------------------------------------------------------------------------

@test "golden: 'GPU occupancy' returns gpu-perf results" {
  run "$KB" search "GPU occupancy"
  assert_success
  assert_output --partial "gpu-perf"
}

@test "golden: 'bandwidth optimization' returns gpu-perf results" {
  run "$KB" search "bandwidth optimization"
  assert_success
  assert_output --partial "gpu-perf"
}

@test "golden: 'profiling Metal' returns gpu-perf results" {
  run "$KB" search "profiling Metal"
  assert_success
  assert_output --partial "gpu-perf"
}

@test "golden: 'performance counter' returns gpu-perf results" {
  run "$KB" search "performance counter"
  assert_success
  assert_output --partial "gpu-perf"
}

@test "golden: 'memory coalescing' returns gpu-perf results" {
  run "$KB" search "memory coalescing"
  assert_success
  assert_output --partial "gpu-perf"
}

# ---------------------------------------------------------------------------
# simd-wave (5 queries)
# ---------------------------------------------------------------------------

@test "golden: 'simdgroup reduction' returns simd-wave results" {
  run "$KB" search "simdgroup reduction"
  assert_success
  assert_output --partial "simd-wave"
}

@test "golden: 'simd_shuffle' returns simd-wave results" {
  run "$KB" search "simd_shuffle"
  assert_success
  assert_output --partial "simd-wave"
}

@test "golden: 'simdgroup_matrix' returns simd-wave results" {
  run "$KB" search "simdgroup_matrix"
  assert_success
  assert_output --partial "simd-wave"
}

@test "golden: 'quad group' returns simd-wave results" {
  run "$KB" search "quad group"
  assert_success
  assert_output --partial "simd-wave"
}

@test "golden: 'simd_prefix_exclusive_sum' returns simd-wave results" {
  run "$KB" search "simd_prefix_exclusive_sum"
  assert_success
  assert_output --partial "simd-wave"
}

# ---------------------------------------------------------------------------
# mlx-compute (5 queries)
# ---------------------------------------------------------------------------

@test "golden: 'MLX custom kernel' returns mlx-compute results" {
  run "$KB" search "MLX custom kernel"
  assert_success
  assert_output --partial "mlx-compute"
}

@test "golden: 'metal_kernel' returns mlx-compute results" {
  run "$KB" search "metal_kernel"
  assert_success
  assert_output --partial "mlx-compute"
}

@test "golden: 'lazy evaluation' returns mlx-compute results" {
  run "$KB" search "lazy evaluation"
  assert_success
  assert_output --partial "mlx-compute"
}

@test "golden: 'MLX streams' returns mlx-compute results" {
  run "$KB" search "MLX streams"
  assert_success
  assert_output --partial "mlx-compute"
}

@test "golden: 'thread_position_in_grid' returns mlx-compute results" {
  run "$KB" search "thread_position_in_grid"
  assert_success
  assert_output --partial "mlx-compute"
}

# ---------------------------------------------------------------------------
# metal4-api (5 queries)
# ---------------------------------------------------------------------------

@test "golden: 'MTLTensor' returns metal4-api results" {
  run "$KB" search "MTLTensor"
  assert_success
  assert_output --partial "metal4-api"
}

@test "golden: 'Metal 4' returns metal4-api results" {
  run "$KB" search "Metal 4"
  assert_success
  assert_output --partial "metal4-api"
}

@test "golden: 'cooperative tensors' returns metal4-api results" {
  run "$KB" search "cooperative tensors"
  assert_success
  assert_output --partial "metal4-api"
}

@test "golden: 'unified encoder' returns metal4-api results" {
  run "$KB" search "unified encoder"
  assert_success
  assert_output --partial "metal4-api"
}

@test "golden: 'residency set' returns metal4-api results" {
  run "$KB" search "residency set"
  assert_success
  assert_output --partial "metal4-api"
}

# ---------------------------------------------------------------------------
# gpu-distributed (5 queries)
# ---------------------------------------------------------------------------

@test "golden: 'RDMA Thunderbolt' returns gpu-distributed results" {
  run "$KB" search "RDMA Thunderbolt"
  assert_success
  assert_output --partial "gpu-distributed"
}

@test "golden: 'MLX distributed' returns gpu-distributed results" {
  run "$KB" search "MLX distributed"
  assert_success
  assert_output --partial "gpu-distributed"
}

@test "golden: 'Thunderbolt cluster' returns gpu-distributed results" {
  run "$KB" search "Thunderbolt cluster"
  assert_success
  assert_output --partial "gpu-distributed"
}

@test "golden: 'allreduce' returns gpu-distributed results" {
  run "$KB" search "allreduce"
  assert_success
  assert_output --partial "gpu-distributed"
}

@test "golden: 'collective communication' returns gpu-distributed results" {
  run "$KB" search "collective communication"
  assert_success
  assert_output --partial "gpu-distributed"
}

# ---------------------------------------------------------------------------
# gpu-centric-arch (5 queries)
# ---------------------------------------------------------------------------

@test "golden: 'persistent kernel' returns gpu-centric-arch results" {
  run "$KB" search "persistent kernel"
  assert_success
  assert_output --partial "gpu-centric-arch"
}

@test "golden: 'GPU centric' returns gpu-centric-arch results" {
  run "$KB" search "GPU centric"
  assert_success
  assert_output --partial "gpu-centric-arch"
}

@test "golden: 'reverse offloading' returns gpu-centric-arch results" {
  run "$KB" search "reverse offloading"
  assert_success
  assert_output --partial "gpu-centric-arch"
}

@test "golden: 'GPU filesystem' returns gpu-centric-arch results" {
  run "$KB" search "GPU filesystem"
  assert_success
  assert_output --partial "gpu-centric-arch"
}

@test "golden: 'megakernel' returns gpu-centric-arch results" {
  run "$KB" search "megakernel"
  assert_success
  assert_output --partial "gpu-centric-arch"
}
