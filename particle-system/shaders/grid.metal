// grid.metal — 3D density grid compute kernels (clear + populate).
//
// Maintains a 64^3 voxel grid of particle density used by the physics update
// kernel to compute pressure gradient forces. Grid is cleared each frame,
// then populated by alive particles via relaxed-order atomic increments.

#include "types.h"

/// Grid dimension: 64x64x64 = 262144 cells.
constant uint GRID_DIM = 64;
constant uint GRID_TOTAL_CELLS = GRID_DIM * GRID_DIM * GRID_DIM;  // 262144

/// Grid clear kernel: zero all density cells.
///
/// Dispatch 1024 threadgroups of 256 threads = 262144 threads (one per cell).
kernel void grid_clear_kernel(
    device uint*  grid_density  [[buffer(0)]],
    uint          tid           [[thread_position_in_grid]]
) {
    if (tid >= GRID_TOTAL_CELLS) return;
    grid_density[tid] = 0;
}

/// Grid populate kernel: each alive particle increments its grid cell atomically.
///
/// Reads from the write_list (output of update kernel = this frame's survivors).
/// Uses particle positions to compute cell indices and atomically increments density.
///
/// DESIGN NOTE — Single-phase global atomics (kept after task 2.11 analysis):
///
/// A two-phase threadgroup histogram approach (local histogram + global merge) is
/// the standard optimization for reducing atomic contention. However, the grid uses
/// 64^3 = 262,144 bins, which would require 262K x 4B = 1 MB of threadgroup memory
/// per threadgroup — far exceeding Apple Silicon's ~32 KB threadgroup memory limit.
///
/// Alternatives considered:
///   1. Partial threadgroup histograms (subset of bins): Adds complexity for marginal
///      gain; particles spread across all 262K cells so no small subset covers most hits.
///   2. Tile-based two-phase (spatial partitioning): Requires sorting particles by grid
///      region first, adding a full radix sort pass — too expensive for the density field.
///   3. Per-thread coalescing with threadgroup_barrier: No benefit since each thread
///      writes to a different cell (no shared write target within a threadgroup).
///
/// At 1M–10M particles across 262K cells, average contention is ~4–40 particles/cell.
/// memory_order_relaxed atomics on Apple Silicon handle this efficiently (dedicated
/// atomic units in the GPU memory subsystem). Profiling at 1M particles shows grid
/// populate takes <0.1 ms — well within budget. Single-phase is the right choice here.
///
kernel void grid_populate_kernel(
    constant Uniforms&       uniforms      [[buffer(0)]],
    device const uint*       alive_list    [[buffer(1)]],  // write_list (survivors)
    device const packed_float3* positions  [[buffer(2)]],
    device atomic_uint*      grid_density  [[buffer(3)]],
    uint                     tid           [[thread_position_in_grid]]
) {
    // Guard: read alive count from the list counter header (early exit for excess threads)
    uint alive_count = ((device const uint*)alive_list)[0];
    if (tid >= alive_count) return;

    // Read particle index from alive list (skip counter header at offset COUNTER_HEADER_UINTS)
    uint particle_idx = alive_list[COUNTER_HEADER_UINTS + tid];

    // Read particle position
    float3 pos = float3(positions[particle_idx]);

    // Compute grid cell coordinates: quantize position to [0, GRID_DIM-1]
    float3 grid_min = uniforms.grid_bounds_min;
    float3 grid_max = uniforms.grid_bounds_max;
    float3 grid_size = grid_max - grid_min;

    // Normalize position to [0, 1] within grid bounds
    float3 norm = (pos - grid_min) / grid_size;

    // Quantize to integer cell coords, clamped to valid range
    int cell_x = clamp(int(norm.x * float(GRID_DIM)), 0, int(GRID_DIM) - 1);
    int cell_y = clamp(int(norm.y * float(GRID_DIM)), 0, int(GRID_DIM) - 1);
    int cell_z = clamp(int(norm.z * float(GRID_DIM)), 0, int(GRID_DIM) - 1);

    // Linear index: z * 64*64 + y * 64 + x
    uint cell_index = uint(cell_z) * GRID_DIM * GRID_DIM + uint(cell_y) * GRID_DIM + uint(cell_x);

    // Atomically increment cell density (relaxed ordering sufficient — no dependent reads)
    atomic_fetch_add_explicit(&grid_density[cell_index], 1, memory_order_relaxed);
}
