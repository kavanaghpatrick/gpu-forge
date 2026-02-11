// prepare_dispatch.metal -- GPU-side emission count computation and dispatch arg generation.
//
// Single-thread kernel: reads dead count, computes emission count, writes indirect
// dispatch arguments and GPU emission parameters. Also resets write list counter.
//
// Runs at the start of each frame's GPU pipeline, replacing CPU readback of dead_count
// and CPU computation of emission threadgroups. See design Section 3.1, 3.2.
//
// KB 134: ~120us empty kernel overhead on M2 Max, dominated by encoder setup.
// KB 277: Indirect dispatch eliminates CPU-GPU sync for GPU-generated dispatch args.

#include "types.h"

/// Threadgroup size for emission kernel (must match emission_kernel dispatch).
constant uint EMISSION_THREADGROUP_SIZE = 256;

kernel void prepare_dispatch(
    device const uint*         dead_list         [[buffer(0)]],  // read: dead_count at offset 0
    constant Uniforms&         uniforms          [[buffer(1)]],  // read: base_emission_rate, burst_count, pool_size
    device uint*               write_list        [[buffer(2)]],  // write: reset counter to 0
    device DispatchArgs*       emission_args     [[buffer(3)]],  // write: indirect dispatch args for emission
    device GpuEmissionParams*  emission_params   [[buffer(4)]],  // write: computed emission count for emission kernel
    uint                       tid               [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // --- Reset write list counter to 0 ---
    // The write list counter is at offset 0 (first uint of CounterHeader).
    // This replaces the CPU-side: (*write_ptr).count = 0
    write_list[0] = 0;

    // --- Read dead count ---
    // dead_list[0] is the atomic counter value. At this point in the pipeline,
    // no other kernel is running, so a non-atomic read is safe.
    uint dead_count = dead_list[0];

    // --- Compute emission count (GPU-side, no CPU readback needed) ---
    uint base_emission = uniforms.base_emission_rate;
    uint burst = uniforms.burst_count;
    uint total_requested = base_emission + burst;
    uint emission_count = min(total_requested, dead_count);

    // Actual burst count must not exceed total emission
    uint actual_burst = min(burst, emission_count);

    // --- Write GPU emission parameters (for emission kernel to read) ---
    emission_params->emission_count = emission_count;
    emission_params->actual_burst_count = actual_burst;

    // --- Write indirect dispatch args for emission kernel ---
    // Clamp threadgroups to pool_size / THREADGROUP_SIZE as defense-in-depth
    uint threadgroups = (emission_count + EMISSION_THREADGROUP_SIZE - 1) / EMISSION_THREADGROUP_SIZE;
    uint max_threadgroups = (uniforms.pool_size + EMISSION_THREADGROUP_SIZE - 1) / EMISSION_THREADGROUP_SIZE;
    threadgroups = min(threadgroups, max_threadgroups);

    emission_args->threadgroupsPerGridX = threadgroups;
    emission_args->threadgroupsPerGridY = 1;
    emission_args->threadgroupsPerGridZ = 1;
}
