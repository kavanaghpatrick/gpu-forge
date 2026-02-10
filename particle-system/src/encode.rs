//! Compute dispatch encoding helpers for the GPU particle pipeline.
//!
//! Each function encapsulates one compute pass: creates encoder, sets pipeline,
//! binds buffers, dispatches, and ends encoding. This keeps render() as a
//! concise high-level orchestration of the 6-pass compute pipeline.

use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};

use crate::buffers::ParticlePool;
use crate::gpu::GpuState;

/// Encode the prepare_dispatch compute pass.
///
/// GPU-side computation of emission parameters and dispatch args.
/// Reads dead_count, computes emission_count, writes indirect dispatch args.
/// Also resets write_list counter to 0 (replaces CPU write_list reset).
///
/// Buffer bindings:
///   buffer(0) = dead_list, buffer(1) = uniform_ring, buffer(2) = write_list,
///   buffer(3) = emission_dispatch_args, buffer(4) = gpu_emission_params
pub fn encode_prepare_dispatch(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    gpu: &GpuState,
    pool: &ParticlePool,
    write_list: &ProtocolObject<dyn MTLBuffer>,
    uniform_offset: usize,
) {
    if let Some(encoder) = cmd_buf.computeCommandEncoder() {
        encoder.setLabel(Some(ns_string!("Prepare Dispatch")));
        encoder.setComputePipelineState(&gpu.prepare_dispatch_pipeline);

        // Safety: buffer pointers are valid for the lifetime of the command buffer.
        // All buffers are StorageModeShared and allocated by ParticlePool.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&pool.dead_list), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&pool.uniform_ring), uniform_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(write_list), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&pool.emission_dispatch_args), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&pool.gpu_emission_params), 0, 4);
        }

        // Single threadgroup of 32 threads (SIMD-aligned, only thread 0 does work)
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );

        encoder.endEncoding();
    }
}

/// Encode the emission compute pass.
///
/// Emission appends new particles to the read list (alongside last frame's survivors).
/// Uses indirect dispatch: threadgroup count from GPU-computed emission_dispatch_args.
///
/// Buffer bindings:
///   buffer(0) = uniform_ring, buffer(1) = dead_list, buffer(2) = read_list,
///   buffer(3) = positions, buffer(4) = velocities, buffer(5) = lifetimes,
///   buffer(6) = colors, buffer(7) = sizes, buffer(8) = gpu_emission_params
pub fn encode_emission(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    gpu: &GpuState,
    pool: &ParticlePool,
    read_list: &ProtocolObject<dyn MTLBuffer>,
    uniform_offset: usize,
) {
    if let Some(encoder) = cmd_buf.computeCommandEncoder() {
        encoder.setLabel(Some(ns_string!("Emission")));
        encoder.setComputePipelineState(&gpu.emission_pipeline);

        // Safety: buffer pointers are valid for the lifetime of the command buffer.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&pool.uniform_ring), uniform_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(&pool.dead_list), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(read_list), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&pool.positions), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&pool.velocities), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&pool.lifetimes), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&pool.colors), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&pool.sizes), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&pool.gpu_emission_params), 0, 8);
        }

        // Indirect dispatch: threadgroup count from prepare_dispatch kernel output
        unsafe {
            encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &pool.emission_dispatch_args,
                0,
                MTLSize { width: 256, height: 1, depth: 1 },
            );
        }

        encoder.endEncoding();
    }
}

/// Encode the grid clear compute pass.
///
/// Zeros all 262144 cells in the grid density buffer.
/// Must run BEFORE grid_populate and update so update kernel reads fresh grid data.
///
/// Buffer bindings: buffer(0) = grid_density
pub fn encode_grid_clear(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    gpu: &GpuState,
    pool: &ParticlePool,
) {
    if let Some(encoder) = cmd_buf.computeCommandEncoder() {
        encoder.setLabel(Some(ns_string!("Grid Clear")));
        encoder.setComputePipelineState(&gpu.grid_clear_pipeline);
        // Safety: grid_density buffer is valid for the lifetime of the command buffer.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&pool.grid_density), 0, 0);
        }
        // 262144 cells / 256 threads per group = 1024 threadgroups
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1024, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        encoder.endEncoding();
    }
}

/// Encode the grid populate compute pass.
///
/// Each alive particle atomically increments its grid cell density.
/// Reads from read_list (last frame's survivors + this frame's new emissions).
/// Must run BEFORE update so the update kernel can read the density field.
///
/// Buffer bindings:
///   buffer(0) = uniform_ring, buffer(1) = read_list,
///   buffer(2) = positions, buffer(3) = grid_density
pub fn encode_grid_populate(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    gpu: &GpuState,
    pool: &ParticlePool,
    read_list: &ProtocolObject<dyn MTLBuffer>,
    uniform_offset: usize,
) {
    if let Some(encoder) = cmd_buf.computeCommandEncoder() {
        encoder.setLabel(Some(ns_string!("Grid Populate")));
        encoder.setComputePipelineState(&gpu.grid_populate_pipeline);
        // Safety: buffer pointers are valid for the lifetime of the command buffer.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&pool.uniform_ring), uniform_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(read_list), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&pool.positions), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&pool.grid_density), 0, 3);
        }
        // Indirect dispatch: threadgroup count from sync_indirect_args output (update_dispatch_args)
        unsafe {
            encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &pool.update_dispatch_args,
                0,
                MTLSize { width: 256, height: 1, depth: 1 },
            );
        }
        encoder.endEncoding();
    }
}

/// Encode the physics update compute pass.
///
/// Reads from read_list (last frame's survivors + this frame's new emissions).
/// Reads grid_density for pressure gradient force.
/// Writes survivors to write_list, writes dead particles back to dead_list.
///
/// Buffer bindings:
///   buffer(0) = uniform_ring, buffer(1) = dead_list, buffer(2) = read_list,
///   buffer(3) = write_list, buffer(4-8) = SoA data, buffer(9) = grid_density
pub fn encode_update(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    gpu: &GpuState,
    pool: &ParticlePool,
    read_list: &ProtocolObject<dyn MTLBuffer>,
    write_list: &ProtocolObject<dyn MTLBuffer>,
    uniform_offset: usize,
) {
    if let Some(encoder) = cmd_buf.computeCommandEncoder() {
        encoder.setLabel(Some(ns_string!("Physics Update")));
        encoder.setComputePipelineState(&gpu.update_pipeline);

        // Safety: buffer pointers are valid for the lifetime of the command buffer.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&pool.uniform_ring), uniform_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(&pool.dead_list), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(read_list), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(write_list), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&pool.positions), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&pool.velocities), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&pool.lifetimes), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&pool.colors), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&pool.sizes), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(&pool.grid_density), 0, 9);
        }

        // Indirect dispatch: threadgroup count from sync_indirect_args output (update_dispatch_args)
        unsafe {
            encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &pool.update_dispatch_args,
                0,
                MTLSize { width: 256, height: 1, depth: 1 },
            );
        }

        encoder.endEncoding();
    }
}

/// Encode the sync_indirect_args compute pass.
///
/// Reads from write_list (update kernel output = this frame's survivors).
/// Writes draw args for indirect rendering and next-frame update_dispatch_args.
///
/// Buffer bindings:
///   buffer(0) = write_list, buffer(1) = indirect_args, buffer(2) = update_dispatch_args
pub fn encode_sync_indirect(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    gpu: &GpuState,
    pool: &ParticlePool,
    write_list: &ProtocolObject<dyn MTLBuffer>,
) {
    if let Some(encoder) = cmd_buf.computeCommandEncoder() {
        encoder.setLabel(Some(ns_string!("Compaction")));
        encoder.setComputePipelineState(&gpu.sync_indirect_pipeline);
        // Safety: buffer pointers are valid for the lifetime of the command buffer.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(write_list), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&pool.indirect_args), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&pool.update_dispatch_args), 0, 2);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: 1, height: 1, depth: 1 },
        );
        encoder.endEncoding();
    }
}
