//! GPU timing infrastructure for kernel benchmarking.
//!
//! Provides:
//! - `benchmark_kernel_gpu_time`: Accurate GPU-side timing via GPUStartTime/GPUEndTime
//! - `gpu_warmup`: Prime GPU caches and pipeline state with throwaway dispatches
//! - `ProtoResult`: Benchmark result struct for prototype comparison
//! - Optional `MTLCounterSampleBuffer` support behind `gpu-counters` feature

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
};

/// Benchmark result from a single prototype measurement.
///
/// Used across all 8 prototypes to produce comparable findings for KB ingestion.
#[derive(Debug, Clone)]
pub struct ProtoResult {
    /// Prototype identifier (e.g., "proto1_flash", "proto3_paged")
    pub prototype: &'static str,
    /// Variant description (e.g., "Br=32,Bc=64,D=128")
    pub variant: String,
    /// Wall-clock GPU time in microseconds (from GPUEndTime - GPUStartTime)
    pub wall_time_us: f64,
    /// Throughput in GFLOPS
    pub throughput_gflops: f64,
    /// Memory bandwidth in GB/s
    pub bandwidth_gbps: f64,
    /// Memory allocated in bytes for this dispatch
    pub memory_bytes: u64,
    /// Free-form notes (tile config, occupancy, etc.)
    pub notes: String,
}

impl ProtoResult {
    /// Create a ProtoResult from GPU time in seconds and compute parameters.
    ///
    /// `flops` is the total floating-point operations for this kernel dispatch.
    /// `bytes_transferred` is the total bytes read+written by the kernel.
    pub fn from_gpu_time(
        prototype: &'static str,
        variant: String,
        gpu_time_secs: f64,
        flops: u64,
        bytes_transferred: u64,
        memory_bytes: u64,
        notes: String,
    ) -> Self {
        let wall_time_us = gpu_time_secs * 1_000_000.0;
        let throughput_gflops = if gpu_time_secs > 0.0 {
            flops as f64 / gpu_time_secs / 1e9
        } else {
            0.0
        };
        let bandwidth_gbps = if gpu_time_secs > 0.0 {
            bytes_transferred as f64 / gpu_time_secs / 1e9
        } else {
            0.0
        };

        Self {
            prototype,
            variant,
            wall_time_us,
            throughput_gflops,
            bandwidth_gbps,
            memory_bytes,
            notes,
        }
    }

    /// GPU time in seconds (convenience for Criterion integration).
    pub fn gpu_time_secs(&self) -> f64 {
        self.wall_time_us / 1_000_000.0
    }

    /// Throughput in TFLOPS (convenience for reporting).
    pub fn throughput_tflops(&self) -> f64 {
        self.throughput_gflops / 1000.0
    }
}

/// Measure GPU kernel execution time using Metal command buffer timestamps.
///
/// Creates a command buffer, encodes one compute dispatch, commits, waits,
/// and returns `GPUEndTime - GPUStartTime` in seconds.
///
/// This gives hardware-accurate GPU execution time, excluding CPU-side
/// dispatch overhead and command buffer scheduling latency.
///
/// # Arguments
/// * `queue` - Metal command queue to create the command buffer from
/// * `pso` - Compiled compute pipeline state for the kernel
/// * `buffers` - Slice of (buffer, argument_index) pairs to bind
/// * `grid_size` - Total thread grid dimensions
/// * `threadgroup_size` - Threads per threadgroup dimensions
///
/// # Returns
/// GPU execution time in seconds (f64), or panics on Metal errors.
pub fn benchmark_kernel_gpu_time(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
) -> f64 {
    let cmd_buf = queue
        .commandBuffer()
        .expect("Failed to create command buffer for timing");

    // Encode a single compute dispatch
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute command encoder");

        encoder.setComputePipelineState(pso);

        unsafe {
            for (buffer, index) in buffers {
                encoder.setBuffer_offset_atIndex(Some(*buffer), 0, *index);
            }
        }

        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
    }

    // Commit and wait for GPU to complete
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    // Extract GPU-side timestamps (seconds since system boot, f64)
    let gpu_start = cmd_buf.GPUStartTime();
    let gpu_end = cmd_buf.GPUEndTime();

    gpu_end - gpu_start
}

/// Warm up the GPU by running throwaway dispatches.
///
/// Performs 4 dispatch-commit-wait cycles to:
/// - Prime the GPU pipeline state cache
/// - Warm up shader instruction caches
/// - Stabilize GPU clock frequency (prevent thermal throttle ramp)
/// - Ensure consistent subsequent measurements
///
/// # Arguments
/// * `queue` - Metal command queue
/// * `pso` - Compiled compute pipeline state
/// * `buffers` - Slice of (buffer, argument_index) pairs to bind
/// * `grid_size` - Total thread grid dimensions
/// * `threadgroup_size` - Threads per threadgroup dimensions
pub fn gpu_warmup(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
) {
    for _ in 0..4 {
        let cmd_buf = queue
            .commandBuffer()
            .expect("Failed to create command buffer for warmup");

        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute command encoder for warmup");

            encoder.setComputePipelineState(pso);

            unsafe {
                for (buffer, index) in buffers {
                    encoder.setBuffer_offset_atIndex(Some(*buffer), 0, *index);
                }
            }

            encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }
}

// --- Optional GPU Counter Support ---

/// GPU counter sampling support via MTLCounterSampleBuffer.
///
/// When the `gpu-counters` feature is enabled, this module provides
/// hardware performance counter access for detailed GPU profiling
/// (ALU utilization, cache hit rates, occupancy).
///
/// Currently stubbed out — actual implementation deferred until
/// prototype benchmarks identify which counters are most valuable.
#[cfg(feature = "gpu-counters")]
pub mod counters {
    /// Placeholder for GPU counter configuration.
    ///
    /// Will be populated with MTLCounterSampleBuffer setup when needed.
    /// Requires Metal GPU Family Apple 7+ (M1 and later).
    pub struct GpuCounterConfig {
        /// Whether the device supports counter sampling.
        pub supported: bool,
    }

    impl GpuCounterConfig {
        /// Check if the current device supports GPU counter sampling.
        ///
        /// Returns a config indicating support status. Actual counter
        /// buffer creation is deferred to when benchmarks need it.
        pub fn probe() -> Self {
            // TODO: Check device.supportsCounterSampling(.atStageBoundary)
            // and enumerate available counter sets
            Self { supported: false }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::pipeline::{PsoCache, PsoKey};
    use objc2_metal::MTLSize;

    #[test]
    fn test_proto_result() {
        let result = ProtoResult::from_gpu_time(
            "proto1_flash",
            "Br=32,Bc=64,D=128".to_string(),
            0.001, // 1ms
            1_000_000_000,
            500_000_000,
            64 * 1024 * 1024,
            "test config".to_string(),
        );

        assert_eq!(result.prototype, "proto1_flash");
        assert!((result.wall_time_us - 1000.0).abs() < 0.01);
        assert!((result.throughput_gflops - 1000.0).abs() < 0.01);
        assert!((result.bandwidth_gbps - 500.0).abs() < 0.01);
        assert_eq!(result.memory_bytes, 64 * 1024 * 1024);
        assert!((result.gpu_time_secs() - 0.001).abs() < 1e-9);
        assert!((result.throughput_tflops() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gpu_warmup() {
        let gpu = GpuDevice::shared();

        // Compile the _stub kernel from the stub metallib
        let mut cache = PsoCache::new(gpu.library.clone());
        let key = PsoKey::simple("_stub");
        let pso = cache.get_or_compile(&key);

        let grid_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        // _stub kernel takes no buffers — dispatch with empty buffer list
        gpu_warmup(&gpu.command_queue, pso, &[], grid_size, tg_size);

        // Also test benchmark_kernel_gpu_time returns a valid duration
        let gpu_time =
            benchmark_kernel_gpu_time(&gpu.command_queue, pso, &[], grid_size, tg_size);

        // GPU time should be non-negative (could be very small for stub kernel)
        assert!(
            gpu_time >= 0.0,
            "GPU time should be non-negative, got: {}",
            gpu_time
        );
        // Sanity: should be less than 1 second for a trivial stub
        assert!(
            gpu_time < 1.0,
            "GPU time unreasonably large for stub kernel: {}",
            gpu_time
        );
    }

    #[cfg(feature = "gpu-counters")]
    #[test]
    fn test_gpu_counter_probe() {
        let config = counters::GpuCounterConfig::probe();
        // Just verify it doesn't crash — actual support check is stubbed
        let _ = config.supported;
    }
}
