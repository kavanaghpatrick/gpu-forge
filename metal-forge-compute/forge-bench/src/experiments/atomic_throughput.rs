//! Atomic Throughput Scaling Sweep.
//!
//! Complements the existing `atomic_contention` experiment by sweeping
//! the NUMBER of target locations from 1 (max contention) to 32768
//! (near-zero contention) to map the full contention-throughput curve.
//!
//! Each thread does `num_passes` atomic_fetch_add to `counter[tid % N]`.
//!   N=1:     all threads → one location (serialized)
//!   N=32:    one target per SIMD group (moderate)
//!   N=1024:  ~one per threadgroup (low)
//!   N=32768: near-zero contention (max throughput)
//!
//! Reveals: at what contention level do atomics become effectively free?

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, ExploitParams, GpuTimer, MetalContext,
    PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

const TARGET_COUNTS: &[u32] = &[1, 4, 16, 32, 128, 1024, 8192, 32768];
const NUM_PASSES: u32 = 1000;

pub struct AtomicThroughputExperiment {
    buf_counters: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    target_times: Vec<(u32, f64)>, // (num_targets, ms)
    size: usize,
}

impl AtomicThroughputExperiment {
    pub fn new() -> Self {
        Self {
            buf_counters: None,
            pso_cache: PsoCache::new(),
            target_times: Vec::new(),
            size: 0,
        }
    }
}

impl Experiment for AtomicThroughputExperiment {
    fn name(&self) -> &str {
        "atomic_throughput"
    }

    fn description(&self) -> &str {
        "Atomic throughput sweep: ops/sec from max contention (1 target) to zero (32K targets)"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, _gen: &mut DataGenerator) {
        self.size = size;
        self.target_times.clear();

        // Allocate counter buffer for max target count
        let max_targets = *TARGET_COUNTS.last().unwrap() as usize;
        self.buf_counters = Some(alloc_buffer(
            &ctx.device,
            max_targets * std::mem::size_of::<u32>(),
        ));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_atomic_sweep");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let counters = self.buf_counters.as_ref().unwrap();
        self.target_times.clear();

        for &num_targets in TARGET_COUNTS {
            // Zero the counter buffer
            unsafe {
                let ptr = counters.contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(ptr, 0, counters.length());
            }

            let params = ExploitParams {
                element_count: self.size as u32,
                num_passes: NUM_PASSES,
                mode: num_targets,
                _pad: 0,
            };
            let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_atomic_sweep");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[(counters.as_ref(), 0), (buf_params.as_ref(), 1)],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let gpu_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
            self.target_times.push((num_targets, gpu_ms));
        }

        // Return 1-target time as primary
        self.target_times
            .first()
            .map(|(_, ms)| *ms)
            .unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        use std::sync::atomic::{AtomicU32, Ordering};

        let timer = BenchTimer::start();

        let counter = AtomicU32::new(0);
        for _ in 0..self.size.min(1_000_000) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        std::hint::black_box(counter.load(Ordering::Relaxed));

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.target_times.len() == TARGET_COUNTS.len() {
            Ok(())
        } else {
            Err(format!(
                "Expected {} results, got {}",
                TARGET_COUNTS.len(),
                self.target_times.len()
            ))
        }
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        let total_ops = self.size as f64 * NUM_PASSES as f64;

        for (targets, ms) in &self.target_times {
            m.insert(format!("targets_{}_ms", targets), *ms);

            // Atomic ops per second
            if *ms > 0.0 {
                let ops_per_sec = total_ops / (*ms / 1000.0);
                m.insert(format!("targets_{}_gops", targets), ops_per_sec / 1e9);
            }
        }

        // Contention penalty: time @ 1 target / time @ max targets
        if let (Some(worst), Some(best)) = (
            self.target_times.first(),
            self.target_times.last(),
        ) {
            if best.1 > 0.0 {
                m.insert("contention_penalty_x".to_string(), worst.1 / best.1);
            }
        }

        // Find the "knee" where adding targets stops helping significantly
        for i in 1..self.target_times.len() {
            let (_, prev_ms) = &self.target_times[i - 1];
            let (targets, curr_ms) = &self.target_times[i];
            if *prev_ms > 0.0 {
                let improvement = 1.0 - curr_ms / prev_ms;
                m.insert(format!("improvement_at_{}_pct", targets), improvement * 100.0);
            }
        }

        m
    }
}
