//! Cross-Threadgroup Communication via Device Memory.
//!
//! Tests whether threadgroups within a single Metal dispatch can communicate
//! through device memory using atomic release/acquire semantics.
//!
//! Design: even TGs = producers, odd TGs = consumers.
//!   Producer: writes data pattern, sets atomic flag (release)
//!   Consumer: polls atomic flag (acquire), reads data, verifies
//!
//! Measures:
//!   - Correctness: did data arrive intact?
//!   - Latency: how many spin iterations before flag visible?
//!   - Coverage: what % of consumer TGs succeeded vs timed out?
//!
//! This is potentially the biggest breakthrough: if cross-TG communication
//! works reliably, it enables GPU-side schedulers, queues, and coordination.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, read_buffer, ExploitParams, GpuTimer,
    MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

// Number of TG pairs: each pair = 1 producer + 1 consumer
const TG_PAIR_COUNTS: &[usize] = &[16, 64, 256, 1024, 4096];

pub struct CrossTgExperiment {
    buf_data: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_flags: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_results: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    // (num_pairs, matches, avg_spins, success_pct)
    pair_results: Vec<(usize, u32, f64, f64)>,
    size: usize,
}

impl CrossTgExperiment {
    pub fn new() -> Self {
        Self {
            buf_data: None,
            buf_flags: None,
            buf_results: None,
            pso_cache: PsoCache::new(),
            pair_results: Vec::new(),
            size: 0,
        }
    }
}

impl Experiment for CrossTgExperiment {
    fn name(&self) -> &str {
        "cross_tg"
    }

    fn description(&self) -> &str {
        "Cross-TG communication: producer-consumer via device memory atomics"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1024]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, _gen: &mut DataGenerator) {
        self.size = size;
        self.pair_results.clear();

        let max_pairs = *TG_PAIR_COUNTS.last().unwrap();

        // Data buffer: 256 u32 per pair
        self.buf_data = Some(alloc_buffer(
            &ctx.device,
            max_pairs * 256 * std::mem::size_of::<u32>(),
        ));

        // Flags buffer: 1 atomic u32 per pair
        self.buf_flags = Some(alloc_buffer(
            &ctx.device,
            max_pairs * std::mem::size_of::<u32>(),
        ));

        // Results buffer: [matches, total_spins, success_count, timeout_count]
        self.buf_results = Some(alloc_buffer(
            &ctx.device,
            4 * std::mem::size_of::<u32>(),
        ));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_cross_tg");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        self.pair_results.clear();

        for &num_pairs in TG_PAIR_COUNTS {
            let data_buf = self.buf_data.as_ref().unwrap();
            let flags_buf = self.buf_flags.as_ref().unwrap();
            let results_buf = self.buf_results.as_ref().unwrap();

            // Zero flags and results
            unsafe {
                let ptr = flags_buf.contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(ptr, 0, num_pairs * 4);
                let ptr = results_buf.contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(ptr, 0, 16);
            }

            let num_tgs = num_pairs * 2; // producer + consumer per pair
            let threads_per_tg = 256usize;
            let total_threads = num_tgs * threads_per_tg;

            let params = ExploitParams {
                element_count: total_threads as u32,
                num_passes: 1,
                mode: num_pairs as u32,
                _pad: 0,
            };
            let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_cross_tg");
            let enc = cmd.computeCommandEncoder().unwrap();

            // Manual dispatch with explicit TG size
            use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

            enc.setComputePipelineState(pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(data_buf.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(flags_buf.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(results_buf.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_params.as_ref()), 0, 3);
            }

            let grid = MTLSize {
                width: num_tgs,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();

            cmd.commit();
            cmd.waitUntilCompleted();

            // Read results
            let results: [u32; 4] = unsafe {
                let ptr = results_buf.contents().as_ptr() as *const [u32; 4];
                *ptr
            };

            let matches = results[0];
            let total_spins = results[1];
            let successes = results[2];
            let timeouts = results[3];

            let avg_spins = if successes > 0 {
                total_spins as f64 / successes as f64
            } else {
                f64::INFINITY
            };

            let total_consumers = successes + timeouts;
            let success_pct = if total_consumers > 0 {
                successes as f64 / total_consumers as f64 * 100.0
            } else {
                0.0
            };

            self.pair_results
                .push((num_pairs, matches, avg_spins, success_pct));
        }

        // Return success percentage of the middle test as primary metric
        let mid = self.pair_results.len() / 2;
        self.pair_results
            .get(mid)
            .map(|(_, _, _, pct)| *pct)
            .unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU: simple producer-consumer with atomic flag
        use std::sync::atomic::{AtomicU32, Ordering};
        let flag = AtomicU32::new(0);
        flag.store(1, Ordering::Release);
        let v = flag.load(Ordering::Acquire);
        std::hint::black_box(v);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.pair_results.is_empty() {
            return Err("No results collected".to_string());
        }
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        for (pairs, matches, avg_spins, success_pct) in &self.pair_results {
            m.insert(format!("pairs_{}_matches", pairs), *matches as f64);
            m.insert(format!("pairs_{}_avg_spins", pairs), *avg_spins);
            m.insert(format!("pairs_{}_success_pct", pairs), *success_pct);

            let expected_matches = *pairs * 256;
            let correctness_pct = if expected_matches > 0 {
                *matches as f64 / expected_matches as f64 * 100.0
            } else {
                0.0
            };
            m.insert(
                format!("pairs_{}_correctness_pct", pairs),
                correctness_pct,
            );
        }

        // Overall: is cross-TG communication reliable?
        let all_success = self
            .pair_results
            .iter()
            .all(|(_, _, _, pct)| *pct > 95.0);
        m.insert(
            "cross_tg_reliable".to_string(),
            if all_success { 1.0 } else { 0.0 },
        );

        m
    }
}
