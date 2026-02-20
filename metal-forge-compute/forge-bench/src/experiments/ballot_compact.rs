//! SIMD Ballot 32x Boolean Compaction experiment.
//!
//! simd_ballot(condition) packs 32 booleans into a single uint32.
//! This is a SINGLE INSTRUCTION that does 32x data compaction.
//! Combined with popcount, it gives free histogram bin counting.
//!
//! Compare: ballot + popcount  vs  per-element atomic_fetch_add

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

pub struct BallotCompactExperiment {
    data: Vec<u32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_ballots: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_count_ballot: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_count_atomic: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_ballot_ms: f64,
    gpu_atomic_ms: f64,
    ballot_count: u32,
    atomic_count: u32,
    size: usize,
}

impl BallotCompactExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_ballots: None,
            buf_count_ballot: None,
            buf_count_atomic: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_ballot_ms: 0.0,
            gpu_atomic_ms: 0.0,
            ballot_count: 0,
            atomic_count: 0,
            size: 0,
        }
    }
}

impl Experiment for BallotCompactExperiment {
    fn name(&self) -> &str {
        "ballot_compact"
    }

    fn description(&self) -> &str {
        "SIMD ballot (32x bool compaction + popcount) vs per-element atomic counting"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        self.data = gen.uniform_u32(size);

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Ballot output: one uint per simdgroup (32 threads)
        let num_simdgroups = size.div_ceil(32);
        self.buf_ballots = Some(alloc_buffer(
            &ctx.device,
            num_simdgroups * std::mem::size_of::<u32>(),
        ));

        // Count buffers (1 atomic_uint each)
        self.buf_count_ballot = Some(alloc_buffer(&ctx.device, std::mem::size_of::<u32>()));
        self.buf_count_atomic = Some(alloc_buffer(&ctx.device, std::mem::size_of::<u32>()));

        // Threshold at ~50% of u32 range
        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 1,
            mode: u32::MAX / 2,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_ballot_count");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_atomic_count");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let ballots = self.buf_ballots.as_ref().unwrap();
        let count_ballot = self.buf_count_ballot.as_ref().unwrap();
        let count_atomic = self.buf_count_atomic.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        // Zero count buffers
        unsafe {
            *(count_ballot.contents().as_ptr() as *mut u32) = 0;
            *(count_atomic.contents().as_ptr() as *mut u32) = 0;
        }

        // --- Ballot + popcount ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_ballot_count");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (ballots.as_ref(), 1),
                    (count_ballot.as_ref(), 2),
                    (params.as_ref(), 3),
                ],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_ballot_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
            self.ballot_count = unsafe { *(count_ballot.contents().as_ptr() as *const u32) };
        }

        // --- Per-element atomic ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_atomic_count");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (ballots.as_ref(), 1), // unused but needed for API compat
                    (count_atomic.as_ref(), 2),
                    (params.as_ref(), 3),
                ],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_atomic_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
            self.atomic_count = unsafe { *(count_atomic.contents().as_ptr() as *const u32) };
        }

        self.gpu_ballot_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        let threshold = u32::MAX / 2;
        let mut count = 0u64;
        for &v in &self.data {
            if v > threshold {
                count += 1;
            }
        }
        std::hint::black_box(count);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.ballot_count == self.atomic_count {
            Ok(())
        } else {
            Err(format!(
                "Count mismatch: ballot={}, atomic={}",
                self.ballot_count, self.atomic_count
            ))
        }
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("ballot_ms".to_string(), self.gpu_ballot_ms);
        m.insert("atomic_ms".to_string(), self.gpu_atomic_ms);

        let speedup = if self.gpu_ballot_ms > 0.0 {
            self.gpu_atomic_ms / self.gpu_ballot_ms
        } else {
            0.0
        };
        m.insert("ballot_speedup_x".to_string(), speedup);

        m.insert("matches_found".to_string(), self.ballot_count as f64);
        m.insert(
            "match_pct".to_string(),
            (self.ballot_count as f64 / size as f64) * 100.0,
        );

        // Bandwidth: reading input data
        let bytes = size as f64 * 4.0; // uint = 4 bytes
        let gbs = if self.gpu_ballot_ms > 0.0 {
            bytes / (self.gpu_ballot_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
