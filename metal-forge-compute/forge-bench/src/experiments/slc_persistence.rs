//! SLC Data Persistence Across Dispatches.
//!
//! Tests whether data written by one Metal dispatch remains in the System Level
//! Cache (SLC) when a second dispatch reads it back via a NEW command buffer.
//!
//! For each buffer size (256KB → 128MB):
//!   1. Dispatch WRITE (fills buffer, warms SLC for sizes that fit)
//!   2. Dispatch READ (new command buffer) — measure bandwidth
//!
//! If SLC persists: small-buffer reads show higher bandwidth than DRAM limit.
//! Large-buffer reads show DRAM bandwidth (~273 GB/s theoretical on M4 Pro).
//! The crossover reveals SLC effective capacity for cross-dispatch persistence.

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

// Buffer sizes: 256KB to 128MB (in u32 elements = bytes/4)
const BUFFER_SIZES: &[(usize, &str)] = &[
    (65536, "256KB"),       // 256 KB — deep L1 / SLC entry
    (262144, "1MB"),        // 1 MB — SLC
    (1048576, "4MB"),       // 4 MB — SLC
    (4194304, "16MB"),      // 16 MB — SLC boundary
    (8388608, "32MB"),      // 32 MB — borderline SLC
    (16777216, "64MB"),     // 64 MB — past SLC
    (33554432, "128MB"),    // 128 MB — pure DRAM
];

pub struct SlcPersistenceExperiment {
    bufs_data: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_read_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    bandwidths: Vec<(String, f64)>, // (label, GB/s)
    size: usize,
}

impl SlcPersistenceExperiment {
    pub fn new() -> Self {
        Self {
            bufs_data: Vec::new(),
            buf_read_output: None,
            pso_cache: PsoCache::new(),
            bandwidths: Vec::new(),
            size: 0,
        }
    }
}

impl Experiment for SlcPersistenceExperiment {
    fn name(&self) -> &str {
        "slc_persistence"
    }

    fn description(&self) -> &str {
        "SLC persistence: does data survive across Metal dispatches? Bandwidth at 256KB-128MB"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1024]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, _gen: &mut DataGenerator) {
        self.size = size;
        self.bufs_data.clear();
        self.bandwidths.clear();

        // Allocate buffers for each test size
        for (elems, _label) in BUFFER_SIZES {
            self.bufs_data
                .push(alloc_buffer(&ctx.device, *elems * 4));
        }

        // Read output buffer (for the reduce kernel)
        let max_simdgroups = BUFFER_SIZES.last().unwrap().0 / 4 / 32 + 1;
        self.buf_read_output = Some(alloc_buffer(
            &ctx.device,
            max_simdgroups * std::mem::size_of::<f32>(),
        ));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_slc_write");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_slc_read_bw");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let read_output = self.buf_read_output.as_ref().unwrap();
        self.bandwidths.clear();

        for (i, (elems, label)) in BUFFER_SIZES.iter().enumerate() {
            let data_buf = &self.bufs_data[i];

            // ── DISPATCH 1: Write (warms SLC) ──
            let write_params = ExploitParams {
                element_count: *elems as u32,
                num_passes: 1,
                mode: 0,
                _pad: 0,
            };
            let buf_wp = alloc_buffer_with_data(&ctx.device, &[write_params]);

            let cmd_write = ctx.queue.commandBuffer().unwrap();
            let pso_write = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_slc_write");
            let enc_w = cmd_write.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc_w,
                pso_write,
                &[(data_buf.as_ref(), 0), (buf_wp.as_ref(), 1)],
                *elems,
            );
            enc_w.endEncoding();
            cmd_write.commit();
            cmd_write.waitUntilCompleted();

            // ── DISPATCH 2: Read bandwidth (NEW command buffer) ──
            let read_params = ExploitParams {
                element_count: *elems as u32,
                num_passes: 1,
                mode: 0,
                _pad: 0,
            };
            let buf_rp = alloc_buffer_with_data(&ctx.device, &[read_params]);

            let cmd_read = ctx.queue.commandBuffer().unwrap();
            let pso_read = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_slc_read_bw");
            let enc_r = cmd_read.computeCommandEncoder().unwrap();
            // Read kernel expects float4* input — same buffer reinterpreted
            dispatch_1d(
                &enc_r,
                pso_read,
                &[
                    (data_buf.as_ref(), 0),
                    (read_output.as_ref(), 1),
                    (buf_rp.as_ref(), 2),
                ],
                *elems / 4, // float4 elements
            );
            enc_r.endEncoding();
            cmd_read.commit();
            cmd_read.waitUntilCompleted();

            let gpu_ms = GpuTimer::elapsed_ms(&cmd_read).unwrap_or(0.0);
            let bytes = *elems * 4;
            let gbs = if gpu_ms > 0.0 {
                (bytes as f64 / 1e9) / (gpu_ms / 1000.0)
            } else {
                0.0
            };

            self.bandwidths.push((label.to_string(), gbs));
        }

        // Return 4MB bandwidth as representative
        self.bandwidths
            .iter()
            .find(|(l, _)| l == "4MB")
            .map(|(_, gbs)| *gbs)
            .unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU sequential read of 4MB
        let data: Vec<f32> = (0..1048576).map(|i| i as f32).collect();
        let mut sum = 0.0f64;
        for &v in &data {
            sum += v as f64;
        }
        std::hint::black_box(sum);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.bandwidths.len() == BUFFER_SIZES.len() {
            Ok(())
        } else {
            Err(format!(
                "Expected {} results, got {}",
                BUFFER_SIZES.len(),
                self.bandwidths.len()
            ))
        }
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        for (label, gbs) in &self.bandwidths {
            m.insert(format!("bw_{}_gbs", label), *gbs);
        }

        // Compute ratio: small-buffer BW / large-buffer BW
        // Ratio > 1 proves SLC persistence
        if let (Some(small), Some(large)) = (
            self.bandwidths.iter().find(|(l, _)| l == "1MB"),
            self.bandwidths.iter().find(|(l, _)| l == "128MB"),
        ) {
            if large.1 > 0.0 {
                m.insert("slc_vs_dram_ratio".to_string(), small.1 / large.1);
            }
        }

        // Find the crossover point (where bandwidth drops by >20%)
        for i in 1..self.bandwidths.len() {
            let (_, prev_bw) = &self.bandwidths[i - 1];
            let (curr_label, curr_bw) = &self.bandwidths[i];
            if *prev_bw > 0.0 {
                let drop = 1.0 - curr_bw / prev_bw;
                m.insert(format!("drop_at_{}_pct", curr_label), drop * 100.0);
            }
        }

        m
    }
}
