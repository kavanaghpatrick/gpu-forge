//! GPU Work-Stealing Scheduler.
//!
//! Per-threadgroup work queues in device memory. Each TG drains its local
//! queue first, then steals from random victims via device atomics.
//!
//! Configurations:
//!   mode 0: Local-only processing (no stealing) — baseline
//!   mode 1: Work-stealing enabled (balanced workload)
//!   mode 2: Imbalanced workload (TG 0 gets 90%) + stealing
//!
//! The "holy shit" moment: work-stealing should redistribute the imbalanced
//! workload, matching or exceeding the balanced local-only throughput.
//!
//! KB foundations:
//!   Finding 3229: cross-TG device atomics 100% reliable
//!   Finding 3239: per-SIMD atomic aggregation 2.26x relief
//!   Finding 3231: SLC retains 16MB at 465 GB/s

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, GpuOsParams, GpuTimer, MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

const THREADS_PER_TG: usize = 256;
const QUEUE_CAPACITY: u32 = 1024; // per-TG queue capacity

// Sweep configurations: (num_tgs, mode, label)
const CONFIGS: &[(u32, u32, &str)] = &[
    (64, 0, "64tg_local"),
    (64, 1, "64tg_steal"),
    (64, 2, "64tg_imbal"),
    (256, 0, "256tg_local"),
    (256, 1, "256tg_steal"),
    (256, 2, "256tg_imbal"),
];

pub struct GpuSchedulerExperiment {
    buf_queues: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_steal_counts: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_total_done: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    // Results: (label, tasks_done, steal_count, gpu_ms)
    results: Vec<(String, u32, u32, f64)>,

    pso_cache: PsoCache,
    size: usize,
}

impl GpuSchedulerExperiment {
    pub fn new() -> Self {
        Self {
            buf_queues: None,
            buf_steal_counts: None,
            buf_total_done: None,
            results: Vec::new(),
            pso_cache: PsoCache::new(),
            size: 0,
        }
    }
}

impl Experiment for GpuSchedulerExperiment {
    fn name(&self) -> &str {
        "gpu_scheduler"
    }

    fn description(&self) -> &str {
        "GPU work-stealing scheduler: per-TG queues, cross-TG steal via device atomics"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1024] // Special experiment — internally sweeps TG configs
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, _gen: &mut DataGenerator) {
        self.size = size;
        self.results.clear();

        let max_tgs = CONFIGS.iter().map(|(t, _, _)| *t).max().unwrap() as usize;
        let queue_stride = 2 + QUEUE_CAPACITY as usize;

        // Allocate for largest config
        self.buf_queues = Some(alloc_buffer(
            &ctx.device,
            max_tgs * queue_stride * std::mem::size_of::<u32>(),
        ));
        self.buf_steal_counts = Some(alloc_buffer(
            &ctx.device,
            max_tgs * std::mem::size_of::<u32>(),
        ));
        self.buf_total_done = Some(alloc_buffer(
            &ctx.device,
            std::mem::size_of::<u32>(),
        ));

        self.pso_cache
            .get_or_create(ctx.library(), "gpuos_ws_init");
        self.pso_cache
            .get_or_create(ctx.library(), "gpuos_ws_process");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        self.results.clear();

        let queues = self.buf_queues.as_ref().unwrap();
        let steal_counts = self.buf_steal_counts.as_ref().unwrap();
        let total_done = self.buf_total_done.as_ref().unwrap();

        for &(num_tgs, mode, label) in CONFIGS {
            let total_tasks = num_tgs * QUEUE_CAPACITY / 2; // 50% fill

            let params = GpuOsParams {
                capacity: QUEUE_CAPACITY,
                num_ops: total_tasks,
                num_queues: num_tgs,
                mode,
            };
            let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

            // Zero total_done
            unsafe {
                let ptr = total_done.contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(ptr, 0, 4);
            }

            let cmd = ctx.queue.commandBuffer().unwrap();

            // ── INIT: one TG per queue, 256 threads each ──
            let pso_init = self
                .pso_cache
                .get_or_create(ctx.library(), "gpuos_ws_init");
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_init);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(queues.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(steal_counts.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_params.as_ref()), 0, 2);
            }
            let grid = MTLSize {
                width: num_tgs as usize,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: THREADS_PER_TG,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();

            // ── PROCESS: same TG layout ──
            let pso_proc = self
                .pso_cache
                .get_or_create(ctx.library(), "gpuos_ws_process");
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_proc);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(queues.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(steal_counts.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(total_done.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_params.as_ref()), 0, 3);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();

            cmd.commit();
            cmd.waitUntilCompleted();
            let gpu_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);

            // Read results
            let tasks_done = unsafe { *(total_done.contents().as_ptr() as *const u32) };

            let mut total_steals: u32 = 0;
            unsafe {
                let ptr = steal_counts.contents().as_ptr() as *const u32;
                for i in 0..num_tgs as usize {
                    total_steals += *ptr.add(i);
                }
            }

            self.results
                .push((label.to_string(), tasks_done, total_steals, gpu_ms));
        }

        // Return the balanced-steal config as primary metric
        self.results
            .iter()
            .find(|(l, _, _, _)| l == "256tg_steal")
            .map(|(_, _, _, ms)| *ms)
            .unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU: sequential task processing (hash each task ID)
        let total_tasks = 256 * QUEUE_CAPACITY / 2;
        let mut sum = 0u64;
        for i in 0..total_tasks {
            let mut h = i;
            h ^= h >> 16;
            h = h.wrapping_mul(0x45d9f3b);
            h ^= h >> 16;
            sum = sum.wrapping_add(h as u64);
        }
        std::hint::black_box(sum);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.results.is_empty() {
            return Err("No results collected".to_string());
        }
        // At least one config should process >0 tasks
        let any_done = self.results.iter().any(|(_, done, _, _)| *done > 0);
        if !any_done {
            return Err("No tasks were processed in any configuration".to_string());
        }
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        for (label, tasks_done, steals, gpu_ms) in &self.results {
            m.insert(format!("{}_tasks_done", label), *tasks_done as f64);
            m.insert(format!("{}_steals", label), *steals as f64);
            m.insert(format!("{}_ms", label), *gpu_ms);

            if *gpu_ms > 0.0 {
                let tasks_per_ms = *tasks_done as f64 / gpu_ms;
                m.insert(format!("{}_tasks_per_ms", label), tasks_per_ms);
            }
        }

        // Key metric: does work-stealing recover imbalanced performance?
        // Compare imbalanced+steal tasks_done to balanced+local tasks_done
        for num_tgs in &["64tg", "256tg"] {
            let local = self
                .results
                .iter()
                .find(|(l, _, _, _)| l == &format!("{}_local", num_tgs));
            let imbal = self
                .results
                .iter()
                .find(|(l, _, _, _)| l == &format!("{}_imbal", num_tgs));

            if let (Some((_, local_done, _, _)), Some((_, imbal_done, _, _))) = (local, imbal) {
                if *local_done > 0 {
                    let recovery_pct = *imbal_done as f64 / *local_done as f64 * 100.0;
                    m.insert(format!("{}_recovery_pct", num_tgs), recovery_pct);
                }
            }
        }

        m
    }
}
