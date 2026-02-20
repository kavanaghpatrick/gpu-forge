//! Multi-Tick Persistent State Chain.
//!
//! Simulates persistent kernels by chaining N compute encoders in a single
//! Metal command buffer. State persists in device memory between ticks,
//! staying SLC-resident for sizes < 16MB (KB finding 3231).
//!
//! Comparison:
//!   A) Single command buffer, N encoders → one dispatch overhead
//!   B) N separate command buffers → N dispatch overheads (~30us each)
//!
//! Per-encoder overhead should be ~1us (KB finding 3224: encoder creation),
//! vs ~30us per separate command buffer (firmware scheduling).
//!
//! This is the building block for "persistent GPU kernels" on Metal,
//! where the 5-second command buffer timeout prevents truly persistent threads.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, BenchTimer, GpuOsParams, GpuTimer,
    MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

const NUM_TICKS: u32 = 100;

pub struct TickChainExperiment {
    buf_state: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_tick_counter: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    multi_encoder_gpu_ms: f64,
    separate_cmdbuf_gpu_ms: f64,
    separate_cmdbuf_wall_ms: f64,
    tick_count_verified: u32,

    pso_cache: PsoCache,
    size: usize,
}

impl TickChainExperiment {
    pub fn new() -> Self {
        Self {
            buf_state: None,
            buf_tick_counter: None,
            multi_encoder_gpu_ms: 0.0,
            separate_cmdbuf_gpu_ms: 0.0,
            separate_cmdbuf_wall_ms: 0.0,
            tick_count_verified: 0,
            pso_cache: PsoCache::new(),
            size: 0,
        }
    }

    fn zero_buffers(&self) {
        if let Some(ref buf) = self.buf_state {
            unsafe {
                let ptr = buf.contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(ptr, 0, self.size * 4);
            }
        }
        if let Some(ref buf) = self.buf_tick_counter {
            unsafe {
                let ptr = buf.contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(ptr, 0, 4);
            }
        }
    }

    fn init_state(&self) {
        if let Some(ref buf) = self.buf_state {
            unsafe {
                let ptr = buf.contents().as_ptr() as *mut u32;
                for i in 0..self.size {
                    *ptr.add(i) = i as u32;
                }
            }
        }
    }
}

impl Experiment for TickChainExperiment {
    fn name(&self) -> &str {
        "tick_chain"
    }

    fn description(&self) -> &str {
        "Tick chain: N encoders in one cmd buf vs N separate cmd bufs — persistent state"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000] // 1M state elements = 4MB (SLC-resident)
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, _gen: &mut DataGenerator) {
        self.size = size;

        self.buf_state = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));
        self.buf_tick_counter = Some(alloc_buffer(
            &ctx.device,
            std::mem::size_of::<u32>(),
        ));

        self.pso_cache
            .get_or_create(ctx.library(), "gpuos_tick_process");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let state = self.buf_state.as_ref().unwrap();
        let tick_counter = self.buf_tick_counter.as_ref().unwrap();

        let params = GpuOsParams {
            capacity: self.size as u32,
            num_ops: NUM_TICKS,
            num_queues: 0,
            mode: 1, // hash-based state machine (more interesting than simple increment)
        };
        let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

        // ── APPROACH A: Single command buffer, N encoders ──
        self.init_state();
        self.zero_buffers();
        // Re-init state after zero_buffers cleared it
        self.init_state();
        // Zero just the tick counter
        unsafe {
            let ptr = tick_counter.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 4);
        }

        let cmd = ctx.queue.commandBuffer().unwrap();
        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "gpuos_tick_process");

        for _tick in 0..NUM_TICKS {
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (state.as_ref(), 0),
                    (tick_counter.as_ref(), 1),
                    (buf_params.as_ref(), 2),
                ],
                self.size,
            );
            enc.endEncoding();
        }

        cmd.commit();
        cmd.waitUntilCompleted();
        self.multi_encoder_gpu_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);

        // Verify tick count
        self.tick_count_verified =
            unsafe { *(tick_counter.contents().as_ptr() as *const u32) };

        // ── APPROACH B: N separate command buffers ──
        self.init_state();
        unsafe {
            let ptr = tick_counter.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 4);
        }

        let wall_timer = BenchTimer::start();
        let mut total_gpu_ms = 0.0;

        for _tick in 0..NUM_TICKS {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "gpuos_tick_process");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (state.as_ref(), 0),
                    (tick_counter.as_ref(), 1),
                    (buf_params.as_ref(), 2),
                ],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            total_gpu_ms += GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        self.separate_cmdbuf_wall_ms = wall_timer.stop();
        self.separate_cmdbuf_gpu_ms = total_gpu_ms;

        // Return multi-encoder GPU time as primary metric
        self.multi_encoder_gpu_ms
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();

        // CPU: sequential state updates (hash-based FSM)
        let mut state: Vec<u32> = (0..self.size as u32).collect();
        for _tick in 0..NUM_TICKS {
            for (i, s) in state.iter_mut().enumerate() {
                let next_state = (*s >> 28) & 0x7;
                let mut data = *s & 0x0FFFFFFF;
                match next_state {
                    0 => {
                        data ^= 0x12345678;
                    }
                    1 => {
                        data = data.wrapping_add(0x9E3779B9);
                    }
                    2 => {
                        data ^= data >> 16;
                    }
                    3 => {
                        data = data.wrapping_mul(0x45d9f3b);
                    }
                    4 => {
                        data ^= data >> 13;
                    }
                    5 => {
                        data = data.wrapping_add(i as u32);
                    }
                    6 => {
                        data ^= 0xDEADBEEF;
                    }
                    _ => {
                        data = data.wrapping_add(1);
                    }
                }
                let ns = (next_state + 1) & 0x7;
                *s = (ns << 28) | (data & 0x0FFFFFFF);
            }
        }
        std::hint::black_box(&state);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.tick_count_verified != NUM_TICKS {
            return Err(format!(
                "Tick count mismatch: expected {}, got {}",
                NUM_TICKS, self.tick_count_verified
            ));
        }
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("multi_encoder_gpu_ms".to_string(), self.multi_encoder_gpu_ms);
        m.insert("separate_cmdbuf_gpu_ms".to_string(), self.separate_cmdbuf_gpu_ms);
        m.insert("separate_cmdbuf_wall_ms".to_string(), self.separate_cmdbuf_wall_ms);
        m.insert("num_ticks".to_string(), NUM_TICKS as f64);
        m.insert("tick_count_verified".to_string(), self.tick_count_verified as f64);

        // Per-tick costs
        let n = NUM_TICKS as f64;
        if self.multi_encoder_gpu_ms > 0.0 {
            let per_tick_us = self.multi_encoder_gpu_ms * 1000.0 / n;
            m.insert("multi_per_tick_us".to_string(), per_tick_us);
        }
        if self.separate_cmdbuf_wall_ms > 0.0 {
            let per_tick_us = self.separate_cmdbuf_wall_ms * 1000.0 / n;
            m.insert("separate_per_tick_us".to_string(), per_tick_us);
        }

        // Overhead reduction: how much faster is multi-encoder?
        if self.multi_encoder_gpu_ms > 0.0 && self.separate_cmdbuf_wall_ms > 0.0 {
            let speedup = self.separate_cmdbuf_wall_ms / self.multi_encoder_gpu_ms;
            m.insert("overhead_reduction_x".to_string(), speedup);
        }

        // Pure GPU compute time should be similar for both
        if self.separate_cmdbuf_gpu_ms > 0.0 && self.multi_encoder_gpu_ms > 0.0 {
            let gpu_ratio = self.separate_cmdbuf_gpu_ms / self.multi_encoder_gpu_ms;
            m.insert("gpu_time_ratio".to_string(), gpu_ratio);
        }

        // State throughput
        let total_elements = self.size as f64 * n;
        if self.multi_encoder_gpu_ms > 0.0 {
            let gbs = (total_elements * 8.0) / (self.multi_encoder_gpu_ms / 1000.0) / 1e9;
            m.insert("state_bw_gbs".to_string(), gbs);
        }

        m
    }
}
