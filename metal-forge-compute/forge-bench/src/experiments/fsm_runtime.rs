//! SIMD FSM Runtime — 16K+ Finite State Machines.
//!
//! Each thread runs an independent 16-state FSM with 4 input symbols.
//! 32 FSMs per SIMD group = 32 independent programs in one warp.
//! Transition table in constant memory for broadcast efficiency.
//!
//! Configurations:
//!   Independent: each FSM runs in isolation (max throughput)
//!   Coupled: every 4 steps, read neighbor TG's lane-0 state via device memory
//!            (KB finding 3229: cross-TG atomics 100% reliable)
//!
//! At 1M threads with 100 transitions each = 100M FSM transitions.
//! The "holy shit": each SIMD lane is a CPU running its own program.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, GpuOsParams, GpuTimer, MetalContext,
    PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

const NUM_TRANSITIONS: u32 = 100;
const THREADS_PER_TG: usize = 256;

// CPU-side transition table (matches Metal constant)
const FSM_TABLE: [u32; 64] = [
    1, 3, 5, 0, 2, 0, 4, 6, 7, 1, 3, 5, 4, 8, 0, 2, 9, 5, 1, 7, 6, 10, 2, 4, 11, 7, 3, 9, 8,
    12, 4, 6, 13, 9, 5, 11, 10, 14, 6, 8, 15, 11, 7, 13, 12, 0, 8, 10, 1, 13, 9, 15, 14, 2, 10,
    12, 3, 15, 11, 1, 0, 4, 12, 14,
];

pub struct FsmRuntimeExperiment {
    input_data: Vec<u32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_global_states: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_total_trans: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    independent_ms: f64,
    coupled_ms: f64,
    independent_transitions: u32,
    coupled_transitions: u32,

    pso_cache: PsoCache,
    size: usize,
}

impl FsmRuntimeExperiment {
    pub fn new() -> Self {
        Self {
            input_data: Vec::new(),
            buf_input: None,
            buf_output: None,
            buf_global_states: None,
            buf_total_trans: None,
            independent_ms: 0.0,
            coupled_ms: 0.0,
            independent_transitions: 0,
            coupled_transitions: 0,
            pso_cache: PsoCache::new(),
            size: 0,
        }
    }
}

impl Experiment for FsmRuntimeExperiment {
    fn name(&self) -> &str {
        "fsm_runtime"
    }

    fn description(&self) -> &str {
        "SIMD FSM runtime: 16K+ independent FSMs, 32 per SIMD group, cross-TG coupling"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;
        self.input_data = gen.uniform_u32(size);

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.input_data));
        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));

        let num_tgs = (size + THREADS_PER_TG - 1) / THREADS_PER_TG;
        self.buf_global_states = Some(alloc_buffer(
            &ctx.device,
            num_tgs * std::mem::size_of::<u32>(),
        ));
        self.buf_total_trans = Some(alloc_buffer(
            &ctx.device,
            std::mem::size_of::<u32>(),
        ));

        self.pso_cache
            .get_or_create(ctx.library(), "gpuos_fsm_independent");
        self.pso_cache
            .get_or_create(ctx.library(), "gpuos_fsm_coupled");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let global_states = self.buf_global_states.as_ref().unwrap();
        let total_trans = self.buf_total_trans.as_ref().unwrap();

        let num_tgs = (self.size + THREADS_PER_TG - 1) / THREADS_PER_TG;

        // ── INDEPENDENT FSMs ──
        let params = GpuOsParams {
            capacity: self.size as u32,
            num_ops: NUM_TRANSITIONS,
            num_queues: num_tgs as u32,
            mode: 0,
        };
        let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

        // Zero transition counter
        unsafe {
            let ptr = total_trans.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 4);
        }

        let cmd = ctx.queue.commandBuffer().unwrap();
        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "gpuos_fsm_independent");
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc,
            pso,
            &[
                (input.as_ref(), 0),
                (output.as_ref(), 1),
                (total_trans.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            self.size,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        self.independent_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        self.independent_transitions =
            unsafe { *(total_trans.contents().as_ptr() as *const u32) };

        // ── COUPLED FSMs ──
        unsafe {
            let ptr = total_trans.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 4);
            let ptr = global_states.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, num_tgs * 4);
        }

        let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "gpuos_fsm_coupled");
        let enc = cmd.computeCommandEncoder().unwrap();

        // Manual dispatch for explicit TG control
        enc.setComputePipelineState(pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(global_states.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(input.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(output.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(total_trans.as_ref()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(buf_params.as_ref()), 0, 4);
        }
        let grid = MTLSize {
            width: num_tgs,
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

        cmd.commit();
        cmd.waitUntilCompleted();
        self.coupled_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        self.coupled_transitions =
            unsafe { *(total_trans.contents().as_ptr() as *const u32) };

        // Return independent time as primary metric
        self.independent_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU: scalar FSM execution
        let mut total_trans = 0u64;
        for tid in 0..self.size {
            let mut state = (tid & 0xF) as u32;
            for step in 0..NUM_TRANSITIONS {
                let input_idx = (tid * NUM_TRANSITIONS as usize + step as usize) % self.size;
                let input = self.input_data[input_idx] & 0x3;
                state = FSM_TABLE[(state * 4 + input) as usize];
                total_trans += 1;
            }
            std::hint::black_box(state);
        }
        std::hint::black_box(total_trans);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // Independent should produce transitions = size * NUM_TRANSITIONS
        let expected = self.size as u32 * NUM_TRANSITIONS;
        if self.independent_transitions < expected / 2 {
            return Err(format!(
                "Too few independent transitions: {} (expected ~{})",
                self.independent_transitions, expected
            ));
        }
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("independent_ms".to_string(), self.independent_ms);
        m.insert("coupled_ms".to_string(), self.coupled_ms);
        m.insert(
            "independent_transitions".to_string(),
            self.independent_transitions as f64,
        );
        m.insert(
            "coupled_transitions".to_string(),
            self.coupled_transitions as f64,
        );

        // Throughput: transitions per millisecond (M transitions/sec)
        if self.independent_ms > 0.0 {
            let mtps =
                self.independent_transitions as f64 / self.independent_ms / 1000.0;
            m.insert("independent_mtps".to_string(), mtps);
        }
        if self.coupled_ms > 0.0 {
            let mtps = self.coupled_transitions as f64 / self.coupled_ms / 1000.0;
            m.insert("coupled_mtps".to_string(), mtps);
        }

        // Coupling overhead
        if self.independent_ms > 0.0 && self.coupled_ms > 0.0 {
            let overhead = self.coupled_ms / self.independent_ms;
            m.insert("coupling_overhead_x".to_string(), overhead);
        }

        // FSMs per SIMD group
        m.insert("fsms_per_simdgroup".to_string(), 32.0);
        m.insert("total_fsms".to_string(), self.size as f64);

        m
    }
}
