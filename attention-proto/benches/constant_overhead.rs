//! Criterion benchmark for Proto 4: Function constant compilation overhead.
//!
//! Measures cold PSO compilation time for N variants (N=1/10/50/100)
//! with different function constant values (HEAD_DIM, BLOCK_R, BLOCK_C).
//! Uses Instant::now() for CPU-side timing since we're measuring the
//! Metal compiler, not GPU execution.
//!
//! Important: Does NOT use PsoCache — each iteration compiles from scratch
//! to measure true cold-compilation overhead.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use attention_proto::device::GpuDevice;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
};

/// A single function constant configuration to compile.
struct ConstantConfig {
    head_dim: u32,
    block_r: u32,
    block_c: u32,
}

/// Generate N unique function constant configurations.
///
/// Cycles through HEAD_DIM in [32, 64, 128], BLOCK_R in [8, 16, 32, 64],
/// BLOCK_C in [32, 64, 128] — 36 unique combinations. For N > 36, we
/// use arbitrary uint values (compilation overhead is independent of
/// whether tile sizes are "realistic").
fn generate_configs(n: usize) -> Vec<ConstantConfig> {
    let head_dims = [32u32, 64, 128];
    let block_rs = [8u32, 16, 32, 64];
    let block_cs = [32u32, 64, 128];

    let mut configs = Vec::with_capacity(n);

    // First pass: all realistic combinations (3 * 4 * 3 = 36)
    for &hd in &head_dims {
        for &br in &block_rs {
            for &bc in &block_cs {
                configs.push(ConstantConfig {
                    head_dim: hd,
                    block_r: br,
                    block_c: bc,
                });
                if configs.len() == n {
                    return configs;
                }
            }
        }
    }

    // If N > 36, generate additional unique configs with arbitrary values.
    // Metal compiler still has to specialize the kernel for each unique
    // combination, so compilation cost is representative.
    let mut idx = configs.len();
    while configs.len() < n {
        configs.push(ConstantConfig {
            head_dim: 32 + (idx as u32 * 7) % 256,
            block_r: 4 + (idx as u32 * 13) % 128,
            block_c: 16 + (idx as u32 * 11) % 256,
        });
        idx += 1;
    }

    configs
}

/// Compile a single PSO with the given function constants.
/// Returns the compiled PSO (held to prevent deallocation during timing).
fn compile_one(
    library: &ProtocolObject<dyn MTLLibrary>,
    device: &ProtocolObject<dyn MTLDevice>,
    config: &ConstantConfig,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let constant_values = MTLFunctionConstantValues::new();

    // Index 0: HEAD_DIM
    let hd = config.head_dim;
    unsafe {
        let ptr = NonNull::new(&hd as *const u32 as *mut std::ffi::c_void).unwrap();
        constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::UInt, 0);
    }

    // Index 1: BLOCK_R
    let br = config.block_r;
    unsafe {
        let ptr = NonNull::new(&br as *const u32 as *mut std::ffi::c_void).unwrap();
        constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::UInt, 1);
    }

    // Index 2: BLOCK_C
    let bc = config.block_c;
    unsafe {
        let ptr = NonNull::new(&bc as *const u32 as *mut std::ffi::c_void).unwrap();
        constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::UInt, 2);
    }

    // Create specialized function
    let fn_name = NSString::from_str("flash_attention");
    let function = library
        .newFunctionWithName_constantValues_error(&fn_name, &constant_values)
        .unwrap_or_else(|e| panic!("Failed to create function with constants: {:?}", e));

    // Compile to PSO
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|e| panic!("Failed to compile PSO: {:?}", e))
}

/// Benchmark cold compilation of N function constant variants.
fn bench_cold_compile(c: &mut Criterion) {
    let gpu = GpuDevice::shared();
    let library = &*gpu.library;
    let device = &*gpu.device;

    let mut group = c.benchmark_group("cold_compile");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &n in &[1usize, 10, 50, 100] {
        let configs = generate_configs(n);

        group.bench_with_input(
            BenchmarkId::new("variants", n),
            &n,
            |b, &_n| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;

                    for _ in 0..iters {
                        // Each iteration compiles all N variants from scratch.
                        // We hold the PSOs in a vec to prevent deallocation
                        // during timing (simulates real usage).
                        let start = Instant::now();
                        let _psos: Vec<_> = configs
                            .iter()
                            .map(|cfg| compile_one(library, device, cfg))
                            .collect();
                        total += start.elapsed();

                        // Drop PSOs after timing to free resources
                        drop(_psos);
                    }

                    total
                });
            },
        );

        // Quick summary: compile once and report timing
        let start = Instant::now();
        let _psos: Vec<_> = configs
            .iter()
            .map(|cfg| compile_one(library, device, cfg))
            .collect();
        let elapsed = start.elapsed();

        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let per_variant_ms = total_ms / n as f64;

        eprintln!(
            "[COMPILE] N={n}: total={total_ms:.1}ms, per_variant={per_variant_ms:.2}ms",
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cold_compile);
criterion_main!(benches);
