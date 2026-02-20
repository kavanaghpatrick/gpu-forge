//! Criterion benchmark for Proto 4: Function constant compilation overhead.
//!
//! Two benchmark groups:
//! 1. `cold_compile`: Measures cold PSO compilation time for N variants (N=1/10/50/100)
//! 2. `binary_archive`: Measures archive creation, archive-accelerated PSO load,
//!    and PsoCache hit time for 72 variants
//!
//! Uses Instant::now() for CPU-side timing since we're measuring the
//! Metal compiler, not GPU execution.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::PathBuf;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use attention_proto::device::GpuDevice;
use attention_proto::pipeline::{PsoCache, PsoKey};

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

/// Generate 72 unique configs for binary archive benchmark.
///
/// Produces exactly 72 configs using:
/// HEAD_DIM in [64, 128, 256] x BLOCK_R in [16, 32, 64] x BLOCK_C in [64, 128]
/// = 3 * 3 * 2 = 18 realistic combos, then extends with arbitrary values to 72.
fn generate_archive_configs() -> Vec<ConstantConfig> {
    let head_dims = [64u32, 128, 256];
    let block_rs = [16u32, 32, 64];
    let block_cs = [64u32, 128];

    let mut configs = Vec::with_capacity(72);

    // 18 realistic combinations
    for &hd in &head_dims {
        for &br in &block_rs {
            for &bc in &block_cs {
                configs.push(ConstantConfig {
                    head_dim: hd,
                    block_r: br,
                    block_c: bc,
                });
            }
        }
    }

    // 54 more unique configs to reach 72 (simulates VARIANT dimension)
    let mut idx = configs.len();
    while configs.len() < 72 {
        configs.push(ConstantConfig {
            head_dim: 32 + (idx as u32 * 7) % 256,
            block_r: 4 + (idx as u32 * 13) % 128,
            block_c: 16 + (idx as u32 * 11) % 256,
        });
        idx += 1;
    }

    configs
}

/// Convert a ConstantConfig to a PsoKey for use with PsoCache.
fn config_to_pso_key(config: &ConstantConfig) -> PsoKey {
    PsoKey::simple("flash_attention")
        .with_uint(0, config.head_dim)
        .with_uint(1, config.block_r)
        .with_uint(2, config.block_c)
}

/// Benchmark binary archive creation and archive-accelerated PSO loading.
///
/// Measures three scenarios for 72 function constant variants:
/// 1. Archive creation: compile all 72 PSOs + serialize to disk
/// 2. Archive load: load PSO from binary archive (should be faster than cold compile)
/// 3. PsoCache hit: HashMap lookup time (near-instant, provides upper bound comparison)
fn bench_binary_archive(c: &mut Criterion) {
    let gpu = GpuDevice::shared();
    let library = &*gpu.library;
    let device = &*gpu.device;

    let configs = generate_archive_configs();
    let archive_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_archive.metallib");

    let mut group = c.benchmark_group("binary_archive");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    // --- 1. Archive creation benchmark ---
    // Compile all 72 PSOs, create archive, serialize to disk.
    group.bench_function("create_72_variants", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;

            for _ in 0..iters {
                // Pre-compile all 72 PSOs into PsoCache
                let mut cache = PsoCache::new(gpu.library.clone());
                for cfg in &configs {
                    let key = config_to_pso_key(cfg);
                    cache.get_or_compile(&key);
                }

                // Time: archive creation + serialization
                let start = Instant::now();
                cache
                    .save_archive(&archive_path)
                    .unwrap_or_else(|e| panic!("Failed to save archive: {}", e));
                total += start.elapsed();

                // Cleanup for next iteration
                drop(cache);
            }

            total
        });
    });

    // --- Pre-create archive for load benchmarks ---
    // Compile and save once outside the benchmark loop.
    {
        let mut cache = PsoCache::new(gpu.library.clone());
        for cfg in &configs {
            let key = config_to_pso_key(cfg);
            cache.get_or_compile(&key);
        }
        cache
            .save_archive(&archive_path)
            .unwrap_or_else(|e| panic!("Failed to save archive for load bench: {}", e));
    }

    // --- 2. Archive-accelerated PSO load benchmark ---
    // Load archive from disk, then compile PSOs with archive hint.
    // Metal should find pre-compiled code in archive, skipping recompilation.
    group.bench_function("load_from_archive_72", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;

            for _ in 0..iters {
                let start = Instant::now();

                // Load archive from disk
                let archive = PsoCache::load_archive(library, &archive_path)
                    .unwrap_or_else(|e| panic!("Failed to load archive: {}", e));

                // Compile all 72 PSOs using archive (should be faster)
                let _psos: Vec<_> = configs
                    .iter()
                    .map(|cfg| {
                        let key = config_to_pso_key(cfg);
                        PsoCache::compile_with_archive(library, &key, &archive)
                    })
                    .collect();

                total += start.elapsed();

                drop(_psos);
            }

            total
        });
    });

    // --- 3. Cold compile (no archive) for direct comparison ---
    group.bench_function("cold_compile_72", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;

            for _ in 0..iters {
                let start = Instant::now();
                let _psos: Vec<_> = configs
                    .iter()
                    .map(|cfg| compile_one(library, device, cfg))
                    .collect();
                total += start.elapsed();

                drop(_psos);
            }

            total
        });
    });

    // --- 4. PsoCache hit benchmark (HashMap lookup, near-instant) ---
    // Pre-populate cache, then measure lookup time.
    {
        let mut cache = PsoCache::new(gpu.library.clone());
        for cfg in &configs {
            let key = config_to_pso_key(cfg);
            cache.get_or_compile(&key);
        }

        group.bench_function("cache_hit_72", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;

                for _ in 0..iters {
                    let start = Instant::now();
                    for cfg in &configs {
                        let key = config_to_pso_key(cfg);
                        let _pso = cache.get_or_compile(&key);
                    }
                    total += start.elapsed();
                }

                total
            });
        });
    }

    group.finish();

    // --- Summary: one-shot timing for all three paths ---
    eprintln!("\n=== Binary Archive Summary (72 variants) ===\n");

    // Cold compile
    let start = Instant::now();
    let _psos: Vec<_> = configs
        .iter()
        .map(|cfg| compile_one(library, device, cfg))
        .collect();
    let cold_elapsed = start.elapsed();
    let cold_ms = cold_elapsed.as_secs_f64() * 1000.0;
    let cold_per_ms = cold_ms / 72.0;
    eprintln!("[COLD COMPILE]   total={cold_ms:.2}ms  per_variant={cold_per_ms:.3}ms");
    drop(_psos);

    // Archive load
    let start = Instant::now();
    let archive = PsoCache::load_archive(library, &archive_path)
        .unwrap_or_else(|e| panic!("Failed to load archive: {}", e));
    let _psos: Vec<_> = configs
        .iter()
        .map(|cfg| {
            let key = config_to_pso_key(cfg);
            PsoCache::compile_with_archive(library, &key, &archive)
        })
        .collect();
    let archive_elapsed = start.elapsed();
    let archive_ms = archive_elapsed.as_secs_f64() * 1000.0;
    let archive_per_ms = archive_ms / 72.0;
    eprintln!("[ARCHIVE LOAD]   total={archive_ms:.2}ms  per_variant={archive_per_ms:.3}ms");
    drop(_psos);

    // Cache hit
    let mut cache = PsoCache::new(gpu.library.clone());
    for cfg in &configs {
        let key = config_to_pso_key(cfg);
        cache.get_or_compile(&key);
    }
    let start = Instant::now();
    for cfg in &configs {
        let key = config_to_pso_key(cfg);
        cache.get_or_compile(&key);
    }
    let cache_elapsed = start.elapsed();
    let cache_us = cache_elapsed.as_secs_f64() * 1_000_000.0;
    let cache_per_ns = cache_elapsed.as_nanos() as f64 / 72.0;
    eprintln!("[CACHE HIT]      total={cache_us:.1}us  per_variant={cache_per_ns:.0}ns");

    // Speedup comparison
    if archive_elapsed.as_nanos() > 0 {
        let speedup = cold_elapsed.as_secs_f64() / archive_elapsed.as_secs_f64();
        eprintln!("\n[SPEEDUP] archive vs cold: {speedup:.2}x");
    }
    if cache_elapsed.as_nanos() > 0 {
        let speedup = cold_elapsed.as_secs_f64() / cache_elapsed.as_secs_f64();
        eprintln!("[SPEEDUP] cache hit vs cold: {speedup:.0}x");
    }

    eprintln!("\n[VERDICT] Cold compile for 72 variants takes {cold_ms:.1}ms total.");
    if cold_ms < 10.0 {
        eprintln!("[VERDICT] Function constants are fast enough that binary archives are unnecessary.");
        eprintln!("[VERDICT] Use PsoCache (HashMap) for runtime dispatch — cache hit is ~{cache_per_ns:.0}ns/lookup.");
    } else {
        eprintln!("[VERDICT] Consider binary archives for faster startup if > 50ms threshold.");
    }

    // Cleanup archive file
    let _ = std::fs::remove_file(&archive_path);
}

criterion_group!(benches, bench_cold_compile, bench_binary_archive);
criterion_main!(benches);
