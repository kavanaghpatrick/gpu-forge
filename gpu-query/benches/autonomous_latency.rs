//! Criterion benchmark suite for the autonomous query engine.
//!
//! BM-01: COUNT(*) 1M rows
//! BM-02: SUM/MIN/MAX 1M rows
//! BM-03: Compound filter + GROUP BY + 3 aggs 1M rows (headline benchmark)
//! BM-04: Fused (AOT) vs autonomous (JIT) comparison
//! BM-05: Work queue write latency
//! BM-06: JIT compilation time (cache miss)
//! BM-07: JIT cache hit
//! BM-08: Binary loading time (1M rows)
//! BM-09: poll_ready latency
//! BM-10: read_result latency
//!
//! All benchmarks use deterministic data:
//!   Column 0 (amount): INT64, values = (i*7+13)%1000
//!   Column 1 (region): INT64, values = i%5

use criterion::{criterion_group, criterion_main, Criterion};

use gpu_query::gpu::autonomous::executor::{
    execute_fused_oneshot, execute_jit_oneshot, AutonomousExecutor, FusedPsoCache,
};
use gpu_query::gpu::autonomous::jit::JitCompiler;
use gpu_query::gpu::autonomous::loader::{BinaryColumnarLoader, ColumnInfo};
use gpu_query::gpu::autonomous::types::{AggSpec, FilterSpec, QueryParamsSlot};
use gpu_query::gpu::autonomous::work_queue::WorkQueue;
use gpu_query::gpu::device::GpuDevice;
use gpu_query::sql::physical_plan::PhysicalPlan;
use gpu_query::sql::types::{AggFunc, CompareOp, LogicalOp, Value};
use gpu_query::storage::columnar::ColumnarBatch;
use gpu_query::storage::schema::{ColumnDef, DataType, RuntimeSchema};

use objc2_metal::MTLBuffer;

// ============================================================================
// Constants
// ============================================================================

const BENCH_ROW_COUNT: usize = 1_000_000;

// ============================================================================
// Helpers
// ============================================================================

fn test_schema() -> RuntimeSchema {
    RuntimeSchema::new(vec![
        ColumnDef {
            name: "amount".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "region".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
    ])
}

fn make_test_batch(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
    schema: &RuntimeSchema,
    row_count: usize,
) -> ColumnarBatch {
    let mut batch = ColumnarBatch::allocate(device, schema, row_count);
    batch.row_count = row_count;

    unsafe {
        let ptr = batch.int_buffer.contents().as_ptr() as *mut i64;
        // Column 0 (amount): (i*7+13)%1000
        for i in 0..row_count {
            *ptr.add(i) = ((i * 7 + 13) % 1000) as i64;
        }
        // Column 1 (region): i%5
        let offset = batch.max_rows;
        for i in 0..row_count {
            *ptr.add(offset + i) = (i % 5) as i64;
        }
    }

    batch
}

fn jit_schema() -> Vec<ColumnInfo> {
    vec![
        ColumnInfo {
            name: "amount".into(),
            data_type: DataType::Int64,
        },
        ColumnInfo {
            name: "region".into(),
            data_type: DataType::Int64,
        },
    ]
}

fn plan_scan(table: &str) -> PhysicalPlan {
    PhysicalPlan::GpuScan {
        table: table.into(),
        columns: vec!["amount".into(), "region".into()],
    }
}

fn plan_filter_gt(column: &str, val: i64, input: PhysicalPlan) -> PhysicalPlan {
    PhysicalPlan::GpuFilter {
        compare_op: CompareOp::Gt,
        column: column.into(),
        value: Value::Int(val),
        input: Box::new(input),
    }
}

fn plan_filter_lt(column: &str, val: i64, input: PhysicalPlan) -> PhysicalPlan {
    PhysicalPlan::GpuFilter {
        compare_op: CompareOp::Lt,
        column: column.into(),
        value: Value::Int(val),
        input: Box::new(input),
    }
}

fn plan_compound_and(left: PhysicalPlan, right: PhysicalPlan) -> PhysicalPlan {
    PhysicalPlan::GpuCompoundFilter {
        op: LogicalOp::And,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn plan_aggregate(
    functions: Vec<(AggFunc, &str)>,
    group_by: Vec<&str>,
    input: PhysicalPlan,
) -> PhysicalPlan {
    PhysicalPlan::GpuAggregate {
        functions: functions
            .into_iter()
            .map(|(f, c)| (f, c.to_string()))
            .collect(),
        group_by: group_by.into_iter().map(|s| s.to_string()).collect(),
        input: Box::new(input),
    }
}

/// Build a COUNT(*) params slot.
fn params_count_star(row_count: u32) -> QueryParamsSlot {
    let mut params = QueryParamsSlot::default();
    params.row_count = row_count;
    params.filter_count = 0;
    params.agg_count = 1;
    params.aggs[0] = AggSpec {
        agg_func: AggFunc::Count.to_gpu_code(),
        column_idx: 0,
        column_type: 0,
        _pad0: 0,
    };
    params
}

/// Build SUM + MIN + MAX params slot (no filter, no GROUP BY).
fn params_sum_min_max(row_count: u32) -> QueryParamsSlot {
    let mut params = QueryParamsSlot::default();
    params.row_count = row_count;
    params.filter_count = 0;
    params.agg_count = 3;
    params.aggs[0] = AggSpec {
        agg_func: AggFunc::Sum.to_gpu_code(),
        column_idx: 0,
        column_type: 0,
        _pad0: 0,
    };
    params.aggs[1] = AggSpec {
        agg_func: AggFunc::Min.to_gpu_code(),
        column_idx: 0,
        column_type: 0,
        _pad0: 0,
    };
    params.aggs[2] = AggSpec {
        agg_func: AggFunc::Max.to_gpu_code(),
        column_idx: 0,
        column_type: 0,
        _pad0: 0,
    };
    params
}

/// Build headline query params: compound filter (amount>200 AND amount<800)
/// + GROUP BY region + COUNT/SUM/MIN/MAX.
fn params_headline(row_count: u32) -> QueryParamsSlot {
    let mut params = QueryParamsSlot::default();
    params.row_count = row_count;
    params.filter_count = 2;
    params.filters[0] = FilterSpec {
        column_idx: 0,
        compare_op: CompareOp::Gt as u32,
        column_type: 0,
        _pad0: 0,
        value_int: 200,
        value_float_bits: (200.0f32).to_bits(),
        _pad1: 0,
        has_null_check: 0,
        _pad2: [0; 3],
    };
    params.filters[1] = FilterSpec {
        column_idx: 0,
        compare_op: CompareOp::Lt as u32,
        column_type: 0,
        _pad0: 0,
        value_int: 800,
        value_float_bits: (800.0f32).to_bits(),
        _pad1: 0,
        has_null_check: 0,
        _pad2: [0; 3],
    };
    params.agg_count = 3;
    params.aggs[0] = AggSpec {
        agg_func: AggFunc::Count.to_gpu_code(),
        column_idx: 0,
        column_type: 0,
        _pad0: 0,
    };
    params.aggs[1] = AggSpec {
        agg_func: AggFunc::Sum.to_gpu_code(),
        column_idx: 0,
        column_type: 0,
        _pad0: 0,
    };
    params.aggs[2] = AggSpec {
        agg_func: AggFunc::Max.to_gpu_code(),
        column_idx: 0,
        column_type: 0,
        _pad0: 0,
    };
    params.group_by_col = 1; // region
    params.has_group_by = 1;
    params
}

/// Headline PhysicalPlan: compound filter + GROUP BY + 3 aggs.
fn headline_plan() -> PhysicalPlan {
    let left = plan_filter_gt("amount", 200, plan_scan("sales"));
    let right = plan_filter_lt("amount", 800, plan_scan("sales"));
    let compound = plan_compound_and(left, right);
    plan_aggregate(
        vec![
            (AggFunc::Count, "*"),
            (AggFunc::Sum, "amount"),
            (AggFunc::Max, "amount"),
        ],
        vec!["region"],
        compound,
    )
}

// ============================================================================
// BM-01: COUNT(*) 1M rows
// ============================================================================

fn bm01_count_star(c: &mut Criterion) {
    let mut group = c.benchmark_group("autonomous_bm01_count_star");
    group.sample_size(50);

    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, BENCH_ROW_COUNT);
    let resident =
        BinaryColumnarLoader::load_table(&gpu.device, "sales", &schema, &batch, None)
            .expect("load_table");

    let params = params_count_star(BENCH_ROW_COUNT as u32);
    let plan = plan_aggregate(vec![(AggFunc::Count, "*")], vec![], plan_scan("sales"));
    let col_schema = jit_schema();
    let mut jit = JitCompiler::new(gpu.device.clone());
    let _ = jit.compile(&plan, &col_schema).expect("jit compile");

    group.bench_function("1M_rows", |b| {
        b.iter(|| {
            execute_jit_oneshot(
                &gpu.device,
                &gpu.command_queue,
                &mut jit,
                &plan,
                &col_schema,
                &params,
                &resident,
            )
            .expect("execute")
        });
    });

    group.finish();
}

// ============================================================================
// BM-02: SUM/MIN/MAX 1M rows
// ============================================================================

fn bm02_sum_min_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("autonomous_bm02_sum_min_max");
    group.sample_size(50);

    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, BENCH_ROW_COUNT);
    let resident =
        BinaryColumnarLoader::load_table(&gpu.device, "sales", &schema, &batch, None)
            .expect("load_table");

    let params = params_sum_min_max(BENCH_ROW_COUNT as u32);
    let plan = plan_aggregate(
        vec![
            (AggFunc::Sum, "amount"),
            (AggFunc::Min, "amount"),
            (AggFunc::Max, "amount"),
        ],
        vec![],
        plan_scan("sales"),
    );
    let col_schema = jit_schema();
    let mut jit = JitCompiler::new(gpu.device.clone());
    let _ = jit.compile(&plan, &col_schema).expect("jit compile");

    group.bench_function("1M_rows", |b| {
        b.iter(|| {
            execute_jit_oneshot(
                &gpu.device,
                &gpu.command_queue,
                &mut jit,
                &plan,
                &col_schema,
                &params,
                &resident,
            )
            .expect("execute")
        });
    });

    group.finish();
}

// ============================================================================
// BM-03: Headline benchmark — compound filter + GROUP BY + 3 aggs, 1M rows
// Target: p50 < 1ms
// ============================================================================

fn bm03_headline(c: &mut Criterion) {
    let mut group = c.benchmark_group("autonomous_bm03_headline");
    group.sample_size(100);
    group.warm_up_time(std::time::Duration::from_secs(5));
    group.measurement_time(std::time::Duration::from_secs(30));

    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, BENCH_ROW_COUNT);

    let plan = headline_plan();
    let col_schema = jit_schema();

    // Use AutonomousExecutor -- the actual autonomous hot path
    let mut executor = AutonomousExecutor::new(gpu.device.clone());
    executor
        .load_table("sales", &schema, &batch)
        .expect("load_table");

    // Warm the JIT cache with a first query
    let _ = executor
        .submit_query(&plan, &col_schema, "sales")
        .expect("submit");
    while !executor.poll_ready() {
        std::hint::spin_loop();
    }
    let _ = executor.read_result();

    group.bench_function("1M_rows", |b| {
        b.iter(|| {
            let _ = executor
                .submit_query(&plan, &col_schema, "sales")
                .expect("submit");
            while !executor.poll_ready() {
                std::hint::spin_loop();
            }
            executor.read_result()
        });
    });

    group.finish();
}

// ============================================================================
// BM-04: Fused (AOT) vs autonomous (JIT) comparison
// ============================================================================

fn bm04_fused_vs_jit(c: &mut Criterion) {
    let mut group = c.benchmark_group("autonomous_bm04_fused_vs_jit");
    group.sample_size(50);

    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, BENCH_ROW_COUNT);
    let resident =
        BinaryColumnarLoader::load_table(&gpu.device, "sales", &schema, &batch, None)
            .expect("load_table");

    let params = params_headline(BENCH_ROW_COUNT as u32);
    let plan = headline_plan();
    let col_schema = jit_schema();

    // AOT path
    let mut aot_cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());
    let _ = aot_cache.get_or_compile(2, 3, true).expect("pso");

    group.bench_function("aot_fused", |b| {
        b.iter(|| {
            execute_fused_oneshot(
                &gpu.device,
                &gpu.command_queue,
                aot_cache.get_or_compile(2, 3, true).unwrap(),
                &params,
                &resident,
            )
            .expect("execute")
        });
    });

    // JIT path (pre-warm cache)
    let mut jit = JitCompiler::new(gpu.device.clone());
    let _ = jit.compile(&plan, &col_schema).expect("jit compile");

    group.bench_function("jit_compiled", |b| {
        b.iter(|| {
            execute_jit_oneshot(
                &gpu.device,
                &gpu.command_queue,
                &mut jit,
                &plan,
                &col_schema,
                &params,
                &resident,
            )
            .expect("execute")
        });
    });

    group.finish();
}

// ============================================================================
// BM-05: Work queue write latency
// ============================================================================

fn bm05_work_queue_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("autonomous_bm05_work_queue_write");
    group.sample_size(100);

    let gpu = GpuDevice::new();
    let mut wq = WorkQueue::new(&gpu.device);
    let params = params_headline(BENCH_ROW_COUNT as u32);

    group.bench_function("write_params", |b| {
        b.iter(|| {
            wq.write_params(&params);
        });
    });

    group.finish();
}

// ============================================================================
// BM-06: JIT compilation time (cache miss)
// ============================================================================

fn bm06_jit_compile_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("autonomous_bm06_jit_compile_miss");
    group.sample_size(20);

    let gpu = GpuDevice::new();
    let plan = headline_plan();
    let col_schema = jit_schema();

    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            // Create fresh compiler for each iteration to force cache miss
            let mut jit = JitCompiler::new(gpu.device.clone());
            let _ = jit.compile(&plan, &col_schema).expect("compile");
        });
    });

    group.finish();
}

// ============================================================================
// BM-07: JIT cache hit
// ============================================================================

fn bm07_jit_cache_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("autonomous_bm07_jit_cache_hit");
    group.sample_size(100);

    let gpu = GpuDevice::new();
    let plan = headline_plan();
    let col_schema = jit_schema();
    let mut jit = JitCompiler::new(gpu.device.clone());

    // Populate cache
    let _ = jit.compile(&plan, &col_schema).expect("jit compile");

    group.bench_function("cache_hit", |b| {
        b.iter(|| {
            let _ = jit.compile(&plan, &col_schema).expect("compile");
        });
    });

    group.finish();
}

// ============================================================================
// BM-08: Binary loading time (1M rows)
// ============================================================================

fn bm08_binary_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("autonomous_bm08_binary_loading");
    group.sample_size(20);

    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, BENCH_ROW_COUNT);

    group.bench_function("1M_rows", |b| {
        b.iter(|| {
            BinaryColumnarLoader::load_table(&gpu.device, "sales", &schema, &batch, None)
                .expect("load_table")
        });
    });

    group.finish();
}

// ============================================================================
// BM-09: poll_ready latency
// ============================================================================

fn bm09_poll_ready(c: &mut Criterion) {
    let mut group = c.benchmark_group("autonomous_bm09_poll_ready");
    group.sample_size(100);

    let gpu = GpuDevice::new();
    let executor = AutonomousExecutor::new(gpu.device.clone());

    // Measure raw poll_ready cost (no query submitted, just reads unified memory)
    group.bench_function("no_pending_query", |b| {
        b.iter(|| executor.poll_ready());
    });

    group.finish();
}

// ============================================================================
// BM-10: read_result latency
// ============================================================================

fn bm10_read_result(c: &mut Criterion) {
    let mut group = c.benchmark_group("autonomous_bm10_read_result");
    group.sample_size(100);

    let gpu = GpuDevice::new();
    let executor = AutonomousExecutor::new(gpu.device.clone());

    // Measure raw read_result cost (reads ~22KB OutputBuffer from unified memory)
    group.bench_function("read_output_buffer", |b| {
        b.iter(|| executor.read_result());
    });

    group.finish();
}

// ============================================================================
// Group all benchmarks
// ============================================================================

criterion_group!(
    benches,
    bm01_count_star,
    bm02_sum_min_max,
    bm03_headline,
    bm04_fused_vs_jit,
    bm05_work_queue_write,
    bm06_jit_compile_miss,
    bm07_jit_cache_hit,
    bm08_binary_loading,
    bm09_poll_ready,
    bm10_read_result,
);
criterion_main!(benches);
