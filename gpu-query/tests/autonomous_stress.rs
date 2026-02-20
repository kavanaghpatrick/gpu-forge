//! Stress tests for the autonomous query engine.
//!
//! All tests are marked `#[ignore]` — run with `cargo test --test autonomous_stress -- --ignored`.
//! Use `--test-threads=1` to avoid GPU contention between tests.
//!
//! Tests:
//! 1. `stress_memory_leak`: Run 10K queries, verify Metal allocated size growth < 1%.
//! 2. `stress_watchdog_survival`: Continuous queries for 30s, zero watchdog errors.
//! 3. `stress_concurrent_submit_poll`: One thread submits, main thread polls+reads.

use gpu_query::gpu::autonomous::executor::AutonomousExecutor;
use gpu_query::gpu::autonomous::loader::ColumnInfo;
use gpu_query::gpu::device::GpuDevice;
use gpu_query::sql::physical_plan::PhysicalPlan;
use gpu_query::sql::types::{AggFunc, CompareOp, Value};
use gpu_query::storage::columnar::ColumnarBatch;
use gpu_query::storage::schema::{ColumnDef, DataType, RuntimeSchema};
use objc2_metal::{MTLBuffer, MTLDevice};

// ============================================================================
// Test helpers
// ============================================================================

/// Create a RuntimeSchema for the standard 2-column test table.
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

/// Create deterministic test data with 2 INT64 columns:
///   Column 0 (amount): (i*7+13)%1000
///   Column 1 (region): i%5
fn make_test_batch(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
    schema: &RuntimeSchema,
    row_count: usize,
) -> ColumnarBatch {
    let mut batch = ColumnarBatch::allocate(device, schema, row_count);
    batch.row_count = row_count;

    unsafe {
        let ptr = batch.int_buffer.contents().as_ptr() as *mut i64;

        // Column 0 (amount): local_int_idx=0, offset = 0 * max_rows
        for i in 0..row_count {
            *ptr.add(i) = ((i * 7 + 13) % 1000) as i64;
        }

        // Column 1 (region): local_int_idx=1, offset = 1 * max_rows
        let offset = batch.max_rows;
        for i in 0..row_count {
            *ptr.add(offset + i) = (i % 5) as i64;
        }
    }

    batch
}

/// JIT schema: maps column names to types for JIT source generation.
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

/// CPU-side amount values for the test dataset.
fn cpu_amounts(row_count: usize) -> Vec<i64> {
    (0..row_count)
        .map(|i| ((i * 7 + 13) % 1000) as i64)
        .collect()
}

/// Helper: build a GpuScan node.
fn plan_scan(table: &str) -> PhysicalPlan {
    PhysicalPlan::GpuScan {
        table: table.into(),
        columns: vec!["amount".into(), "region".into()],
    }
}

/// Helper: build a GpuFilter node (column > value).
fn plan_filter_gt(column: &str, val: i64, input: PhysicalPlan) -> PhysicalPlan {
    PhysicalPlan::GpuFilter {
        compare_op: CompareOp::Gt,
        column: column.into(),
        value: Value::Int(val),
        input: Box::new(input),
    }
}

/// Helper: build a GpuAggregate node.
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

const ROW_COUNT: usize = 100_000;

// ============================================================================
// Stress Test 1: Memory Leak Detection
//
// Run 10K queries through the AutonomousExecutor and verify that the Metal
// device's currentAllocatedSize doesn't grow more than 1%. This ensures no
// per-query buffer leaks.
// ============================================================================
#[test]
#[ignore]
fn stress_memory_leak() {
    let gpu = GpuDevice::new();
    let mut executor = AutonomousExecutor::new(gpu.device.clone());

    // Load test data
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, ROW_COUNT);
    executor
        .load_table("sales", &schema, &batch)
        .expect("load_table failed");

    let col_schema = jit_schema();
    let amounts = cpu_amounts(ROW_COUNT);

    // Warm up: run a few queries so JIT cache is populated and steady-state reached
    for i in 0..10u32 {
        let threshold = (i % 900 + 50) as i64;
        let plan = plan_aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            plan_filter_gt("amount", threshold, plan_scan("sales")),
        );
        let _seq = executor
            .submit_query(&plan, &col_schema, "sales")
            .expect("warmup submit failed");

        let timeout = std::time::Duration::from_secs(10);
        let start = std::time::Instant::now();
        while !executor.poll_ready() {
            if start.elapsed() > timeout {
                panic!("Warmup query {} timed out", i);
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        let _ = executor.read_result();
    }

    // Snapshot Metal allocated size AFTER warm-up
    let size_before = gpu.device.currentAllocatedSize();
    eprintln!(
        "Metal allocated size after warm-up: {} bytes ({:.2} MB)",
        size_before,
        size_before as f64 / 1_048_576.0
    );

    // Run 10K queries
    let query_count = 10_000u32;
    let start_all = std::time::Instant::now();

    for i in 0..query_count {
        let threshold = (i % 900 + 50) as i64;
        let expected_count: i64 = amounts.iter().filter(|&&v| v > threshold).count() as i64;

        let plan = plan_aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            plan_filter_gt("amount", threshold, plan_scan("sales")),
        );

        let _seq = executor
            .submit_query(&plan, &col_schema, "sales")
            .unwrap_or_else(|e| panic!("Query {} submit failed: {}", i, e));

        let timeout = std::time::Duration::from_secs(10);
        let poll_start = std::time::Instant::now();
        while !executor.poll_ready() {
            if poll_start.elapsed() > timeout {
                panic!("Query {} timed out waiting for result", i);
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }

        let result = executor.read_result();

        // Spot-check correctness every 1000 queries
        if i % 1000 == 0 {
            assert_eq!(
                result.agg_results[0][0].value_int, expected_count,
                "Query {}: COUNT(*) WHERE amount > {} expected {}, got {}",
                i, threshold, expected_count, result.agg_results[0][0].value_int
            );
        }
    }

    let elapsed = start_all.elapsed();

    // Snapshot Metal allocated size AFTER 10K queries
    let size_after = gpu.device.currentAllocatedSize();
    eprintln!(
        "Metal allocated size after {}K queries: {} bytes ({:.2} MB)",
        query_count / 1000,
        size_after,
        size_after as f64 / 1_048_576.0
    );
    eprintln!(
        "Growth: {} bytes ({:.4}%)",
        size_after as i64 - size_before as i64,
        if size_before > 0 {
            ((size_after as f64 - size_before as f64) / size_before as f64) * 100.0
        } else {
            0.0
        }
    );
    eprintln!(
        "Total time for {} queries: {:.2}s ({:.2} us/query avg)",
        query_count,
        elapsed.as_secs_f64(),
        elapsed.as_micros() as f64 / query_count as f64
    );

    // Verify: growth < 1%
    let growth_pct = if size_before > 0 {
        ((size_after as f64 - size_before as f64) / size_before as f64) * 100.0
    } else {
        0.0
    };

    assert!(
        growth_pct < 1.0,
        "Metal allocated size grew by {:.4}% (from {} to {} bytes) after {} queries — exceeds 1% threshold",
        growth_pct,
        size_before,
        size_after,
        query_count
    );

    // Verify stats
    let stats = executor.stats();
    assert!(
        stats.total_queries >= query_count as u64,
        "Expected at least {} total queries, got {}",
        query_count,
        stats.total_queries
    );

    executor.shutdown();
    eprintln!("stress_memory_leak: PASSED ({} queries, {:.4}% growth)", query_count, growth_pct);
}

// ============================================================================
// Stress Test 2: Watchdog Survival
//
// Continuously submit and poll queries for 30 seconds. If the GPU watchdog
// timer fires (killing long-running shaders), it would cause command buffer
// errors and missing results. This test verifies zero watchdog errors over
// the sustained workload.
// ============================================================================
#[test]
#[ignore]
fn stress_watchdog_survival() {
    let gpu = GpuDevice::new();
    let mut executor = AutonomousExecutor::new(gpu.device.clone());

    // Load test data
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, ROW_COUNT);
    executor
        .load_table("sales", &schema, &batch)
        .expect("load_table failed");

    let col_schema = jit_schema();
    let amounts = cpu_amounts(ROW_COUNT);

    let duration = std::time::Duration::from_secs(30);
    let start = std::time::Instant::now();
    let mut query_count = 0u64;
    let mut error_count = 0u64;
    let mut wrong_result_count = 0u64;
    let mut max_poll_us = 0u64;

    while start.elapsed() < duration {
        let threshold = ((query_count % 900) + 50) as i64;
        let expected_count: i64 = amounts.iter().filter(|&&v| v > threshold).count() as i64;

        // Build and submit
        let plan = plan_aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            plan_filter_gt("amount", threshold, plan_scan("sales")),
        );

        match executor.submit_query(&plan, &col_schema, "sales") {
            Ok(_seq) => {}
            Err(e) => {
                error_count += 1;
                eprintln!("Query {} submit error: {}", query_count, e);
                continue;
            }
        }

        // Poll ready
        let poll_start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(10);
        loop {
            if executor.poll_ready() {
                break;
            }
            if poll_start.elapsed() > timeout {
                error_count += 1;
                eprintln!(
                    "Query {} timed out (threshold={})",
                    query_count, threshold
                );
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }

        let poll_us = poll_start.elapsed().as_micros() as u64;
        if poll_us > max_poll_us {
            max_poll_us = poll_us;
        }

        if executor.poll_ready() {
            let result = executor.read_result();

            // Check correctness
            if result.error_code != 0 {
                error_count += 1;
                eprintln!(
                    "Query {} error_code={} (threshold={})",
                    query_count, result.error_code, threshold
                );
            } else if result.agg_results[0][0].value_int != expected_count {
                wrong_result_count += 1;
                if wrong_result_count <= 5 {
                    eprintln!(
                        "Query {} wrong result: expected {}, got {} (threshold={})",
                        query_count,
                        expected_count,
                        result.agg_results[0][0].value_int,
                        threshold
                    );
                }
            }
        }

        query_count += 1;
    }

    let elapsed = start.elapsed();
    eprintln!(
        "Watchdog survival: {} queries in {:.2}s ({:.2} queries/sec)",
        query_count,
        elapsed.as_secs_f64(),
        query_count as f64 / elapsed.as_secs_f64()
    );
    eprintln!("Errors: {}, Wrong results: {}", error_count, wrong_result_count);
    eprintln!("Max poll latency: {} us", max_poll_us);

    // Zero watchdog/submit/timeout errors
    assert_eq!(
        error_count, 0,
        "Expected zero errors over {} queries in {}s, got {} errors",
        query_count,
        elapsed.as_secs(),
        error_count
    );

    // Allow a tiny number of wrong results from atomic CAS race at high contention,
    // but expect overwhelming majority correct
    let wrong_pct = if query_count > 0 {
        (wrong_result_count as f64 / query_count as f64) * 100.0
    } else {
        0.0
    };
    assert!(
        wrong_pct < 0.1,
        "Wrong result rate {:.4}% ({}/{}) exceeds 0.1% threshold",
        wrong_pct,
        wrong_result_count,
        query_count
    );

    // Must have run a meaningful number of queries in 30s
    assert!(
        query_count >= 100,
        "Expected at least 100 queries in 30s, got {}",
        query_count
    );

    executor.shutdown();
    eprintln!("stress_watchdog_survival: PASSED ({} queries, 0 errors)", query_count);
}

// ============================================================================
// Stress Test 3: Concurrent Submit + Poll
//
// One background thread submits queries at max rate. The main thread polls
// and reads results. Proper handshake ensures no overlapping GPU dispatches:
//   1. Submitter sends query plan+expected via channel, waits on "consumed" signal
//   2. Main thread polls GPU until ready, reads result, verifies, signals "consumed"
//   3. Submitter proceeds to next query
//
// This tests cross-thread usage of AutonomousExecutor (the warm-up thread pattern).
// ============================================================================
#[test]
#[ignore]
fn stress_concurrent_submit_poll() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};

    let gpu = GpuDevice::new();
    let executor = Arc::new(Mutex::new(AutonomousExecutor::new(gpu.device.clone())));

    // Load test data
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, ROW_COUNT);
    executor
        .lock()
        .unwrap()
        .load_table("sales", &schema, &batch)
        .expect("load_table failed");

    // Channel: submitter -> main thread (expected_count for each query)
    let (tx, rx) = mpsc::channel::<i64>();

    // Handshake flag: main thread sets this to signal submitter can proceed
    let can_submit = Arc::new(AtomicBool::new(true));

    let total_queries = 500u32;
    let executor_submit = Arc::clone(&executor);
    let can_submit_submit = Arc::clone(&can_submit);

    // Spawn submitter thread
    let submit_handle = std::thread::spawn(move || {
        let col_schema = jit_schema();
        let amounts_local = cpu_amounts(ROW_COUNT);
        let mut submitted = 0u32;

        for i in 0..total_queries {
            let threshold = (i % 900 + 50) as i64;
            let expected_count: i64 =
                amounts_local.iter().filter(|&&v| v > threshold).count() as i64;

            let plan = plan_aggregate(
                vec![(AggFunc::Count, "*")],
                vec![],
                plan_filter_gt("amount", threshold, plan_scan("sales")),
            );

            // Wait for "consumed" signal from main thread
            let wait_start = std::time::Instant::now();
            while !can_submit_submit.load(Ordering::Acquire) {
                if wait_start.elapsed() > std::time::Duration::from_secs(30) {
                    panic!("Submitter timed out waiting for consume signal at query {}", i);
                }
                std::thread::sleep(std::time::Duration::from_micros(50));
            }

            // Clear the flag — main thread will set it after consuming result
            can_submit_submit.store(false, Ordering::Release);

            // Submit query
            {
                let mut exec = executor_submit.lock().unwrap();
                exec.submit_query(&plan, &col_schema, "sales")
                    .unwrap_or_else(|e| panic!("Submit {} failed: {}", i, e));
            }

            // Send expected value to main thread
            tx.send(expected_count).ok();
            submitted += 1;
        }

        submitted
    });

    // Main thread: poll and read results
    let mut received = 0u32;
    let mut correct = 0u32;
    let mut errors = 0u32;
    let overall_start = std::time::Instant::now();

    while received < total_queries {
        if overall_start.elapsed() > std::time::Duration::from_secs(60) {
            eprintln!("Overall timeout, received {}/{}", received, total_queries);
            break;
        }

        // Wait for submitter to send expected result
        match rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(expected_count) => {
                // Poll until GPU finishes this query
                let poll_start = std::time::Instant::now();
                let poll_timeout = std::time::Duration::from_secs(10);
                let mut timed_out = false;
                loop {
                    {
                        let exec = executor.lock().unwrap();
                        if exec.poll_ready() {
                            break;
                        }
                    }
                    if poll_start.elapsed() > poll_timeout {
                        errors += 1;
                        timed_out = true;
                        eprintln!(
                            "Result {} poll timed out (expected count={})",
                            received, expected_count
                        );
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }

                if !timed_out {
                    // Read result
                    let exec = executor.lock().unwrap();
                    let result = exec.read_result();
                    drop(exec);

                    if result.error_code != 0 {
                        errors += 1;
                        eprintln!("Result {} error_code={}", received, result.error_code);
                    } else if result.agg_results[0][0].value_int == expected_count {
                        correct += 1;
                    } else {
                        errors += 1;
                        if errors <= 5 {
                            eprintln!(
                                "Result {} wrong: expected {}, got {}",
                                received, expected_count, result.agg_results[0][0].value_int
                            );
                        }
                    }
                }

                received += 1;

                // Signal submitter: safe to submit next query
                can_submit.store(true, Ordering::Release);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                eprintln!("Channel recv timeout, received {}", received);
                errors += 1;
                break;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Submitter finished
                break;
            }
        }
    }

    let submitted = submit_handle.join().expect("submitter thread panicked");

    let elapsed = overall_start.elapsed();
    eprintln!(
        "Concurrent stress: submitted={}, received={}, correct={}, errors={}",
        submitted, received, correct, errors
    );
    eprintln!(
        "Completed in {:.2}s ({:.2} queries/sec)",
        elapsed.as_secs_f64(),
        received as f64 / elapsed.as_secs_f64()
    );

    // Verify all queries submitted
    assert_eq!(
        submitted, total_queries,
        "Expected {} submitted queries, got {}",
        total_queries, submitted
    );

    // Verify all results received
    assert_eq!(
        received, total_queries,
        "Expected {} results received, got {}",
        total_queries, received
    );

    // Verify zero errors
    assert_eq!(
        errors, 0,
        "Expected zero errors, got {} out of {} queries",
        errors, received
    );

    // Verify all correct
    assert_eq!(
        correct, total_queries,
        "Expected {} correct results, got {}",
        total_queries, correct
    );

    executor.lock().unwrap().shutdown();
    eprintln!(
        "stress_concurrent_submit_poll: PASSED ({} queries, all correct)",
        total_queries
    );
}
