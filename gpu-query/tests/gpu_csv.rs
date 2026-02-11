//! Integration tests for the GPU CSV parser pipeline.
//!
//! Tests the full path: CSV file -> mmap -> GPU newline detection ->
//! GPU field extraction -> SoA columnar buffers -> CPU readback + verify.

use std::io::Write;
use tempfile::NamedTempFile;

use gpu_query::gpu::device::GpuDevice;
use gpu_query::gpu::encode;
use gpu_query::gpu::types::CsvParseParams;
use gpu_query::io::mmap::MmapFile;
use gpu_query::storage::columnar::ColumnarBatch;
use gpu_query::storage::schema::{ColumnDef, DataType, RuntimeSchema};

// Import Metal traits so methods are in scope.
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState,
};

/// Helper: create a temp CSV file with given content.
fn make_csv(content: &str) -> NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("create temp csv");
    f.write_all(content.as_bytes()).expect("write csv");
    f.flush().expect("flush csv");
    f
}

/// Run the full GPU CSV parse pipeline and return the ColumnarBatch.
///
/// This function:
/// 1. Mmaps the file
/// 2. Runs csv_detect_newlines to find row boundaries
/// 3. Sorts row offsets (atomic adds produce unordered results)
/// 4. Runs csv_parse_fields to extract typed columns
/// 5. Returns the batch with parsed data
fn gpu_parse_csv(
    gpu: &GpuDevice,
    mmap: &MmapFile,
    schema: &RuntimeSchema,
    delimiter: u8,
    has_header: bool,
) -> ColumnarBatch {
    let file_size = mmap.file_size();
    let max_rows = 1024u32; // plenty for tests

    // Create the input buffer (mmap'd file data)
    let data_buffer = mmap.as_metal_buffer(&gpu.device);

    // Create params
    let params = CsvParseParams {
        file_size: file_size as u32,
        num_columns: schema.num_columns() as u32,
        delimiter: delimiter as u32,
        has_header: if has_header { 1 } else { 0 },
        max_rows,
        _pad0: 0,
    };

    // Allocate output buffers for pass 1
    let row_count_buffer = encode::alloc_buffer(&gpu.device, std::mem::size_of::<u32>());
    // Zero the row count
    unsafe {
        let ptr = row_count_buffer.contents().as_ptr() as *mut u32;
        *ptr = 0;
    }

    let row_offsets_buffer = encode::alloc_buffer(
        &gpu.device,
        (max_rows as usize + 1) * std::mem::size_of::<u32>(),
    );

    let params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[params]);

    // Create pipeline states
    let detect_pipeline = encode::make_pipeline(&gpu.library, "csv_detect_newlines");
    let parse_pipeline = encode::make_pipeline(&gpu.library, "csv_parse_fields");

    // ---- Pass 1: Detect newlines ----
    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        encode::dispatch_threads_1d(
            &encoder,
            &detect_pipeline,
            &[
                (&data_buffer, 0),
                (&row_count_buffer, 1),
                (&row_offsets_buffer, 2),
                (&params_buffer, 3),
            ],
            file_size,
        );

        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    // Read back row count from pass 1
    let newline_count = unsafe {
        let ptr = row_count_buffer.contents().as_ptr() as *const u32;
        *ptr
    };

    // Row offsets from atomic adds are unordered -- sort them on CPU.
    let mut sorted_offsets = unsafe {
        let ptr = row_offsets_buffer.contents().as_ptr() as *const u32;
        std::slice::from_raw_parts(ptr, newline_count as usize).to_vec()
    };
    sorted_offsets.sort();

    // Compute data row count:
    // If has_header: first newline ends header, so data rows = newline_count - 1
    // If no header: data rows = newline_count (each newline terminates a data row)
    let num_data_rows = if has_header {
        (newline_count as usize).saturating_sub(1)
    } else {
        newline_count as usize
    };

    if num_data_rows == 0 {
        let mut batch = ColumnarBatch::allocate(&gpu.device, schema, max_rows as usize);
        batch.row_count = 0;
        return batch;
    }

    // Build data row offsets for pass 2.
    // sorted_offsets[0] = byte after first newline = start of row after header (or data row 1)
    // For has_header: data rows start at sorted_offsets[0..] (skip none, first is data row 0)
    // For no header: need to prepend offset 0 (file start = first row).
    let final_offsets: Vec<u32> = if has_header {
        // sorted_offsets[0] = start of data row 0 (after header newline)
        // sorted_offsets[1] = start of data row 1, etc.
        sorted_offsets
    } else {
        let mut v = vec![0u32];
        v.extend_from_slice(&sorted_offsets);
        v
    };

    let data_row_offsets_buffer = encode::alloc_buffer_with_data(&gpu.device, &final_offsets);

    // Row count buffer for pass 2
    let data_row_count_buffer = encode::alloc_buffer(&gpu.device, std::mem::size_of::<u32>());
    unsafe {
        let ptr = data_row_count_buffer.contents().as_ptr() as *mut u32;
        *ptr = num_data_rows as u32;
    }

    // Allocate SoA columnar batch
    let mut batch = ColumnarBatch::allocate(&gpu.device, schema, max_rows as usize);

    // Create column schemas buffer for GPU
    let gpu_schemas = schema.to_gpu_schemas();
    let schemas_buffer = encode::alloc_buffer_with_data(&gpu.device, &gpu_schemas);

    // Params for pass 2: max_rows must match batch.max_rows for SoA stride consistency.
    // The shader uses max_rows as the stride for column layout: [col_idx * max_rows + row_idx].
    let parse_params = CsvParseParams {
        file_size: file_size as u32,
        num_columns: schema.num_columns() as u32,
        delimiter: delimiter as u32,
        has_header: if has_header { 1 } else { 0 },
        max_rows,
        _pad0: 0,
    };
    let parse_params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[parse_params]);

    // ---- Pass 2: Parse fields ----
    let cmd_buf2 = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf2
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        encoder.setComputePipelineState(&parse_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&data_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&data_row_offsets_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&data_row_count_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&batch.int_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&batch.float_buffer), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&parse_params_buffer), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&schemas_buffer), 0, 6);
        }

        let threads_per_tg = parse_pipeline.maxTotalThreadsPerThreadgroup().min(256);
        let tg_count = num_data_rows.div_ceil(threads_per_tg);

        let grid = objc2_metal::MTLSize {
            width: tg_count,
            height: 1,
            depth: 1,
        };
        let tg = objc2_metal::MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        encoder.endEncoding();
    }
    cmd_buf2.commit();
    cmd_buf2.waitUntilCompleted();

    batch.row_count = num_data_rows;
    batch
}

// ---- Tests ----

#[test]
fn test_gpu_csv_parse_basic() {
    let csv = "id,name,amount\n1,alice,100\n2,bob,200\n3,charlie,300\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    // Schema: id=INT64, name=VARCHAR(skip), amount=INT64
    let schema = RuntimeSchema::new(vec![
        ColumnDef {
            name: "id".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "name".to_string(),
            data_type: DataType::Varchar,
            nullable: false,
        },
        ColumnDef {
            name: "amount".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
    ]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 3, "Should have 3 data rows");

    // Read back INT64 columns.
    // INT64 columns: id (local idx 0), amount (local idx 1)
    unsafe {
        let ids = batch.read_int_column(0);
        assert_eq!(ids, vec![1, 2, 3], "id column mismatch");

        let amounts = batch.read_int_column(1);
        assert_eq!(amounts, vec![100, 200, 300], "amount column mismatch");
    }
}

#[test]
fn test_gpu_csv_parse_floats() {
    let csv = "x,y\n1.5,2.5\n3.0,4.0\n10.25,20.75\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![
        ColumnDef {
            name: "x".to_string(),
            data_type: DataType::Float64,
            nullable: false,
        },
        ColumnDef {
            name: "y".to_string(),
            data_type: DataType::Float64,
            nullable: false,
        },
    ]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 3);

    // Float columns are f32 on GPU (Metal lacks double)
    unsafe {
        let x_vals = batch.read_float_column(0);
        let y_vals = batch.read_float_column(1);

        assert!((x_vals[0] - 1.5).abs() < 1e-5, "x[0] = {}", x_vals[0]);
        assert!((x_vals[1] - 3.0).abs() < 1e-5, "x[1] = {}", x_vals[1]);
        assert!((x_vals[2] - 10.25).abs() < 1e-5, "x[2] = {}", x_vals[2]);

        assert!((y_vals[0] - 2.5).abs() < 1e-5, "y[0] = {}", y_vals[0]);
        assert!((y_vals[1] - 4.0).abs() < 1e-5, "y[1] = {}", y_vals[1]);
        assert!((y_vals[2] - 20.75).abs() < 1e-5, "y[2] = {}", y_vals[2]);
    }
}

#[test]
fn test_gpu_csv_parse_mixed_types() {
    let csv = "id,price,count\n1,9.99,10\n2,19.50,20\n3,5.25,30\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![
        ColumnDef {
            name: "id".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "price".to_string(),
            data_type: DataType::Float64,
            nullable: false,
        },
        ColumnDef {
            name: "count".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
    ]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 3);

    unsafe {
        // INT64 columns: id (idx 0), count (idx 1)
        let ids = batch.read_int_column(0);
        assert_eq!(ids, vec![1, 2, 3]);

        let counts = batch.read_int_column(1);
        assert_eq!(counts, vec![10, 20, 30]);

        // FLOAT32 columns: price (idx 0)
        let prices = batch.read_float_column(0);
        assert!((prices[0] - 9.99).abs() < 1e-2, "price[0] = {}", prices[0]);
        assert!((prices[1] - 19.50).abs() < 1e-2, "price[1] = {}", prices[1]);
        assert!((prices[2] - 5.25).abs() < 1e-2, "price[2] = {}", prices[2]);
    }
}

#[test]
fn test_gpu_csv_parse_negative_values() {
    let csv = "val\n-42\n0\n100\n-1\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![ColumnDef {
        name: "val".to_string(),
        data_type: DataType::Int64,
        nullable: false,
    }]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 4);

    unsafe {
        let vals = batch.read_int_column(0);
        assert_eq!(vals, vec![-42, 0, 100, -1]);
    }
}

#[test]
fn test_gpu_csv_parse_single_row() {
    let csv = "x\n42\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![ColumnDef {
        name: "x".to_string(),
        data_type: DataType::Int64,
        nullable: false,
    }]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 1);

    unsafe {
        let vals = batch.read_int_column(0);
        assert_eq!(vals, vec![42]);
    }
}

#[test]
fn test_gpu_csv_parse_many_rows() {
    // Generate a CSV with 100 rows
    let mut csv = String::from("id,value\n");
    for i in 0..100 {
        csv.push_str(&format!("{},{}\n", i, i * 10));
    }
    let tmp = make_csv(&csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![
        ColumnDef {
            name: "id".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "value".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
    ]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 100);

    unsafe {
        let ids = batch.read_int_column(0);
        let values = batch.read_int_column(1);

        for i in 0..100 {
            assert_eq!(ids[i], i as i64, "id[{}] mismatch", i);
            assert_eq!(values[i], (i * 10) as i64, "value[{}] mismatch", i);
        }
    }
}

// ---- Edge case tests ----

#[test]
fn test_gpu_csv_parse_single_column() {
    let csv = "value\n42\n100\n-7\n0\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![ColumnDef {
        name: "value".to_string(),
        data_type: DataType::Int64,
        nullable: false,
    }]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 4);
    unsafe {
        let vals = batch.read_int_column(0);
        assert_eq!(vals, vec![42, 100, -7, 0]);
    }
}

#[test]
fn test_gpu_csv_parse_tab_delimiter() {
    let csv = "id\tvalue\n1\t100\n2\t200\n3\t300\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![
        ColumnDef {
            name: "id".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "value".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
    ]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b'\t', true);

    assert_eq!(batch.row_count, 3);
    unsafe {
        let ids = batch.read_int_column(0);
        assert_eq!(ids, vec![1, 2, 3]);
        let values = batch.read_int_column(1);
        assert_eq!(values, vec![100, 200, 300]);
    }
}

#[test]
fn test_gpu_csv_parse_large_int_values() {
    let csv = "val\n1000000\n-999999\n2147483647\n-2147483648\n0\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![ColumnDef {
        name: "val".to_string(),
        data_type: DataType::Int64,
        nullable: false,
    }]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 5);
    unsafe {
        let vals = batch.read_int_column(0);
        assert_eq!(vals[0], 1_000_000);
        assert_eq!(vals[1], -999_999);
        assert_eq!(vals[2], 2_147_483_647);
        assert_eq!(vals[3], -2_147_483_648);
        assert_eq!(vals[4], 0);
    }
}

#[test]
fn test_gpu_csv_parse_all_zeros() {
    let csv = "a,b\n0,0\n0,0\n0,0\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![
        ColumnDef {
            name: "a".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "b".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
    ]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 3);
    unsafe {
        let a_vals = batch.read_int_column(0);
        let b_vals = batch.read_int_column(1);
        assert_eq!(a_vals, vec![0, 0, 0]);
        assert_eq!(b_vals, vec![0, 0, 0]);
    }
}

#[test]
fn test_gpu_csv_parse_negative_floats() {
    let csv = "val\n-1.5\n-0.25\n-100.75\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![ColumnDef {
        name: "val".to_string(),
        data_type: DataType::Float64,
        nullable: false,
    }]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 3);
    unsafe {
        let vals = batch.read_float_column(0);
        assert!((vals[0] - (-1.5)).abs() < 1e-5, "val[0] = {}", vals[0]);
        assert!((vals[1] - (-0.25)).abs() < 1e-5, "val[1] = {}", vals[1]);
        assert!((vals[2] - (-100.75)).abs() < 1e-2, "val[2] = {}", vals[2]);
    }
}

#[test]
fn test_gpu_csv_parse_many_columns() {
    let csv = "a,b,c,d,e\n1,2,3,4,5\n10,20,30,40,50\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![
        ColumnDef {
            name: "a".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "b".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "c".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "d".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "e".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
    ]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 2);
    unsafe {
        let a = batch.read_int_column(0);
        let e = batch.read_int_column(4);
        assert_eq!(a, vec![1, 10]);
        assert_eq!(e, vec![5, 50]);
    }
}

#[test]
fn test_gpu_csv_parse_pipe_delimiter() {
    let csv = "id|value\n1|100\n2|200\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![
        ColumnDef {
            name: "id".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "value".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
    ]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b'|', true);

    assert_eq!(batch.row_count, 2);
    unsafe {
        let ids = batch.read_int_column(0);
        let vals = batch.read_int_column(1);
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(vals, vec![100, 200]);
    }
}

#[test]
fn test_gpu_csv_parse_two_rows() {
    let csv = "a,b\n10,20\n30,40\n";
    let tmp = make_csv(csv);

    let gpu = GpuDevice::new();
    let mmap = MmapFile::open(tmp.path()).expect("mmap failed");

    let schema = RuntimeSchema::new(vec![
        ColumnDef {
            name: "a".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "b".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
    ]);

    let batch = gpu_parse_csv(&gpu, &mmap, &schema, b',', true);

    assert_eq!(batch.row_count, 2);
    unsafe {
        let a = batch.read_int_column(0);
        let b = batch.read_int_column(1);
        assert_eq!(a, vec![10, 30]);
        assert_eq!(b, vec![20, 40]);
    }
}
