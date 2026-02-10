//! GPU query execution engine: connects SQL physical plan to Metal kernel dispatch.
//!
//! Walks the physical plan tree bottom-up, dispatching Metal compute kernels
//! in sequence within a single command buffer per stage. Uses `waitUntilCompleted`
//! for synchronous result readback (POC simplification).
//!
//! Pipeline: SQL parse -> PhysicalPlan -> GpuScan (CSV parse) -> GpuFilter
//! (column_filter) -> GpuAggregate (aggregate_count/aggregate_sum_int64) -> result.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState,
};

use crate::gpu::device::GpuDevice;
use crate::gpu::encode;
use crate::gpu::pipeline::{filter_pso_key, ColumnTypeCode, PsoCache};
use crate::gpu::pipeline::CompareOp as GpuCompareOp;
use crate::gpu::types::{AggParams, CsvParseParams, FilterParams};
use crate::io::catalog::TableEntry;
use crate::io::format_detect::FileFormat;
use crate::io::mmap::MmapFile;
use crate::io::json;
use crate::io::parquet::{self, ColumnData};
use crate::sql::physical_plan::PhysicalPlan;
use crate::sql::types::{AggFunc, CompareOp, Value};
use crate::storage::columnar::ColumnarBatch;
use crate::storage::schema::{ColumnDef, DataType, RuntimeSchema};

/// Result of executing a query.
pub struct QueryResult {
    /// Column names in result.
    pub columns: Vec<String>,
    /// Formatted row values.
    pub rows: Vec<Vec<String>>,
    /// Number of result rows.
    pub row_count: usize,
}

impl QueryResult {
    /// Print results as a formatted table.
    pub fn print(&self) {
        if self.columns.is_empty() {
            println!("(empty result)");
            return;
        }

        // Compute column widths
        let mut widths: Vec<usize> = self.columns.iter().map(|c| c.len()).collect();
        for row in &self.rows {
            for (i, val) in row.iter().enumerate() {
                if i < widths.len() && val.len() > widths[i] {
                    widths[i] = val.len();
                }
            }
        }

        // Print header
        let header: Vec<String> = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{:>width$}", c, width = widths[i]))
            .collect();
        println!("{}", header.join(" | "));

        // Print separator
        let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
        println!("{}", sep.join("-+-"));

        // Print rows
        for row in &self.rows {
            let formatted: Vec<String> = row
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let w = if i < widths.len() { widths[i] } else { v.len() };
                    format!("{:>width$}", v, width = w)
                })
                .collect();
            println!("{}", formatted.join(" | "));
        }

        println!("({} row{})", self.row_count, if self.row_count == 1 { "" } else { "s" });
    }
}

/// Intermediate result from a scan stage.
struct ScanResult {
    /// The mmap'd file (must outlive Metal buffers).
    _mmap: MmapFile,
    /// Parsed columnar data.
    batch: ColumnarBatch,
    /// Runtime schema for the parsed data.
    schema: RuntimeSchema,
    /// CSV metadata delimiter.
    delimiter: u8,
}

/// Intermediate result from a filter stage.
struct FilterResult {
    /// Selection bitmask (1 bit per row).
    bitmask_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Number of matching rows.
    match_count: u32,
    /// Total row count from input.
    row_count: u32,
}

/// The GPU query execution engine.
pub struct QueryExecutor {
    device: GpuDevice,
    pso_cache: PsoCache,
}

impl QueryExecutor {
    /// Create a new query executor with Metal device and empty PSO cache.
    pub fn new() -> Result<Self, String> {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new();
        Ok(Self { device, pso_cache })
    }

    /// Execute a physical plan against data in the catalog.
    pub fn execute(
        &mut self,
        plan: &PhysicalPlan,
        catalog: &[TableEntry],
    ) -> Result<QueryResult, String> {
        match plan {
            PhysicalPlan::GpuAggregate {
                functions,
                group_by: _,
                input,
            } => {
                // Walk down to get scan and optional filter results
                let (scan_result, filter_result) = self.resolve_input(input, catalog)?;
                self.execute_aggregate(&scan_result, filter_result.as_ref(), functions)
            }

            PhysicalPlan::GpuFilter { .. } => {
                // A filter without aggregate on top: return filtered row count
                let (scan_result, filter_result) = self.resolve_input(plan, catalog)?;
                let count = filter_result
                    .as_ref()
                    .map(|f| f.match_count)
                    .unwrap_or(scan_result.batch.row_count as u32);
                Ok(QueryResult {
                    columns: vec!["count".to_string()],
                    rows: vec![vec![count.to_string()]],
                    row_count: 1,
                })
            }

            PhysicalPlan::GpuScan { table, .. } => {
                let scan_result = self.execute_scan(table, catalog)?;
                let row_count = scan_result.batch.row_count;
                Ok(QueryResult {
                    columns: vec!["count(*)".to_string()],
                    rows: vec![vec![row_count.to_string()]],
                    row_count: 1,
                })
            }

            PhysicalPlan::GpuLimit { count, input } => {
                let mut result = self.execute(input, catalog)?;
                if result.rows.len() > *count {
                    result.rows.truncate(*count);
                    result.row_count = *count;
                }
                Ok(result)
            }

            _ => Err(format!("Unsupported plan node for POC: {:?}", plan)),
        }
    }

    /// Recursively resolve the input chain to get scan + optional filter results.
    fn resolve_input(
        &mut self,
        plan: &PhysicalPlan,
        catalog: &[TableEntry],
    ) -> Result<(ScanResult, Option<FilterResult>), String> {
        match plan {
            PhysicalPlan::GpuScan { table, .. } => {
                let scan = self.execute_scan(table, catalog)?;
                Ok((scan, None))
            }

            PhysicalPlan::GpuFilter {
                compare_op,
                column,
                value,
                input,
            } => {
                // Get the scan from below the filter
                let scan = match input.as_ref() {
                    PhysicalPlan::GpuScan { table, .. } => self.execute_scan(table, catalog)?,
                    other => {
                        // Recurse for nested plans
                        let (scan, _) = self.resolve_input(other, catalog)?;
                        scan
                    }
                };

                let filter = self.execute_filter(&scan, compare_op, column, value)?;
                Ok((scan, Some(filter)))
            }

            PhysicalPlan::GpuAggregate { input, .. } => {
                // Pass through to the input
                self.resolve_input(input, catalog)
            }

            _ => Err(format!("Unsupported input plan node: {:?}", plan)),
        }
    }

    /// Execute a table scan: mmap file -> GPU CSV/Parquet parse -> ColumnarBatch.
    fn execute_scan(
        &self,
        table: &str,
        catalog: &[TableEntry],
    ) -> Result<ScanResult, String> {
        // Find the table in the catalog
        let entry = catalog
            .iter()
            .find(|e| e.name.eq_ignore_ascii_case(table))
            .ok_or_else(|| format!("Table '{}' not found in catalog", table))?;

        match entry.format {
            FileFormat::Csv => {
                let csv_meta = entry
                    .csv_metadata
                    .as_ref()
                    .ok_or_else(|| format!("No CSV metadata for table '{}'", table))?;

                // Build runtime schema: infer types from first data rows
                let schema = infer_schema_from_csv(&entry.path, csv_meta)?;

                // Memory-map the file
                let mmap = MmapFile::open(&entry.path)
                    .map_err(|e| format!("Failed to mmap '{}': {}", entry.path.display(), e))?;

                // Run GPU CSV parse pipeline
                let batch = self.gpu_parse_csv(&mmap, &schema, csv_meta.delimiter);

                Ok(ScanResult {
                    _mmap: mmap,
                    batch,
                    schema,
                    delimiter: csv_meta.delimiter,
                })
            }
            FileFormat::Parquet => {
                self.execute_parquet_scan(&entry.path)
            }
            FileFormat::Json => {
                self.execute_json_scan(&entry.path)
            }
            other => {
                Err(format!(
                    "File format {:?} not yet supported for table '{}'",
                    other, table
                ))
            }
        }
    }

    /// Execute a Parquet scan: read metadata -> read column data -> upload to GPU buffers.
    fn execute_parquet_scan(
        &self,
        path: &std::path::Path,
    ) -> Result<ScanResult, String> {
        let meta = parquet::read_metadata(path)?;
        let columns_data = parquet::read_columns(path, &meta, None)?;

        let schema = meta.schema.clone();
        let row_count = meta.row_count;
        let max_rows = std::cmp::max(row_count, 1);

        // Allocate ColumnarBatch and upload data
        let mut batch = ColumnarBatch::allocate(&self.device.device, &schema, max_rows);
        batch.row_count = row_count;

        // Upload column data into the ColumnarBatch buffers

        for (col_name, col_data) in &columns_data {
            let col_idx = schema
                .column_index(col_name)
                .ok_or_else(|| format!("Column '{}' not found in schema", col_name))?;
            let col_def = &schema.columns[col_idx];

            match col_def.data_type {
                DataType::Int64 => {
                    // Find the local int column index
                    let local_idx = schema.columns[..col_idx]
                        .iter()
                        .filter(|c| c.data_type == DataType::Int64)
                        .count();

                    if let ColumnData::Int64(ref values) = col_data {
                        // Write directly to the int_buffer at the correct offset
                        let offset = local_idx * max_rows;
                        unsafe {
                            let ptr = batch.int_buffer.contents().as_ptr() as *mut i64;
                            for (i, &v) in values.iter().enumerate() {
                                *ptr.add(offset + i) = v;
                            }
                        }
                    }
                }
                DataType::Float64 => {
                    let local_idx = schema.columns[..col_idx]
                        .iter()
                        .filter(|c| c.data_type == DataType::Float64)
                        .count();

                    if let ColumnData::Float64(ref values) = col_data {
                        // Metal uses float32, so downcast
                        let offset = local_idx * max_rows;
                        unsafe {
                            let ptr = batch.float_buffer.contents().as_ptr() as *mut f32;
                            for (i, &v) in values.iter().enumerate() {
                                *ptr.add(offset + i) = v as f32;
                            }
                        }
                    }
                }
                _ => {
                    // Skip unsupported types (Varchar, etc.)
                }
            }
        }

        // We need a dummy mmap for the ScanResult. Create a minimal one.
        // Since Parquet data is already in GPU buffers, the mmap is just a placeholder.
        let mmap = MmapFile::open(path)
            .map_err(|e| format!("Failed to mmap '{}': {}", path.display(), e))?;

        Ok(ScanResult {
            _mmap: mmap,
            batch,
            schema,
            delimiter: b',',
        })
    }

    /// Execute an NDJSON scan: mmap file -> GPU structural index -> GPU field extraction -> ColumnarBatch.
    fn execute_json_scan(
        &self,
        path: &std::path::Path,
    ) -> Result<ScanResult, String> {
        let json_meta = json::parse_ndjson_header(path)
            .map_err(|e| format!("Failed to parse NDJSON header '{}': {}", path.display(), e))?;

        let schema = json_meta.to_schema();

        // Memory-map the file
        let mmap = MmapFile::open(path)
            .map_err(|e| format!("Failed to mmap '{}': {}", path.display(), e))?;

        let batch = self.gpu_parse_json(&mmap, &schema);

        Ok(ScanResult {
            _mmap: mmap,
            batch,
            schema,
            delimiter: b',', // unused for JSON
        })
    }

    /// Run the two-pass GPU NDJSON parse pipeline.
    ///
    /// Pass 1: json_structural_index - detect newlines for row boundaries
    /// Pass 2: json_extract_columns - extract fields to SoA column buffers
    fn gpu_parse_json(
        &self,
        mmap: &MmapFile,
        schema: &RuntimeSchema,
    ) -> ColumnarBatch {
        let file_size = mmap.file_size();
        let max_rows = 1_048_576u32; // 1M rows max for POC

        let data_buffer = mmap.as_metal_buffer(&self.device.device);

        // Reuse CsvParseParams for JSON: delimiter=0, has_header=0
        let params = CsvParseParams {
            file_size: file_size as u32,
            num_columns: schema.num_columns() as u32,
            delimiter: 0, // unused for JSON
            has_header: 0, // NDJSON has no header
            max_rows,
            _pad0: 0,
        };

        // Allocate output buffers for pass 1
        let row_count_buffer =
            encode::alloc_buffer(&self.device.device, std::mem::size_of::<u32>());
        unsafe {
            let ptr = row_count_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;
        }

        let row_offsets_buffer = encode::alloc_buffer(
            &self.device.device,
            (max_rows as usize + 1) * std::mem::size_of::<u32>(),
        );
        let params_buffer = encode::alloc_buffer_with_data(&self.device.device, &[params]);

        let index_pipeline =
            encode::make_pipeline(&self.device.library, "json_structural_index");
        let extract_pipeline =
            encode::make_pipeline(&self.device.library, "json_extract_columns");

        // ---- Pass 1: Structural indexing (newline detection) ----
        let cmd_buf = encode::make_command_buffer(&self.device.command_queue);
        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("compute encoder");

            encode::dispatch_threads_1d(
                &encoder,
                &index_pipeline,
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

        let newline_count = unsafe {
            let ptr = row_count_buffer.contents().as_ptr() as *const u32;
            *ptr
        };

        // Sort offsets (atomics produce unordered results)
        let mut sorted_offsets = unsafe {
            let ptr = row_offsets_buffer.contents().as_ptr() as *const u32;
            std::slice::from_raw_parts(ptr, newline_count as usize).to_vec()
        };
        sorted_offsets.sort();

        // For NDJSON: each newline terminates a JSON object, so data rows = newline_count.
        // Row 0 starts at byte 0, row 1 starts after first newline, etc.
        let num_data_rows = newline_count as usize;

        if num_data_rows == 0 {
            let mut batch =
                ColumnarBatch::allocate(&self.device.device, schema, max_rows as usize);
            batch.row_count = 0;
            return batch;
        }

        // Build row start offsets: [0, sorted_offsets[0], sorted_offsets[1], ...]
        // Row 0 starts at byte 0 (beginning of file)
        let mut data_row_offsets = Vec::with_capacity(num_data_rows + 1);
        data_row_offsets.push(0u32); // First row starts at beginning of file
        data_row_offsets.extend_from_slice(&sorted_offsets);

        let data_row_offsets_buffer =
            encode::alloc_buffer_with_data(&self.device.device, &data_row_offsets);

        let data_row_count_buffer =
            encode::alloc_buffer(&self.device.device, std::mem::size_of::<u32>());
        unsafe {
            let ptr = data_row_count_buffer.contents().as_ptr() as *mut u32;
            *ptr = num_data_rows as u32;
        }

        let mut batch =
            ColumnarBatch::allocate(&self.device.device, schema, max_rows as usize);

        let gpu_schemas = schema.to_gpu_schemas();
        let schemas_buffer =
            encode::alloc_buffer_with_data(&self.device.device, &gpu_schemas);

        let extract_params = CsvParseParams {
            file_size: file_size as u32,
            num_columns: schema.num_columns() as u32,
            delimiter: 0,
            has_header: 0,
            max_rows,
            _pad0: 0,
        };
        let extract_params_buffer =
            encode::alloc_buffer_with_data(&self.device.device, &[extract_params]);

        // ---- Pass 2: Field extraction ----
        let cmd_buf2 = encode::make_command_buffer(&self.device.command_queue);
        {
            let encoder = cmd_buf2
                .computeCommandEncoder()
                .expect("compute encoder");

            encoder.setComputePipelineState(&extract_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&data_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&data_row_offsets_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&data_row_count_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&batch.int_buffer), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&batch.float_buffer), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&extract_params_buffer), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&schemas_buffer), 0, 6);
            }

            let threads_per_tg = extract_pipeline.maxTotalThreadsPerThreadgroup().min(256);
            let tg_count = (num_data_rows + threads_per_tg - 1) / threads_per_tg;

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

    /// Run the two-pass GPU CSV parse pipeline.
    fn gpu_parse_csv(
        &self,
        mmap: &MmapFile,
        schema: &RuntimeSchema,
        delimiter: u8,
    ) -> ColumnarBatch {
        let file_size = mmap.file_size();
        let max_rows = 1_048_576u32; // 1M rows max for POC

        let data_buffer = mmap.as_metal_buffer(&self.device.device);

        let params = CsvParseParams {
            file_size: file_size as u32,
            num_columns: schema.num_columns() as u32,
            delimiter: delimiter as u32,
            has_header: 1,
            max_rows,
            _pad0: 0,
        };

        // Allocate output buffers for pass 1
        let row_count_buffer =
            encode::alloc_buffer(&self.device.device, std::mem::size_of::<u32>());
        unsafe {
            let ptr = row_count_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;
        }

        let row_offsets_buffer = encode::alloc_buffer(
            &self.device.device,
            (max_rows as usize + 1) * std::mem::size_of::<u32>(),
        );
        let params_buffer = encode::alloc_buffer_with_data(&self.device.device, &[params]);

        let detect_pipeline =
            encode::make_pipeline(&self.device.library, "csv_detect_newlines");
        let parse_pipeline =
            encode::make_pipeline(&self.device.library, "csv_parse_fields");

        // ---- Pass 1: Detect newlines ----
        let cmd_buf = encode::make_command_buffer(&self.device.command_queue);
        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("compute encoder");

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

        let newline_count = unsafe {
            let ptr = row_count_buffer.contents().as_ptr() as *const u32;
            *ptr
        };

        let mut sorted_offsets = unsafe {
            let ptr = row_offsets_buffer.contents().as_ptr() as *const u32;
            std::slice::from_raw_parts(ptr, newline_count as usize).to_vec()
        };
        sorted_offsets.sort();

        // Data rows = newline_count - 1 (first newline ends header)
        let num_data_rows = (newline_count as usize).saturating_sub(1);

        if num_data_rows == 0 {
            let mut batch =
                ColumnarBatch::allocate(&self.device.device, schema, max_rows as usize);
            batch.row_count = 0;
            return batch;
        }

        let data_row_offsets_buffer =
            encode::alloc_buffer_with_data(&self.device.device, &sorted_offsets);

        let data_row_count_buffer =
            encode::alloc_buffer(&self.device.device, std::mem::size_of::<u32>());
        unsafe {
            let ptr = data_row_count_buffer.contents().as_ptr() as *mut u32;
            *ptr = num_data_rows as u32;
        }

        let mut batch =
            ColumnarBatch::allocate(&self.device.device, schema, max_rows as usize);

        let gpu_schemas = schema.to_gpu_schemas();
        let schemas_buffer =
            encode::alloc_buffer_with_data(&self.device.device, &gpu_schemas);

        let parse_params = CsvParseParams {
            file_size: file_size as u32,
            num_columns: schema.num_columns() as u32,
            delimiter: delimiter as u32,
            has_header: 1,
            max_rows,
            _pad0: 0,
        };
        let parse_params_buffer =
            encode::alloc_buffer_with_data(&self.device.device, &[parse_params]);

        // ---- Pass 2: Parse fields ----
        let cmd_buf2 = encode::make_command_buffer(&self.device.command_queue);
        {
            let encoder = cmd_buf2
                .computeCommandEncoder()
                .expect("compute encoder");

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
            let tg_count = (num_data_rows + threads_per_tg - 1) / threads_per_tg;

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

    /// Execute a filter: encode column_filter kernel with function constants.
    fn execute_filter(
        &mut self,
        scan: &ScanResult,
        compare_op: &CompareOp,
        column: &str,
        value: &Value,
    ) -> Result<FilterResult, String> {
        let row_count = scan.batch.row_count as u32;
        let bitmask_words = ((row_count + 31) / 32) as usize;

        // Resolve column index and type
        let col_idx = scan
            .schema
            .column_index(column)
            .ok_or_else(|| format!("Column '{}' not found in schema", column))?;
        let col_def = &scan.schema.columns[col_idx];

        // Determine the GPU column type and which buffer to use
        let (gpu_col_type, data_buffer, compare_int, compare_float) = match col_def.data_type {
            DataType::Int64 => {
                // Find the local index among INT64 columns
                let local_idx = scan
                    .schema
                    .columns[..col_idx]
                    .iter()
                    .filter(|c| c.data_type == DataType::Int64)
                    .count();

                let int_val = match value {
                    Value::Int(v) => *v,
                    Value::Float(v) => *v as i64,
                    _ => return Err(format!("Cannot compare INT64 column to {:?}", value)),
                };

                // The int_buffer holds all INT64 columns in SoA layout.
                // For column at local_idx, data starts at offset local_idx * max_rows * sizeof(i64).
                let offset = local_idx * scan.batch.max_rows * std::mem::size_of::<i64>();

                (ColumnTypeCode::Int64, &scan.batch.int_buffer, int_val, 0.0f64)
            }
            DataType::Float64 => {
                let local_idx = scan
                    .schema
                    .columns[..col_idx]
                    .iter()
                    .filter(|c| c.data_type == DataType::Float64)
                    .count();

                let float_val = match value {
                    Value::Float(v) => *v as f32,
                    Value::Int(v) => *v as f32,
                    _ => return Err(format!("Cannot compare FLOAT64 column to {:?}", value)),
                };

                let offset = local_idx * scan.batch.max_rows * std::mem::size_of::<f32>();
                let float_bits = float_val.to_bits() as i32;

                (
                    ColumnTypeCode::Float64,
                    &scan.batch.float_buffer,
                    0i64,
                    f64::from_bits(float_bits as u32 as u64),
                )
            }
            _ => {
                return Err(format!(
                    "Filter on {:?} columns not supported in POC",
                    col_def.data_type
                ))
            }
        };

        // Convert SQL CompareOp to GPU CompareOp
        let gpu_compare_op = match compare_op {
            CompareOp::Eq => GpuCompareOp::Eq,
            CompareOp::Ne => GpuCompareOp::Ne,
            CompareOp::Lt => GpuCompareOp::Lt,
            CompareOp::Le => GpuCompareOp::Le,
            CompareOp::Gt => GpuCompareOp::Gt,
            CompareOp::Ge => GpuCompareOp::Ge,
        };

        // Allocate output buffers
        let bitmask_buffer =
            encode::alloc_buffer(&self.device.device, bitmask_words * 4);
        let match_count_buffer = encode::alloc_buffer(&self.device.device, 4);
        let null_bitmap_buffer =
            encode::alloc_buffer(&self.device.device, std::cmp::max(bitmask_words * 4, 4));

        // Zero buffers
        unsafe {
            let bitmask_ptr = bitmask_buffer.contents().as_ptr() as *mut u32;
            for i in 0..bitmask_words {
                *bitmask_ptr.add(i) = 0;
            }
            let count_ptr = match_count_buffer.contents().as_ptr() as *mut u32;
            *count_ptr = 0;
            let null_ptr = null_bitmap_buffer.contents().as_ptr() as *mut u32;
            for i in 0..bitmask_words {
                *null_ptr.add(i) = 0;
            }
        }

        // Build FilterParams
        let params = FilterParams {
            compare_value_int: compare_int,
            compare_value_float: compare_float,
            row_count,
            column_stride: match gpu_col_type {
                ColumnTypeCode::Int64 => 8,
                ColumnTypeCode::Float64 => 4,
            },
            null_bitmap_present: 0,
            _pad0: 0,
            compare_value_int_hi: 0,
        };
        let params_buffer =
            encode::alloc_buffer_with_data(&self.device.device, &[params]);

        // Get specialized PSO
        let key = filter_pso_key(gpu_compare_op, gpu_col_type, false);
        let pipeline = self.pso_cache.get_or_create(&self.device.library, &key);

        // For multi-column SoA buffers, we need to set buffer offset to point at
        // the correct column's data within the shared buffer.
        let col_local_idx = match col_def.data_type {
            DataType::Int64 => scan.schema.columns[..col_idx]
                .iter()
                .filter(|c| c.data_type == DataType::Int64)
                .count(),
            DataType::Float64 => scan.schema.columns[..col_idx]
                .iter()
                .filter(|c| c.data_type == DataType::Float64)
                .count(),
            _ => 0,
        };

        let data_offset = match col_def.data_type {
            DataType::Int64 => col_local_idx * scan.batch.max_rows * std::mem::size_of::<i64>(),
            DataType::Float64 => {
                col_local_idx * scan.batch.max_rows * std::mem::size_of::<f32>()
            }
            _ => 0,
        };

        // Dispatch filter kernel
        let cmd_buf = encode::make_command_buffer(&self.device.command_queue);
        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("compute encoder");

            encoder.setComputePipelineState(pipeline);
            unsafe {
                // buffer(0): column data with offset for the specific column
                encoder.setBuffer_offset_atIndex(Some(data_buffer), data_offset, 0);
                encoder.setBuffer_offset_atIndex(Some(&bitmask_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&match_count_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&null_bitmap_buffer), 0, 4);
            }

            let threads_per_tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
            let grid_size = objc2_metal::MTLSize {
                width: row_count as usize,
                height: 1,
                depth: 1,
            };
            let tg_size = objc2_metal::MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            };

            encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
            encoder.endEncoding();
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Read back match count
        let match_count = unsafe {
            let ptr = match_count_buffer.contents().as_ptr() as *const u32;
            *ptr
        };

        Ok(FilterResult {
            bitmask_buffer,
            match_count,
            row_count,
        })
    }

    /// Execute aggregate functions using the selection mask from filter.
    fn execute_aggregate(
        &self,
        scan: &ScanResult,
        filter: Option<&FilterResult>,
        functions: &[(AggFunc, String)],
    ) -> Result<QueryResult, String> {
        let row_count = scan.batch.row_count as u32;

        // Build an all-ones mask if no filter was applied
        let default_mask;
        let mask_buffer = if let Some(f) = filter {
            &f.bitmask_buffer
        } else {
            default_mask = build_all_ones_mask(&self.device.device, row_count as usize);
            &default_mask
        };

        let mut columns = Vec::new();
        let mut values = Vec::new();

        for (func, col_name) in functions {
            let (col_label, result_str) = match func {
                AggFunc::Count => {
                    let label = if col_name == "*" {
                        "count(*)".to_string()
                    } else {
                        format!("count({})", col_name)
                    };
                    let count = self.run_aggregate_count(mask_buffer, row_count)?;
                    (label, count.to_string())
                }
                AggFunc::Sum => {
                    let label = format!("sum({})", col_name);

                    // Find column in schema
                    let col_idx = scan
                        .schema
                        .column_index(col_name)
                        .ok_or_else(|| format!("Column '{}' not found", col_name))?;
                    let col_def = &scan.schema.columns[col_idx];

                    match col_def.data_type {
                        DataType::Int64 => {
                            let local_idx = scan.schema.columns[..col_idx]
                                .iter()
                                .filter(|c| c.data_type == DataType::Int64)
                                .count();
                            let sum = self.run_aggregate_sum_int64(
                                &scan.batch,
                                mask_buffer,
                                row_count,
                                local_idx,
                            )?;
                            (label, sum.to_string())
                        }
                        _ => {
                            return Err(format!(
                                "SUM on {:?} columns not yet supported",
                                col_def.data_type
                            ))
                        }
                    }
                }
                _ => {
                    return Err(format!(
                        "{:?} aggregate not supported in POC",
                        func
                    ))
                }
            };

            columns.push(col_label);
            values.push(result_str);
        }

        Ok(QueryResult {
            columns,
            rows: vec![values],
            row_count: 1,
        })
    }

    /// Run the aggregate_count kernel on a selection mask.
    fn run_aggregate_count(
        &self,
        mask_buffer: &ProtocolObject<dyn MTLBuffer>,
        row_count: u32,
    ) -> Result<u64, String> {
        let num_words = ((row_count + 31) / 32) as usize;

        let result_buffer = encode::alloc_buffer(&self.device.device, 4);
        unsafe {
            let ptr = result_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;
        }

        let params = AggParams {
            row_count,
            group_count: 0,
            agg_function: 0, // COUNT
            _pad0: 0,
        };
        let params_buffer =
            encode::alloc_buffer_with_data(&self.device.device, &[params]);

        let pipeline =
            encode::make_pipeline(&self.device.library, "aggregate_count");

        let cmd_buf = encode::make_command_buffer(&self.device.command_queue);
        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("compute encoder");

            encode::dispatch_1d(
                &encoder,
                &pipeline,
                &[
                    (mask_buffer, 0),
                    (&result_buffer, 1),
                    (&params_buffer, 2),
                ],
                num_words,
            );

            encoder.endEncoding();
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let count = unsafe {
            let ptr = result_buffer.contents().as_ptr() as *const u32;
            *ptr as u64
        };

        Ok(count)
    }

    /// Run the aggregate_sum_int64 kernel.
    fn run_aggregate_sum_int64(
        &self,
        batch: &ColumnarBatch,
        mask_buffer: &ProtocolObject<dyn MTLBuffer>,
        row_count: u32,
        int_col_local_idx: usize,
    ) -> Result<i64, String> {
        // Result: two uint32 atomics (lo + hi) = 8 bytes
        let result_buffer = encode::alloc_buffer(&self.device.device, 8);
        unsafe {
            let ptr = result_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0; // lo
            *ptr.add(1) = 0; // hi
        }

        let params = AggParams {
            row_count,
            group_count: 0,
            agg_function: 1, // SUM
            _pad0: 0,
        };
        let params_buffer =
            encode::alloc_buffer_with_data(&self.device.device, &[params]);

        let pipeline =
            encode::make_pipeline(&self.device.library, "aggregate_sum_int64");

        let data_offset =
            int_col_local_idx * batch.max_rows * std::mem::size_of::<i64>();

        let cmd_buf = encode::make_command_buffer(&self.device.command_queue);
        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("compute encoder");

            encoder.setComputePipelineState(&pipeline);

            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(&batch.int_buffer),
                    data_offset,
                    0,
                );
                encoder.setBuffer_offset_atIndex(Some(mask_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&result_buffer), 0, 2); // result_lo
                encoder.setBuffer_offset_atIndex(Some(&result_buffer), 4, 3); // result_hi
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 4);
            }

            let threads_per_tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
            let threadgroup_count =
                (row_count as usize + threads_per_tg - 1) / threads_per_tg;

            let grid_size = objc2_metal::MTLSize {
                width: threadgroup_count,
                height: 1,
                depth: 1,
            };
            let tg_size = objc2_metal::MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            };

            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
            encoder.endEncoding();
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Read back lo + hi and reconstruct int64
        let sum = unsafe {
            let ptr = result_buffer.contents().as_ptr() as *const u32;
            let lo = *ptr as u64;
            let hi = *ptr.add(1) as u64;
            let val = lo | (hi << 32);
            val as i64
        };

        Ok(sum)
    }
}

/// Build an all-ones selection mask (no filtering).
fn build_all_ones_mask(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    row_count: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let num_words = (row_count + 31) / 32;
    let buf = encode::alloc_buffer(device, std::cmp::max(num_words * 4, 4));
    unsafe {
        let ptr = buf.contents().as_ptr() as *mut u32;
        for i in 0..num_words {
            if i == num_words - 1 {
                let valid_bits = row_count % 32;
                if valid_bits == 0 {
                    *ptr.add(i) = 0xFFFFFFFF;
                } else {
                    *ptr.add(i) = (1u32 << valid_bits) - 1;
                }
            } else {
                *ptr.add(i) = 0xFFFFFFFF;
            }
        }
    }
    buf
}

/// Infer a runtime schema from a CSV file by sampling the first data row.
///
/// For each column, tries to parse as i64, then f64, falling back to Varchar.
fn infer_schema_from_csv(
    path: &std::path::Path,
    csv_meta: &crate::io::csv::CsvMetadata,
) -> Result<RuntimeSchema, String> {
    use std::io::BufRead;

    let file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open '{}': {}", path.display(), e))?;
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header
    let _header = lines.next();

    // Read first data row for type inference
    let first_row = lines
        .next()
        .ok_or_else(|| "CSV has no data rows".to_string())?
        .map_err(|e| format!("Cannot read CSV data row: {}", e))?;

    let delimiter = csv_meta.delimiter as char;
    let fields: Vec<&str> = first_row.trim_end().split(delimiter).collect();

    let mut columns = Vec::new();
    for (i, name) in csv_meta.column_names.iter().enumerate() {
        let field = fields.get(i).unwrap_or(&"");
        let data_type = if field.parse::<i64>().is_ok() {
            DataType::Int64
        } else if field.parse::<f64>().is_ok() {
            DataType::Float64
        } else {
            DataType::Varchar
        };

        columns.push(ColumnDef {
            name: name.clone(),
            data_type,
            nullable: false,
        });
    }

    Ok(RuntimeSchema::new(columns))
}
