//! GPU query execution engine: connects SQL physical plan to Metal kernel dispatch.
//!
//! Walks the physical plan tree bottom-up, dispatching Metal compute kernels
//! in sequence within a single command buffer per stage. Uses `waitUntilCompleted`
//! for synchronous result readback (POC simplification).
//!
//! Pipeline: SQL parse -> PhysicalPlan -> GpuScan (CSV parse) -> GpuFilter
//! (column_filter) -> GpuAggregate (aggregate_count/aggregate_sum_int64) -> result.

use std::collections::HashMap;

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
use crate::storage::dictionary::Dictionary;
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
    /// GPU-resident match count buffer (avoids CPU readback between stages).
    match_count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Indirect dispatch args buffer for downstream kernels.
    /// Written by `prepare_query_dispatch` kernel.
    dispatch_args_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Cached CPU-side match count (populated lazily on first read).
    match_count_cached: std::cell::Cell<Option<u32>>,
    /// Total row count from input.
    row_count: u32,
}

impl FilterResult {
    /// Read match_count from GPU buffer (lazy, cached after first read).
    fn match_count(&self) -> u32 {
        if let Some(cached) = self.match_count_cached.get() {
            return cached;
        }
        let count = unsafe {
            let ptr = self.match_count_buffer.contents().as_ptr() as *const u32;
            *ptr
        };
        self.match_count_cached.set(Some(count));
        count
    }
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
                group_by,
                input,
            } => {
                // Walk down to get scan and optional filter results
                let (scan_result, filter_result) = self.resolve_input(input, catalog)?;
                if group_by.is_empty() {
                    self.execute_aggregate(&scan_result, filter_result.as_ref(), functions)
                } else {
                    self.execute_aggregate_grouped(
                        &scan_result,
                        filter_result.as_ref(),
                        functions,
                        group_by,
                    )
                }
            }

            PhysicalPlan::GpuFilter { .. } => {
                // A filter without aggregate on top: return filtered row count
                let (scan_result, filter_result) = self.resolve_input(plan, catalog)?;
                let count = filter_result
                    .as_ref()
                    .map(|f| f.match_count())
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

            PhysicalPlan::GpuSort { order_by, input } => {
                // CPU-side sort: get results from inner plan, sort rows
                self.execute_sort(order_by, input, catalog)
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

            PhysicalPlan::GpuSort { input, .. } => {
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

                // Run GPU CSV parse pipeline (handles int/float columns)
                let mut batch = self.gpu_parse_csv(&mmap, &schema, csv_meta.delimiter);

                // Build dictionaries for VARCHAR columns from raw CSV data (CPU-side)
                build_csv_dictionaries(
                    &entry.path,
                    csv_meta,
                    &schema,
                    &mut batch,
                )?;

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

        let mut batch = self.gpu_parse_json(&mmap, &schema);

        // Build dictionaries for VARCHAR columns from raw NDJSON data (CPU-side)
        build_json_dictionaries(path, &schema, &mut batch)?;

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

        // Determine the GPU column type and which buffer to use.
        // For Varchar, we create a temporary buffer with dict codes as i64.
        // The _temp_owned_buffer keeps it alive for the GPU dispatch.
        let (gpu_col_type, compare_int, compare_float);
        let _temp_owned_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>;
        let data_buffer: &ProtocolObject<dyn MTLBuffer>;

        match col_def.data_type {
            DataType::Int64 => {
                let int_val = match value {
                    Value::Int(v) => *v,
                    Value::Float(v) => *v as i64,
                    _ => return Err(format!("Cannot compare INT64 column to {:?}", value)),
                };
                gpu_col_type = ColumnTypeCode::Int64;
                compare_int = int_val;
                compare_float = 0.0f64;
                data_buffer = &scan.batch.int_buffer;
                _temp_owned_buffer = None;
            }
            DataType::Float64 => {
                let float_val = match value {
                    Value::Float(v) => *v as f32,
                    Value::Int(v) => *v as f32,
                    _ => return Err(format!("Cannot compare FLOAT64 column to {:?}", value)),
                };
                let float_bits = float_val.to_bits() as i32;
                gpu_col_type = ColumnTypeCode::Float64;
                compare_int = 0i64;
                compare_float = f64::from_bits(float_bits as u32 as u64);
                data_buffer = &scan.batch.float_buffer;
                _temp_owned_buffer = None;
            }
            DataType::Varchar => {
                // Dictionary-encoded string column: look up the comparison value's
                // dict code on CPU, then filter on dict codes using INT64 filter kernel.
                let local_idx = scan
                    .schema
                    .columns[..col_idx]
                    .iter()
                    .filter(|c| c.data_type == DataType::Varchar)
                    .count();

                let str_val = match value {
                    Value::Str(s) => s.as_str(),
                    _ => return Err(format!("Cannot compare VARCHAR column to {:?}", value)),
                };

                // Look up string value in the column's dictionary
                let dict = scan
                    .batch
                    .dictionaries
                    .get(col_idx)
                    .and_then(|d| d.as_ref())
                    .ok_or_else(|| {
                        format!("No dictionary for VARCHAR column '{}'", column)
                    })?;

                // Encode the comparison value. If not in dict, no rows can match.
                let dict_code = match dict.encode(str_val) {
                    Some(code) => code as i64,
                    None => {
                        // Value not in dictionary -> 0 matches. Return empty result.
                        let bitmask_buffer =
                            encode::alloc_buffer(&self.device.device, std::cmp::max(bitmask_words * 4, 4));
                        let match_count_buffer = encode::alloc_buffer(&self.device.device, 4);
                        let dispatch_args_buffer = encode::alloc_buffer(
                            &self.device.device,
                            std::mem::size_of::<crate::gpu::types::DispatchArgs>(),
                        );
                        unsafe {
                            let ptr = bitmask_buffer.contents().as_ptr() as *mut u32;
                            for i in 0..bitmask_words {
                                *ptr.add(i) = 0;
                            }
                            let cp = match_count_buffer.contents().as_ptr() as *mut u32;
                            *cp = 0;
                        }
                        return Ok(FilterResult {
                            bitmask_buffer,
                            match_count_buffer,
                            dispatch_args_buffer,
                            match_count_cached: std::cell::Cell::new(Some(0)),
                            row_count,
                        });
                    }
                };

                // Re-upload dict codes as i64 values to a temporary buffer
                // for the INT64 filter kernel.
                let dict_codes = unsafe { scan.batch.read_string_dict_column(local_idx) };
                let codes_as_i64: Vec<i64> = dict_codes.iter().map(|&c| c as i64).collect();
                let temp = encode::alloc_buffer_with_data(&self.device.device, &codes_as_i64);

                gpu_col_type = ColumnTypeCode::Int64;
                compare_int = dict_code;
                compare_float = 0.0f64;
                _temp_owned_buffer = Some(temp);
                // This is safe: _temp_owned_buffer keeps the Retained alive
                // until end of function, and the GPU dispatch completes synchronously
                // (waitUntilCompleted) before we return.
                data_buffer = _temp_owned_buffer.as_ref().unwrap();
            }
            _ => {
                return Err(format!(
                    "Filter on {:?} columns not supported in POC",
                    col_def.data_type
                ))
            }
        }

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
        // For Varchar, the temp buffer contains only this column's data at offset 0.
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
            DataType::Varchar => 0, // temp buffer has data at offset 0
            _ => 0,
        };

        // Allocate indirect dispatch args buffer for downstream kernels
        let dispatch_args_buffer = encode::alloc_buffer(
            &self.device.device,
            std::mem::size_of::<crate::gpu::types::DispatchArgs>(),
        );

        // Dispatch filter kernel + prepare_query_dispatch in one command buffer.
        // This eliminates the GPU→CPU→GPU round-trip for match_count readback.
        let cmd_buf = encode::make_command_buffer(&self.device.command_queue);
        {
            // --- Stage 1: Filter kernel ---
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
        {
            // --- Stage 2: Prepare dispatch args from match_count (GPU-autonomous) ---
            let prepare_pipeline =
                encode::make_pipeline(&self.device.library, "prepare_query_dispatch");

            let tpt = 256u32; // threads per threadgroup for downstream kernels
            let tpt_buffer =
                encode::alloc_buffer_with_data(&self.device.device, &[tpt]);

            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("compute encoder");

            encoder.setComputePipelineState(&prepare_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&match_count_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&dispatch_args_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&tpt_buffer), 0, 2);
            }

            // Single thread dispatch
            let grid = objc2_metal::MTLSize { width: 1, height: 1, depth: 1 };
            let tg = objc2_metal::MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // match_count is NOT read back here -- stays GPU-resident.
        // Use FilterResult.match_count() for lazy CPU readback when needed.
        Ok(FilterResult {
            bitmask_buffer,
            match_count_buffer,
            dispatch_args_buffer,
            match_count_cached: std::cell::Cell::new(None),
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
            let (col_label, result_str) =
                self.compute_aggregate(func, col_name, scan, mask_buffer, row_count)?;
            columns.push(col_label);
            values.push(result_str);
        }

        Ok(QueryResult {
            columns,
            rows: vec![values],
            row_count: 1,
        })
    }

    /// Compute a single aggregate function and return (label, formatted result).
    fn compute_aggregate(
        &self,
        func: &AggFunc,
        col_name: &str,
        scan: &ScanResult,
        mask_buffer: &ProtocolObject<dyn MTLBuffer>,
        row_count: u32,
    ) -> Result<(String, String), String> {
        match func {
            AggFunc::Count => {
                let label = if col_name == "*" {
                    "count(*)".to_string()
                } else {
                    format!("count({})", col_name)
                };
                let count = self.run_aggregate_count(mask_buffer, row_count)?;
                Ok((label, count.to_string()))
            }
            AggFunc::Sum => {
                let label = format!("sum({})", col_name);
                let (col_idx, col_def) = self.resolve_column(scan, col_name)?;
                match col_def.data_type {
                    DataType::Int64 => {
                        let local_idx = self.int_local_idx(scan, col_idx);
                        let sum = self.run_aggregate_sum_int64(
                            &scan.batch, mask_buffer, row_count, local_idx,
                        )?;
                        Ok((label, sum.to_string()))
                    }
                    DataType::Float64 => {
                        let local_idx = self.float_local_idx(scan, col_idx);
                        let sum = self.run_aggregate_sum_float(
                            &scan.batch, mask_buffer, row_count, local_idx,
                        )?;
                        // Format float with reasonable precision
                        Ok((label, format_float(sum as f64)))
                    }
                    _ => Err(format!("SUM on {:?} not supported", col_def.data_type)),
                }
            }
            AggFunc::Avg => {
                let label = format!("avg({})", col_name);
                let (col_idx, col_def) = self.resolve_column(scan, col_name)?;
                // AVG = SUM / COUNT
                let count = self.run_aggregate_count(mask_buffer, row_count)?;
                if count == 0 {
                    return Ok((label, "NULL".to_string()));
                }
                match col_def.data_type {
                    DataType::Int64 => {
                        let local_idx = self.int_local_idx(scan, col_idx);
                        let sum = self.run_aggregate_sum_int64(
                            &scan.batch, mask_buffer, row_count, local_idx,
                        )?;
                        let avg = sum as f64 / count as f64;
                        Ok((label, format_float(avg)))
                    }
                    DataType::Float64 => {
                        let local_idx = self.float_local_idx(scan, col_idx);
                        let sum = self.run_aggregate_sum_float(
                            &scan.batch, mask_buffer, row_count, local_idx,
                        )?;
                        let avg = sum as f64 / count as f64;
                        Ok((label, format_float(avg)))
                    }
                    _ => Err(format!("AVG on {:?} not supported", col_def.data_type)),
                }
            }
            AggFunc::Min => {
                let label = format!("min({})", col_name);
                let (col_idx, col_def) = self.resolve_column(scan, col_name)?;
                match col_def.data_type {
                    DataType::Int64 => {
                        let local_idx = self.int_local_idx(scan, col_idx);
                        let min = self.run_aggregate_min_int64(
                            &scan.batch, mask_buffer, row_count, local_idx,
                        )?;
                        Ok((label, min.to_string()))
                    }
                    DataType::Float64 => {
                        // CPU fallback for float MIN
                        let local_idx = self.float_local_idx(scan, col_idx);
                        let vals = self.read_masked_floats(
                            &scan.batch, mask_buffer, row_count, local_idx,
                        );
                        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                        Ok((label, format_float(min as f64)))
                    }
                    _ => Err(format!("MIN on {:?} not supported", col_def.data_type)),
                }
            }
            AggFunc::Max => {
                let label = format!("max({})", col_name);
                let (col_idx, col_def) = self.resolve_column(scan, col_name)?;
                match col_def.data_type {
                    DataType::Int64 => {
                        let local_idx = self.int_local_idx(scan, col_idx);
                        let max = self.run_aggregate_max_int64(
                            &scan.batch, mask_buffer, row_count, local_idx,
                        )?;
                        Ok((label, max.to_string()))
                    }
                    DataType::Float64 => {
                        // CPU fallback for float MAX
                        let local_idx = self.float_local_idx(scan, col_idx);
                        let vals = self.read_masked_floats(
                            &scan.batch, mask_buffer, row_count, local_idx,
                        );
                        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        Ok((label, format_float(max as f64)))
                    }
                    _ => Err(format!("MAX on {:?} not supported", col_def.data_type)),
                }
            }
        }
    }

    /// Resolve column name to (index, definition).
    fn resolve_column<'a>(
        &self,
        scan: &'a ScanResult,
        col_name: &str,
    ) -> Result<(usize, &'a ColumnDef), String> {
        let col_idx = scan
            .schema
            .column_index(col_name)
            .ok_or_else(|| format!("Column '{}' not found", col_name))?;
        Ok((col_idx, &scan.schema.columns[col_idx]))
    }

    /// Get the local index among INT64 columns for a given global column index.
    fn int_local_idx(&self, scan: &ScanResult, col_idx: usize) -> usize {
        scan.schema.columns[..col_idx]
            .iter()
            .filter(|c| c.data_type == DataType::Int64)
            .count()
    }

    /// Get the local index among FLOAT64 columns for a given global column index.
    fn float_local_idx(&self, scan: &ScanResult, col_idx: usize) -> usize {
        scan.schema.columns[..col_idx]
            .iter()
            .filter(|c| c.data_type == DataType::Float64)
            .count()
    }

    /// Read masked float values from GPU buffer (CPU readback for fallback aggregates).
    fn read_masked_floats(
        &self,
        batch: &ColumnarBatch,
        mask_buffer: &ProtocolObject<dyn MTLBuffer>,
        row_count: u32,
        float_col_local_idx: usize,
    ) -> Vec<f32> {
        let offset = float_col_local_idx * batch.max_rows;
        let mut result = Vec::new();
        unsafe {
            let data_ptr = batch.float_buffer.contents().as_ptr() as *const f32;
            let mask_ptr = mask_buffer.contents().as_ptr() as *const u32;
            for i in 0..row_count as usize {
                let word = *mask_ptr.add(i / 32);
                let bit = (word >> (i % 32)) & 1;
                if bit == 1 {
                    result.push(*data_ptr.add(offset + i));
                }
            }
        }
        result
    }

    /// Execute aggregate functions with GROUP BY (CPU-side grouping).
    ///
    /// Reads back column data from GPU, groups on CPU using HashMap,
    /// then computes aggregates per group.
    fn execute_aggregate_grouped(
        &self,
        scan: &ScanResult,
        filter: Option<&FilterResult>,
        functions: &[(AggFunc, String)],
        group_by: &[String],
    ) -> Result<QueryResult, String> {
        let row_count = scan.batch.row_count as u32;

        // Build selection mask
        let default_mask;
        let mask_buffer = if let Some(f) = filter {
            &f.bitmask_buffer
        } else {
            default_mask = build_all_ones_mask(&self.device.device, row_count as usize);
            &default_mask
        };

        // Read group-by column values from GPU buffers
        // For now, support grouping by INT64 or VARCHAR columns (via string keys)
        let group_keys = self.read_group_keys(scan, mask_buffer, row_count, group_by)?;

        // group_keys: Vec<(row_index, group_key_string)>
        // Build HashMap: group_key -> Vec<row_index>
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (row_idx, key) in &group_keys {
            groups.entry(key.clone()).or_default().push(*row_idx);
        }

        // Sort groups for deterministic output
        let mut sorted_keys: Vec<String> = groups.keys().cloned().collect();
        sorted_keys.sort();

        // Build result columns: group_by columns + aggregate columns
        let mut result_columns = Vec::new();
        for g in group_by {
            result_columns.push(g.clone());
        }
        for (func, col_name) in functions {
            let label = match func {
                AggFunc::Count => {
                    if col_name == "*" {
                        "count(*)".to_string()
                    } else {
                        format!("count({})", col_name)
                    }
                }
                AggFunc::Sum => format!("sum({})", col_name),
                AggFunc::Avg => format!("avg({})", col_name),
                AggFunc::Min => format!("min({})", col_name),
                AggFunc::Max => format!("max({})", col_name),
            };
            result_columns.push(label);
        }

        // Read aggregate column data once (CPU-side for grouped aggregation)
        let mut agg_data: Vec<AggColumnData> = Vec::new();
        for (func, col_name) in functions {
            if *func == AggFunc::Count && col_name == "*" {
                agg_data.push(AggColumnData::CountStar);
                continue;
            }
            let (col_idx, col_def) = self.resolve_column(scan, col_name)?;
            match col_def.data_type {
                DataType::Int64 => {
                    let local_idx = self.int_local_idx(scan, col_idx);
                    let all_vals = unsafe { scan.batch.read_int_column(local_idx) };
                    agg_data.push(AggColumnData::Int64(all_vals));
                }
                DataType::Float64 => {
                    let local_idx = self.float_local_idx(scan, col_idx);
                    let all_vals = unsafe { scan.batch.read_float_column(local_idx) };
                    agg_data.push(AggColumnData::Float64(all_vals));
                }
                _ => return Err(format!("Aggregate on {:?} not supported", col_def.data_type)),
            }
        }

        // Compute aggregates per group
        let mut rows = Vec::new();
        for key in &sorted_keys {
            let row_indices = &groups[key];
            let mut row = Vec::new();

            // Add group-by values (from the key)
            // For multi-column GROUP BY, the key is "val1|val2|..." - split back
            let key_parts: Vec<&str> = key.split('|').collect();
            for part in &key_parts {
                row.push(part.to_string());
            }

            // Add aggregate values
            for (i, (func, _)) in functions.iter().enumerate() {
                let val = compute_cpu_aggregate(func, &agg_data[i], row_indices);
                row.push(val);
            }

            rows.push(row);
        }

        Ok(QueryResult {
            columns: result_columns,
            rows: rows.clone(),
            row_count: rows.len(),
        })
    }

    /// Execute ORDER BY via CPU-side sorting.
    ///
    /// Materializes all selected rows from the inner plan (scan + optional filter),
    /// then sorts them by the specified columns using Rust's sort_by.
    fn execute_sort(
        &mut self,
        order_by: &[(String, bool)],
        input: &PhysicalPlan,
        catalog: &[TableEntry],
    ) -> Result<QueryResult, String> {
        let (scan_result, filter_result) = self.resolve_input(input, catalog)?;
        let row_count = scan_result.batch.row_count as u32;

        // Build selection mask
        let default_mask;
        let mask_buffer = if let Some(f) = filter_result.as_ref() {
            &f.bitmask_buffer
        } else {
            default_mask = build_all_ones_mask(&self.device.device, row_count as usize);
            &default_mask
        };

        // Read all columns and build row-oriented data for sorting
        let schema = &scan_result.schema;
        let mut col_names: Vec<String> = Vec::new();
        let mut col_data: Vec<ColumnValues> = Vec::new();

        for col_def in &schema.columns {
            col_names.push(col_def.name.clone());
            let col_idx = schema.column_index(&col_def.name).unwrap();
            match col_def.data_type {
                DataType::Int64 => {
                    let local_idx = self.int_local_idx(&scan_result, col_idx);
                    let vals = unsafe { scan_result.batch.read_int_column(local_idx) };
                    col_data.push(ColumnValues::Int64(vals));
                }
                DataType::Float64 => {
                    let local_idx = self.float_local_idx(&scan_result, col_idx);
                    let vals = unsafe { scan_result.batch.read_float_column(local_idx) };
                    col_data.push(ColumnValues::Float64(vals));
                }
                DataType::Varchar => {
                    // Read dict-encoded strings from the string_dict_buffer
                    let local_idx = scan_result.schema.columns[..col_idx]
                        .iter()
                        .filter(|c| c.data_type == DataType::Varchar)
                        .count();
                    let dict_codes = unsafe {
                        scan_result.batch.read_string_dict_column(local_idx)
                    };
                    let dict = scan_result.batch.dictionaries.get(col_idx)
                        .and_then(|d| d.as_ref());
                    let vals: Vec<String> = dict_codes
                        .iter()
                        .map(|&code| {
                            if let Some(d) = dict {
                                if (code as usize) < d.len() {
                                    d.decode(code).to_string()
                                } else {
                                    "NULL".to_string()
                                }
                            } else {
                                "".to_string()
                            }
                        })
                        .collect();
                    col_data.push(ColumnValues::Varchar(vals));
                }
                _ => {
                    // Other unsupported types: placeholder empty strings
                    let vals = vec!["".to_string(); row_count as usize];
                    col_data.push(ColumnValues::Varchar(vals));
                }
            }
        }

        // Collect selected row indices
        let mut selected_rows: Vec<usize> = Vec::new();
        unsafe {
            let mask_ptr = mask_buffer.contents().as_ptr() as *const u32;
            for i in 0..row_count as usize {
                let word = *mask_ptr.add(i / 32);
                let bit = (word >> (i % 32)) & 1;
                if bit == 1 {
                    selected_rows.push(i);
                }
            }
        }

        // Resolve sort column indices
        let sort_specs: Vec<(usize, bool)> = order_by
            .iter()
            .map(|(col_name, ascending)| {
                let idx = schema
                    .column_index(col_name)
                    .ok_or_else(|| format!("Sort column '{}' not found", col_name))?;
                Ok((idx, *ascending))
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Sort selected row indices by the sort columns
        selected_rows.sort_by(|&a, &b| {
            for &(col_idx, ascending) in &sort_specs {
                let cmp = match &col_data[col_idx] {
                    ColumnValues::Int64(vals) => vals[a].cmp(&vals[b]),
                    ColumnValues::Float64(vals) => {
                        vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    ColumnValues::Varchar(vals) => vals[a].cmp(&vals[b]),
                };
                let cmp = if ascending { cmp } else { cmp.reverse() };
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });

        // Build result rows from sorted indices
        let mut rows = Vec::new();
        for &row_idx in &selected_rows {
            let mut row = Vec::new();
            for col_val in &col_data {
                match col_val {
                    ColumnValues::Int64(vals) => row.push(vals[row_idx].to_string()),
                    ColumnValues::Float64(vals) => row.push(format_float(vals[row_idx] as f64)),
                    ColumnValues::Varchar(vals) => row.push(vals[row_idx].clone()),
                }
            }
            rows.push(row);
        }

        let result_count = rows.len();
        Ok(QueryResult {
            columns: col_names,
            rows,
            row_count: result_count,
        })
    }

    /// Read group-by column keys as strings for CPU-side grouping.
    /// Returns (row_index, group_key) for each selected row.
    fn read_group_keys(
        &self,
        scan: &ScanResult,
        mask_buffer: &ProtocolObject<dyn MTLBuffer>,
        row_count: u32,
        group_by: &[String],
    ) -> Result<Vec<(usize, String)>, String> {
        // Pre-read all group columns
        let mut col_readers: Vec<GroupColumnReader> = Vec::new();
        for g_col_name in group_by {
            let (col_idx, col_def) = self.resolve_column(scan, g_col_name)?;
            match col_def.data_type {
                DataType::Int64 => {
                    let local_idx = self.int_local_idx(scan, col_idx);
                    let vals = unsafe { scan.batch.read_int_column(local_idx) };
                    col_readers.push(GroupColumnReader::Int64(vals));
                }
                DataType::Float64 => {
                    let local_idx = self.float_local_idx(scan, col_idx);
                    let vals = unsafe { scan.batch.read_float_column(local_idx) };
                    col_readers.push(GroupColumnReader::Float64(vals));
                }
                DataType::Varchar => {
                    // Read dict-encoded VARCHAR column and decode to strings
                    let local_idx = scan.schema.columns[..col_idx]
                        .iter()
                        .filter(|c| c.data_type == DataType::Varchar)
                        .count();
                    let dict_codes = unsafe { scan.batch.read_string_dict_column(local_idx) };
                    let dict = scan.batch.dictionaries.get(col_idx)
                        .and_then(|d| d.as_ref());
                    let vals: Vec<String> = dict_codes
                        .iter()
                        .map(|&code| {
                            if let Some(d) = dict {
                                if (code as usize) < d.len() {
                                    d.decode(code).to_string()
                                } else {
                                    "NULL".to_string()
                                }
                            } else {
                                "".to_string()
                            }
                        })
                        .collect();
                    col_readers.push(GroupColumnReader::Varchar(vals));
                }
                _ => {
                    return Err(format!(
                        "GROUP BY on {:?} not supported",
                        col_def.data_type
                    ));
                }
            }
        }

        let mut result = Vec::new();
        unsafe {
            let mask_ptr = mask_buffer.contents().as_ptr() as *const u32;
            for i in 0..row_count as usize {
                let word = *mask_ptr.add(i / 32);
                let bit = (word >> (i % 32)) & 1;
                if bit == 1 {
                    let mut key_parts = Vec::new();
                    for reader in &col_readers {
                        match reader {
                            GroupColumnReader::Int64(vals) => {
                                key_parts.push(vals[i].to_string());
                            }
                            GroupColumnReader::Float64(vals) => {
                                key_parts.push(format!("{}", vals[i]));
                            }
                            GroupColumnReader::Varchar(vals) => {
                                key_parts.push(vals[i].clone());
                            }
                        }
                    }
                    result.push((i, key_parts.join("|")));
                }
            }
        }

        Ok(result)
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

    /// Run the aggregate_min_int64 kernel.
    /// GPU does SIMD + threadgroup reduction, writes per-threadgroup partials.
    /// Host reads partials and does final MIN on CPU.
    fn run_aggregate_min_int64(
        &self,
        batch: &ColumnarBatch,
        mask_buffer: &ProtocolObject<dyn MTLBuffer>,
        row_count: u32,
        int_col_local_idx: usize,
    ) -> Result<i64, String> {
        let pipeline =
            encode::make_pipeline(&self.device.library, "aggregate_min_int64");

        let threads_per_tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
        let threadgroup_count =
            (row_count as usize + threads_per_tg - 1) / threads_per_tg;

        // Allocate partials buffer: one i64 per threadgroup, initialized to INT64_MAX
        let partials_buffer = encode::alloc_buffer(
            &self.device.device,
            threadgroup_count * std::mem::size_of::<i64>(),
        );
        unsafe {
            let ptr = partials_buffer.contents().as_ptr() as *mut i64;
            for i in 0..threadgroup_count {
                *ptr.add(i) = i64::MAX;
            }
        }

        let params = AggParams {
            row_count,
            group_count: 0,
            agg_function: 3, // MIN
            _pad0: 0,
        };
        let params_buffer =
            encode::alloc_buffer_with_data(&self.device.device, &[params]);

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
                    Some(&batch.int_buffer), data_offset, 0,
                );
                encoder.setBuffer_offset_atIndex(Some(mask_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&partials_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
            }

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

        // CPU-side final reduction over threadgroup partials
        let result = unsafe {
            let ptr = partials_buffer.contents().as_ptr() as *const i64;
            let mut min_val = i64::MAX;
            for i in 0..threadgroup_count {
                let v = *ptr.add(i);
                if v < min_val {
                    min_val = v;
                }
            }
            min_val
        };

        Ok(result)
    }

    /// Run the aggregate_max_int64 kernel.
    /// GPU does SIMD + threadgroup reduction, writes per-threadgroup partials.
    /// Host reads partials and does final MAX on CPU.
    fn run_aggregate_max_int64(
        &self,
        batch: &ColumnarBatch,
        mask_buffer: &ProtocolObject<dyn MTLBuffer>,
        row_count: u32,
        int_col_local_idx: usize,
    ) -> Result<i64, String> {
        let pipeline =
            encode::make_pipeline(&self.device.library, "aggregate_max_int64");

        let threads_per_tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
        let threadgroup_count =
            (row_count as usize + threads_per_tg - 1) / threads_per_tg;

        // Allocate partials buffer: one i64 per threadgroup, initialized to INT64_MIN
        let partials_buffer = encode::alloc_buffer(
            &self.device.device,
            threadgroup_count * std::mem::size_of::<i64>(),
        );
        unsafe {
            let ptr = partials_buffer.contents().as_ptr() as *mut i64;
            for i in 0..threadgroup_count {
                *ptr.add(i) = i64::MIN;
            }
        }

        let params = AggParams {
            row_count,
            group_count: 0,
            agg_function: 4, // MAX
            _pad0: 0,
        };
        let params_buffer =
            encode::alloc_buffer_with_data(&self.device.device, &[params]);

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
                    Some(&batch.int_buffer), data_offset, 0,
                );
                encoder.setBuffer_offset_atIndex(Some(mask_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&partials_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
            }

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

        // CPU-side final reduction over threadgroup partials
        let result = unsafe {
            let ptr = partials_buffer.contents().as_ptr() as *const i64;
            let mut max_val = i64::MIN;
            for i in 0..threadgroup_count {
                let v = *ptr.add(i);
                if v > max_val {
                    max_val = v;
                }
            }
            max_val
        };

        Ok(result)
    }

    /// Run the aggregate_sum_float kernel for FLOAT columns.
    fn run_aggregate_sum_float(
        &self,
        batch: &ColumnarBatch,
        mask_buffer: &ProtocolObject<dyn MTLBuffer>,
        row_count: u32,
        float_col_local_idx: usize,
    ) -> Result<f32, String> {
        // Result: single uint32 storing float bits, initialized to 0.0f
        let result_buffer = encode::alloc_buffer(&self.device.device, 4);
        unsafe {
            let ptr = result_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0; // 0.0f as bits = 0
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
            encode::make_pipeline(&self.device.library, "aggregate_sum_float");

        let data_offset =
            float_col_local_idx * batch.max_rows * std::mem::size_of::<f32>();

        let cmd_buf = encode::make_command_buffer(&self.device.command_queue);
        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("compute encoder");

            encoder.setComputePipelineState(&pipeline);

            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(&batch.float_buffer),
                    data_offset,
                    0,
                );
                encoder.setBuffer_offset_atIndex(Some(mask_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&result_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
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

        let sum = unsafe {
            let ptr = result_buffer.contents().as_ptr() as *const u32;
            f32::from_bits(*ptr)
        };

        Ok(sum)
    }
}

/// Column values for row materialization (ORDER BY).
enum ColumnValues {
    Int64(Vec<i64>),
    Float64(Vec<f32>),
    Varchar(Vec<String>),
}

/// Column data reader for GROUP BY key extraction.
enum GroupColumnReader {
    Int64(Vec<i64>),
    Float64(Vec<f32>),
    Varchar(Vec<String>),
}

/// Column data for CPU-side grouped aggregation.
enum AggColumnData {
    CountStar,
    Int64(Vec<i64>),
    Float64(Vec<f32>),
}

/// Compute a CPU-side aggregate over the given row indices.
fn compute_cpu_aggregate(
    func: &AggFunc,
    data: &AggColumnData,
    row_indices: &[usize],
) -> String {
    if row_indices.is_empty() {
        return "NULL".to_string();
    }

    match (func, data) {
        (AggFunc::Count, AggColumnData::CountStar) => row_indices.len().to_string(),
        (AggFunc::Count, _) => row_indices.len().to_string(),

        (AggFunc::Sum, AggColumnData::Int64(vals)) => {
            let sum: i64 = row_indices.iter().map(|&i| vals[i]).sum();
            sum.to_string()
        }
        (AggFunc::Sum, AggColumnData::Float64(vals)) => {
            let sum: f64 = row_indices.iter().map(|&i| vals[i] as f64).sum();
            format_float(sum)
        }

        (AggFunc::Avg, AggColumnData::Int64(vals)) => {
            let sum: f64 = row_indices.iter().map(|&i| vals[i] as f64).sum();
            let avg = sum / row_indices.len() as f64;
            format_float(avg)
        }
        (AggFunc::Avg, AggColumnData::Float64(vals)) => {
            let sum: f64 = row_indices.iter().map(|&i| vals[i] as f64).sum();
            let avg = sum / row_indices.len() as f64;
            format_float(avg)
        }

        (AggFunc::Min, AggColumnData::Int64(vals)) => {
            let min = row_indices.iter().map(|&i| vals[i]).min().unwrap();
            min.to_string()
        }
        (AggFunc::Min, AggColumnData::Float64(vals)) => {
            let min = row_indices
                .iter()
                .map(|&i| vals[i])
                .fold(f32::INFINITY, f32::min);
            format_float(min as f64)
        }

        (AggFunc::Max, AggColumnData::Int64(vals)) => {
            let max = row_indices.iter().map(|&i| vals[i]).max().unwrap();
            max.to_string()
        }
        (AggFunc::Max, AggColumnData::Float64(vals)) => {
            let max = row_indices
                .iter()
                .map(|&i| vals[i])
                .fold(f32::NEG_INFINITY, f32::max);
            format_float(max as f64)
        }

        _ => "NULL".to_string(),
    }
}

/// Format a float value for display: remove trailing zeros after decimal.
fn format_float(val: f64) -> String {
    if val == val.floor() && val.abs() < 1e15 {
        // Integer-valued float: show as integer
        format!("{}", val as i64)
    } else {
        // Show with reasonable precision
        let s = format!("{:.6}", val);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
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

/// Maximum number of sample rows to examine for schema inference.
const SCHEMA_INFER_SAMPLE_ROWS: usize = 100;

/// Infer a runtime schema from a CSV file by sampling up to 100 data rows.
///
/// For each column, uses type voting across all sampled rows:
/// - If ALL non-empty values parse as i64 -> Int64
/// - If ALL non-empty values parse as f64 (but not all as i64) -> Float64
/// - Otherwise -> Varchar
/// - If any field is empty, the column is marked nullable
///
/// This multi-row sampling is more robust than single-row inference,
/// catching cases where the first row might have atypical values.
pub fn infer_schema_from_csv(
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

    let num_columns = csv_meta.column_names.len();
    let delimiter = csv_meta.delimiter as char;

    // Per-column vote counters
    let mut int_votes = vec![0usize; num_columns];
    let mut float_votes = vec![0usize; num_columns];
    let mut varchar_votes = vec![0usize; num_columns];
    let mut null_votes = vec![0usize; num_columns];
    let mut rows_sampled = 0usize;

    // Sample up to SCHEMA_INFER_SAMPLE_ROWS data rows
    for line_result in lines.take(SCHEMA_INFER_SAMPLE_ROWS) {
        let line = line_result.map_err(|e| format!("Cannot read CSV data row: {}", e))?;
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }

        let fields: Vec<&str> = trimmed.split(delimiter).collect();
        rows_sampled += 1;

        for (i, _name) in csv_meta.column_names.iter().enumerate() {
            let field = fields.get(i).map(|s| s.trim()).unwrap_or("");

            if field.is_empty() {
                null_votes[i] += 1;
            } else if field.parse::<i64>().is_ok() {
                int_votes[i] += 1;
            } else if field.parse::<f64>().is_ok() {
                float_votes[i] += 1;
            } else {
                varchar_votes[i] += 1;
            }
        }
    }

    if rows_sampled == 0 {
        return Err("CSV has no data rows".to_string());
    }

    // Determine final type for each column via voting
    let mut columns = Vec::new();
    for (i, name) in csv_meta.column_names.iter().enumerate() {
        let nullable = null_votes[i] > 0;

        // Type precedence: if ANY row has a varchar value -> Varchar
        // If ANY row has a float (but not varchar) -> Float64
        // If ALL non-empty rows are int -> Int64
        let data_type = if varchar_votes[i] > 0 {
            DataType::Varchar
        } else if float_votes[i] > 0 {
            DataType::Float64
        } else if int_votes[i] > 0 {
            DataType::Int64
        } else {
            // All values were empty/null -- default to Varchar
            DataType::Varchar
        };

        columns.push(ColumnDef {
            name: name.clone(),
            data_type,
            nullable,
        });
    }

    Ok(RuntimeSchema::new(columns))
}

/// Build dictionaries for VARCHAR columns in a CSV file (CPU-side).
///
/// After the GPU has parsed int/float columns, this function reads the raw CSV
/// on CPU to extract string column values, builds dictionaries for each VARCHAR
/// column, encodes values as u32 dict codes, and writes them to the batch's
/// string_dict_buffer.
fn build_csv_dictionaries(
    path: &std::path::Path,
    csv_meta: &crate::io::csv::CsvMetadata,
    schema: &RuntimeSchema,
    batch: &mut ColumnarBatch,
) -> Result<(), String> {
    use std::io::BufRead;

    // Check if there are any VARCHAR columns
    let varchar_count = schema.count_type(DataType::Varchar);
    if varchar_count == 0 {
        return Ok(());
    }

    // Read the entire CSV to extract string column values
    let file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open '{}': {}", path.display(), e))?;
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines();
    let delimiter = csv_meta.delimiter as char;

    // Skip header
    let _header = lines.next();

    // Identify which columns are VARCHAR and their local indices
    let varchar_cols: Vec<(usize, usize)> = schema
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.data_type == DataType::Varchar)
        .enumerate()
        .map(|(local_idx, (global_idx, _))| (global_idx, local_idx))
        .collect();

    // Collect string values for each VARCHAR column
    let mut col_values: Vec<Vec<String>> = vec![Vec::new(); varchar_count];

    for line_result in lines {
        let line = line_result.map_err(|e| format!("Cannot read CSV row: {}", e))?;
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }

        let fields: Vec<&str> = trimmed.split(delimiter).collect();

        for &(global_idx, local_idx) in &varchar_cols {
            let field = fields.get(global_idx).map(|s| s.trim()).unwrap_or("");
            col_values[local_idx].push(field.to_string());
        }
    }

    // Build dictionaries and encode values for each VARCHAR column
    for &(global_idx, local_idx) in &varchar_cols {
        let values = &col_values[local_idx];

        if let Some(dict) = Dictionary::build(values) {
            let encoded = dict.encode_column(values);

            // Write encoded dict codes to the string_dict_buffer at the correct offset
            let offset = local_idx * batch.max_rows;
            unsafe {
                let ptr = batch.string_dict_buffer.contents().as_ptr() as *mut u32;
                for (i, &code) in encoded.iter().enumerate() {
                    *ptr.add(offset + i) = code;
                }
            }

            batch.dictionaries[global_idx] = Some(dict);
        }
    }

    Ok(())
}

/// Build dictionaries for VARCHAR columns in an NDJSON file (CPU-side).
///
/// Similar to build_csv_dictionaries but parses NDJSON records.
fn build_json_dictionaries(
    path: &std::path::Path,
    schema: &RuntimeSchema,
    batch: &mut ColumnarBatch,
) -> Result<(), String> {
    use std::io::BufRead;

    let varchar_count = schema.count_type(DataType::Varchar);
    if varchar_count == 0 {
        return Ok(());
    }

    let file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open '{}': {}", path.display(), e))?;
    let reader = std::io::BufReader::new(file);

    // Identify VARCHAR columns
    let varchar_cols: Vec<(usize, usize)> = schema
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.data_type == DataType::Varchar)
        .enumerate()
        .map(|(local_idx, (global_idx, _))| (global_idx, local_idx))
        .collect();

    let mut col_values: Vec<Vec<String>> = vec![Vec::new(); varchar_count];

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| format!("Cannot read JSON line: {}", e))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Simple JSON field extraction: find "field_name":"value" or "field_name":value
        for &(global_idx, local_idx) in &varchar_cols {
            let col_name = &schema.columns[global_idx].name;
            let pattern = format!("\"{}\":", col_name);
            if let Some(pos) = trimmed.find(&pattern) {
                let after = &trimmed[pos + pattern.len()..];
                let value = extract_json_string_value(after);
                col_values[local_idx].push(value);
            } else {
                col_values[local_idx].push(String::new());
            }
        }
    }

    // Build dictionaries and encode
    for &(global_idx, local_idx) in &varchar_cols {
        let values = &col_values[local_idx];

        if let Some(dict) = Dictionary::build(values) {
            let encoded = dict.encode_column(values);

            let offset = local_idx * batch.max_rows;
            unsafe {
                let ptr = batch.string_dict_buffer.contents().as_ptr() as *mut u32;
                for (i, &code) in encoded.iter().enumerate() {
                    *ptr.add(offset + i) = code;
                }
            }

            batch.dictionaries[global_idx] = Some(dict);
        }
    }

    Ok(())
}

/// Extract a JSON string value from a position right after `"key":`.
/// Handles both quoted strings and unquoted values.
fn extract_json_string_value(s: &str) -> String {
    let s = s.trim_start();
    if s.starts_with('"') {
        // Quoted string
        let inner = &s[1..];
        if let Some(end) = inner.find('"') {
            inner[..end].to_string()
        } else {
            inner.to_string()
        }
    } else {
        // Unquoted value (number, null, bool)
        let end = s.find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
            .unwrap_or(s.len());
        s[..end].to_string()
    }
}
