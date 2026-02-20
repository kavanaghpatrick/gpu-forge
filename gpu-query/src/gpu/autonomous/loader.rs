//! Binary columnar data loader for GPU-resident tables.
//!
//! Converts a `ColumnarBatch` (per-type separate buffers) into a single contiguous
//! Metal buffer with `ColumnMeta` array for the autonomous query engine. This is the
//! bridge between the existing CSV/JSON parse pipeline and the GPU-resident format.
//!
//! Buffer layout:
//! - Single contiguous Metal buffer for ALL column data
//! - Each column starts at a 16-byte aligned offset
//! - ColumnMeta.offset stores byte offset within data_buffer
//! - ColumnMeta.stride stores bytes per element
//!
//! Column type mapping:
//! - INT64  -> i64[] (8 bytes per element, stride=8, column_type=0)
//! - FLOAT64 -> f32[] downcast (4 bytes per element, stride=4, column_type=1)
//! - VARCHAR -> u32[] dictionary codes (4 bytes per element, stride=4, column_type=2)

use std::collections::HashMap;
use std::sync::mpsc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::types::ColumnMeta;
use crate::storage::columnar::ColumnarBatch;
use crate::storage::schema::{DataType, RuntimeSchema};

/// Column type constants matching MSL autonomous_types.h.
const COLUMN_TYPE_INT64: u32 = 0;
const COLUMN_TYPE_FLOAT32: u32 = 1;
const COLUMN_TYPE_DICT_U32: u32 = 2;

/// Page size for Metal buffer alignment (16 KB).
const PAGE_SIZE: usize = 16384;

/// Alignment for individual columns within the data buffer.
const COLUMN_ALIGNMENT: usize = 16;

/// Schema information for a single column in a ResidentTable.
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: DataType,
}

/// A table loaded into GPU-resident Metal buffers for autonomous query execution.
///
/// Contains a single contiguous data buffer with all column data, a ColumnMeta
/// array describing each column's location and type, and dictionaries for
/// VARCHAR columns.
pub struct ResidentTable {
    /// Single contiguous Metal buffer containing all column data.
    pub data_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Per-column metadata (byte offset, type, stride, row_count).
    pub column_metas: Vec<ColumnMeta>,
    /// Metal buffer containing the ColumnMeta array for GPU binding.
    pub column_meta_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Number of rows in the table.
    pub row_count: u32,
    /// Column names and types.
    pub schema: Vec<ColumnInfo>,
    /// Dictionary values for VARCHAR columns (col_idx -> sorted dict values).
    pub dictionaries: HashMap<usize, Vec<String>>,
}

/// Progress update sent during table loading.
#[derive(Debug, Clone)]
pub enum LoadProgress {
    /// Starting to load a column (col_idx, col_name).
    LoadingColumn(usize, String),
    /// Column loaded successfully (col_idx, bytes_written).
    ColumnLoaded(usize, usize),
    /// All columns loaded, table ready.
    Complete(u32),
}

/// Binary columnar data loader: converts ColumnarBatch to GPU-resident format.
pub struct BinaryColumnarLoader;

impl BinaryColumnarLoader {
    /// Load a ColumnarBatch into a GPU-resident table.
    ///
    /// Converts per-type separate buffers into a single contiguous Metal buffer
    /// with ColumnMeta array. Reports progress via the provided channel.
    ///
    /// # Column type mapping
    /// - INT64  -> i64[] (8 bytes/element)
    /// - FLOAT64 -> f32[] downcast (4 bytes/element)
    /// - VARCHAR -> u32[] dictionary codes (4 bytes/element)
    ///
    /// # Errors
    /// Returns Err if buffer allocation fails or data is inconsistent.
    pub fn load_table(
        device: &ProtocolObject<dyn MTLDevice>,
        _table_name: &str,
        schema: &RuntimeSchema,
        batch: &ColumnarBatch,
        progress_tx: Option<&mpsc::Sender<LoadProgress>>,
    ) -> Result<ResidentTable, String> {
        let row_count = batch.row_count;
        let num_cols = schema.columns.len();

        // First pass: compute total buffer size with 16-byte alignment per column
        let mut total_size: usize = 0;
        let mut offsets: Vec<u64> = Vec::with_capacity(num_cols);
        let mut strides: Vec<u32> = Vec::with_capacity(num_cols);
        let mut col_types: Vec<u32> = Vec::with_capacity(num_cols);

        for col in &schema.columns {
            // Align current offset to COLUMN_ALIGNMENT
            let aligned_offset = align_up(total_size, COLUMN_ALIGNMENT);
            offsets.push(aligned_offset as u64);

            let (stride, col_type) = match col.data_type {
                DataType::Int64 | DataType::Date => {
                    (std::mem::size_of::<i64>() as u32, COLUMN_TYPE_INT64)
                }
                DataType::Float64 => {
                    // Downcast to f32 for GPU
                    (std::mem::size_of::<f32>() as u32, COLUMN_TYPE_FLOAT32)
                }
                DataType::Varchar => {
                    // Dictionary codes as u32
                    (std::mem::size_of::<u32>() as u32, COLUMN_TYPE_DICT_U32)
                }
                DataType::Bool => {
                    // Store as u32 on GPU (4 bytes)
                    (std::mem::size_of::<u32>() as u32, COLUMN_TYPE_INT64)
                }
            };

            strides.push(stride);
            col_types.push(col_type);

            let col_size = row_count * stride as usize;
            total_size = aligned_offset + col_size;
        }

        // Round up to page alignment (16KB) or minimum 1 byte to avoid zero-size buffer
        let buffer_size = if total_size == 0 {
            PAGE_SIZE
        } else {
            align_up(total_size, PAGE_SIZE)
        };

        // Allocate single contiguous Metal buffer
        let options = MTLResourceOptions::StorageModeShared;
        let data_buffer = device
            .newBufferWithLength_options(buffer_size, options)
            .ok_or_else(|| "Failed to allocate data Metal buffer".to_string())?;

        // Zero-initialize the buffer
        unsafe {
            let ptr = data_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, buffer_size);
        }

        // Second pass: copy data into contiguous buffer
        let mut int_local_idx = 0usize;
        let mut float_local_idx = 0usize;
        let mut string_local_idx = 0usize;

        let mut column_metas: Vec<ColumnMeta> = Vec::with_capacity(num_cols);
        let mut schema_info: Vec<ColumnInfo> = Vec::with_capacity(num_cols);
        let mut dictionaries: HashMap<usize, Vec<String>> = HashMap::new();

        for (col_idx, col) in schema.columns.iter().enumerate() {
            let offset = offsets[col_idx];
            let stride = strides[col_idx];
            let col_type = col_types[col_idx];

            // Send progress
            if let Some(tx) = progress_tx {
                let _ = tx.send(LoadProgress::LoadingColumn(col_idx, col.name.clone()));
            }

            let bytes_written = match col.data_type {
                DataType::Int64 | DataType::Date => {
                    let bytes = unsafe {
                        Self::copy_int_column(
                            &data_buffer,
                            batch,
                            int_local_idx,
                            offset as usize,
                            row_count,
                        )
                    };
                    int_local_idx += 1;
                    bytes
                }
                DataType::Float64 => {
                    let bytes = unsafe {
                        Self::copy_float_column(
                            &data_buffer,
                            batch,
                            float_local_idx,
                            offset as usize,
                            row_count,
                        )
                    };
                    float_local_idx += 1;
                    bytes
                }
                DataType::Varchar => {
                    let (bytes, dict_values) = unsafe {
                        Self::copy_varchar_column(
                            &data_buffer,
                            batch,
                            string_local_idx,
                            offset as usize,
                            row_count,
                            col_idx,
                        )
                    };
                    if let Some(values) = dict_values {
                        dictionaries.insert(col_idx, values);
                    }
                    string_local_idx += 1;
                    bytes
                }
                DataType::Bool => {
                    // Not yet used in autonomous path, zero-fill
                    0
                }
            };

            // Build ColumnMeta
            column_metas.push(ColumnMeta {
                offset,
                column_type: col_type,
                stride,
                null_offset: 0, // No null bitmaps yet
                row_count: row_count as u32,
                _pad: 0,
            });

            schema_info.push(ColumnInfo {
                name: col.name.clone(),
                data_type: col.data_type,
            });

            // Send progress
            if let Some(tx) = progress_tx {
                let _ = tx.send(LoadProgress::ColumnLoaded(col_idx, bytes_written));
            }
        }

        // Allocate Metal buffer for ColumnMeta array
        let meta_size = std::cmp::max(
            num_cols * std::mem::size_of::<ColumnMeta>(),
            std::mem::size_of::<ColumnMeta>(), // At least 1 ColumnMeta worth
        );
        let column_meta_buffer = device
            .newBufferWithLength_options(meta_size, options)
            .ok_or_else(|| "Failed to allocate column meta Metal buffer".to_string())?;

        // Copy ColumnMeta array into Metal buffer
        if !column_metas.is_empty() {
            unsafe {
                let dst = column_meta_buffer.contents().as_ptr() as *mut ColumnMeta;
                std::ptr::copy_nonoverlapping(column_metas.as_ptr(), dst, column_metas.len());
            }
        }

        // Send completion
        if let Some(tx) = progress_tx {
            let _ = tx.send(LoadProgress::Complete(row_count as u32));
        }

        Ok(ResidentTable {
            data_buffer,
            column_metas,
            column_meta_buffer,
            row_count: row_count as u32,
            schema: schema_info,
            dictionaries,
        })
    }

    /// Copy INT64 column data from ColumnarBatch into the contiguous buffer.
    ///
    /// # Safety
    /// Caller must ensure offset + row_count * 8 <= buffer size.
    unsafe fn copy_int_column(
        data_buffer: &ProtocolObject<dyn MTLBuffer>,
        batch: &ColumnarBatch,
        local_col_idx: usize,
        offset: usize,
        row_count: usize,
    ) -> usize {
        if row_count == 0 {
            return 0;
        }
        let src_ptr = batch.int_buffer.contents().as_ptr() as *const i64;
        let src_offset = local_col_idx * batch.max_rows;
        let src = src_ptr.add(src_offset);

        let dst_base = data_buffer.contents().as_ptr() as *mut u8;
        let dst = dst_base.add(offset) as *mut i64;

        std::ptr::copy_nonoverlapping(src, dst, row_count);
        row_count * std::mem::size_of::<i64>()
    }

    /// Copy FLOAT64 column data from ColumnarBatch, downcasting f64->f32 (GPU float).
    ///
    /// The existing ColumnarBatch stores floats as f32 already (Metal uses 32-bit float),
    /// so this is a direct copy.
    ///
    /// # Safety
    /// Caller must ensure offset + row_count * 4 <= buffer size.
    unsafe fn copy_float_column(
        data_buffer: &ProtocolObject<dyn MTLBuffer>,
        batch: &ColumnarBatch,
        local_col_idx: usize,
        offset: usize,
        row_count: usize,
    ) -> usize {
        if row_count == 0 {
            return 0;
        }
        // ColumnarBatch stores floats as f32 already
        let src_ptr = batch.float_buffer.contents().as_ptr() as *const f32;
        let src_offset = local_col_idx * batch.max_rows;
        let src = src_ptr.add(src_offset);

        let dst_base = data_buffer.contents().as_ptr() as *mut u8;
        let dst = dst_base.add(offset) as *mut f32;

        std::ptr::copy_nonoverlapping(src, dst, row_count);
        row_count * std::mem::size_of::<f32>()
    }

    /// Copy VARCHAR dictionary codes from ColumnarBatch into the contiguous buffer.
    ///
    /// Returns the bytes written and optionally the dictionary values if present.
    ///
    /// # Safety
    /// Caller must ensure offset + row_count * 4 <= buffer size.
    unsafe fn copy_varchar_column(
        data_buffer: &ProtocolObject<dyn MTLBuffer>,
        batch: &ColumnarBatch,
        local_col_idx: usize,
        offset: usize,
        row_count: usize,
        global_col_idx: usize,
    ) -> (usize, Option<Vec<String>>) {
        if row_count == 0 {
            return (0, None);
        }
        let src_ptr = batch.string_dict_buffer.contents().as_ptr() as *const u32;
        let src_offset = local_col_idx * batch.max_rows;
        let src = src_ptr.add(src_offset);

        let dst_base = data_buffer.contents().as_ptr() as *mut u8;
        let dst = dst_base.add(offset) as *mut u32;

        std::ptr::copy_nonoverlapping(src, dst, row_count);

        // Extract dictionary values if available
        let dict_values = batch
            .dictionaries
            .get(global_col_idx)
            .and_then(|d| d.as_ref())
            .map(|d| d.values().to_vec());

        (row_count * std::mem::size_of::<u32>(), dict_values)
    }
}

/// Align `value` up to the next multiple of `alignment`.
fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::columnar::ColumnarBatch;
    use crate::storage::dictionary::Dictionary;
    use crate::storage::schema::{ColumnDef, DataType, RuntimeSchema};
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use std::sync::mpsc;

    fn get_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
        MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)")
    }

    /// Create a test schema with the specified column types.
    fn make_schema(cols: &[(&str, DataType)]) -> RuntimeSchema {
        RuntimeSchema::new(
            cols.iter()
                .map(|(name, dt)| ColumnDef {
                    name: name.to_string(),
                    data_type: *dt,
                    nullable: false,
                })
                .collect(),
        )
    }

    /// Create a ColumnarBatch with deterministic test data.
    /// INT64: value = (i * 7 + 13) % 1000
    /// FLOAT64 (as f32): value = ((i * 7 + 13) % 1000) as f32
    /// VARCHAR: dict codes for ["a", "b", "c", "d", "e"][i % 5]
    fn make_test_batch(
        device: &ProtocolObject<dyn MTLDevice>,
        schema: &RuntimeSchema,
        row_count: usize,
    ) -> ColumnarBatch {
        let mut batch = ColumnarBatch::allocate(device, schema, row_count);
        batch.row_count = row_count;

        // Fill INT64 columns
        let mut int_local_idx = 0usize;
        let mut float_local_idx = 0usize;
        let mut string_local_idx = 0usize;

        for (global_idx, col) in schema.columns.iter().enumerate() {
            match col.data_type {
                DataType::Int64 | DataType::Date => {
                    unsafe {
                        let ptr = batch.int_buffer.contents().as_ptr() as *mut i64;
                        let offset = int_local_idx * batch.max_rows;
                        for i in 0..row_count {
                            *ptr.add(offset + i) = ((i * 7 + 13) % 1000) as i64;
                        }
                    }
                    int_local_idx += 1;
                }
                DataType::Float64 => {
                    unsafe {
                        let ptr = batch.float_buffer.contents().as_ptr() as *mut f32;
                        let offset = float_local_idx * batch.max_rows;
                        for i in 0..row_count {
                            *ptr.add(offset + i) = ((i * 7 + 13) % 1000) as f32;
                        }
                    }
                    float_local_idx += 1;
                }
                DataType::Varchar => {
                    // Build dictionary for ["a", "b", "c", "d", "e"]
                    let dict_values: Vec<String> = (0..row_count)
                        .map(|i| ["a", "b", "c", "d", "e"][i % 5].to_string())
                        .collect();
                    let dict = Dictionary::build(&dict_values).unwrap();
                    let codes = dict.encode_column(&dict_values);

                    unsafe {
                        let ptr = batch.string_dict_buffer.contents().as_ptr() as *mut u32;
                        let offset = string_local_idx * batch.max_rows;
                        for i in 0..row_count {
                            *ptr.add(offset + i) = codes[i];
                        }
                    }

                    batch.dictionaries[global_idx] = Some(dict);
                    string_local_idx += 1;
                }
                _ => {}
            }
        }

        batch
    }

    // ================================================================
    // Test 1: Load single INT64 column, verify round-trip
    // ================================================================
    #[test]
    fn load_single_int64_column() {
        let device = get_device();
        let schema = make_schema(&[("id", DataType::Int64)]);
        let batch = make_test_batch(&device, &schema, 100);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        assert_eq!(table.row_count, 100);
        assert_eq!(table.column_metas.len(), 1);
        assert_eq!(table.column_metas[0].column_type, COLUMN_TYPE_INT64);
        assert_eq!(table.column_metas[0].stride, 8);

        // Read back and verify values
        let meta = &table.column_metas[0];
        unsafe {
            let base = table.data_buffer.contents().as_ptr() as *const u8;
            let col_ptr = base.add(meta.offset as usize) as *const i64;
            for i in 0..100usize {
                let expected = ((i * 7 + 13) % 1000) as i64;
                let actual = *col_ptr.add(i);
                assert_eq!(actual, expected, "INT64 mismatch at row {}", i);
            }
        }
    }

    // ================================================================
    // Test 2: Load single FLOAT64 column, verify f32 downcast
    // ================================================================
    #[test]
    fn load_single_float64_column() {
        let device = get_device();
        let schema = make_schema(&[("amount", DataType::Float64)]);
        let batch = make_test_batch(&device, &schema, 100);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        assert_eq!(table.row_count, 100);
        assert_eq!(table.column_metas[0].column_type, COLUMN_TYPE_FLOAT32);
        assert_eq!(table.column_metas[0].stride, 4);

        // Read back f32 values
        let meta = &table.column_metas[0];
        unsafe {
            let base = table.data_buffer.contents().as_ptr() as *const u8;
            let col_ptr = base.add(meta.offset as usize) as *const f32;
            for i in 0..100usize {
                let expected = ((i * 7 + 13) % 1000) as f32;
                let actual = *col_ptr.add(i);
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "FLOAT32 mismatch at row {}: expected {}, got {}",
                    i,
                    expected,
                    actual
                );
            }
        }
    }

    // ================================================================
    // Test 3: Load single VARCHAR column, verify dictionary codes
    // ================================================================
    #[test]
    fn load_single_varchar_column() {
        let device = get_device();
        let schema = make_schema(&[("region", DataType::Varchar)]);
        let batch = make_test_batch(&device, &schema, 100);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        assert_eq!(table.row_count, 100);
        assert_eq!(table.column_metas[0].column_type, COLUMN_TYPE_DICT_U32);
        assert_eq!(table.column_metas[0].stride, 4);

        // Verify dictionary was captured
        assert!(table.dictionaries.contains_key(&0), "Should have dict for col 0");
        let dict_values = &table.dictionaries[&0];
        assert_eq!(dict_values.len(), 5, "5 distinct values");
        // Dictionary is sorted: a=0, b=1, c=2, d=3, e=4
        assert_eq!(dict_values, &["a", "b", "c", "d", "e"]);

        // Read back dict codes
        let meta = &table.column_metas[0];
        unsafe {
            let base = table.data_buffer.contents().as_ptr() as *const u8;
            let col_ptr = base.add(meta.offset as usize) as *const u32;
            // Build expected dict mapping: sorted ["a","b","c","d","e"] -> a=0,b=1,c=2,d=3,e=4
            let labels = ["a", "b", "c", "d", "e"];
            for i in 0..100usize {
                let label = labels[i % 5];
                let expected_code = match label {
                    "a" => 0u32,
                    "b" => 1,
                    "c" => 2,
                    "d" => 3,
                    "e" => 4,
                    _ => panic!("unexpected label"),
                };
                let actual = *col_ptr.add(i);
                assert_eq!(actual, expected_code, "VARCHAR code mismatch at row {}", i);
            }
        }
    }

    // ================================================================
    // Test 4: Load multi-column table (INT64 + FLOAT64 + VARCHAR)
    // ================================================================
    #[test]
    fn load_multi_column_table() {
        let device = get_device();
        let schema = make_schema(&[
            ("id", DataType::Int64),
            ("amount", DataType::Float64),
            ("region", DataType::Varchar),
        ]);
        let batch = make_test_batch(&device, &schema, 50);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        assert_eq!(table.row_count, 50);
        assert_eq!(table.column_metas.len(), 3);
        assert_eq!(table.schema.len(), 3);

        // Verify types
        assert_eq!(table.column_metas[0].column_type, COLUMN_TYPE_INT64);
        assert_eq!(table.column_metas[1].column_type, COLUMN_TYPE_FLOAT32);
        assert_eq!(table.column_metas[2].column_type, COLUMN_TYPE_DICT_U32);

        // Verify strides
        assert_eq!(table.column_metas[0].stride, 8);
        assert_eq!(table.column_metas[1].stride, 4);
        assert_eq!(table.column_metas[2].stride, 4);

        // Verify data round-trip for each column
        unsafe {
            let base = table.data_buffer.contents().as_ptr() as *const u8;

            // INT64 column
            let int_ptr = base.add(table.column_metas[0].offset as usize) as *const i64;
            for i in 0..50usize {
                let expected = ((i * 7 + 13) % 1000) as i64;
                assert_eq!(*int_ptr.add(i), expected, "INT64 mismatch at row {}", i);
            }

            // FLOAT32 column
            let float_ptr = base.add(table.column_metas[1].offset as usize) as *const f32;
            for i in 0..50usize {
                let expected = ((i * 7 + 13) % 1000) as f32;
                assert!(
                    (*float_ptr.add(i) - expected).abs() < 1e-5,
                    "FLOAT32 mismatch at row {}",
                    i
                );
            }

            // VARCHAR dict codes
            let dict_ptr = base.add(table.column_metas[2].offset as usize) as *const u32;
            let labels = ["a", "b", "c", "d", "e"];
            for i in 0..50usize {
                let label = labels[i % 5];
                let expected_code = match label {
                    "a" => 0u32,
                    "b" => 1,
                    "c" => 2,
                    "d" => 3,
                    "e" => 4,
                    _ => unreachable!(),
                };
                assert_eq!(
                    *dict_ptr.add(i),
                    expected_code,
                    "VARCHAR code mismatch at row {}",
                    i
                );
            }
        }
    }

    // ================================================================
    // Test 5: Verify ColumnMeta offsets are 16-byte aligned
    // ================================================================
    #[test]
    fn column_meta_offsets_are_16_byte_aligned() {
        let device = get_device();
        let schema = make_schema(&[
            ("id", DataType::Int64),
            ("amount", DataType::Float64),
            ("region", DataType::Varchar),
            ("count", DataType::Int64),
        ]);
        let batch = make_test_batch(&device, &schema, 100);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        for (idx, meta) in table.column_metas.iter().enumerate() {
            assert_eq!(
                meta.offset % COLUMN_ALIGNMENT as u64,
                0,
                "Column {} offset {} is not 16-byte aligned",
                idx,
                meta.offset
            );
        }
    }

    // ================================================================
    // Test 6: Verify buffer size is page-aligned
    // ================================================================
    #[test]
    fn buffer_size_is_page_aligned() {
        let device = get_device();
        let schema = make_schema(&[("id", DataType::Int64), ("amount", DataType::Float64)]);
        let batch = make_test_batch(&device, &schema, 100);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        let buf_len = table.data_buffer.length() as usize;
        assert!(
            buf_len >= PAGE_SIZE,
            "Buffer should be at least one page ({} bytes), got {}",
            PAGE_SIZE,
            buf_len
        );
        assert_eq!(
            buf_len % PAGE_SIZE,
            0,
            "Buffer size {} is not page-aligned (16KB)",
            buf_len
        );
    }

    // ================================================================
    // Test 7: Verify row_count correct
    // ================================================================
    #[test]
    fn row_count_matches_input() {
        let device = get_device();
        let schema = make_schema(&[("id", DataType::Int64)]);

        for count in [0usize, 1, 10, 100, 999, 1000] {
            let batch = make_test_batch(&device, &schema, count);
            let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
                .expect("load_table failed");
            assert_eq!(
                table.row_count, count as u32,
                "row_count mismatch for input {}",
                count
            );
        }
    }

    // ================================================================
    // Test 8: Load 1K rows, verify all values round-trip
    // ================================================================
    #[test]
    fn load_1k_rows_round_trip() {
        let device = get_device();
        let schema = make_schema(&[
            ("id", DataType::Int64),
            ("amount", DataType::Float64),
            ("region", DataType::Varchar),
        ]);
        let batch = make_test_batch(&device, &schema, 1000);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        assert_eq!(table.row_count, 1000);

        unsafe {
            let base = table.data_buffer.contents().as_ptr() as *const u8;

            // Verify all 1000 INT64 values
            let int_ptr = base.add(table.column_metas[0].offset as usize) as *const i64;
            for i in 0..1000usize {
                let expected = ((i * 7 + 13) % 1000) as i64;
                assert_eq!(*int_ptr.add(i), expected, "INT64 mismatch at row {}", i);
            }

            // Verify all 1000 FLOAT32 values
            let float_ptr = base.add(table.column_metas[1].offset as usize) as *const f32;
            for i in 0..1000usize {
                let expected = ((i * 7 + 13) % 1000) as f32;
                assert!(
                    (*float_ptr.add(i) - expected).abs() < 1e-5,
                    "FLOAT32 mismatch at row {}",
                    i
                );
            }

            // Verify all 1000 VARCHAR dict codes
            let dict_ptr = base.add(table.column_metas[2].offset as usize) as *const u32;
            let labels = ["a", "b", "c", "d", "e"];
            for i in 0..1000usize {
                let label = labels[i % 5];
                let expected_code = match label {
                    "a" => 0u32,
                    "b" => 1,
                    "c" => 2,
                    "d" => 3,
                    "e" => 4,
                    _ => unreachable!(),
                };
                assert_eq!(
                    *dict_ptr.add(i),
                    expected_code,
                    "VARCHAR code mismatch at row {}",
                    i
                );
            }
        }
    }

    // ================================================================
    // Test 9: Progress channel receives updates
    // ================================================================
    #[test]
    fn progress_channel_receives_updates() {
        let device = get_device();
        let schema = make_schema(&[
            ("id", DataType::Int64),
            ("amount", DataType::Float64),
        ]);
        let batch = make_test_batch(&device, &schema, 10);

        let (tx, rx) = mpsc::channel();

        let _table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, Some(&tx))
            .expect("load_table failed");

        drop(tx); // Close sender so we can drain

        let messages: Vec<LoadProgress> = rx.iter().collect();

        // Should have: LoadingColumn(0, ..), ColumnLoaded(0, ..), LoadingColumn(1, ..), ColumnLoaded(1, ..), Complete(..)
        assert!(messages.len() >= 5, "Expected at least 5 progress messages, got {}", messages.len());

        // Verify first message is LoadingColumn
        assert!(matches!(&messages[0], LoadProgress::LoadingColumn(0, _)));
        // Verify last message is Complete
        assert!(matches!(&messages[messages.len() - 1], LoadProgress::Complete(10)));
    }

    // ================================================================
    // Test 10: Empty table (0 rows) doesn't crash
    // ================================================================
    #[test]
    fn empty_table_no_crash() {
        let device = get_device();
        let schema = make_schema(&[
            ("id", DataType::Int64),
            ("amount", DataType::Float64),
            ("region", DataType::Varchar),
        ]);
        let batch = make_test_batch(&device, &schema, 0);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        assert_eq!(table.row_count, 0);
        assert_eq!(table.column_metas.len(), 3);
        // Buffer should still be allocated (page-aligned minimum)
        assert!(table.data_buffer.length() > 0);
    }

    // ================================================================
    // Test 11: ColumnMeta row_count matches input
    // ================================================================
    #[test]
    fn column_meta_row_count_matches() {
        let device = get_device();
        let schema = make_schema(&[
            ("id", DataType::Int64),
            ("amount", DataType::Float64),
        ]);
        let batch = make_test_batch(&device, &schema, 500);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        for (idx, meta) in table.column_metas.iter().enumerate() {
            assert_eq!(
                meta.row_count, 500,
                "ColumnMeta[{}].row_count should be 500",
                idx
            );
        }
    }

    // ================================================================
    // Test 12: ColumnMeta buffer contains correct data
    // ================================================================
    #[test]
    fn column_meta_buffer_round_trip() {
        let device = get_device();
        let schema = make_schema(&[
            ("id", DataType::Int64),
            ("amount", DataType::Float64),
            ("region", DataType::Varchar),
        ]);
        let batch = make_test_batch(&device, &schema, 100);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        // Read ColumnMeta from Metal buffer and compare with Vec
        unsafe {
            let ptr = table.column_meta_buffer.contents().as_ptr() as *const ColumnMeta;
            for i in 0..3usize {
                let meta_from_buf = *ptr.add(i);
                let meta_from_vec = &table.column_metas[i];
                assert_eq!(
                    meta_from_buf.offset, meta_from_vec.offset,
                    "ColumnMeta[{}].offset mismatch",
                    i
                );
                assert_eq!(
                    meta_from_buf.column_type, meta_from_vec.column_type,
                    "ColumnMeta[{}].column_type mismatch",
                    i
                );
                assert_eq!(
                    meta_from_buf.stride, meta_from_vec.stride,
                    "ColumnMeta[{}].stride mismatch",
                    i
                );
                assert_eq!(
                    meta_from_buf.row_count, meta_from_vec.row_count,
                    "ColumnMeta[{}].row_count mismatch",
                    i
                );
            }
        }
    }

    // ================================================================
    // Test 13: Schema info preserved
    // ================================================================
    #[test]
    fn schema_info_preserved() {
        let device = get_device();
        let schema = make_schema(&[
            ("id", DataType::Int64),
            ("amount", DataType::Float64),
            ("region", DataType::Varchar),
        ]);
        let batch = make_test_batch(&device, &schema, 10);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        assert_eq!(table.schema.len(), 3);
        assert_eq!(table.schema[0].name, "id");
        assert_eq!(table.schema[0].data_type, DataType::Int64);
        assert_eq!(table.schema[1].name, "amount");
        assert_eq!(table.schema[1].data_type, DataType::Float64);
        assert_eq!(table.schema[2].name, "region");
        assert_eq!(table.schema[2].data_type, DataType::Varchar);
    }

    // ================================================================
    // Test 14: Column offsets are non-overlapping
    // ================================================================
    #[test]
    fn column_offsets_non_overlapping() {
        let device = get_device();
        let schema = make_schema(&[
            ("id", DataType::Int64),
            ("amount", DataType::Float64),
            ("region", DataType::Varchar),
        ]);
        let batch = make_test_batch(&device, &schema, 100);

        let table = BinaryColumnarLoader::load_table(&device, "test", &schema, &batch, None)
            .expect("load_table failed");

        // For each pair of columns, verify their data regions don't overlap
        for i in 0..table.column_metas.len() {
            let meta_i = &table.column_metas[i];
            let end_i = meta_i.offset + (meta_i.row_count as u64 * meta_i.stride as u64);

            for j in (i + 1)..table.column_metas.len() {
                let meta_j = &table.column_metas[j];
                let end_j = meta_j.offset + (meta_j.row_count as u64 * meta_j.stride as u64);

                // Either i ends before j starts, or j ends before i starts
                assert!(
                    end_i <= meta_j.offset || end_j <= meta_i.offset,
                    "Columns {} and {} overlap: [{}, {}) and [{}, {})",
                    i,
                    j,
                    meta_i.offset,
                    end_i,
                    meta_j.offset,
                    end_j
                );
            }
        }
    }

    // ================================================================
    // Test 15: align_up helper function
    // ================================================================
    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(0, PAGE_SIZE), 0);
        assert_eq!(align_up(1, PAGE_SIZE), PAGE_SIZE);
        assert_eq!(align_up(PAGE_SIZE, PAGE_SIZE), PAGE_SIZE);
        assert_eq!(align_up(PAGE_SIZE + 1, PAGE_SIZE), 2 * PAGE_SIZE);
    }
}
