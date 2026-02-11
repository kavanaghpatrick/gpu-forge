//! Columnar (SoA) buffer allocation and management for GPU-parsed data.
//!
//! A `ColumnarBatch` holds Metal buffers for each column in Structure-of-Arrays
//! layout. Integer columns are stored as `int64_t` arrays, float columns as
//! `double` arrays. The SoA layout ensures coalesced GPU memory access.
//!
//! Buffer layout: `column_buffer[col_local_idx * max_rows + row_idx]`
//! where `col_local_idx` is the index within that type's columns (not global).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

use crate::gpu::encode;
use crate::storage::dictionary::Dictionary;
use crate::storage::schema::{DataType, RuntimeSchema};

/// A batch of columnar data stored in Metal buffers (SoA layout).
///
/// All integer columns share a single contiguous buffer.
/// All float columns share a single contiguous buffer.
/// String columns are dictionary-encoded: dict codes stored in string_dict_buffer
/// as u32 values, with dictionaries providing the code-to-string mapping.
/// This matches the GPU kernel's SoA write pattern.
pub struct ColumnarBatch {
    /// Combined buffer for all INT64 columns: layout [col_idx * max_rows + row_idx].
    pub int_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Combined buffer for all FLOAT64 columns: layout [col_idx * max_rows + row_idx].
    pub float_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Combined buffer for all dictionary-encoded VARCHAR columns: layout [col_idx * max_rows + row_idx].
    /// Each element is a u32 dictionary code.
    pub string_dict_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Dictionaries for each column (indexed by global column index).
    /// `None` for non-string columns, `Some(Dictionary)` for string columns.
    pub dictionaries: Vec<Option<Dictionary>>,
    /// Maximum rows this batch can hold.
    pub max_rows: usize,
    /// Number of INT64 columns.
    pub int_col_count: usize,
    /// Number of FLOAT64 columns.
    pub float_col_count: usize,
    /// Number of VARCHAR (dictionary-encoded) columns.
    pub string_col_count: usize,
    /// Actual number of rows parsed (set after GPU parse completes).
    pub row_count: usize,
}

impl ColumnarBatch {
    /// Allocate a new columnar batch for the given schema and max row count.
    ///
    /// Allocates Metal buffers sized for `max_rows * column_count * element_size`.
    /// Buffers are StorageModeShared for CPU readback of results.
    pub fn allocate(
        device: &ProtocolObject<dyn MTLDevice>,
        schema: &RuntimeSchema,
        max_rows: usize,
    ) -> Self {
        let int_col_count = schema.count_type(DataType::Int64);
        let float_col_count = schema.count_type(DataType::Float64);
        let string_col_count = schema.count_type(DataType::Varchar);

        // Minimum 1 element to avoid zero-size buffer allocation
        let int_size = std::cmp::max(int_col_count * max_rows, 1) * std::mem::size_of::<i64>();
        // Metal uses float (32-bit), not double, for float columns.
        let float_size = std::cmp::max(float_col_count * max_rows, 1) * std::mem::size_of::<f32>();
        // String dict buffer: u32 codes per row per varchar column
        let string_dict_size =
            std::cmp::max(string_col_count * max_rows, 1) * std::mem::size_of::<u32>();

        let int_buffer = encode::alloc_buffer(device, int_size);
        let float_buffer = encode::alloc_buffer(device, float_size);
        let string_dict_buffer = encode::alloc_buffer(device, string_dict_size);

        // Initialize dictionaries: None for all columns (populated later by executor)
        let dictionaries = vec![None; schema.num_columns()];

        Self {
            int_buffer,
            float_buffer,
            string_dict_buffer,
            dictionaries,
            max_rows,
            int_col_count,
            float_col_count,
            string_col_count,
            row_count: 0,
        }
    }

    /// Read back an INT64 column as a Vec<i64>.
    ///
    /// `col_local_idx` is the index among INT64 columns only (0-based).
    ///
    /// # Safety
    /// Must be called after GPU work completes (waitUntilCompleted).
    pub unsafe fn read_int_column(&self, col_local_idx: usize) -> Vec<i64> {
        assert!(
            col_local_idx < self.int_col_count,
            "INT64 column index out of bounds"
        );
        let ptr = self.int_buffer.contents().as_ptr() as *const i64;
        let offset = col_local_idx * self.max_rows;
        let slice = std::slice::from_raw_parts(ptr.add(offset), self.row_count);
        slice.to_vec()
    }

    /// Read back a FLOAT column as a Vec<f32>.
    ///
    /// Metal shaders use 32-bit float (no double support), so float columns
    /// are stored as f32 on the GPU side.
    ///
    /// `col_local_idx` is the index among FLOAT64 columns only (0-based).
    ///
    /// # Safety
    /// Must be called after GPU work completes (waitUntilCompleted).
    pub unsafe fn read_float_column(&self, col_local_idx: usize) -> Vec<f32> {
        assert!(
            col_local_idx < self.float_col_count,
            "FLOAT64 column index out of bounds"
        );
        let ptr = self.float_buffer.contents().as_ptr() as *const f32;
        let offset = col_local_idx * self.max_rows;
        let slice = std::slice::from_raw_parts(ptr.add(offset), self.row_count);
        slice.to_vec()
    }

    /// Read back a dictionary-encoded VARCHAR column as a Vec<u32> of dict codes.
    ///
    /// `col_local_idx` is the index among VARCHAR columns only (0-based).
    ///
    /// # Safety
    /// Must be called after GPU work completes (waitUntilCompleted).
    pub unsafe fn read_string_dict_column(&self, col_local_idx: usize) -> Vec<u32> {
        assert!(
            col_local_idx < self.string_col_count,
            "VARCHAR column index out of bounds"
        );
        let ptr = self.string_dict_buffer.contents().as_ptr() as *const u32;
        let offset = col_local_idx * self.max_rows;
        let slice = std::slice::from_raw_parts(ptr.add(offset), self.row_count);
        slice.to_vec()
    }
}
