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
use crate::storage::schema::{DataType, RuntimeSchema};

/// A batch of columnar data stored in Metal buffers (SoA layout).
///
/// All integer columns share a single contiguous buffer.
/// All float columns share a single contiguous buffer.
/// This matches the GPU kernel's SoA write pattern.
pub struct ColumnarBatch {
    /// Combined buffer for all INT64 columns: layout [col_idx * max_rows + row_idx].
    pub int_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Combined buffer for all FLOAT64 columns: layout [col_idx * max_rows + row_idx].
    pub float_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Maximum rows this batch can hold.
    pub max_rows: usize,
    /// Number of INT64 columns.
    pub int_col_count: usize,
    /// Number of FLOAT64 columns.
    pub float_col_count: usize,
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

        // Minimum 1 element to avoid zero-size buffer allocation
        let int_size =
            std::cmp::max(int_col_count * max_rows, 1) * std::mem::size_of::<i64>();
        // Metal uses float (32-bit), not double, for float columns.
        let float_size =
            std::cmp::max(float_col_count * max_rows, 1) * std::mem::size_of::<f32>();

        let int_buffer = encode::alloc_buffer(device, int_size);
        let float_buffer = encode::alloc_buffer(device, float_size);

        Self {
            int_buffer,
            float_buffer,
            max_rows,
            int_col_count,
            float_col_count,
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
        assert!(col_local_idx < self.int_col_count, "INT64 column index out of bounds");
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
}
