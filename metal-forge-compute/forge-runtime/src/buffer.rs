//! Common buffer trait and conversion functions for forge GPU buffers.
//!
//! [`ForgeBuffer`] provides a uniform interface over [`SortBuffer`] and [`FilterBuffer`],
//! while the free functions [`filter_result_to_sort_buffer`] and [`sort_buffer_to_filter_buffer`]
//! enable zero-copy buffer hand-off between filter and sort stages.

use forge_filter::{FilterBuffer, FilterKey, FilterResult};
use forge_sort::{SortBuffer, SortKey};
use objc2_metal::MTLBuffer;
use objc2::runtime::ProtocolObject;

/// Uniform read-only interface over GPU-resident typed buffers.
///
/// Implemented for both [`SortBuffer<T>`] and [`FilterBuffer<T>`], allowing
/// pipeline stages to accept either buffer type generically.
pub trait ForgeBuffer<T> {
    /// Reference to the underlying Metal buffer.
    fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer>;

    /// Number of valid elements in the buffer.
    fn len(&self) -> usize;

    /// Allocated capacity in elements.
    fn capacity(&self) -> usize;

    /// Whether the buffer contains zero elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: SortKey> ForgeBuffer<T> for SortBuffer<T> {
    fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.metal_buffer()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn capacity(&self) -> usize {
        self.capacity()
    }
}

impl<T: FilterKey> ForgeBuffer<T> for FilterBuffer<T> {
    fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.metal_buffer()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn capacity(&self) -> usize {
        self.capacity()
    }
}

/// Convert a [`FilterResult`] into a [`SortBuffer`] by taking ownership of the
/// values buffer. Returns `None` if the result has no values buffer.
///
/// This enables a filter→sort pipeline: filter elements first, then sort
/// the survivors without copying data.
///
/// # Type constraint
///
/// `T` must implement both [`FilterKey`] and [`SortKey`] (true for u32, i32,
/// f32, u64, i64, f64).
pub fn filter_result_to_sort_buffer<T: FilterKey + SortKey>(
    result: FilterResult<T>,
) -> Option<SortBuffer<T>> {
    let (buffer, len, capacity) = result.take_values_buffer()?;
    Some(SortBuffer::from_raw_parts(buffer, len, capacity))
}

/// Convert a [`SortBuffer`] into a [`FilterBuffer`] by transferring ownership
/// of the underlying Metal buffer.
///
/// This enables a sort→filter pipeline: sort elements first, then filter
/// the sorted data without copying.
///
/// # Type constraint
///
/// `T` must implement both [`SortKey`] and [`FilterKey`] (true for u32, i32,
/// f32, u64, i64, f64).
pub fn sort_buffer_to_filter_buffer<T: SortKey + FilterKey>(
    sort_buf: SortBuffer<T>,
) -> FilterBuffer<T> {
    let (buffer, len, capacity) = sort_buf.into_raw_parts();
    FilterBuffer::from_raw_parts(buffer, len, capacity)
}
