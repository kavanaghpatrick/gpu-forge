//! Arrow integration for forge-filter.
//!
//! Provides [`ArrowFilterKey`] — a sealed mapping from [`FilterKey`] types to
//! their Arrow primitive type equivalents, enabling zero-copy interop with
//! `arrow-array` `PrimitiveArray<T>`.
//!
//! Also provides [`GpuFilter::filter_arrow`] and [`GpuFilter::filter_arrow_nullable`]
//! for filtering Arrow `PrimitiveArray` types directly on the GPU.
//!
//! Gated behind `#[cfg(feature = "arrow")]`.

use arrow_array::types::{
    ArrowPrimitiveType, Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, PrimitiveArray};
use arrow_buffer::ArrowNativeType;

use crate::{FilterError, FilterKey, GpuFilter, Predicate};

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
    impl Sealed for u64 {}
    impl Sealed for i64 {}
    impl Sealed for f64 {}
}

/// Maps a [`FilterKey`] type to its corresponding Arrow primitive type.
///
/// This is a sealed trait — only the six GPU-supported numeric types implement it.
///
/// # Type mappings
///
/// | Rust type | Arrow type |
/// |-----------|------------|
/// | `u32` | [`UInt32Type`](arrow_array::types::UInt32Type) |
/// | `i32` | [`Int32Type`](arrow_array::types::Int32Type) |
/// | `f32` | [`Float32Type`](arrow_array::types::Float32Type) |
/// | `u64` | [`UInt64Type`](arrow_array::types::UInt64Type) |
/// | `i64` | [`Int64Type`](arrow_array::types::Int64Type) |
/// | `f64` | [`Float64Type`](arrow_array::types::Float64Type) |
pub trait ArrowFilterKey: FilterKey + ArrowNativeType + sealed::Sealed {
    /// The Arrow primitive type that corresponds to this filter key.
    type ArrowType: ArrowPrimitiveType<Native = Self>;
}

impl ArrowFilterKey for u32 {
    type ArrowType = UInt32Type;
}

impl ArrowFilterKey for i32 {
    type ArrowType = Int32Type;
}

impl ArrowFilterKey for f32 {
    type ArrowType = Float32Type;
}

impl ArrowFilterKey for u64 {
    type ArrowType = UInt64Type;
}

impl ArrowFilterKey for i64 {
    type ArrowType = Int64Type;
}

impl ArrowFilterKey for f64 {
    type ArrowType = Float64Type;
}

// --- GpuFilter Arrow methods ---

impl GpuFilter {
    /// Filter an Arrow [`PrimitiveArray`], returning a new `PrimitiveArray` with matching values.
    ///
    /// Copies the array's values to a page-aligned Metal buffer, runs the GPU filter,
    /// and constructs the output `PrimitiveArray` from the result. The array's offset
    /// is handled automatically (sliced arrays work correctly).
    ///
    /// # Type parameter
    ///
    /// `T` must implement [`ArrowFilterKey`], mapping the six GPU-supported numeric types
    /// (`u32`, `i32`, `f32`, `u64`, `i64`, `f64`) to their Arrow equivalents.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if GPU execution fails.
    pub fn filter_arrow<T: ArrowFilterKey>(
        &mut self,
        array: &PrimitiveArray<T::ArrowType>,
        pred: &Predicate<T>,
    ) -> Result<PrimitiveArray<T::ArrowType>, FilterError> {
        let len = array.len();
        if len == 0 {
            return Ok(PrimitiveArray::<T::ArrowType>::from_iter_values(
                std::iter::empty::<T>(),
            ));
        }

        // Extract values — ScalarBuffer<T> derefs to &[T] since T: ArrowNativeType.
        // Offset already accounted for by ScalarBuffer after a slice().
        let values: &[T] = array.values();

        // Copy to page-aligned Metal buffer
        let mut buf = self.alloc_filter_buffer::<T>(len);
        buf.copy_from_slice(values);

        // Run GPU filter
        let result = self.filter(&buf, pred)?;

        // Convert result to PrimitiveArray
        Ok(PrimitiveArray::<T::ArrowType>::from_iter_values(
            result.as_slice().iter().copied(),
        ))
    }

    /// Filter an Arrow [`PrimitiveArray`] with NULL handling, excluding NULL elements.
    ///
    /// Extracts the validity bitmap from the Arrow array's null buffer and converts it
    /// to the packed u32 format expected by the GPU kernel. Elements where the validity
    /// bit is 0 (NULL) are excluded from the output regardless of whether they match
    /// the predicate.
    ///
    /// The returned `PrimitiveArray` has no null buffer (all NULLs are excluded).
    ///
    /// # Type parameter
    ///
    /// `T` must implement [`ArrowFilterKey`].
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if GPU execution fails.
    pub fn filter_arrow_nullable<T: ArrowFilterKey>(
        &mut self,
        array: &PrimitiveArray<T::ArrowType>,
        pred: &Predicate<T>,
    ) -> Result<PrimitiveArray<T::ArrowType>, FilterError> {
        let len = array.len();
        if len == 0 {
            return Ok(PrimitiveArray::<T::ArrowType>::from_iter_values(
                std::iter::empty::<T>(),
            ));
        }

        // Extract values — offset already accounted for by ScalarBuffer after slice.
        let values: &[T] = array.values();

        // Copy to page-aligned Metal buffer
        let mut buf = self.alloc_filter_buffer::<T>(len);
        buf.copy_from_slice(values);

        // Extract validity bitmap and convert to &[u8] for filter_nullable
        match array.nulls() {
            Some(null_buf) => {
                // NullBuffer wraps BooleanBuffer which may have a bit offset.
                // We need to re-pack bits starting from the offset to produce
                // a byte-aligned validity bitmap for the GPU kernel.
                let boolean_buf = null_buf.inner();
                let bit_offset = boolean_buf.offset();
                let raw_bytes = boolean_buf.values();

                let validity_bytes = if bit_offset == 0 {
                    // Fast path: no bit offset, just take the raw bytes
                    let needed_bytes = (len + 7) / 8;
                    raw_bytes[..needed_bytes.min(raw_bytes.len())].to_vec()
                } else {
                    // Slow path: bit offset from slice — re-pack bits
                    let needed_bytes = (len + 7) / 8;
                    let mut packed = vec![0u8; needed_bytes];
                    for i in 0..len {
                        let src_bit = bit_offset + i;
                        let src_byte = src_bit / 8;
                        let src_bit_idx = src_bit % 8;
                        let is_valid = (raw_bytes[src_byte] >> src_bit_idx) & 1;

                        let dst_byte = i / 8;
                        let dst_bit_idx = i % 8;
                        packed[dst_byte] |= is_valid << dst_bit_idx;
                    }
                    packed
                };

                // Run GPU filter with validity bitmap
                let result = self.filter_nullable(&buf, pred, &validity_bytes)?;
                Ok(PrimitiveArray::<T::ArrowType>::from_iter_values(
                    result.as_slice().iter().copied(),
                ))
            }
            None => {
                // No null buffer — all elements valid. Use filter() instead.
                let result = self.filter(&buf, pred)?;
                Ok(PrimitiveArray::<T::ArrowType>::from_iter_values(
                    result.as_slice().iter().copied(),
                ))
            }
        }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::{
        Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type, UInt64Type,
    };

    #[test]
    fn test_filter_arrow_u32_basic() {
        let mut gpu = GpuFilter::new().unwrap();

        // Create Arrow array
        let data: Vec<u32> = (0..10_000).collect();
        let arrow_array = PrimitiveArray::<UInt32Type>::from_iter_values(data.clone());

        // Filter via Arrow API
        let pred = Predicate::Gt(5000u32);
        let arrow_result = gpu.filter_arrow::<u32>(&arrow_array, &pred).unwrap();

        // Filter via direct API for comparison
        let mut buf = gpu.alloc_filter_buffer::<u32>(data.len());
        buf.copy_from_slice(&data);
        let direct_result = gpu.filter(&buf, &pred).unwrap();

        // Results must match
        assert_eq!(arrow_result.len(), direct_result.len());
        let arrow_vals: &[u32] = arrow_result.values();
        let direct_vals = direct_result.as_slice();
        assert_eq!(arrow_vals, direct_vals);
    }

    #[test]
    fn test_filter_arrow_all_types() {
        let mut gpu = GpuFilter::new().unwrap();
        let n = 10_000usize;

        // u32
        {
            let data: Vec<u32> = (0..n as u32).collect();
            let arr = PrimitiveArray::<UInt32Type>::from_iter_values(data);
            let result = gpu.filter_arrow::<u32>(&arr, &Predicate::Gt(5000u32)).unwrap();
            assert_eq!(result.len(), 4999);
        }

        // i32
        {
            let data: Vec<i32> = (0..n as i32).collect();
            let arr = PrimitiveArray::<Int32Type>::from_iter_values(data);
            let result = gpu.filter_arrow::<i32>(&arr, &Predicate::Lt(100i32)).unwrap();
            assert_eq!(result.len(), 100);
        }

        // f32
        {
            let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let arr = PrimitiveArray::<Float32Type>::from_iter_values(data);
            let result = gpu.filter_arrow::<f32>(&arr, &Predicate::Ge(9000.0f32)).unwrap();
            assert_eq!(result.len(), 1000);
        }

        // u64
        {
            let data: Vec<u64> = (0..n as u64).collect();
            let arr = PrimitiveArray::<UInt64Type>::from_iter_values(data);
            let result = gpu.filter_arrow::<u64>(&arr, &Predicate::Le(99u64)).unwrap();
            assert_eq!(result.len(), 100);
        }

        // i64
        {
            let data: Vec<i64> = (0..n as i64).collect();
            let arr = PrimitiveArray::<Int64Type>::from_iter_values(data);
            let result = gpu.filter_arrow::<i64>(&arr, &Predicate::Eq(42i64)).unwrap();
            assert_eq!(result.len(), 1);
            assert_eq!(result.values()[0], 42i64);
        }

        // f64
        {
            let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let arr = PrimitiveArray::<Float64Type>::from_iter_values(data);
            let result = gpu.filter_arrow::<f64>(&arr, &Predicate::Ne(500.0f64)).unwrap();
            assert_eq!(result.len(), n - 1);
        }
    }

    #[test]
    fn test_filter_arrow_nullable_basic() {
        let mut gpu = GpuFilter::new().unwrap();

        // Create Arrow array with NULLs: Some values, None for NULLs
        let data: Vec<Option<u32>> = (0..1000u32)
            .map(|i| if i % 100 == 0 { None } else { Some(i) })
            .collect();
        let arrow_array = PrimitiveArray::<UInt32Type>::from(data.clone());

        // Filter: Gt(500) with NULLs excluded
        let pred = Predicate::Gt(500u32);
        let result = gpu
            .filter_arrow_nullable::<u32>(&arrow_array, &pred)
            .unwrap();

        // CPU reference: elements > 500 that are not NULL
        let expected: Vec<u32> = data
            .iter()
            .filter_map(|opt| opt.filter(|&v| v > 500))
            .collect();

        assert_eq!(result.len(), expected.len());
        let result_vals: &[u32] = result.values();
        assert_eq!(result_vals, &expected[..]);

        // Verify no null buffer in output
        assert!(result.nulls().is_none());
    }

    #[test]
    fn test_filter_arrow_sliced() {
        let mut gpu = GpuFilter::new().unwrap();

        // Create a large array and slice it with non-zero offset
        let data: Vec<u32> = (0..10_000).collect();
        let full_array = PrimitiveArray::<UInt32Type>::from_iter_values(data);

        // Slice: offset=2000, length=3000 -> values 2000..5000
        let sliced = full_array.slice(2000, 3000);

        // Filter: Gt(4000) on sliced array
        let pred = Predicate::Gt(4000u32);
        let result = gpu.filter_arrow::<u32>(&sliced, &pred).unwrap();

        // Expected: values 4001..4999 (from the sliced range 2000..5000)
        let expected: Vec<u32> = (4001..5000).collect();
        assert_eq!(result.len(), expected.len());
        let result_vals: &[u32] = result.values();
        assert_eq!(result_vals, &expected[..]);
    }

    #[test]
    fn test_filter_arrow_empty() {
        let mut gpu = GpuFilter::new().unwrap();

        // Empty Arrow array
        let arr = PrimitiveArray::<UInt32Type>::from_iter_values(Vec::<u32>::new());
        assert_eq!(arr.len(), 0);

        let result = gpu
            .filter_arrow::<u32>(&arr, &Predicate::Gt(0u32))
            .unwrap();
        assert_eq!(result.len(), 0);

        // Empty nullable
        let arr_nullable = PrimitiveArray::<UInt32Type>::from(Vec::<Option<u32>>::new());
        let result_nullable = gpu
            .filter_arrow_nullable::<u32>(&arr_nullable, &Predicate::Gt(0u32))
            .unwrap();
        assert_eq!(result_nullable.len(), 0);
    }

    /// Verify that `filter_arrow` produces identical output to the direct
    /// `filter()` API for the same u32 data and predicate.
    #[test]
    fn test_filter_arrow_matches_filter_u32() {
        let mut gpu = GpuFilter::new().unwrap();

        // Use a non-trivial dataset with a Between predicate
        let n = 100_000usize;
        let data: Vec<u32> = (0..n as u32).collect();
        let arrow_array = PrimitiveArray::<UInt32Type>::from_iter_values(data.clone());

        let pred = Predicate::Between(10_000u32, 50_000u32);

        // Arrow path
        let arrow_result = gpu.filter_arrow::<u32>(&arrow_array, &pred).unwrap();

        // Direct path
        let mut buf = gpu.alloc_filter_buffer::<u32>(n);
        buf.copy_from_slice(&data);
        let direct_result = gpu.filter(&buf, &pred).unwrap();

        // Must be identical
        assert_eq!(arrow_result.len(), direct_result.len());
        assert_eq!(arrow_result.values().as_ref(), direct_result.as_slice());
    }

    /// All elements are NULL — output must be empty.
    #[test]
    fn test_filter_arrow_nullable_all_null() {
        let mut gpu = GpuFilter::new().unwrap();

        // 1000 elements, every one NULL
        let data: Vec<Option<u32>> = vec![None; 1000];
        let arrow_array = PrimitiveArray::<UInt32Type>::from(data);

        // Even Gt(0) should return nothing — everything is NULL
        let pred = Predicate::Gt(0u32);
        let result = gpu
            .filter_arrow_nullable::<u32>(&arrow_array, &pred)
            .unwrap();

        assert_eq!(result.len(), 0);
        assert!(result.nulls().is_none());
    }

    /// No NULLs at all — nullable path must produce identical output to non-nullable.
    #[test]
    fn test_filter_arrow_nullable_no_null() {
        let mut gpu = GpuFilter::new().unwrap();

        let n = 10_000usize;
        let data: Vec<u32> = (0..n as u32).collect();
        let pred = Predicate::Lt(5000u32);

        // Non-nullable Arrow array
        let arr_nonnull = PrimitiveArray::<UInt32Type>::from_iter_values(data.clone());
        let result_nonnull = gpu.filter_arrow::<u32>(&arr_nonnull, &pred).unwrap();

        // Nullable Arrow array with NO nulls (all Some)
        let data_opt: Vec<Option<u32>> = data.iter().map(|&v| Some(v)).collect();
        let arr_nullable = PrimitiveArray::<UInt32Type>::from(data_opt);
        let result_nullable = gpu
            .filter_arrow_nullable::<u32>(&arr_nullable, &pred)
            .unwrap();

        // Must be identical
        assert_eq!(result_nonnull.len(), result_nullable.len());
        assert_eq!(
            result_nonnull.values().as_ref(),
            result_nullable.values().as_ref()
        );
    }

    /// Verify that copying 64 MB of data (16M u32 values) through the Arrow path
    /// completes in reasonable time (< 10ms including GPU dispatch).
    #[test]
    fn test_filter_arrow_copy_overhead() {
        let mut gpu = GpuFilter::new().unwrap();

        let n = 16_000_000usize; // 16M elements * 4 bytes = 64 MB
        let data: Vec<u32> = (0..n as u32).collect();
        let arrow_array = PrimitiveArray::<UInt32Type>::from_iter_values(data);

        // Warm up: first call compiles PSO
        let pred = Predicate::Gt(u32::MAX - 1);
        let _ = gpu.filter_arrow::<u32>(&arrow_array, &pred).unwrap();

        // Timed run — predicate selects ~0% to minimize scatter work
        let start = std::time::Instant::now();
        let result = gpu.filter_arrow::<u32>(&arrow_array, &pred).unwrap();
        let elapsed = start.elapsed();

        // Very few matches expected (only u32::MAX - 1 is excluded, but data ends at 15999999)
        assert!(result.len() <= n);

        // 64 MB copy + GPU dispatch should complete well under 10ms on Apple Silicon
        assert!(
            elapsed.as_millis() < 10,
            "Arrow filter took {}ms (expected <10ms for 64MB)",
            elapsed.as_millis()
        );
    }
}
