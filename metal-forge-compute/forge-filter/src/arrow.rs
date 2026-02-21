//! Arrow integration for forge-filter.
//!
//! Provides [`ArrowFilterKey`] — a sealed mapping from [`FilterKey`] types to
//! their Arrow primitive type equivalents, enabling zero-copy interop with
//! `arrow-array` `PrimitiveArray<T>`.
//!
//! Gated behind `#[cfg(feature = "arrow")]`.

use arrow_array::types::{
    ArrowPrimitiveType, Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type, UInt64Type,
};

use crate::FilterKey;

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
pub trait ArrowFilterKey: FilterKey + sealed::Sealed {
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
