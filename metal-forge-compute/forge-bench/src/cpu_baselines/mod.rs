//! CPU baseline implementations for benchmark comparison.
//!
//! Each module provides parallel (rayon) or sequential CPU implementations
//! matching the GPU kernels.

pub mod accelerate;
pub mod rayon_filter;
pub mod rayon_reduce;
pub mod rayon_sort;
pub mod sequential;
