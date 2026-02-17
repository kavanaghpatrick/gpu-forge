//! CPU baseline implementations for benchmark comparison.
//!
//! Each module provides parallel (rayon) or sequential CPU implementations
//! matching the GPU kernels.

pub mod rayon_reduce;
pub mod sequential;
