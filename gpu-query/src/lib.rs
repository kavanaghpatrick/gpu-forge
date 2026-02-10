//! gpu-query: GPU-native local data analytics engine for Apple Silicon.
//!
//! Query local files (CSV, Parquet, JSON) at GPU memory bandwidth using
//! Metal compute kernels. Zero-copy via mmap + makeBuffer(bytesNoCopy:).

pub mod gpu;