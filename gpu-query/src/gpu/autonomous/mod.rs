//! GPU-autonomous query engine: persistent kernel, JIT compilation, zero-readback output.

pub mod types;
pub mod work_queue;
pub mod loader;
pub mod executor;
pub mod jit;
