//! forge-runtime: shared GPU context and pipeline builder for forge-sort + forge-filter.
//!
//! Provides [`ForgeContext`] for shared Metal device/queue ownership,
//! buffer conversion utilities, and (future) single-command-buffer pipeline execution.

pub mod buffer;
pub mod context;

pub use buffer::{filter_result_to_sort_buffer, sort_buffer_to_filter_buffer, ForgeBuffer};
pub use context::ForgeContext;
