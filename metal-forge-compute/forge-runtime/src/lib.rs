//! forge-runtime: shared GPU context and pipeline builder for forge-sort + forge-filter.
//!
//! Provides [`ForgeContext`] for shared Metal device/queue ownership,
//! buffer conversion utilities, and (future) single-command-buffer pipeline execution.

pub mod context;

pub use context::ForgeContext;
