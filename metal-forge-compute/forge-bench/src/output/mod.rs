//! Output modules for rendering benchmark results.
//!
//! Supports table (comfy-table), JSON, CSV, summary, roofline, and progress bar output.

pub mod csv;
pub mod json;
pub mod progress;
pub mod roofline;
pub mod summary;
pub mod table;
