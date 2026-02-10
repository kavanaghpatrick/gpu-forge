//! SQL parsing and query planning for gpu-query.
//!
//! Wraps `sqlparser-rs` to parse an MVP SQL subset and converts the AST into
//! logical and physical plans suitable for GPU kernel dispatch.
//!
//! Supported SQL subset:
//! - SELECT: columns, aggregates (COUNT, SUM, AVG, MIN, MAX), COUNT(*)
//! - FROM: single table name (maps to file name)
//! - WHERE: simple predicates, compound (AND, OR)
//! - GROUP BY: column list
//! - ORDER BY: column ASC/DESC
//! - LIMIT: integer value

pub mod logical_plan;
pub mod parser;
pub mod physical_plan;
pub mod types;
