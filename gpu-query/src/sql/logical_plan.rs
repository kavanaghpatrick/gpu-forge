//! Logical query plan representation.
//!
//! Logical plans describe *what* to compute without specifying *how* (no GPU
//! kernel references). The planner converts SQL AST into a logical plan tree,
//! which is then lowered to a physical plan for GPU execution.

use std::fmt;

use super::types::{AggFunc, Expr};

/// A node in the logical query plan tree.
///
/// Plans are nested bottom-up: `Scan` is always a leaf, operators wrap their
/// input plan via `Box<LogicalPlan>`.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// Read rows from a table (file). Leaf node.
    Scan {
        /// Table name (maps to file name without extension).
        table: String,
        /// Column names to project at scan time. Empty = all columns.
        projection: Vec<String>,
    },

    /// Apply a filter predicate (WHERE clause).
    Filter {
        /// Boolean predicate expression.
        predicate: Expr,
        /// Input plan producing rows to filter.
        input: Box<LogicalPlan>,
    },

    /// Compute aggregate functions, optionally grouped.
    Aggregate {
        /// GROUP BY expressions (empty = single-group aggregation).
        group_by: Vec<Expr>,
        /// Aggregate function calls: (function, argument expression).
        aggregates: Vec<(AggFunc, Expr)>,
        /// Input plan producing rows to aggregate.
        input: Box<LogicalPlan>,
    },

    /// Sort rows by one or more expressions.
    Sort {
        /// Order-by columns: (expression, ascending).
        order_by: Vec<(Expr, bool)>,
        /// Input plan producing rows to sort.
        input: Box<LogicalPlan>,
    },

    /// Limit output to N rows.
    Limit {
        /// Maximum number of rows to return.
        count: usize,
        /// Input plan producing rows to limit.
        input: Box<LogicalPlan>,
    },

    /// Project specific columns/expressions from the input.
    Projection {
        /// Expressions to project.
        columns: Vec<Expr>,
        /// Input plan.
        input: Box<LogicalPlan>,
    },
}

impl LogicalPlan {
    /// Get a reference to the input plan, if any.
    pub fn input(&self) -> Option<&LogicalPlan> {
        match self {
            LogicalPlan::Scan { .. } => None,
            LogicalPlan::Filter { input, .. } => Some(input),
            LogicalPlan::Aggregate { input, .. } => Some(input),
            LogicalPlan::Sort { input, .. } => Some(input),
            LogicalPlan::Limit { input, .. } => Some(input),
            LogicalPlan::Projection { input, .. } => Some(input),
        }
    }

    /// Return the table name if this plan reads from a single table.
    pub fn table_name(&self) -> Option<&str> {
        match self {
            LogicalPlan::Scan { table, .. } => Some(table),
            _ => self.input().and_then(|i| i.table_name()),
        }
    }
}

impl fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, 0)
    }
}

impl LogicalPlan {
    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);
        match self {
            LogicalPlan::Scan { table, projection } => {
                write!(f, "{}Scan: table={}", pad, table)?;
                if !projection.is_empty() {
                    write!(f, ", projection=[{}]", projection.join(", "))?;
                }
                Ok(())
            }
            LogicalPlan::Filter { predicate, input } => {
                writeln!(f, "{}Filter: {}", pad, predicate)?;
                input.fmt_indent(f, indent + 1)
            }
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                input,
            } => {
                write!(f, "{}Aggregate:", pad)?;
                if !group_by.is_empty() {
                    let groups: Vec<String> = group_by.iter().map(|e| e.to_string()).collect();
                    write!(f, " group_by=[{}]", groups.join(", "))?;
                }
                let aggs: Vec<String> = aggregates
                    .iter()
                    .map(|(func, arg)| format!("{}({})", func, arg))
                    .collect();
                writeln!(f, " funcs=[{}]", aggs.join(", "))?;
                input.fmt_indent(f, indent + 1)
            }
            LogicalPlan::Sort { order_by, input } => {
                let cols: Vec<String> = order_by
                    .iter()
                    .map(|(e, asc)| {
                        format!("{} {}", e, if *asc { "ASC" } else { "DESC" })
                    })
                    .collect();
                writeln!(f, "{}Sort: [{}]", pad, cols.join(", "))?;
                input.fmt_indent(f, indent + 1)
            }
            LogicalPlan::Limit { count, input } => {
                writeln!(f, "{}Limit: {}", pad, count)?;
                input.fmt_indent(f, indent + 1)
            }
            LogicalPlan::Projection { columns, input } => {
                let cols: Vec<String> = columns.iter().map(|e| e.to_string()).collect();
                writeln!(f, "{}Projection: [{}]", pad, cols.join(", "))?;
                input.fmt_indent(f, indent + 1)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::types::{CompareOp, Value};

    #[test]
    fn test_scan_plan() {
        let plan = LogicalPlan::Scan {
            table: "sales".into(),
            projection: vec![],
        };
        assert_eq!(plan.table_name(), Some("sales"));
        assert!(plan.input().is_none());
    }

    #[test]
    fn test_filter_plan() {
        let plan = LogicalPlan::Filter {
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Column("amount".into())),
                op: CompareOp::Gt,
                right: Box::new(Expr::Literal(Value::Int(100))),
            },
            input: Box::new(LogicalPlan::Scan {
                table: "sales".into(),
                projection: vec![],
            }),
        };
        assert_eq!(plan.table_name(), Some("sales"));
        assert!(plan.input().is_some());
    }

    #[test]
    fn test_aggregate_plan() {
        let plan = LogicalPlan::Aggregate {
            group_by: vec![],
            aggregates: vec![(AggFunc::Count, Expr::Wildcard)],
            input: Box::new(LogicalPlan::Scan {
                table: "t".into(),
                projection: vec![],
            }),
        };
        assert_eq!(plan.table_name(), Some("t"));
    }

    #[test]
    fn test_nested_plan_table_name() {
        let plan = LogicalPlan::Limit {
            count: 10,
            input: Box::new(LogicalPlan::Sort {
                order_by: vec![(Expr::Column("id".into()), false)],
                input: Box::new(LogicalPlan::Scan {
                    table: "orders".into(),
                    projection: vec![],
                }),
            }),
        };
        assert_eq!(plan.table_name(), Some("orders"));
    }

    #[test]
    fn test_plan_display() {
        let plan = LogicalPlan::Filter {
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Column("x".into())),
                op: CompareOp::Gt,
                right: Box::new(Expr::Literal(Value::Int(5))),
            },
            input: Box::new(LogicalPlan::Scan {
                table: "t".into(),
                projection: vec![],
            }),
        };
        let s = plan.to_string();
        assert!(s.contains("Filter: x > 5"));
        assert!(s.contains("Scan: table=t"));
    }

    #[test]
    fn test_projection_plan() {
        let plan = LogicalPlan::Projection {
            columns: vec![Expr::Column("a".into()), Expr::Column("b".into())],
            input: Box::new(LogicalPlan::Scan {
                table: "t".into(),
                projection: vec![],
            }),
        };
        assert_eq!(plan.table_name(), Some("t"));
    }
}
