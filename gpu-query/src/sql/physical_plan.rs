//! Physical query plan mapping logical operators to GPU kernel dispatches.
//!
//! Each physical plan node corresponds to one or more Metal compute kernel
//! dispatches. The executor walks the physical plan tree bottom-up, dispatching
//! kernels in sequence within a single command buffer.

use super::logical_plan::LogicalPlan;
use super::types::{AggFunc, CompareOp, Expr, LogicalOp, Value};

/// A node in the physical query plan tree.
///
/// Each variant maps directly to GPU kernel dispatch(es):
/// - `GpuScan` -> CSV/Parquet/JSON parse kernels
/// - `GpuFilter` -> `column_filter` kernel with function constants
/// - `GpuAggregate` -> `aggregate_*` kernels with hierarchical reduction
/// - `GpuSort` -> `radix_sort_*` kernels
/// - `GpuLimit` -> CPU-side truncation (no kernel needed)
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalPlan {
    /// GPU scan: parse file into columnar buffers.
    GpuScan {
        /// Table name (file name without extension).
        table: String,
        /// Columns to materialize.
        columns: Vec<String>,
    },

    /// GPU filter: apply comparison predicate to a column.
    GpuFilter {
        /// Comparison operator (maps to function constant COMPARE_OP).
        compare_op: CompareOp,
        /// Column name to filter on.
        column: String,
        /// Comparison value.
        value: Value,
        /// Input physical plan.
        input: Box<PhysicalPlan>,
    },

    /// GPU compound filter: AND/OR of two filter bitmasks.
    GpuCompoundFilter {
        /// Logical operator (AND = bitwise AND, OR = bitwise OR of masks).
        op: LogicalOp,
        /// Left filter producing a bitmask.
        left: Box<PhysicalPlan>,
        /// Right filter producing a bitmask.
        right: Box<PhysicalPlan>,
    },

    /// GPU aggregation: compute aggregate functions over filtered rows.
    GpuAggregate {
        /// Aggregate function calls: (function, column name).
        /// For COUNT(*), column is "*".
        functions: Vec<(AggFunc, String)>,
        /// GROUP BY column names.
        group_by: Vec<String>,
        /// Input physical plan.
        input: Box<PhysicalPlan>,
    },

    /// GPU sort: radix sort by column values.
    GpuSort {
        /// Order-by columns: (column name, ascending).
        order_by: Vec<(String, bool)>,
        /// Input physical plan.
        input: Box<PhysicalPlan>,
    },

    /// Limit output rows (CPU-side truncation).
    GpuLimit {
        /// Maximum number of rows.
        count: usize,
        /// Input physical plan.
        input: Box<PhysicalPlan>,
    },
}

/// Errors that can occur during physical plan generation.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanError {
    /// Expression type not supported in physical plan.
    UnsupportedExpr(String),
    /// Aggregate argument not supported.
    UnsupportedAggregate(String),
    /// Filter predicate not in expected form.
    InvalidPredicate(String),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::UnsupportedExpr(msg) => write!(f, "unsupported expression: {}", msg),
            PlanError::UnsupportedAggregate(msg) => write!(f, "unsupported aggregate: {}", msg),
            PlanError::InvalidPredicate(msg) => write!(f, "invalid predicate: {}", msg),
        }
    }
}

impl std::error::Error for PlanError {}

/// Convert a logical plan to a physical plan for GPU execution.
pub fn plan(logical: &LogicalPlan) -> Result<PhysicalPlan, PlanError> {
    match logical {
        LogicalPlan::Scan { table, projection } => Ok(PhysicalPlan::GpuScan {
            table: table.clone(),
            columns: projection.clone(),
        }),

        LogicalPlan::Filter { predicate, input } => {
            let input_plan = plan(input)?;
            lower_predicate(predicate, input_plan)
        }

        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            input,
        } => {
            let input_plan = plan(input)?;
            let functions = aggregates
                .iter()
                .map(|(func, arg)| {
                    let col_name = match arg {
                        Expr::Column(name) => name.clone(),
                        Expr::Wildcard => "*".to_string(),
                        other => {
                            return Err(PlanError::UnsupportedAggregate(format!(
                                "aggregate argument must be column or *, got: {}",
                                other
                            )))
                        }
                    };
                    Ok((*func, col_name))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let group_cols = group_by
                .iter()
                .map(|e| match e {
                    Expr::Column(name) => Ok(name.clone()),
                    other => Err(PlanError::UnsupportedExpr(format!(
                        "GROUP BY must be column name, got: {}",
                        other
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(PhysicalPlan::GpuAggregate {
                functions,
                group_by: group_cols,
                input: Box::new(input_plan),
            })
        }

        LogicalPlan::Sort { order_by, input } => {
            let input_plan = plan(input)?;
            let cols = order_by
                .iter()
                .map(|(expr, asc)| match expr {
                    Expr::Column(name) => Ok((name.clone(), *asc)),
                    other => Err(PlanError::UnsupportedExpr(format!(
                        "ORDER BY must be column name, got: {}",
                        other
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(PhysicalPlan::GpuSort {
                order_by: cols,
                input: Box::new(input_plan),
            })
        }

        LogicalPlan::Limit { count, input } => {
            let input_plan = plan(input)?;
            Ok(PhysicalPlan::GpuLimit {
                count: *count,
                input: Box::new(input_plan),
            })
        }

        LogicalPlan::Projection { columns: _, input } => {
            // In the MVP, projection is handled by the scan node's column list
            // or by the executor selecting which result columns to output.
            // Pass through to the input plan.
            plan(input)
        }
    }
}

/// Lower a predicate expression to physical filter plan(s).
fn lower_predicate(
    predicate: &Expr,
    input_plan: PhysicalPlan,
) -> Result<PhysicalPlan, PlanError> {
    match predicate {
        // Simple comparison: col op value or value op col
        Expr::BinaryOp { left, op, right } => {
            let (column, value) = extract_column_value(left, right, *op)?;
            Ok(PhysicalPlan::GpuFilter {
                compare_op: *op,
                column,
                value,
                input: Box::new(input_plan),
            })
        }

        // Compound predicate: AND/OR
        Expr::Compound {
            left,
            op,
            right,
        } => {
            // For compound predicates, we need the same input for both sides.
            // The executor will dispatch two separate filter kernels on the same
            // data and combine their bitmasks.
            let left_plan = lower_predicate(left, input_plan.clone())?;
            let right_plan = lower_predicate(right, input_plan)?;
            Ok(PhysicalPlan::GpuCompoundFilter {
                op: *op,
                left: Box::new(left_plan),
                right: Box::new(right_plan),
            })
        }

        other => Err(PlanError::InvalidPredicate(format!(
            "expected comparison or compound predicate, got: {}",
            other
        ))),
    }
}

/// Extract column name and literal value from a binary comparison.
///
/// Handles both `col op value` and `value op col` forms.
fn extract_column_value(
    left: &Expr,
    right: &Expr,
    _op: CompareOp,
) -> Result<(String, Value), PlanError> {
    match (left, right) {
        (Expr::Column(name), Expr::Literal(val)) => Ok((name.clone(), val.clone())),
        (Expr::Literal(val), Expr::Column(name)) => {
            // value op col -> flip to col op value (caller already has original op)
            Ok((name.clone(), val.clone()))
        }
        _ => Err(PlanError::InvalidPredicate(format!(
            "expected column op literal, got: {} and {}",
            left, right
        ))),
    }
}

impl PhysicalPlan {
    /// Get a reference to the primary input plan, if any.
    pub fn input(&self) -> Option<&PhysicalPlan> {
        match self {
            PhysicalPlan::GpuScan { .. } => None,
            PhysicalPlan::GpuFilter { input, .. } => Some(input),
            PhysicalPlan::GpuCompoundFilter { left, .. } => Some(left),
            PhysicalPlan::GpuAggregate { input, .. } => Some(input),
            PhysicalPlan::GpuSort { input, .. } => Some(input),
            PhysicalPlan::GpuLimit { input, .. } => Some(input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_plan(table: &str) -> LogicalPlan {
        LogicalPlan::Scan {
            table: table.into(),
            projection: vec![],
        }
    }

    #[test]
    fn test_plan_scan() {
        let logical = scan_plan("sales");
        let physical = plan(&logical).unwrap();
        match physical {
            PhysicalPlan::GpuScan { table, columns } => {
                assert_eq!(table, "sales");
                assert!(columns.is_empty());
            }
            _ => panic!("expected GpuScan"),
        }
    }

    #[test]
    fn test_plan_filter() {
        let logical = LogicalPlan::Filter {
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Column("amount".into())),
                op: CompareOp::Gt,
                right: Box::new(Expr::Literal(Value::Int(100))),
            },
            input: Box::new(scan_plan("sales")),
        };
        let physical = plan(&logical).unwrap();
        match &physical {
            PhysicalPlan::GpuFilter {
                compare_op,
                column,
                value,
                ..
            } => {
                assert_eq!(*compare_op, CompareOp::Gt);
                assert_eq!(column, "amount");
                assert_eq!(*value, Value::Int(100));
            }
            _ => panic!("expected GpuFilter"),
        }
    }

    #[test]
    fn test_plan_aggregate_count_star() {
        let logical = LogicalPlan::Aggregate {
            group_by: vec![],
            aggregates: vec![(AggFunc::Count, Expr::Wildcard)],
            input: Box::new(scan_plan("t")),
        };
        let physical = plan(&logical).unwrap();
        match &physical {
            PhysicalPlan::GpuAggregate {
                functions,
                group_by,
                ..
            } => {
                assert_eq!(functions.len(), 1);
                assert_eq!(functions[0], (AggFunc::Count, "*".to_string()));
                assert!(group_by.is_empty());
            }
            _ => panic!("expected GpuAggregate"),
        }
    }

    #[test]
    fn test_plan_aggregate_with_group_by() {
        let logical = LogicalPlan::Aggregate {
            group_by: vec![Expr::Column("region".into())],
            aggregates: vec![(AggFunc::Sum, Expr::Column("amount".into()))],
            input: Box::new(scan_plan("sales")),
        };
        let physical = plan(&logical).unwrap();
        match &physical {
            PhysicalPlan::GpuAggregate {
                functions,
                group_by,
                ..
            } => {
                assert_eq!(functions.len(), 1);
                assert_eq!(functions[0], (AggFunc::Sum, "amount".to_string()));
                assert_eq!(group_by, &["region"]);
            }
            _ => panic!("expected GpuAggregate"),
        }
    }

    #[test]
    fn test_plan_sort() {
        let logical = LogicalPlan::Sort {
            order_by: vec![(Expr::Column("amount".into()), false)],
            input: Box::new(scan_plan("sales")),
        };
        let physical = plan(&logical).unwrap();
        match &physical {
            PhysicalPlan::GpuSort { order_by, .. } => {
                assert_eq!(order_by.len(), 1);
                assert_eq!(order_by[0], ("amount".to_string(), false));
            }
            _ => panic!("expected GpuSort"),
        }
    }

    #[test]
    fn test_plan_limit() {
        let logical = LogicalPlan::Limit {
            count: 10,
            input: Box::new(scan_plan("sales")),
        };
        let physical = plan(&logical).unwrap();
        match &physical {
            PhysicalPlan::GpuLimit { count, .. } => {
                assert_eq!(*count, 10);
            }
            _ => panic!("expected GpuLimit"),
        }
    }

    #[test]
    fn test_plan_compound_filter_and() {
        let logical = LogicalPlan::Filter {
            predicate: Expr::Compound {
                left: Box::new(Expr::BinaryOp {
                    left: Box::new(Expr::Column("a".into())),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Literal(Value::Int(1))),
                }),
                op: LogicalOp::And,
                right: Box::new(Expr::BinaryOp {
                    left: Box::new(Expr::Column("b".into())),
                    op: CompareOp::Lt,
                    right: Box::new(Expr::Literal(Value::Int(10))),
                }),
            },
            input: Box::new(scan_plan("t")),
        };
        let physical = plan(&logical).unwrap();
        match &physical {
            PhysicalPlan::GpuCompoundFilter { op, left, right } => {
                assert_eq!(*op, LogicalOp::And);
                match left.as_ref() {
                    PhysicalPlan::GpuFilter { column, .. } => assert_eq!(column, "a"),
                    _ => panic!("expected GpuFilter for left"),
                }
                match right.as_ref() {
                    PhysicalPlan::GpuFilter { column, .. } => assert_eq!(column, "b"),
                    _ => panic!("expected GpuFilter for right"),
                }
            }
            _ => panic!("expected GpuCompoundFilter"),
        }
    }

    #[test]
    fn test_plan_full_query() {
        // SELECT count(*) FROM sales WHERE amount > 100
        let logical = LogicalPlan::Aggregate {
            group_by: vec![],
            aggregates: vec![(AggFunc::Count, Expr::Wildcard)],
            input: Box::new(LogicalPlan::Filter {
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Column("amount".into())),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Literal(Value::Int(100))),
                },
                input: Box::new(scan_plan("sales")),
            }),
        };
        let physical = plan(&logical).unwrap();

        // Should be GpuAggregate -> GpuFilter -> GpuScan
        match &physical {
            PhysicalPlan::GpuAggregate { input, .. } => match input.as_ref() {
                PhysicalPlan::GpuFilter { input, .. } => match input.as_ref() {
                    PhysicalPlan::GpuScan { table, .. } => assert_eq!(table, "sales"),
                    _ => panic!("expected GpuScan at leaf"),
                },
                _ => panic!("expected GpuFilter"),
            },
            _ => panic!("expected GpuAggregate at root"),
        }
    }

    #[test]
    fn test_plan_error_unsupported_aggregate_arg() {
        let logical = LogicalPlan::Aggregate {
            group_by: vec![],
            aggregates: vec![(
                AggFunc::Sum,
                Expr::BinaryOp {
                    left: Box::new(Expr::Column("a".into())),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Literal(Value::Int(1))),
                },
            )],
            input: Box::new(scan_plan("t")),
        };
        let result = plan(&logical);
        assert!(result.is_err());
        match result.unwrap_err() {
            PlanError::UnsupportedAggregate(_) => {}
            other => panic!("expected UnsupportedAggregate, got: {:?}", other),
        }
    }

    #[test]
    fn test_plan_error_unsupported_group_by() {
        let logical = LogicalPlan::Aggregate {
            group_by: vec![Expr::Literal(Value::Int(1))],
            aggregates: vec![(AggFunc::Count, Expr::Wildcard)],
            input: Box::new(scan_plan("t")),
        };
        let result = plan(&logical);
        assert!(result.is_err());
    }

    #[test]
    fn test_physical_plan_input() {
        let p = PhysicalPlan::GpuScan {
            table: "t".into(),
            columns: vec![],
        };
        assert!(p.input().is_none());

        let p = PhysicalPlan::GpuLimit {
            count: 5,
            input: Box::new(PhysicalPlan::GpuScan {
                table: "t".into(),
                columns: vec![],
            }),
        };
        assert!(p.input().is_some());
    }
}
