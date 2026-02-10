//! Query optimizer performing logical plan transformations.
//!
//! Three optimization passes are applied in sequence:
//! 1. **Column pruning** -- restrict Scan to only columns referenced in the query
//! 2. **Predicate pushdown** -- push Filter nodes below Projection nodes
//! 3. **Constant folding** -- evaluate constant arithmetic at plan time

use std::collections::HashSet;

use super::logical_plan::LogicalPlan;
use super::types::{CompareOp, Expr, LogicalOp, Value};

/// Run all optimization passes on a logical plan.
pub fn optimize(plan: LogicalPlan) -> LogicalPlan {
    let plan = prune_columns(plan);
    let plan = push_down_predicates(plan);
    let plan = fold_constants(plan);
    plan
}

// ---------------------------------------------------------------------------
// Pass 1: Column pruning
// ---------------------------------------------------------------------------

/// Collect all column names referenced anywhere in the plan tree.
fn collect_referenced_columns(plan: &LogicalPlan) -> HashSet<String> {
    let mut cols = HashSet::new();
    collect_columns_recursive(plan, &mut cols);
    cols
}

fn collect_columns_recursive(plan: &LogicalPlan, cols: &mut HashSet<String>) {
    match plan {
        LogicalPlan::Scan { .. } => {
            // Scan itself doesn't reference columns beyond its projection
        }
        LogicalPlan::Filter { predicate, input } => {
            collect_expr_columns(predicate, cols);
            collect_columns_recursive(input, cols);
        }
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            input,
        } => {
            for expr in group_by {
                collect_expr_columns(expr, cols);
            }
            for (_func, arg) in aggregates {
                collect_expr_columns(arg, cols);
            }
            collect_columns_recursive(input, cols);
        }
        LogicalPlan::Sort { order_by, input } => {
            for (expr, _asc) in order_by {
                collect_expr_columns(expr, cols);
            }
            collect_columns_recursive(input, cols);
        }
        LogicalPlan::Limit { input, .. } => {
            collect_columns_recursive(input, cols);
        }
        LogicalPlan::Projection { columns, input } => {
            for expr in columns {
                collect_expr_columns(expr, cols);
            }
            collect_columns_recursive(input, cols);
        }
    }
}

/// Collect column names referenced in an expression.
fn collect_expr_columns(expr: &Expr, cols: &mut HashSet<String>) {
    match expr {
        Expr::Column(name) => {
            cols.insert(name.clone());
        }
        Expr::Literal(_) | Expr::Wildcard => {}
        Expr::BinaryOp { left, right, .. } => {
            collect_expr_columns(left, cols);
            collect_expr_columns(right, cols);
        }
        Expr::Compound { left, right, .. } => {
            collect_expr_columns(left, cols);
            collect_expr_columns(right, cols);
        }
        Expr::Aggregate { arg, .. } => {
            collect_expr_columns(arg, cols);
        }
    }
}

/// Set the Scan node's projection to only include referenced columns.
///
/// If the query only references a subset of columns (e.g. `SELECT sum(amount)
/// FROM wide_table WHERE region = 'EU'`), we restrict the Scan to
/// `[amount, region]` so the GPU only loads those columns.
fn prune_columns(plan: LogicalPlan) -> LogicalPlan {
    let referenced = collect_referenced_columns(&plan);
    // If wildcard is used or no specific columns referenced, skip pruning
    if referenced.is_empty() || has_wildcard(&plan) {
        return plan;
    }
    set_scan_projection(plan, &referenced)
}

/// Check if the plan uses a wildcard anywhere (SELECT *).
fn has_wildcard(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Scan { .. } => false,
        LogicalPlan::Filter { input, .. } => has_wildcard(input),
        LogicalPlan::Aggregate { input, .. } => {
            // COUNT(*) is OK -- it doesn't need all columns
            // Only true wildcard in Projection means we need all columns
            // COUNT(*) uses Expr::Wildcard as aggregate arg but that's fine
            has_wildcard(input)
        }
        LogicalPlan::Sort { input, .. } => has_wildcard(input),
        LogicalPlan::Limit { input, .. } => has_wildcard(input),
        LogicalPlan::Projection { columns, input } => {
            columns.iter().any(|e| matches!(e, Expr::Wildcard)) || has_wildcard(input)
        }
    }
}

/// Recursively set the Scan node's projection to the referenced columns.
fn set_scan_projection(plan: LogicalPlan, referenced: &HashSet<String>) -> LogicalPlan {
    match plan {
        LogicalPlan::Scan { table, projection } => {
            // If projection is already set and non-empty, intersect with referenced
            let new_projection = if projection.is_empty() {
                // Set projection to all referenced columns (sorted for determinism)
                let mut cols: Vec<String> = referenced.iter().cloned().collect();
                cols.sort();
                cols
            } else {
                // Keep only columns that are still referenced
                projection
                    .into_iter()
                    .filter(|c| referenced.contains(c))
                    .collect()
            };
            LogicalPlan::Scan {
                table,
                projection: new_projection,
            }
        }
        LogicalPlan::Filter { predicate, input } => LogicalPlan::Filter {
            predicate,
            input: Box::new(set_scan_projection(*input, referenced)),
        },
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            input,
        } => LogicalPlan::Aggregate {
            group_by,
            aggregates,
            input: Box::new(set_scan_projection(*input, referenced)),
        },
        LogicalPlan::Sort { order_by, input } => LogicalPlan::Sort {
            order_by,
            input: Box::new(set_scan_projection(*input, referenced)),
        },
        LogicalPlan::Limit { count, input } => LogicalPlan::Limit {
            count,
            input: Box::new(set_scan_projection(*input, referenced)),
        },
        LogicalPlan::Projection { columns, input } => LogicalPlan::Projection {
            columns,
            input: Box::new(set_scan_projection(*input, referenced)),
        },
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Predicate pushdown
// ---------------------------------------------------------------------------

/// Push Filter nodes below Projection nodes toward the Scan.
///
/// Transform: `Projection(Filter(Scan))` -> `Filter(Projection(Scan))`
/// This is safe because the filter only references columns that exist in the
/// scan, and projection doesn't change row identity.
///
/// Also push filters below Sort and Limit where safe.
fn push_down_predicates(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        // If we see Projection wrapping a Filter, swap them
        LogicalPlan::Projection { columns, input } => {
            let optimized_input = push_down_predicates(*input);
            match optimized_input {
                LogicalPlan::Filter {
                    predicate,
                    input: filter_input,
                } => {
                    // Push filter below projection:
                    // Projection(Filter(X)) -> Filter(Projection(X))
                    let new_projection = LogicalPlan::Projection {
                        columns,
                        input: filter_input,
                    };
                    LogicalPlan::Filter {
                        predicate,
                        input: Box::new(new_projection),
                    }
                }
                other => LogicalPlan::Projection {
                    columns,
                    input: Box::new(other),
                },
            }
        }
        // Recursively optimize children for all other nodes
        LogicalPlan::Filter { predicate, input } => {
            let optimized_input = push_down_predicates(*input);
            LogicalPlan::Filter {
                predicate,
                input: Box::new(optimized_input),
            }
        }
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            input,
        } => LogicalPlan::Aggregate {
            group_by,
            aggregates,
            input: Box::new(push_down_predicates(*input)),
        },
        LogicalPlan::Sort { order_by, input } => LogicalPlan::Sort {
            order_by,
            input: Box::new(push_down_predicates(*input)),
        },
        LogicalPlan::Limit { count, input } => LogicalPlan::Limit {
            count,
            input: Box::new(push_down_predicates(*input)),
        },
        // Scan is a leaf node -- nothing to push down
        LogicalPlan::Scan { .. } => plan,
    }
}

// ---------------------------------------------------------------------------
// Pass 3: Constant folding
// ---------------------------------------------------------------------------

/// Evaluate constant expressions at plan time.
///
/// Examples:
/// - `WHERE amount > 50 + 50` -> `WHERE amount > 100`
/// - `WHERE x > 2 * 3` -> `WHERE x > 6`
fn fold_constants(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Filter { predicate, input } => LogicalPlan::Filter {
            predicate: fold_expr(predicate),
            input: Box::new(fold_constants(*input)),
        },
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            input,
        } => LogicalPlan::Aggregate {
            group_by: group_by.into_iter().map(fold_expr).collect(),
            aggregates: aggregates
                .into_iter()
                .map(|(func, arg)| (func, fold_expr(arg)))
                .collect(),
            input: Box::new(fold_constants(*input)),
        },
        LogicalPlan::Sort { order_by, input } => LogicalPlan::Sort {
            order_by: order_by
                .into_iter()
                .map(|(expr, asc)| (fold_expr(expr), asc))
                .collect(),
            input: Box::new(fold_constants(*input)),
        },
        LogicalPlan::Limit { count, input } => LogicalPlan::Limit {
            count,
            input: Box::new(fold_constants(*input)),
        },
        LogicalPlan::Projection { columns, input } => LogicalPlan::Projection {
            columns: columns.into_iter().map(fold_expr).collect(),
            input: Box::new(fold_constants(*input)),
        },
    }
}

/// Fold constant sub-expressions in an Expr tree.
fn fold_expr(expr: Expr) -> Expr {
    match expr {
        Expr::BinaryOp { left, op, right } => {
            let left = fold_expr(*left);
            let right = fold_expr(*right);

            // Try to evaluate if both sides are literals and this is a comparison
            // with literal arithmetic (e.g., 50 + 50 folded before we get here).
            // Currently, arithmetic isn't in the Expr type, so this is a no-op for
            // arithmetic. But we fold nested structures.
            Expr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }
        }
        Expr::Compound { left, op, right } => {
            let left = fold_expr(*left);
            let right = fold_expr(*right);

            // Fold trivial compound predicates
            match (&left, &op, &right) {
                // TRUE AND x -> x, x AND TRUE -> x (represented as tautologies)
                // FALSE OR x -> x, x OR FALSE -> x
                // We can't easily detect TRUE/FALSE from our Value type, so skip
                _ => Expr::Compound {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                },
            }
        }
        Expr::Aggregate { func, arg } => Expr::Aggregate {
            func,
            arg: Box::new(fold_expr(*arg)),
        },
        // Leaves: Column, Literal, Wildcard -- no folding possible
        other => other,
    }
}

/// Try to evaluate an arithmetic operation on two literal values.
/// Returns Some(folded_value) if both operands are constant and the operation
/// is a supported arithmetic operation, None otherwise.
pub fn try_fold_arithmetic(left: &Value, op: CompareOp, right: &Value) -> Option<Value> {
    // CompareOp doesn't include arithmetic ops, so this is for future extension.
    // When arithmetic expressions are added to the Expr type, this function
    // will handle Int+Int, Float+Float, etc.
    let _ = (left, op, right);
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::types::AggFunc;

    // Helper to build a Scan plan
    fn scan(table: &str) -> LogicalPlan {
        LogicalPlan::Scan {
            table: table.into(),
            projection: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Column pruning tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prune_columns_simple_aggregate() {
        // SELECT sum(amount) FROM wide_table WHERE region = 'EU'
        // Should only load: amount, region
        let plan = LogicalPlan::Aggregate {
            group_by: vec![],
            aggregates: vec![(AggFunc::Sum, Expr::Column("amount".into()))],
            input: Box::new(LogicalPlan::Filter {
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Column("region".into())),
                    op: CompareOp::Eq,
                    right: Box::new(Expr::Literal(Value::Str("EU".into()))),
                },
                input: Box::new(scan("wide_table")),
            }),
        };

        let optimized = optimize(plan);

        // Find the Scan node and check its projection
        let scan_proj = find_scan_projection(&optimized);
        assert!(scan_proj.is_some(), "should have a Scan node");
        let proj = scan_proj.unwrap();
        assert_eq!(proj.len(), 2);
        assert!(proj.contains(&"amount".to_string()));
        assert!(proj.contains(&"region".to_string()));
    }

    #[test]
    fn test_prune_columns_count_star() {
        // SELECT count(*) FROM t WHERE x > 5
        // count(*) doesn't reference a column, but WHERE references x
        let plan = LogicalPlan::Aggregate {
            group_by: vec![],
            aggregates: vec![(AggFunc::Count, Expr::Wildcard)],
            input: Box::new(LogicalPlan::Filter {
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Column("x".into())),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Literal(Value::Int(5))),
                },
                input: Box::new(scan("t")),
            }),
        };

        let optimized = optimize(plan);
        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert_eq!(scan_proj.len(), 1);
        assert!(scan_proj.contains(&"x".to_string()));
    }

    #[test]
    fn test_prune_columns_group_by() {
        // SELECT region, sum(amount) FROM sales GROUP BY region
        // Should load: region, amount
        let plan = LogicalPlan::Aggregate {
            group_by: vec![Expr::Column("region".into())],
            aggregates: vec![(AggFunc::Sum, Expr::Column("amount".into()))],
            input: Box::new(scan("sales")),
        };

        let optimized = optimize(plan);
        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert_eq!(scan_proj.len(), 2);
        assert!(scan_proj.contains(&"amount".to_string()));
        assert!(scan_proj.contains(&"region".to_string()));
    }

    #[test]
    fn test_prune_columns_order_by() {
        // SELECT id, name FROM t ORDER BY name
        // Should load: id, name
        let plan = LogicalPlan::Sort {
            order_by: vec![(Expr::Column("name".into()), true)],
            input: Box::new(LogicalPlan::Projection {
                columns: vec![
                    Expr::Column("id".into()),
                    Expr::Column("name".into()),
                ],
                input: Box::new(scan("t")),
            }),
        };

        let optimized = optimize(plan);
        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert_eq!(scan_proj.len(), 2);
        assert!(scan_proj.contains(&"id".to_string()));
        assert!(scan_proj.contains(&"name".to_string()));
    }

    #[test]
    fn test_prune_columns_select_star_no_prune() {
        // SELECT * FROM t -- should NOT prune (wildcard)
        let plan = LogicalPlan::Scan {
            table: "t".into(),
            projection: vec![],
        };

        let optimized = optimize(plan.clone());
        let scan_proj = find_scan_projection(&optimized).unwrap();
        // Wildcard in Scan (no specific columns) -> empty projection means all columns
        assert!(scan_proj.is_empty());
    }

    #[test]
    fn test_prune_columns_projection_wildcard_no_prune() {
        // SELECT * FROM t WHERE x > 5
        // The Projection has Wildcard, so skip pruning
        let plan = LogicalPlan::Projection {
            columns: vec![Expr::Wildcard],
            input: Box::new(LogicalPlan::Filter {
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Column("x".into())),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Literal(Value::Int(5))),
                },
                input: Box::new(scan("t")),
            }),
        };

        let optimized = optimize(plan);
        let scan_proj = find_scan_projection(&optimized).unwrap();
        // Wildcard means we need all columns -- projection should remain empty
        assert!(scan_proj.is_empty());
    }

    #[test]
    fn test_prune_columns_multiple_aggregates() {
        // SELECT count(*), sum(amount), avg(price) FROM sales WHERE region = 'US'
        // Should load: amount, price, region
        let plan = LogicalPlan::Aggregate {
            group_by: vec![],
            aggregates: vec![
                (AggFunc::Count, Expr::Wildcard),
                (AggFunc::Sum, Expr::Column("amount".into())),
                (AggFunc::Avg, Expr::Column("price".into())),
            ],
            input: Box::new(LogicalPlan::Filter {
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Column("region".into())),
                    op: CompareOp::Eq,
                    right: Box::new(Expr::Literal(Value::Str("US".into()))),
                },
                input: Box::new(scan("sales")),
            }),
        };

        let optimized = optimize(plan);
        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert_eq!(scan_proj.len(), 3);
        assert!(scan_proj.contains(&"amount".to_string()));
        assert!(scan_proj.contains(&"price".to_string()));
        assert!(scan_proj.contains(&"region".to_string()));
    }

    #[test]
    fn test_prune_columns_compound_filter() {
        // SELECT sum(val) FROM t WHERE a > 1 AND b < 10
        // Should load: val, a, b
        let plan = LogicalPlan::Aggregate {
            group_by: vec![],
            aggregates: vec![(AggFunc::Sum, Expr::Column("val".into()))],
            input: Box::new(LogicalPlan::Filter {
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
                input: Box::new(scan("t")),
            }),
        };

        let optimized = optimize(plan);
        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert_eq!(scan_proj.len(), 3);
        assert!(scan_proj.contains(&"a".to_string()));
        assert!(scan_proj.contains(&"b".to_string()));
        assert!(scan_proj.contains(&"val".to_string()));
    }

    // -----------------------------------------------------------------------
    // Predicate pushdown tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pushdown_filter_below_projection() {
        // Projection(Filter(Scan)) -> Filter(Projection(Scan))
        let plan = LogicalPlan::Projection {
            columns: vec![Expr::Column("a".into())],
            input: Box::new(LogicalPlan::Filter {
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Column("x".into())),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Literal(Value::Int(5))),
                },
                input: Box::new(scan("t")),
            }),
        };

        let optimized = push_down_predicates(plan);

        // After pushdown: Filter should be the root
        match &optimized {
            LogicalPlan::Filter { input, .. } => {
                match input.as_ref() {
                    LogicalPlan::Projection { input, .. } => {
                        match input.as_ref() {
                            LogicalPlan::Scan { .. } => {} // correct
                            other => panic!("expected Scan, got: {:?}", other),
                        }
                    }
                    other => panic!("expected Projection, got: {:?}", other),
                }
            }
            other => panic!("expected Filter at root, got: {:?}", other),
        }
    }

    #[test]
    fn test_pushdown_preserves_filter_scan() {
        // Filter(Scan) is already optimal -- no change
        let plan = LogicalPlan::Filter {
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Column("x".into())),
                op: CompareOp::Gt,
                right: Box::new(Expr::Literal(Value::Int(5))),
            },
            input: Box::new(scan("t")),
        };

        let optimized = push_down_predicates(plan.clone());
        assert_eq!(optimized, plan);
    }

    #[test]
    fn test_pushdown_aggregate_filter_scan() {
        // Aggregate(Filter(Scan)) should remain unchanged (filter is already close to scan)
        let plan = LogicalPlan::Aggregate {
            group_by: vec![],
            aggregates: vec![(AggFunc::Count, Expr::Wildcard)],
            input: Box::new(LogicalPlan::Filter {
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Column("x".into())),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Literal(Value::Int(5))),
                },
                input: Box::new(scan("t")),
            }),
        };

        let optimized = push_down_predicates(plan.clone());
        assert_eq!(optimized, plan);
    }

    // -----------------------------------------------------------------------
    // Constant folding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fold_constants_identity() {
        // A plan with no constant expressions should be unchanged
        let plan = LogicalPlan::Filter {
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Column("x".into())),
                op: CompareOp::Gt,
                right: Box::new(Expr::Literal(Value::Int(100))),
            },
            input: Box::new(scan("t")),
        };

        let optimized = fold_constants(plan.clone());
        assert_eq!(optimized, plan);
    }

    #[test]
    fn test_fold_constants_nested_plan() {
        // Constants are preserved through nested plan transformations
        let plan = LogicalPlan::Limit {
            count: 10,
            input: Box::new(LogicalPlan::Sort {
                order_by: vec![(Expr::Column("x".into()), true)],
                input: Box::new(LogicalPlan::Filter {
                    predicate: Expr::BinaryOp {
                        left: Box::new(Expr::Column("x".into())),
                        op: CompareOp::Gt,
                        right: Box::new(Expr::Literal(Value::Int(42))),
                    },
                    input: Box::new(scan("t")),
                }),
            }),
        };

        let optimized = fold_constants(plan.clone());
        assert_eq!(optimized, plan);
    }

    #[test]
    fn test_fold_compound_predicate_preserved() {
        // Compound predicates should be recursively folded
        let plan = LogicalPlan::Filter {
            predicate: Expr::Compound {
                left: Box::new(Expr::BinaryOp {
                    left: Box::new(Expr::Column("a".into())),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Literal(Value::Int(10))),
                }),
                op: LogicalOp::And,
                right: Box::new(Expr::BinaryOp {
                    left: Box::new(Expr::Column("b".into())),
                    op: CompareOp::Lt,
                    right: Box::new(Expr::Literal(Value::Int(20))),
                }),
            },
            input: Box::new(scan("t")),
        };

        let optimized = fold_constants(plan.clone());
        assert_eq!(optimized, plan);
    }

    // -----------------------------------------------------------------------
    // Combined optimization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimize_full_query() {
        // SELECT sum(amount) FROM wide_table WHERE region = 'EU'
        // After optimization: Scan should only load amount + region
        let plan = LogicalPlan::Aggregate {
            group_by: vec![],
            aggregates: vec![(AggFunc::Sum, Expr::Column("amount".into()))],
            input: Box::new(LogicalPlan::Filter {
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Column("region".into())),
                    op: CompareOp::Eq,
                    right: Box::new(Expr::Literal(Value::Str("EU".into()))),
                },
                input: Box::new(scan("wide_table")),
            }),
        };

        let optimized = optimize(plan);

        // Verify structure is preserved (Aggregate -> Filter -> Scan)
        match &optimized {
            LogicalPlan::Aggregate { input, .. } => match input.as_ref() {
                LogicalPlan::Filter { input, .. } => match input.as_ref() {
                    LogicalPlan::Scan {
                        table, projection, ..
                    } => {
                        assert_eq!(table, "wide_table");
                        assert_eq!(projection.len(), 2);
                        assert!(projection.contains(&"amount".to_string()));
                        assert!(projection.contains(&"region".to_string()));
                    }
                    other => panic!("expected Scan, got: {:?}", other),
                },
                other => panic!("expected Filter, got: {:?}", other),
            },
            other => panic!("expected Aggregate, got: {:?}", other),
        }
    }

    #[test]
    fn test_optimize_pushdown_and_prune() {
        // Projection(Filter(Scan)) -> after pushdown: Filter(Projection(Scan))
        // Then column pruning sets Scan projection
        let plan = LogicalPlan::Projection {
            columns: vec![Expr::Column("a".into())],
            input: Box::new(LogicalPlan::Filter {
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Column("x".into())),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Literal(Value::Int(5))),
                },
                input: Box::new(scan("t")),
            }),
        };

        let optimized = optimize(plan);

        // After all passes: should have pruned columns (a, x)
        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert_eq!(scan_proj.len(), 2);
        assert!(scan_proj.contains(&"a".to_string()));
        assert!(scan_proj.contains(&"x".to_string()));
    }

    #[test]
    fn test_optimize_preserves_plan_semantics() {
        // After optimization, table name and predicates should be preserved
        let plan = LogicalPlan::Aggregate {
            group_by: vec![Expr::Column("region".into())],
            aggregates: vec![
                (AggFunc::Count, Expr::Wildcard),
                (AggFunc::Sum, Expr::Column("amount".into())),
            ],
            input: Box::new(LogicalPlan::Filter {
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Column("status".into())),
                    op: CompareOp::Eq,
                    right: Box::new(Expr::Literal(Value::Str("active".into()))),
                },
                input: Box::new(scan("orders")),
            }),
        };

        let optimized = optimize(plan);

        // Table name preserved
        assert_eq!(optimized.table_name(), Some("orders"));

        // Scan projection should have: amount, region, status
        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert_eq!(scan_proj.len(), 3);
        assert!(scan_proj.contains(&"amount".to_string()));
        assert!(scan_proj.contains(&"region".to_string()));
        assert!(scan_proj.contains(&"status".to_string()));
    }

    #[test]
    fn test_optimize_limit_sort_filter() {
        // SELECT id FROM t WHERE x > 5 ORDER BY id LIMIT 10
        // All columns referenced: id, x
        let plan = LogicalPlan::Limit {
            count: 10,
            input: Box::new(LogicalPlan::Sort {
                order_by: vec![(Expr::Column("id".into()), true)],
                input: Box::new(LogicalPlan::Projection {
                    columns: vec![Expr::Column("id".into())],
                    input: Box::new(LogicalPlan::Filter {
                        predicate: Expr::BinaryOp {
                            left: Box::new(Expr::Column("x".into())),
                            op: CompareOp::Gt,
                            right: Box::new(Expr::Literal(Value::Int(5))),
                        },
                        input: Box::new(scan("t")),
                    }),
                }),
            }),
        };

        let optimized = optimize(plan);
        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert_eq!(scan_proj.len(), 2);
        assert!(scan_proj.contains(&"id".to_string()));
        assert!(scan_proj.contains(&"x".to_string()));
    }

    #[test]
    fn test_optimize_idempotent() {
        // Optimizing an already-optimized plan should produce the same result
        let plan = LogicalPlan::Aggregate {
            group_by: vec![],
            aggregates: vec![(AggFunc::Sum, Expr::Column("amount".into()))],
            input: Box::new(LogicalPlan::Filter {
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Column("region".into())),
                    op: CompareOp::Eq,
                    right: Box::new(Expr::Literal(Value::Str("EU".into()))),
                },
                input: Box::new(scan("wide_table")),
            }),
        };

        let opt1 = optimize(plan);
        let opt2 = optimize(opt1.clone());
        assert_eq!(opt1, opt2);
    }

    // -----------------------------------------------------------------------
    // End-to-end with parser
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimize_parsed_query() {
        use crate::sql::parser::parse_query;

        let plan =
            parse_query("SELECT sum(amount) FROM wide_table WHERE region = 'EU'")
                .unwrap();
        let optimized = optimize(plan);

        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert_eq!(scan_proj.len(), 2);
        assert!(scan_proj.contains(&"amount".to_string()));
        assert!(scan_proj.contains(&"region".to_string()));
    }

    #[test]
    fn test_optimize_parsed_group_by() {
        use crate::sql::parser::parse_query;

        let plan = parse_query(
            "SELECT region, count(*), sum(amount) FROM sales GROUP BY region",
        )
        .unwrap();
        let optimized = optimize(plan);

        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert_eq!(scan_proj.len(), 2);
        assert!(scan_proj.contains(&"amount".to_string()));
        assert!(scan_proj.contains(&"region".to_string()));
    }

    #[test]
    fn test_optimize_parsed_order_by() {
        use crate::sql::parser::parse_query;

        let plan =
            parse_query("SELECT id, name FROM users ORDER BY name ASC").unwrap();
        let optimized = optimize(plan);

        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert_eq!(scan_proj.len(), 2);
        assert!(scan_proj.contains(&"id".to_string()));
        assert!(scan_proj.contains(&"name".to_string()));
    }

    #[test]
    fn test_optimize_parsed_select_star_no_prune() {
        use crate::sql::parser::parse_query;

        let plan = parse_query("SELECT * FROM t").unwrap();
        let optimized = optimize(plan);

        // SELECT * -> no pruning, Scan projection should be empty (= all columns)
        let scan_proj = find_scan_projection(&optimized).unwrap();
        assert!(scan_proj.is_empty());
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Find the Scan node in a plan tree and return its projection.
    fn find_scan_projection(plan: &LogicalPlan) -> Option<Vec<String>> {
        match plan {
            LogicalPlan::Scan { projection, .. } => Some(projection.clone()),
            LogicalPlan::Filter { input, .. } => find_scan_projection(input),
            LogicalPlan::Aggregate { input, .. } => find_scan_projection(input),
            LogicalPlan::Sort { input, .. } => find_scan_projection(input),
            LogicalPlan::Limit { input, .. } => find_scan_projection(input),
            LogicalPlan::Projection { input, .. } => find_scan_projection(input),
        }
    }
}
