//! JIT Metal shader compiler with plan-structure caching.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::sql::physical_plan::PhysicalPlan;

/// Compute a structure-only hash of a physical plan.
///
/// The hash captures plan node types, operators, column references, aggregate
/// functions, and GROUP BY columns. It deliberately does NOT hash literal values
/// (filter thresholds), so two plans that differ only in constants produce the
/// same hash. This enables JIT PSO cache reuse across parameter changes.
pub fn plan_structure_hash(plan: &PhysicalPlan) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_plan_node(plan, &mut hasher);
    hasher.finish()
}

/// Recursively hash a plan node's structure into the hasher.
fn hash_plan_node(plan: &PhysicalPlan, hasher: &mut DefaultHasher) {
    match plan {
        PhysicalPlan::GpuScan { table, columns } => {
            "GpuScan".hash(hasher);
            table.hash(hasher);
            for col in columns {
                col.hash(hasher);
            }
        }

        PhysicalPlan::GpuFilter {
            compare_op,
            column,
            value: _, // deliberately skip literal value
            input,
        } => {
            "GpuFilter".hash(hasher);
            compare_op.hash(hasher);
            column.hash(hasher);
            // Do NOT hash value -- same structure with different literals shares PSO
            hash_plan_node(input, hasher);
        }

        PhysicalPlan::GpuCompoundFilter { op, left, right } => {
            "GpuCompoundFilter".hash(hasher);
            // LogicalOp doesn't derive Hash, so discriminate manually
            match op {
                crate::sql::types::LogicalOp::And => 0u8.hash(hasher),
                crate::sql::types::LogicalOp::Or => 1u8.hash(hasher),
            }
            hash_plan_node(left, hasher);
            hash_plan_node(right, hasher);
        }

        PhysicalPlan::GpuAggregate {
            functions,
            group_by,
            input,
        } => {
            "GpuAggregate".hash(hasher);
            for (func, col) in functions {
                func.hash(hasher);
                col.hash(hasher);
            }
            for col in group_by {
                col.hash(hasher);
            }
            hash_plan_node(input, hasher);
        }

        PhysicalPlan::GpuSort { order_by, input } => {
            "GpuSort".hash(hasher);
            for (col, asc) in order_by {
                col.hash(hasher);
                asc.hash(hasher);
            }
            hash_plan_node(input, hasher);
        }

        PhysicalPlan::GpuLimit { count, input } => {
            "GpuLimit".hash(hasher);
            count.hash(hasher);
            hash_plan_node(input, hasher);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::types::{AggFunc, CompareOp, LogicalOp, Value};

    /// Helper: build a scan node.
    fn scan(table: &str) -> PhysicalPlan {
        PhysicalPlan::GpuScan {
            table: table.into(),
            columns: vec!["amount".into(), "region".into()],
        }
    }

    /// Helper: build a filter node (column > value).
    fn filter_gt(column: &str, val: Value, input: PhysicalPlan) -> PhysicalPlan {
        PhysicalPlan::GpuFilter {
            compare_op: CompareOp::Gt,
            column: column.into(),
            value: val,
            input: Box::new(input),
        }
    }

    /// Helper: build a compound AND filter.
    fn compound_and(left: PhysicalPlan, right: PhysicalPlan) -> PhysicalPlan {
        PhysicalPlan::GpuCompoundFilter {
            op: LogicalOp::And,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Helper: build an aggregate node.
    fn aggregate(
        functions: Vec<(AggFunc, &str)>,
        group_by: Vec<&str>,
        input: PhysicalPlan,
    ) -> PhysicalPlan {
        PhysicalPlan::GpuAggregate {
            functions: functions
                .into_iter()
                .map(|(f, c)| (f, c.to_string()))
                .collect(),
            group_by: group_by.into_iter().map(|s| s.to_string()).collect(),
            input: Box::new(input),
        }
    }

    // -----------------------------------------------------------------------
    // Test: deterministic -- same plan always produces the same hash
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_deterministic() {
        let plan = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(500), scan("sales")),
        );
        let h1 = plan_structure_hash(&plan);
        let h2 = plan_structure_hash(&plan);
        assert_eq!(h1, h2, "same plan must produce same hash");
    }

    // -----------------------------------------------------------------------
    // Test: structural equality -- different literals produce the same hash
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_structural_equality_different_literals() {
        let plan_a = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(500), scan("sales")),
        );
        let plan_b = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(300), scan("sales")),
        );
        assert_eq!(
            plan_structure_hash(&plan_a),
            plan_structure_hash(&plan_b),
            "different literal values must produce same hash"
        );
    }

    // -----------------------------------------------------------------------
    // Test: different structure produces different hash
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_different_structure() {
        // Plan A: COUNT(*) with filter
        let plan_a = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(500), scan("sales")),
        );
        // Plan B: COUNT(*) without filter
        let plan_b = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        assert_ne!(
            plan_structure_hash(&plan_a),
            plan_structure_hash(&plan_b),
            "structurally different plans must produce different hashes"
        );
    }

    // -----------------------------------------------------------------------
    // Test: filter compare_op matters
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_filter_op_matters() {
        let plan_gt = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(500), scan("sales")),
        );
        let plan_lt = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            PhysicalPlan::GpuFilter {
                compare_op: CompareOp::Lt,
                column: "amount".into(),
                value: Value::Int(500),
                input: Box::new(scan("sales")),
            },
        );
        assert_ne!(
            plan_structure_hash(&plan_gt),
            plan_structure_hash(&plan_lt),
            "different compare_op must produce different hash"
        );
    }

    // -----------------------------------------------------------------------
    // Test: aggregate function type matters
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_agg_func_matters() {
        let plan_count = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let plan_sum = aggregate(vec![(AggFunc::Sum, "amount")], vec![], scan("sales"));
        assert_ne!(
            plan_structure_hash(&plan_count),
            plan_structure_hash(&plan_sum),
            "different agg function must produce different hash"
        );
    }

    // -----------------------------------------------------------------------
    // Test: GROUP BY column matters
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_group_by_matters() {
        let plan_no_group = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let plan_group = aggregate(
            vec![(AggFunc::Count, "*")],
            vec!["region"],
            scan("sales"),
        );
        assert_ne!(
            plan_structure_hash(&plan_no_group),
            plan_structure_hash(&plan_group),
            "GROUP BY vs no GROUP BY must produce different hash"
        );
    }

    // -----------------------------------------------------------------------
    // Test: compound filter hashes child ops
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_compound_filter() {
        let left = filter_gt("amount", Value::Int(100), scan("sales"));
        let right = PhysicalPlan::GpuFilter {
            compare_op: CompareOp::Lt,
            column: "amount".into(),
            value: Value::Int(1000),
            input: Box::new(scan("sales")),
        };
        let compound = compound_and(left, right);
        let plan_a = aggregate(vec![(AggFunc::Count, "*")], vec![], compound);

        // Same structure, different literals
        let left2 = filter_gt("amount", Value::Int(200), scan("sales"));
        let right2 = PhysicalPlan::GpuFilter {
            compare_op: CompareOp::Lt,
            column: "amount".into(),
            value: Value::Int(2000),
            input: Box::new(scan("sales")),
        };
        let compound2 = compound_and(left2, right2);
        let plan_b = aggregate(vec![(AggFunc::Count, "*")], vec![], compound2);

        assert_eq!(
            plan_structure_hash(&plan_a),
            plan_structure_hash(&plan_b),
            "compound filters with same structure but different literals must match"
        );
    }

    // -----------------------------------------------------------------------
    // Test: filter column reference matters
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_filter_column_matters() {
        let plan_a = filter_gt("amount", Value::Int(500), scan("sales"));
        let plan_b = filter_gt("price", Value::Int(500), scan("sales"));
        assert_ne!(
            plan_structure_hash(&plan_a),
            plan_structure_hash(&plan_b),
            "different filter column must produce different hash"
        );
    }
}
