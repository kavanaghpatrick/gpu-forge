//! Core type definitions for the SQL engine.
//!
//! These types form the intermediate representation between the sqlparser AST
//! and our logical/physical plans. They are independent of both the parser
//! library and the GPU kernel types.

use std::fmt;

/// Data types supported in SQL expressions and results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Int64,
    Float64,
    Varchar,
    Boolean,
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Int64 => write!(f, "INT64"),
            DataType::Float64 => write!(f, "FLOAT64"),
            DataType::Varchar => write!(f, "VARCHAR"),
            DataType::Boolean => write!(f, "BOOLEAN"),
        }
    }
}

/// Literal values in SQL expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
    Null,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(v) => write!(f, "{}", v),
            Value::Float(v) => write!(f, "{}", v),
            Value::Str(v) => write!(f, "'{}'", v),
            Value::Null => write!(f, "NULL"),
        }
    }
}

/// Comparison operators for WHERE predicates.
///
/// Values match the GPU filter kernel's `compare_op` function constant
/// (see `CompareOp` in `gpu::pipeline`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CompareOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

impl fmt::Display for CompareOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareOp::Eq => write!(f, "="),
            CompareOp::Ne => write!(f, "!="),
            CompareOp::Lt => write!(f, "<"),
            CompareOp::Le => write!(f, "<="),
            CompareOp::Gt => write!(f, ">"),
            CompareOp::Ge => write!(f, ">="),
        }
    }
}

/// Aggregate functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

impl fmt::Display for AggFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggFunc::Count => write!(f, "COUNT"),
            AggFunc::Sum => write!(f, "SUM"),
            AggFunc::Avg => write!(f, "AVG"),
            AggFunc::Min => write!(f, "MIN"),
            AggFunc::Max => write!(f, "MAX"),
        }
    }
}

impl AggFunc {
    /// Convert to the GPU aggregation kernel's function code.
    /// Matches `AggParams.agg_function` values.
    pub fn to_gpu_code(self) -> u32 {
        match self {
            AggFunc::Count => 0,
            AggFunc::Sum => 1,
            AggFunc::Avg => 2,
            AggFunc::Min => 3,
            AggFunc::Max => 4,
        }
    }
}

/// Logical operator for compound predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOp {
    And,
    Or,
}

/// SQL expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Column reference by name.
    Column(String),
    /// Literal value.
    Literal(Value),
    /// Binary comparison: left op right.
    BinaryOp {
        left: Box<Expr>,
        op: CompareOp,
        right: Box<Expr>,
    },
    /// Compound predicate: left AND/OR right.
    Compound {
        left: Box<Expr>,
        op: LogicalOp,
        right: Box<Expr>,
    },
    /// Aggregate function call.
    Aggregate {
        func: AggFunc,
        /// The argument expression. For COUNT(*), this is `Expr::Wildcard`.
        arg: Box<Expr>,
    },
    /// Wildcard (*) in SELECT or COUNT(*).
    Wildcard,
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(name) => write!(f, "{}", name),
            Expr::Literal(val) => write!(f, "{}", val),
            Expr::BinaryOp { left, op, right } => write!(f, "{} {} {}", left, op, right),
            Expr::Compound { left, op, right } => {
                let op_str = match op {
                    LogicalOp::And => "AND",
                    LogicalOp::Or => "OR",
                };
                write!(f, "({} {} {})", left, op_str, right)
            }
            Expr::Aggregate { func, arg } => write!(f, "{}({})", func, arg),
            Expr::Wildcard => write!(f, "*"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_display() {
        assert_eq!(DataType::Int64.to_string(), "INT64");
        assert_eq!(DataType::Float64.to_string(), "FLOAT64");
        assert_eq!(DataType::Varchar.to_string(), "VARCHAR");
        assert_eq!(DataType::Boolean.to_string(), "BOOLEAN");
    }

    #[test]
    fn test_value_display() {
        assert_eq!(Value::Int(42).to_string(), "42");
        #[allow(clippy::approx_constant)]
        let pi_approx = 3.14;
        assert_eq!(Value::Float(pi_approx).to_string(), "3.14");
        assert_eq!(Value::Str("hello".into()).to_string(), "'hello'");
        assert_eq!(Value::Null.to_string(), "NULL");
    }

    #[test]
    fn test_compare_op_gpu_values() {
        assert_eq!(CompareOp::Eq as u32, 0);
        assert_eq!(CompareOp::Ne as u32, 1);
        assert_eq!(CompareOp::Lt as u32, 2);
        assert_eq!(CompareOp::Le as u32, 3);
        assert_eq!(CompareOp::Gt as u32, 4);
        assert_eq!(CompareOp::Ge as u32, 5);
    }

    #[test]
    fn test_agg_func_gpu_codes() {
        assert_eq!(AggFunc::Count.to_gpu_code(), 0);
        assert_eq!(AggFunc::Sum.to_gpu_code(), 1);
        assert_eq!(AggFunc::Avg.to_gpu_code(), 2);
        assert_eq!(AggFunc::Min.to_gpu_code(), 3);
        assert_eq!(AggFunc::Max.to_gpu_code(), 4);
    }

    #[test]
    fn test_expr_display() {
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Column("amount".into())),
            op: CompareOp::Gt,
            right: Box::new(Expr::Literal(Value::Int(100))),
        };
        assert_eq!(expr.to_string(), "amount > 100");
    }

    #[test]
    fn test_aggregate_display() {
        let expr = Expr::Aggregate {
            func: AggFunc::Count,
            arg: Box::new(Expr::Wildcard),
        };
        assert_eq!(expr.to_string(), "COUNT(*)");
    }

    #[test]
    fn test_compound_display() {
        let expr = Expr::Compound {
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
        };
        assert_eq!(expr.to_string(), "(a > 1 AND b < 10)");
    }
}
