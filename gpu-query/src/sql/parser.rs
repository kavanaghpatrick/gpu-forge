//! SQL parser wrapping `sqlparser-rs` for the gpu-query MVP SQL subset.
//!
//! Converts a SQL string into our `LogicalPlan` representation. Only a subset
//! of SQL is supported -- see module-level doc for `sql::mod.rs`.

use sqlparser::ast::{
    self as sp, Expr as SpExpr, FunctionArg, FunctionArgExpr, GroupByExpr,
    SelectItem, SetExpr, Statement, TableFactor,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use super::logical_plan::LogicalPlan;
use super::types::{AggFunc, CompareOp, Expr, LogicalOp, Value};

/// Errors that can occur during SQL parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// sqlparser returned an error.
    SqlParser(String),
    /// The SQL statement is not a SELECT query.
    NotASelect,
    /// Unsupported SQL feature.
    Unsupported(String),
    /// Missing FROM clause.
    MissingFrom,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::SqlParser(msg) => write!(f, "SQL parse error: {}", msg),
            ParseError::NotASelect => write!(f, "only SELECT statements are supported"),
            ParseError::Unsupported(msg) => write!(f, "unsupported SQL: {}", msg),
            ParseError::MissingFrom => write!(f, "missing FROM clause"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse a SQL query string into a LogicalPlan.
///
/// Only SELECT statements are supported. The parser handles:
/// - Column references and wildcards
/// - Aggregate functions: COUNT, SUM, AVG, MIN, MAX (including COUNT(*))
/// - WHERE with comparison predicates and AND/OR compounds
/// - GROUP BY column list
/// - ORDER BY column ASC/DESC
/// - LIMIT integer
pub fn parse_query(sql: &str) -> Result<LogicalPlan, ParseError> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql).map_err(|e| ParseError::SqlParser(e.to_string()))?;

    if statements.len() != 1 {
        return Err(ParseError::Unsupported(format!(
            "expected exactly one statement, got {}",
            statements.len()
        )));
    }

    let statement = &statements[0];
    match statement {
        Statement::Query(query) => convert_query(query),
        _ => Err(ParseError::NotASelect),
    }
}

/// Convert a sqlparser Query to our LogicalPlan.
fn convert_query(query: &sp::Query) -> Result<LogicalPlan, ParseError> {
    let body = query.body.as_ref();

    let select = match body {
        SetExpr::Select(select) => select.as_ref(),
        _ => {
            return Err(ParseError::Unsupported(
                "only simple SELECT queries are supported (no UNION, INTERSECT, etc.)".into(),
            ))
        }
    };

    // 1. FROM clause -> Scan node
    let table_name = extract_table_name(select)?;
    let mut plan = LogicalPlan::Scan {
        table: table_name,
        projection: vec![],
    };

    // 2. WHERE clause -> Filter node
    if let Some(selection) = &select.selection {
        let predicate = convert_expr(selection)?;
        plan = LogicalPlan::Filter {
            predicate,
            input: Box::new(plan),
        };
    }

    // 3. Analyze SELECT items for aggregates and column references
    let (select_exprs, has_aggregates) = convert_select_items(&select.projection)?;

    // 4. GROUP BY -> Aggregate node (or implicit aggregation if agg functions present)
    let group_by_exprs = convert_group_by(&select.group_by)?;

    if has_aggregates || !group_by_exprs.is_empty() {
        // Extract aggregate functions from select expressions
        let mut aggregates = Vec::new();
        let mut _non_agg_columns: Vec<Expr> = Vec::new();
        for expr in &select_exprs {
            collect_aggregates(expr, &mut aggregates);
        }

        plan = LogicalPlan::Aggregate {
            group_by: group_by_exprs,
            aggregates,
            input: Box::new(plan),
        };
    } else {
        // Non-aggregate query: add Projection if not just *
        let has_wildcard = select_exprs.iter().any(|e| matches!(e, Expr::Wildcard));
        if !has_wildcard {
            plan = LogicalPlan::Projection {
                columns: select_exprs.clone(),
                input: Box::new(plan),
            };
        }
    }

    // 5. ORDER BY -> Sort node
    if let Some(order_by) = &query.order_by {
        match order_by {
            sp::OrderBy { exprs, .. } => {
                if !exprs.is_empty() {
                    let order_exprs = exprs
                        .iter()
                        .map(|o| {
                            let expr = convert_expr(&o.expr)?;
                            let asc = o.asc.unwrap_or(true);
                            Ok((expr, asc))
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    plan = LogicalPlan::Sort {
                        order_by: order_exprs,
                        input: Box::new(plan),
                    };
                }
            }
        }
    }

    // 6. LIMIT -> Limit node
    if let Some(limit_expr) = &query.limit {
        let count = extract_limit_value(limit_expr)?;
        plan = LogicalPlan::Limit {
            count,
            input: Box::new(plan),
        };
    }

    Ok(plan)
}

/// Extract the single table name from the FROM clause.
fn extract_table_name(select: &sp::Select) -> Result<String, ParseError> {
    if select.from.is_empty() {
        return Err(ParseError::MissingFrom);
    }
    if select.from.len() > 1 {
        return Err(ParseError::Unsupported("multiple FROM tables (joins) not supported".into()));
    }

    let table_with_joins = &select.from[0];
    if !table_with_joins.joins.is_empty() {
        return Err(ParseError::Unsupported("JOINs not supported".into()));
    }

    match &table_with_joins.relation {
        TableFactor::Table { name, .. } => {
            // name is ObjectName which is a Vec<Ident>
            let parts: Vec<String> = name.0.iter().map(|ident| ident.value.clone()).collect();
            Ok(parts.join("."))
        }
        _ => Err(ParseError::Unsupported(
            "only simple table references are supported in FROM".into(),
        )),
    }
}

/// Convert SELECT items to our Expr types, also detecting if aggregates are present.
fn convert_select_items(
    items: &[SelectItem],
) -> Result<(Vec<Expr>, bool), ParseError> {
    let mut exprs = Vec::new();
    let mut has_aggregates = false;

    for item in items {
        match item {
            SelectItem::UnnamedExpr(expr) => {
                let converted = convert_expr(expr)?;
                if contains_aggregate(&converted) {
                    has_aggregates = true;
                }
                exprs.push(converted);
            }
            SelectItem::ExprWithAlias { expr, .. } => {
                let converted = convert_expr(expr)?;
                if contains_aggregate(&converted) {
                    has_aggregates = true;
                }
                exprs.push(converted);
            }
            SelectItem::Wildcard(_) => {
                exprs.push(Expr::Wildcard);
            }
            SelectItem::QualifiedWildcard(_, _) => {
                exprs.push(Expr::Wildcard);
            }
        }
    }

    Ok((exprs, has_aggregates))
}

/// Check if an expression contains any aggregate function calls.
fn contains_aggregate(expr: &Expr) -> bool {
    match expr {
        Expr::Aggregate { .. } => true,
        Expr::BinaryOp { left, right, .. } => contains_aggregate(left) || contains_aggregate(right),
        Expr::Compound { left, right, .. } => contains_aggregate(left) || contains_aggregate(right),
        _ => false,
    }
}

/// Collect (AggFunc, arg) pairs from an expression tree.
fn collect_aggregates(expr: &Expr, out: &mut Vec<(AggFunc, Expr)>) {
    match expr {
        Expr::Aggregate { func, arg } => {
            out.push((*func, arg.as_ref().clone()));
        }
        Expr::BinaryOp { left, right, .. } => {
            collect_aggregates(left, out);
            collect_aggregates(right, out);
        }
        Expr::Compound { left, right, .. } => {
            collect_aggregates(left, out);
            collect_aggregates(right, out);
        }
        _ => {}
    }
}

/// Convert GROUP BY clause to our Expr types.
fn convert_group_by(group_by: &GroupByExpr) -> Result<Vec<Expr>, ParseError> {
    match group_by {
        GroupByExpr::All(_) => Err(ParseError::Unsupported("GROUP BY ALL not supported".into())),
        GroupByExpr::Expressions(exprs, _modifiers) => {
            exprs.iter().map(convert_expr).collect()
        }
    }
}

/// Convert a sqlparser expression to our Expr type.
fn convert_expr(expr: &SpExpr) -> Result<Expr, ParseError> {
    match expr {
        // Column reference: just an identifier
        SpExpr::Identifier(ident) => Ok(Expr::Column(ident.value.clone())),

        // Compound identifier (e.g., t.col)
        SpExpr::CompoundIdentifier(parts) => {
            // Use the last part as the column name
            let name = parts
                .last()
                .map(|i| i.value.clone())
                .ok_or_else(|| ParseError::Unsupported("empty compound identifier".into()))?;
            Ok(Expr::Column(name))
        }

        // Numeric literal
        SpExpr::Value(val) => convert_value(val),

        // Unary minus (negative numbers)
        SpExpr::UnaryOp {
            op: sp::UnaryOperator::Minus,
            expr: inner,
        } => {
            let inner_val = convert_expr(inner)?;
            match inner_val {
                Expr::Literal(Value::Int(n)) => Ok(Expr::Literal(Value::Int(-n))),
                Expr::Literal(Value::Float(n)) => Ok(Expr::Literal(Value::Float(-n))),
                _ => Err(ParseError::Unsupported(
                    "unary minus only supported on numeric literals".into(),
                )),
            }
        }

        // Binary operator (comparison or logical)
        SpExpr::BinaryOp { left, op, right } => {
            match op {
                // Logical operators
                sp::BinaryOperator::And => {
                    let l = convert_expr(left)?;
                    let r = convert_expr(right)?;
                    Ok(Expr::Compound {
                        left: Box::new(l),
                        op: LogicalOp::And,
                        right: Box::new(r),
                    })
                }
                sp::BinaryOperator::Or => {
                    let l = convert_expr(left)?;
                    let r = convert_expr(right)?;
                    Ok(Expr::Compound {
                        left: Box::new(l),
                        op: LogicalOp::Or,
                        right: Box::new(r),
                    })
                }
                // Comparison operators
                _ => {
                    let compare_op = convert_binop(op)?;
                    let l = convert_expr(left)?;
                    let r = convert_expr(right)?;
                    Ok(Expr::BinaryOp {
                        left: Box::new(l),
                        op: compare_op,
                        right: Box::new(r),
                    })
                }
            }
        }

        // Function call (aggregate or scalar)
        SpExpr::Function(func) => convert_function(func),

        // Nested expression in parentheses
        SpExpr::Nested(inner) => convert_expr(inner),

        _ => Err(ParseError::Unsupported(format!(
            "expression type not supported: {:?}",
            std::mem::discriminant(expr)
        ))),
    }
}

/// Convert a sqlparser Value to our Value type.
fn convert_value(val: &sp::Value) -> Result<Expr, ParseError> {
    match val {
        sp::Value::Number(s, _) => {
            // Try integer first, then float
            if let Ok(i) = s.parse::<i64>() {
                Ok(Expr::Literal(Value::Int(i)))
            } else if let Ok(f) = s.parse::<f64>() {
                Ok(Expr::Literal(Value::Float(f)))
            } else {
                Err(ParseError::Unsupported(format!(
                    "cannot parse number: {}",
                    s
                )))
            }
        }
        sp::Value::SingleQuotedString(s) => Ok(Expr::Literal(Value::Str(s.clone()))),
        sp::Value::DoubleQuotedString(s) => Ok(Expr::Literal(Value::Str(s.clone()))),
        sp::Value::Null => Ok(Expr::Literal(Value::Null)),
        sp::Value::Boolean(b) => {
            // Store booleans as integers for GPU compatibility
            Ok(Expr::Literal(Value::Int(if *b { 1 } else { 0 })))
        }
        _ => Err(ParseError::Unsupported(format!(
            "value type not supported: {:?}",
            val
        ))),
    }
}

/// Convert a sqlparser binary operator to our CompareOp.
fn convert_binop(op: &sp::BinaryOperator) -> Result<CompareOp, ParseError> {
    match op {
        sp::BinaryOperator::Eq => Ok(CompareOp::Eq),
        sp::BinaryOperator::NotEq => Ok(CompareOp::Ne),
        sp::BinaryOperator::Lt => Ok(CompareOp::Lt),
        sp::BinaryOperator::LtEq => Ok(CompareOp::Le),
        sp::BinaryOperator::Gt => Ok(CompareOp::Gt),
        sp::BinaryOperator::GtEq => Ok(CompareOp::Ge),
        _ => Err(ParseError::Unsupported(format!(
            "binary operator not supported: {:?}",
            op
        ))),
    }
}

/// Convert a sqlparser Function to our aggregate Expr.
fn convert_function(func: &sp::Function) -> Result<Expr, ParseError> {
    let name = func
        .name
        .0
        .iter()
        .map(|i| i.value.to_uppercase())
        .collect::<Vec<_>>()
        .join(".");

    let agg_func = match name.as_str() {
        "COUNT" => AggFunc::Count,
        "SUM" => AggFunc::Sum,
        "AVG" => AggFunc::Avg,
        "MIN" => AggFunc::Min,
        "MAX" => AggFunc::Max,
        _ => {
            return Err(ParseError::Unsupported(format!(
                "function not supported: {}",
                name
            )))
        }
    };

    // Extract the argument
    let args = match &func.args {
        sp::FunctionArguments::None => vec![],
        sp::FunctionArguments::Subquery(_) => {
            return Err(ParseError::Unsupported("subquery arguments not supported".into()));
        }
        sp::FunctionArguments::List(arg_list) => arg_list.args.clone(),
    };

    let arg_expr = if args.is_empty() {
        // COUNT() with no args -> treat as COUNT(*)
        Expr::Wildcard
    } else if args.len() == 1 {
        match &args[0] {
            FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => Expr::Wildcard,
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => convert_expr(expr)?,
            FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_)) => Expr::Wildcard,
            FunctionArg::Named { arg, .. } | FunctionArg::ExprNamed { arg, .. } => match arg {
                FunctionArgExpr::Wildcard => Expr::Wildcard,
                FunctionArgExpr::Expr(expr) => convert_expr(expr)?,
                FunctionArgExpr::QualifiedWildcard(_) => Expr::Wildcard,
            },
        }
    } else {
        return Err(ParseError::Unsupported(format!(
            "aggregate function with {} arguments not supported",
            args.len()
        )));
    };

    Ok(Expr::Aggregate {
        func: agg_func,
        arg: Box::new(arg_expr),
    })
}

/// Extract a LIMIT value from a sqlparser expression.
fn extract_limit_value(expr: &SpExpr) -> Result<usize, ParseError> {
    match expr {
        SpExpr::Value(sp::Value::Number(s, _)) => s
            .parse::<usize>()
            .map_err(|_| ParseError::Unsupported(format!("invalid LIMIT value: {}", s))),
        _ => Err(ParseError::Unsupported(
            "LIMIT must be a literal integer".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::physical_plan;

    // ---- Basic parsing tests ----

    #[test]
    fn test_parse_select_star() {
        let plan = parse_query("SELECT * FROM sales").unwrap();
        match &plan {
            LogicalPlan::Scan { table, .. } => assert_eq!(table, "sales"),
            _ => panic!("expected Scan for SELECT *, got: {:?}", plan),
        }
    }

    #[test]
    fn test_parse_select_columns() {
        let plan = parse_query("SELECT id, name FROM users").unwrap();
        match &plan {
            LogicalPlan::Projection { columns, input } => {
                assert_eq!(columns.len(), 2);
                assert_eq!(columns[0], Expr::Column("id".into()));
                assert_eq!(columns[1], Expr::Column("name".into()));
                match input.as_ref() {
                    LogicalPlan::Scan { table, .. } => assert_eq!(table, "users"),
                    _ => panic!("expected Scan"),
                }
            }
            _ => panic!("expected Projection"),
        }
    }

    #[test]
    fn test_parse_count_star_with_where() {
        let plan =
            parse_query("SELECT count(*) FROM sales WHERE amount > 100").unwrap();
        // Should be: Aggregate -> Filter -> Scan
        match &plan {
            LogicalPlan::Aggregate {
                aggregates, input, ..
            } => {
                assert_eq!(aggregates.len(), 1);
                assert_eq!(aggregates[0].0, AggFunc::Count);
                assert_eq!(aggregates[0].1, Expr::Wildcard);
                match input.as_ref() {
                    LogicalPlan::Filter {
                        predicate, input, ..
                    } => {
                        match predicate {
                            Expr::BinaryOp { left, op, right } => {
                                assert_eq!(**left, Expr::Column("amount".into()));
                                assert_eq!(*op, CompareOp::Gt);
                                assert_eq!(**right, Expr::Literal(Value::Int(100)));
                            }
                            _ => panic!("expected BinaryOp predicate"),
                        }
                        match input.as_ref() {
                            LogicalPlan::Scan { table, .. } => assert_eq!(table, "sales"),
                            _ => panic!("expected Scan"),
                        }
                    }
                    _ => panic!("expected Filter"),
                }
            }
            _ => panic!("expected Aggregate, got: {:?}", plan),
        }
    }

    #[test]
    fn test_parse_multiple_aggregates() {
        let plan =
            parse_query("SELECT count(*), sum(amount) FROM sales").unwrap();
        match &plan {
            LogicalPlan::Aggregate {
                aggregates,
                group_by,
                ..
            } => {
                assert_eq!(aggregates.len(), 2);
                assert_eq!(aggregates[0].0, AggFunc::Count);
                assert_eq!(aggregates[0].1, Expr::Wildcard);
                assert_eq!(aggregates[1].0, AggFunc::Sum);
                assert_eq!(aggregates[1].1, Expr::Column("amount".into()));
                assert!(group_by.is_empty());
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn test_parse_group_by() {
        let plan =
            parse_query("SELECT region, sum(amount) FROM sales GROUP BY region")
                .unwrap();
        match &plan {
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                ..
            } => {
                assert_eq!(group_by.len(), 1);
                assert_eq!(group_by[0], Expr::Column("region".into()));
                assert_eq!(aggregates.len(), 1);
                assert_eq!(aggregates[0].0, AggFunc::Sum);
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn test_parse_order_by_desc_limit() {
        let plan = parse_query(
            "SELECT * FROM sales ORDER BY amount DESC LIMIT 10",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Limit { count, input } => {
                assert_eq!(*count, 10);
                match input.as_ref() {
                    LogicalPlan::Sort { order_by, .. } => {
                        assert_eq!(order_by.len(), 1);
                        assert_eq!(order_by[0].0, Expr::Column("amount".into()));
                        assert!(!order_by[0].1); // DESC = false
                    }
                    _ => panic!("expected Sort"),
                }
            }
            _ => panic!("expected Limit, got: {:?}", plan),
        }
    }

    #[test]
    fn test_parse_order_by_asc() {
        let plan =
            parse_query("SELECT * FROM t ORDER BY id ASC").unwrap();
        match &plan {
            LogicalPlan::Sort { order_by, .. } => {
                assert_eq!(order_by[0].1, true); // ASC
            }
            _ => panic!("expected Sort"),
        }
    }

    #[test]
    fn test_parse_compound_where_and() {
        let plan = parse_query(
            "SELECT * FROM t WHERE a > 1 AND b < 10",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::Compound { op, .. } => assert_eq!(*op, LogicalOp::And),
                _ => panic!("expected Compound predicate"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_compound_where_or() {
        let plan = parse_query(
            "SELECT * FROM t WHERE x = 1 OR y = 2",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::Compound { op, .. } => assert_eq!(*op, LogicalOp::Or),
                _ => panic!("expected Compound predicate"),
            },
            _ => panic!("expected Filter"),
        }
    }

    // ---- All aggregate functions ----

    #[test]
    fn test_parse_avg() {
        let plan = parse_query("SELECT avg(price) FROM items").unwrap();
        match &plan {
            LogicalPlan::Aggregate { aggregates, .. } => {
                assert_eq!(aggregates[0].0, AggFunc::Avg);
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn test_parse_min_max() {
        let plan =
            parse_query("SELECT min(price), max(price) FROM items").unwrap();
        match &plan {
            LogicalPlan::Aggregate { aggregates, .. } => {
                assert_eq!(aggregates[0].0, AggFunc::Min);
                assert_eq!(aggregates[1].0, AggFunc::Max);
            }
            _ => panic!("expected Aggregate"),
        }
    }

    // ---- Value types ----

    #[test]
    fn test_parse_float_literal() {
        let plan = parse_query("SELECT * FROM t WHERE x > 3.14").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { right, .. } => {
                    assert_eq!(**right, Expr::Literal(Value::Float(3.14)));
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_string_literal() {
        let plan =
            parse_query("SELECT * FROM t WHERE name = 'Alice'").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { right, .. } => {
                    assert_eq!(
                        **right,
                        Expr::Literal(Value::Str("Alice".into()))
                    );
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_negative_number() {
        let plan = parse_query("SELECT * FROM t WHERE x > -5").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { right, .. } => {
                    assert_eq!(**right, Expr::Literal(Value::Int(-5)));
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    // ---- All comparison operators ----

    #[test]
    fn test_parse_all_compare_ops() {
        let cases = vec![
            ("SELECT * FROM t WHERE x = 1", CompareOp::Eq),
            ("SELECT * FROM t WHERE x != 1", CompareOp::Ne),
            ("SELECT * FROM t WHERE x < 1", CompareOp::Lt),
            ("SELECT * FROM t WHERE x <= 1", CompareOp::Le),
            ("SELECT * FROM t WHERE x > 1", CompareOp::Gt),
            ("SELECT * FROM t WHERE x >= 1", CompareOp::Ge),
        ];
        for (sql, expected_op) in cases {
            let plan = parse_query(sql).unwrap();
            match &plan {
                LogicalPlan::Filter { predicate, .. } => match predicate {
                    Expr::BinaryOp { op, .. } => {
                        assert_eq!(*op, expected_op, "failed for SQL: {}", sql);
                    }
                    _ => panic!("expected BinaryOp for: {}", sql),
                },
                _ => panic!("expected Filter for: {}", sql),
            }
        }
    }

    // ---- Error cases ----

    #[test]
    fn test_parse_error_invalid_sql() {
        let result = parse_query("SELEC * FORM t");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::SqlParser(_) => {}
            other => panic!("expected SqlParser error, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_error_not_select() {
        let result = parse_query("INSERT INTO t VALUES (1)");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::NotASelect => {}
            other => panic!("expected NotASelect, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_error_missing_from() {
        let result = parse_query("SELECT 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_unsupported_function() {
        let result = parse_query("SELECT custom_func(x) FROM t");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::Unsupported(msg) => assert!(msg.contains("not supported")),
            other => panic!("expected Unsupported, got: {:?}", other),
        }
    }

    // ---- End-to-end: parse -> plan ----

    #[test]
    fn test_parse_to_physical_plan() {
        let logical =
            parse_query("SELECT count(*) FROM t WHERE col > 100").unwrap();
        let physical = physical_plan::plan(&logical).unwrap();
        // Should be: GpuAggregate -> GpuFilter -> GpuScan
        match &physical {
            physical_plan::PhysicalPlan::GpuAggregate { functions, input, .. } => {
                assert_eq!(functions[0].0, AggFunc::Count);
                match input.as_ref() {
                    physical_plan::PhysicalPlan::GpuFilter {
                        compare_op,
                        column,
                        value,
                        ..
                    } => {
                        assert_eq!(*compare_op, CompareOp::Gt);
                        assert_eq!(column, "col");
                        assert_eq!(*value, Value::Int(100));
                    }
                    _ => panic!("expected GpuFilter"),
                }
            }
            _ => panic!("expected GpuAggregate"),
        }
    }

    #[test]
    fn test_parse_group_by_to_physical() {
        let logical =
            parse_query("SELECT region, sum(amount) FROM sales GROUP BY region")
                .unwrap();
        let physical = physical_plan::plan(&logical).unwrap();
        match &physical {
            physical_plan::PhysicalPlan::GpuAggregate {
                group_by,
                functions,
                ..
            } => {
                assert_eq!(group_by, &["region"]);
                assert_eq!(functions[0], (AggFunc::Sum, "amount".into()));
            }
            _ => panic!("expected GpuAggregate"),
        }
    }

    #[test]
    fn test_parse_order_limit_to_physical() {
        let logical = parse_query(
            "SELECT * FROM sales ORDER BY amount DESC LIMIT 10",
        )
        .unwrap();
        let physical = physical_plan::plan(&logical).unwrap();
        match &physical {
            physical_plan::PhysicalPlan::GpuLimit { count, input } => {
                assert_eq!(*count, 10);
                match input.as_ref() {
                    physical_plan::PhysicalPlan::GpuSort { order_by, .. } => {
                        assert_eq!(order_by[0], ("amount".into(), false));
                    }
                    _ => panic!("expected GpuSort"),
                }
            }
            _ => panic!("expected GpuLimit"),
        }
    }

    #[test]
    fn test_parse_case_insensitive_keywords() {
        // SQL keywords should be case-insensitive
        let plan = parse_query("select COUNT(*) from t where x > 1").unwrap();
        match &plan {
            LogicalPlan::Aggregate { aggregates, .. } => {
                assert_eq!(aggregates[0].0, AggFunc::Count);
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn test_parse_multiple_group_by_columns() {
        let plan = parse_query(
            "SELECT region, category, sum(amount) FROM sales GROUP BY region, category",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Aggregate { group_by, .. } => {
                assert_eq!(group_by.len(), 2);
                assert_eq!(group_by[0], Expr::Column("region".into()));
                assert_eq!(group_by[1], Expr::Column("category".into()));
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn test_parse_where_eq_string() {
        let plan =
            parse_query("SELECT count(*) FROM t WHERE region = 'EU'").unwrap();
        match &plan {
            LogicalPlan::Aggregate { input, .. } => match input.as_ref() {
                LogicalPlan::Filter { predicate, .. } => match predicate {
                    Expr::BinaryOp { op, right, .. } => {
                        assert_eq!(*op, CompareOp::Eq);
                        assert_eq!(**right, Expr::Literal(Value::Str("EU".into())));
                    }
                    _ => panic!("expected BinaryOp"),
                },
                _ => panic!("expected Filter"),
            },
            _ => panic!("expected Aggregate"),
        }
    }

    // ---- Individual aggregate function tests ----

    #[test]
    fn test_parse_count_column() {
        let plan = parse_query("SELECT count(id) FROM users").unwrap();
        match &plan {
            LogicalPlan::Aggregate { aggregates, .. } => {
                assert_eq!(aggregates.len(), 1);
                assert_eq!(aggregates[0].0, AggFunc::Count);
                assert_eq!(aggregates[0].1, Expr::Column("id".into()));
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn test_parse_sum_column() {
        let plan = parse_query("SELECT sum(revenue) FROM orders").unwrap();
        match &plan {
            LogicalPlan::Aggregate { aggregates, .. } => {
                assert_eq!(aggregates.len(), 1);
                assert_eq!(aggregates[0].0, AggFunc::Sum);
                assert_eq!(aggregates[0].1, Expr::Column("revenue".into()));
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn test_parse_all_five_aggregates() {
        let plan = parse_query(
            "SELECT count(*), sum(a), avg(b), min(c), max(d) FROM t",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Aggregate { aggregates, .. } => {
                assert_eq!(aggregates.len(), 5);
                assert_eq!(aggregates[0].0, AggFunc::Count);
                assert_eq!(aggregates[0].1, Expr::Wildcard);
                assert_eq!(aggregates[1].0, AggFunc::Sum);
                assert_eq!(aggregates[1].1, Expr::Column("a".into()));
                assert_eq!(aggregates[2].0, AggFunc::Avg);
                assert_eq!(aggregates[2].1, Expr::Column("b".into()));
                assert_eq!(aggregates[3].0, AggFunc::Min);
                assert_eq!(aggregates[3].1, Expr::Column("c".into()));
                assert_eq!(aggregates[4].0, AggFunc::Max);
                assert_eq!(aggregates[4].1, Expr::Column("d".into()));
            }
            _ => panic!("expected Aggregate"),
        }
    }

    // ---- GROUP BY tests ----

    #[test]
    fn test_parse_group_by_with_count() {
        let plan =
            parse_query("SELECT status, count(*) FROM orders GROUP BY status")
                .unwrap();
        match &plan {
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                ..
            } => {
                assert_eq!(group_by.len(), 1);
                assert_eq!(group_by[0], Expr::Column("status".into()));
                assert_eq!(aggregates[0].0, AggFunc::Count);
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn test_parse_group_by_three_columns() {
        let plan = parse_query(
            "SELECT region, category, year, sum(sales) FROM data GROUP BY region, category, year",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Aggregate { group_by, .. } => {
                assert_eq!(group_by.len(), 3);
                assert_eq!(group_by[0], Expr::Column("region".into()));
                assert_eq!(group_by[1], Expr::Column("category".into()));
                assert_eq!(group_by[2], Expr::Column("year".into()));
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn test_parse_group_by_with_multiple_aggs() {
        let plan = parse_query(
            "SELECT dept, count(*), avg(salary), max(salary) FROM emp GROUP BY dept",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                ..
            } => {
                assert_eq!(group_by.len(), 1);
                assert_eq!(aggregates.len(), 3);
                assert_eq!(aggregates[0].0, AggFunc::Count);
                assert_eq!(aggregates[1].0, AggFunc::Avg);
                assert_eq!(aggregates[2].0, AggFunc::Max);
            }
            _ => panic!("expected Aggregate"),
        }
    }

    // ---- ORDER BY tests ----

    #[test]
    fn test_parse_order_by_default_asc() {
        // Without explicit ASC/DESC, should default to ASC
        let plan =
            parse_query("SELECT * FROM t ORDER BY name").unwrap();
        match &plan {
            LogicalPlan::Sort { order_by, .. } => {
                assert_eq!(order_by.len(), 1);
                assert_eq!(order_by[0].0, Expr::Column("name".into()));
                assert!(order_by[0].1); // default ASC = true
            }
            _ => panic!("expected Sort"),
        }
    }

    #[test]
    fn test_parse_order_by_multiple_columns() {
        let plan = parse_query(
            "SELECT * FROM t ORDER BY region ASC, amount DESC",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Sort { order_by, .. } => {
                assert_eq!(order_by.len(), 2);
                assert_eq!(order_by[0].0, Expr::Column("region".into()));
                assert!(order_by[0].1); // ASC
                assert_eq!(order_by[1].0, Expr::Column("amount".into()));
                assert!(!order_by[1].1); // DESC
            }
            _ => panic!("expected Sort"),
        }
    }

    #[test]
    fn test_parse_order_by_three_columns() {
        let plan = parse_query(
            "SELECT * FROM t ORDER BY a ASC, b DESC, c ASC",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Sort { order_by, .. } => {
                assert_eq!(order_by.len(), 3);
                assert!(order_by[0].1);  // ASC
                assert!(!order_by[1].1); // DESC
                assert!(order_by[2].1);  // ASC
            }
            _ => panic!("expected Sort"),
        }
    }

    // ---- LIMIT tests ----

    #[test]
    fn test_parse_limit_only() {
        let plan = parse_query("SELECT * FROM t LIMIT 5").unwrap();
        match &plan {
            LogicalPlan::Limit { count, .. } => {
                assert_eq!(*count, 5);
            }
            _ => panic!("expected Limit"),
        }
    }

    #[test]
    fn test_parse_limit_large_value() {
        let plan = parse_query("SELECT * FROM t LIMIT 1000000").unwrap();
        match &plan {
            LogicalPlan::Limit { count, .. } => {
                assert_eq!(*count, 1_000_000);
            }
            _ => panic!("expected Limit"),
        }
    }

    #[test]
    fn test_parse_limit_one() {
        let plan = parse_query("SELECT * FROM t LIMIT 1").unwrap();
        match &plan {
            LogicalPlan::Limit { count, .. } => {
                assert_eq!(*count, 1);
            }
            _ => panic!("expected Limit"),
        }
    }

    // ---- Compound predicate tests ----

    #[test]
    fn test_parse_nested_and_or() {
        // (a > 1 AND b < 10) OR c = 5 -- parser precedence: AND binds tighter
        let plan = parse_query(
            "SELECT * FROM t WHERE a > 1 AND b < 10 OR c = 5",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::Compound { op, left, .. } => {
                    assert_eq!(*op, LogicalOp::Or);
                    // Left side should be AND compound
                    match left.as_ref() {
                        Expr::Compound { op, .. } => assert_eq!(*op, LogicalOp::And),
                        _ => panic!("expected inner AND compound"),
                    }
                }
                _ => panic!("expected Compound predicate"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_parenthesized_or_then_and() {
        // Explicit parens override precedence: (a = 1 OR b = 2) AND c = 3
        let plan = parse_query(
            "SELECT * FROM t WHERE (a = 1 OR b = 2) AND c = 3",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::Compound { op, left, .. } => {
                    assert_eq!(*op, LogicalOp::And);
                    // Left side should be OR compound (from parens)
                    match left.as_ref() {
                        Expr::Compound { op, .. } => assert_eq!(*op, LogicalOp::Or),
                        _ => panic!("expected inner OR compound"),
                    }
                }
                _ => panic!("expected Compound predicate"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_triple_and() {
        let plan = parse_query(
            "SELECT * FROM t WHERE a > 1 AND b > 2 AND c > 3",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => {
                // Should be nested ANDs
                match predicate {
                    Expr::Compound { op, .. } => assert_eq!(*op, LogicalOp::And),
                    _ => panic!("expected Compound"),
                }
            }
            _ => panic!("expected Filter"),
        }
    }

    // ---- WHERE with different comparison ops ----

    #[test]
    fn test_parse_where_ne_string() {
        let plan =
            parse_query("SELECT * FROM t WHERE status != 'closed'").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { op, right, .. } => {
                    assert_eq!(*op, CompareOp::Ne);
                    assert_eq!(**right, Expr::Literal(Value::Str("closed".into())));
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_where_le_float() {
        let plan =
            parse_query("SELECT * FROM t WHERE price <= 9.99").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { op, right, .. } => {
                    assert_eq!(*op, CompareOp::Le);
                    assert_eq!(**right, Expr::Literal(Value::Float(9.99)));
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_where_ge_int() {
        let plan =
            parse_query("SELECT * FROM t WHERE qty >= 100").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { left, op, right } => {
                    assert_eq!(**left, Expr::Column("qty".into()));
                    assert_eq!(*op, CompareOp::Ge);
                    assert_eq!(**right, Expr::Literal(Value::Int(100)));
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    // ---- String literal tests ----

    #[test]
    fn test_parse_string_with_spaces() {
        let plan =
            parse_query("SELECT * FROM t WHERE city = 'New York'").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { right, .. } => {
                    assert_eq!(**right, Expr::Literal(Value::Str("New York".into())));
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_double_quoted_string() {
        let plan =
            parse_query("SELECT * FROM t WHERE name = \"Bob\"").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { right, .. } => {
                    // sqlparser may treat double-quoted as identifier; check what we get
                    // GenericDialect treats double-quoted strings as identifiers
                    match right.as_ref() {
                        Expr::Column(name) => assert_eq!(name, "Bob"),
                        Expr::Literal(Value::Str(s)) => assert_eq!(s, "Bob"),
                        other => panic!("expected Column or Str, got: {:?}", other),
                    }
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    // ---- Error case tests ----

    #[test]
    fn test_parse_error_empty_string() {
        let result = parse_query("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_multiple_statements() {
        let result = parse_query("SELECT * FROM t; SELECT * FROM u");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::Unsupported(msg) => assert!(msg.contains("expected exactly one")),
            other => panic!("expected Unsupported, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_error_update_statement() {
        let result = parse_query("UPDATE t SET x = 1");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::NotASelect => {}
            other => panic!("expected NotASelect, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_error_delete_statement() {
        let result = parse_query("DELETE FROM t WHERE id = 1");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::NotASelect => {}
            other => panic!("expected NotASelect, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_error_join_unsupported() {
        let result = parse_query("SELECT * FROM t JOIN u ON t.id = u.id");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::Unsupported(msg) => assert!(msg.contains("JOIN")),
            other => panic!("expected Unsupported for JOIN, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_error_union_unsupported() {
        let result = parse_query("SELECT * FROM t UNION SELECT * FROM u");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::Unsupported(msg) => assert!(msg.contains("UNION") || msg.contains("simple SELECT")),
            other => panic!("expected Unsupported for UNION, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_error_multiple_from_tables() {
        let result = parse_query("SELECT * FROM t, u");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::Unsupported(msg) => assert!(msg.contains("multiple FROM")),
            other => panic!("expected Unsupported, got: {:?}", other),
        }
    }

    // ---- Edge cases ----

    #[test]
    fn test_parse_select_single_column() {
        let plan = parse_query("SELECT id FROM users").unwrap();
        match &plan {
            LogicalPlan::Projection { columns, .. } => {
                assert_eq!(columns.len(), 1);
                assert_eq!(columns[0], Expr::Column("id".into()));
            }
            _ => panic!("expected Projection"),
        }
    }

    #[test]
    fn test_parse_null_literal() {
        let plan = parse_query("SELECT * FROM t WHERE x = NULL").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { right, .. } => {
                    assert_eq!(**right, Expr::Literal(Value::Null));
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_boolean_literal_true() {
        let plan = parse_query("SELECT * FROM t WHERE active = true").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { right, .. } => {
                    // Booleans stored as Int(1) for GPU compat
                    assert_eq!(**right, Expr::Literal(Value::Int(1)));
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_boolean_literal_false() {
        let plan = parse_query("SELECT * FROM t WHERE active = false").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { right, .. } => {
                    assert_eq!(**right, Expr::Literal(Value::Int(0)));
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_negative_float() {
        let plan = parse_query("SELECT * FROM t WHERE x > -3.14").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { right, .. } => {
                    assert_eq!(**right, Expr::Literal(Value::Float(-3.14)));
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_zero_literal() {
        let plan = parse_query("SELECT * FROM t WHERE x = 0").unwrap();
        match &plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryOp { right, .. } => {
                    assert_eq!(**right, Expr::Literal(Value::Int(0)));
                }
                _ => panic!("expected BinaryOp"),
            },
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn test_parse_compound_identifier() {
        // t.col should extract "col" as the column name
        let plan = parse_query("SELECT t.name FROM t").unwrap();
        match &plan {
            LogicalPlan::Projection { columns, .. } => {
                assert_eq!(columns[0], Expr::Column("name".into()));
            }
            _ => panic!("expected Projection"),
        }
    }

    #[test]
    fn test_parse_error_display() {
        // Verify ParseError Display implementations
        let e = ParseError::SqlParser("bad syntax".into());
        assert!(e.to_string().contains("SQL parse error"));

        let e = ParseError::NotASelect;
        assert!(e.to_string().contains("SELECT"));

        let e = ParseError::Unsupported("feature".into());
        assert!(e.to_string().contains("unsupported"));

        let e = ParseError::MissingFrom;
        assert!(e.to_string().contains("FROM"));
    }

    #[test]
    fn test_parse_mixed_case_functions() {
        // Aggregate function names should be case-insensitive
        let plan = parse_query("SELECT Sum(a), Min(b), MAX(c) FROM t").unwrap();
        match &plan {
            LogicalPlan::Aggregate { aggregates, .. } => {
                assert_eq!(aggregates[0].0, AggFunc::Sum);
                assert_eq!(aggregates[1].0, AggFunc::Min);
                assert_eq!(aggregates[2].0, AggFunc::Max);
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn test_parse_where_with_aggregate() {
        // WHERE clause combined with aggregate: Filter -> Aggregate
        let plan = parse_query(
            "SELECT sum(amount) FROM orders WHERE status = 'shipped'",
        )
        .unwrap();
        match &plan {
            LogicalPlan::Aggregate { aggregates, input, .. } => {
                assert_eq!(aggregates[0].0, AggFunc::Sum);
                match input.as_ref() {
                    LogicalPlan::Filter { predicate, .. } => match predicate {
                        Expr::BinaryOp { op, .. } => assert_eq!(*op, CompareOp::Eq),
                        _ => panic!("expected BinaryOp"),
                    },
                    _ => panic!("expected Filter"),
                }
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn test_parse_full_query_pipeline() {
        // Test a query that exercises all clauses: SELECT + WHERE + GROUP BY + ORDER BY + LIMIT
        let plan = parse_query(
            "SELECT region, sum(amount) FROM sales WHERE year >= 2020 GROUP BY region ORDER BY region ASC LIMIT 10",
        )
        .unwrap();
        // Expected tree: Limit -> Sort -> Aggregate -> Filter -> Scan
        match &plan {
            LogicalPlan::Limit { count, input } => {
                assert_eq!(*count, 10);
                match input.as_ref() {
                    LogicalPlan::Sort { order_by, input } => {
                        assert_eq!(order_by[0].0, Expr::Column("region".into()));
                        assert!(order_by[0].1); // ASC
                        match input.as_ref() {
                            LogicalPlan::Aggregate { group_by, aggregates, input } => {
                                assert_eq!(group_by[0], Expr::Column("region".into()));
                                assert_eq!(aggregates[0].0, AggFunc::Sum);
                                match input.as_ref() {
                                    LogicalPlan::Filter { input, .. } => match input.as_ref() {
                                        LogicalPlan::Scan { table, .. } => {
                                            assert_eq!(table, "sales");
                                        }
                                        _ => panic!("expected Scan"),
                                    },
                                    _ => panic!("expected Filter"),
                                }
                            }
                            _ => panic!("expected Aggregate"),
                        }
                    }
                    _ => panic!("expected Sort"),
                }
            }
            _ => panic!("expected Limit"),
        }
    }
}
