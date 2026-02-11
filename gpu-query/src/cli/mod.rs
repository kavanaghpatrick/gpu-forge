//! CLI module: non-interactive execution and output formatting.
//!
//! Handles the full non-interactive pipeline:
//! parse args -> resolve query -> execute -> format output -> exit.

pub mod args;
pub mod commands;

use crate::gpu::executor::{QueryExecutor, QueryResult};
use args::{CliArgs, OutputFormat};
use std::io::Write;

/// Format a QueryResult according to the specified output format.
pub fn format_result(result: &QueryResult, format: &OutputFormat) -> String {
    match format {
        OutputFormat::Table => format_table(result),
        OutputFormat::Csv => format_csv(result),
        OutputFormat::Json => format_json(result),
        OutputFormat::Jsonl => format_jsonl(result),
    }
}

/// Format as ASCII table (same as QueryResult::print but returns String).
fn format_table(result: &QueryResult) -> String {
    if result.columns.is_empty() {
        return "(empty result)".to_string();
    }

    let mut out = String::new();

    // Compute column widths
    let mut widths: Vec<usize> = result.columns.iter().map(|c| c.len()).collect();
    for row in &result.rows {
        for (i, val) in row.iter().enumerate() {
            if i < widths.len() && val.len() > widths[i] {
                widths[i] = val.len();
            }
        }
    }

    // Header
    let header: Vec<String> = result
        .columns
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{:>width$}", c, width = widths[i]))
        .collect();
    out.push_str(&header.join(" | "));
    out.push('\n');

    // Separator
    let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    out.push_str(&sep.join("-+-"));
    out.push('\n');

    // Rows
    for row in &result.rows {
        let formatted: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let w = if i < widths.len() { widths[i] } else { v.len() };
                format!("{:>width$}", v, width = w)
            })
            .collect();
        out.push_str(&formatted.join(" | "));
        out.push('\n');
    }

    out.push_str(&format!(
        "({} row{})",
        result.row_count,
        if result.row_count == 1 { "" } else { "s" }
    ));

    out
}

/// Format as CSV.
fn format_csv(result: &QueryResult) -> String {
    let mut out = String::new();

    // Header row
    out.push_str(&csv_escape_row(&result.columns));
    out.push('\n');

    // Data rows
    for row in &result.rows {
        out.push_str(&csv_escape_row(row));
        out.push('\n');
    }

    // Remove trailing newline
    if out.ends_with('\n') {
        out.pop();
    }

    out
}

/// Escape and join a row as CSV.
fn csv_escape_row(fields: &[String]) -> String {
    fields
        .iter()
        .map(|f| {
            if f.contains(',') || f.contains('"') || f.contains('\n') {
                format!("\"{}\"", f.replace('"', "\"\""))
            } else {
                f.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Format as JSON array of objects.
fn format_json(result: &QueryResult) -> String {
    let mut out = String::from("[\n");

    for (row_idx, row) in result.rows.iter().enumerate() {
        out.push_str("  {");
        for (col_idx, val) in row.iter().enumerate() {
            if col_idx > 0 {
                out.push_str(", ");
            }
            let col_name = &result.columns[col_idx];
            out.push_str(&format!(
                "\"{}\": {}",
                json_escape(col_name),
                json_value(val)
            ));
        }
        out.push('}');
        if row_idx < result.rows.len() - 1 {
            out.push(',');
        }
        out.push('\n');
    }

    out.push(']');
    out
}

/// Format as JSONL (one JSON object per line).
fn format_jsonl(result: &QueryResult) -> String {
    let mut out = String::new();

    for row in &result.rows {
        out.push('{');
        for (col_idx, val) in row.iter().enumerate() {
            if col_idx > 0 {
                out.push_str(", ");
            }
            let col_name = &result.columns[col_idx];
            out.push_str(&format!(
                "\"{}\": {}",
                json_escape(col_name),
                json_value(val)
            ));
        }
        out.push('}');
        out.push('\n');
    }

    // Remove trailing newline
    if out.ends_with('\n') {
        out.pop();
    }

    out
}

/// Escape a string for JSON.
fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Format a value for JSON output. Attempts numeric parsing.
fn json_value(val: &str) -> String {
    // Try integer
    if let Ok(_) = val.parse::<i64>() {
        return val.to_string();
    }
    // Try float
    if let Ok(_) = val.parse::<f64>() {
        return val.to_string();
    }
    // NULL
    if val == "NULL" || val.is_empty() {
        return "null".to_string();
    }
    // String
    format!("\"{}\"", json_escape(val))
}

/// Run the non-interactive execution path.
/// Returns the exit code.
pub fn run_non_interactive(args: &CliArgs) -> i32 {
    // Validate directory
    if !args.directory.is_dir() {
        eprintln!("Error: '{}' is not a directory", args.directory.display());
        return 1;
    }

    // Resolve SQL query
    let sql = match args.resolve_query() {
        Ok(q) => q,
        Err(e) => {
            eprintln!("Error: {}", e);
            return 1;
        }
    };

    // Scan directory
    let catalog = match crate::io::catalog::scan_directory(&args.directory) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error scanning directory: {}", e);
            return 1;
        }
    };

    if catalog.is_empty() {
        eprintln!("No data files found in '{}'", args.directory.display());
        return 1;
    }

    // Parse SQL
    let logical_plan = match crate::sql::parser::parse_query(&sql) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("SQL parse error: {}", e);
            return 1;
        }
    };

    // Optimize
    let logical_plan = crate::sql::optimizer::optimize(logical_plan);

    // Plan
    let physical_plan = match crate::sql::physical_plan::plan(&logical_plan) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Plan error: {:?}", e);
            return 1;
        }
    };

    // Execute on GPU
    let mut executor = match QueryExecutor::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("GPU init error: {}", e);
            return 1;
        }
    };

    let result = match executor.execute(&physical_plan, &catalog) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Query execution error: {}", e);
            return 1;
        }
    };

    // Format output
    let formatted = format_result(&result, &args.format);

    // Write output
    if let Some(ref path) = args.output {
        match std::fs::write(path, &formatted) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error writing to '{}': {}", path.display(), e);
                return 1;
            }
        }
    } else {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        if let Err(e) = write!(handle, "{}", formatted) {
            // Broken pipe is expected when piping to head/etc
            if e.kind() != std::io::ErrorKind::BrokenPipe {
                eprintln!("Write error: {}", e);
                return 1;
            }
        }
        // Add trailing newline for non-table formats
        if args.format != OutputFormat::Table {
            let _ = writeln!(handle);
        } else {
            let _ = writeln!(handle);
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::executor::QueryResult;

    fn sample_result() -> QueryResult {
        QueryResult {
            columns: vec!["name".into(), "count".into(), "total".into()],
            rows: vec![
                vec!["Alice".into(), "5".into(), "1500".into()],
                vec!["Bob".into(), "3".into(), "900".into()],
            ],
            row_count: 2,
        }
    }

    #[test]
    fn test_format_csv() {
        let r = sample_result();
        let csv = format_csv(&r);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "name,count,total");
        assert_eq!(lines[1], "Alice,5,1500");
        assert_eq!(lines[2], "Bob,3,900");
    }

    #[test]
    fn test_format_json() {
        let r = sample_result();
        let json = format_json(&r);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"name\": \"Alice\""));
        assert!(json.contains("\"count\": 5"));
    }

    #[test]
    fn test_format_jsonl() {
        let r = sample_result();
        let jsonl = format_jsonl(&r);
        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with('{'));
        assert!(lines[0].contains("\"name\": \"Alice\""));
    }

    #[test]
    fn test_format_table() {
        let r = sample_result();
        let table = format_table(&r);
        assert!(table.contains("name"));
        assert!(table.contains("Alice"));
        assert!(table.contains("(2 rows)"));
    }

    #[test]
    fn test_format_empty_result() {
        let r = QueryResult {
            columns: vec![],
            rows: vec![],
            row_count: 0,
        };
        let table = format_table(&r);
        assert_eq!(table, "(empty result)");
    }

    #[test]
    fn test_csv_escape_with_comma() {
        let row = vec!["hello, world".to_string(), "simple".to_string()];
        let escaped = csv_escape_row(&row);
        assert_eq!(escaped, "\"hello, world\",simple");
    }

    #[test]
    fn test_json_value_int() {
        assert_eq!(json_value("42"), "42");
    }

    #[test]
    fn test_json_value_float() {
        assert_eq!(json_value("3.14"), "3.14");
    }

    #[test]
    fn test_json_value_string() {
        assert_eq!(json_value("hello"), "\"hello\"");
    }

    #[test]
    fn test_json_value_null() {
        assert_eq!(json_value("NULL"), "null");
    }
}
