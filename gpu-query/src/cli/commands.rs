//! Dot command handler for the TUI interactive mode.
//!
//! Supports commands like .tables, .schema, .history, .quit, etc.
//! Implements query history persistence to ~/.config/gpu-query/history.

use std::fmt;
use std::path::PathBuf;

use crate::cli::args::OutputFormat;

/// Maximum number of queries retained in history.
pub const MAX_HISTORY_SIZE: usize = 1000;

/// The config directory name under ~/.config/
const CONFIG_DIR: &str = "gpu-query";

/// History file name within the config directory.
const HISTORY_FILE: &str = "history";

/// A parsed dot command with its arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DotCommand {
    /// .tables — list all loaded tables with format and row count.
    Tables,
    /// .schema <table> — show table schema (column names and types).
    Schema(String),
    /// .describe <table> — alias for .schema.
    Describe(String),
    /// .gpu — show GPU device info and memory.
    Gpu,
    /// .profile on|off — toggle profile mode.
    Profile(Option<bool>),
    /// .benchmark — placeholder for CPU benchmark comparison.
    Benchmark,
    /// .timer on|off — toggle query timing display.
    Timer(Option<bool>),
    /// .comparison — show last CPU vs GPU comparison.
    Comparison,
    /// .format csv|json|table — change output format.
    Format(Option<OutputFormat>),
    /// .save <file> — save last result to file.
    Save(String),
    /// .history — show recent queries.
    History,
    /// .clear — clear the screen / results.
    Clear,
    /// .help — show all dot commands.
    Help,
    /// .quit — exit the application.
    Quit,
}

/// Result of executing a dot command.
#[derive(Debug, Clone)]
pub enum DotCommandResult {
    /// Text output to display in the results area.
    Output(String),
    /// State change was applied (with optional status message).
    StateChange(String),
    /// Clear the screen / results.
    ClearScreen,
    /// Quit the application.
    Quit,
    /// Error message.
    Error(String),
    /// Request GPU-parallel DESCRIBE on specified table.
    /// The caller should invoke `executor.execute_describe()` and display the result.
    DescribeTable(String),
}

impl fmt::Display for DotCommandResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DotCommandResult::Output(s) => write!(f, "{}", s),
            DotCommandResult::StateChange(s) => write!(f, "{}", s),
            DotCommandResult::ClearScreen => write!(f, "(screen cleared)"),
            DotCommandResult::Quit => write!(f, "Goodbye."),
            DotCommandResult::Error(s) => write!(f, "Error: {}", s),
            DotCommandResult::DescribeTable(table) => write!(f, "DESCRIBE {}", table),
        }
    }
}

/// Check if a string looks like a dot command.
pub fn is_dot_command(input: &str) -> bool {
    input.trim().starts_with('.')
}

/// Parse a dot command string into a DotCommand enum.
///
/// Returns None if the input is not a recognized dot command.
pub fn parse_dot_command(input: &str) -> Option<DotCommand> {
    let trimmed = input.trim();
    if !trimmed.starts_with('.') {
        return None;
    }

    let parts: Vec<&str> = trimmed.splitn(2, char::is_whitespace).collect();
    let cmd = parts[0].to_lowercase();
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd.as_str() {
        ".tables" => Some(DotCommand::Tables),
        ".schema" => {
            if arg.is_empty() {
                Some(DotCommand::Schema(String::new()))
            } else {
                Some(DotCommand::Schema(arg.to_string()))
            }
        }
        ".describe" => {
            if arg.is_empty() {
                Some(DotCommand::Describe(String::new()))
            } else {
                Some(DotCommand::Describe(arg.to_string()))
            }
        }
        ".gpu" => Some(DotCommand::Gpu),
        ".profile" => {
            let toggle = parse_on_off(arg);
            Some(DotCommand::Profile(toggle))
        }
        ".benchmark" => Some(DotCommand::Benchmark),
        ".timer" => {
            let toggle = parse_on_off(arg);
            Some(DotCommand::Timer(toggle))
        }
        ".comparison" => Some(DotCommand::Comparison),
        ".format" => {
            if arg.is_empty() {
                Some(DotCommand::Format(None))
            } else {
                match arg.parse::<OutputFormat>() {
                    Ok(fmt) => Some(DotCommand::Format(Some(fmt))),
                    Err(_) => Some(DotCommand::Format(None)),
                }
            }
        }
        ".save" => {
            if arg.is_empty() {
                Some(DotCommand::Save(String::new()))
            } else {
                Some(DotCommand::Save(arg.to_string()))
            }
        }
        ".history" => Some(DotCommand::History),
        ".clear" => Some(DotCommand::Clear),
        ".help" | ".h" | ".?" => Some(DotCommand::Help),
        ".quit" | ".q" | ".exit" => Some(DotCommand::Quit),
        _ => None,
    }
}

/// Parse "on"/"off" strings to bool.
fn parse_on_off(s: &str) -> Option<bool> {
    match s.to_lowercase().as_str() {
        "on" | "1" | "true" | "yes" => Some(true),
        "off" | "0" | "false" | "no" => Some(false),
        _ => None,
    }
}

/// Context provided to dot command execution.
/// Contains references to relevant application state needed by commands.
pub struct DotCommandContext<'a> {
    /// Table names and their metadata for .tables / .schema / .describe.
    pub tables: &'a [(String, String, Vec<(String, String)>)], // (name, format, [(col_name, col_type)])
    /// Query history for .history.
    pub history: &'a [String],
    /// Last query result as formatted text for .save.
    pub last_result_text: Option<&'a str>,
    /// Current output format for .format display.
    pub current_format: &'a OutputFormat,
    /// Whether profile mode is currently on.
    pub profile_on: bool,
    /// Whether timer is currently on.
    pub timer_on: bool,
    /// GPU device name.
    pub gpu_device_name: String,
    /// GPU memory allocated in bytes.
    pub gpu_memory_bytes: u64,
    /// Last CPU comparison info.
    pub last_comparison: Option<String>,
}

/// Execute a dot command and return the result.
pub fn handle_dot_command(cmd: &DotCommand, ctx: &DotCommandContext) -> DotCommandResult {
    match cmd {
        DotCommand::Tables => handle_tables(ctx),
        DotCommand::Schema(table) => handle_schema(table, ctx),
        DotCommand::Describe(table) => handle_describe(table),
        DotCommand::Gpu => handle_gpu(ctx),
        DotCommand::Profile(toggle) => handle_profile(*toggle, ctx),
        DotCommand::Benchmark => {
            DotCommandResult::Output("Benchmark mode: run a query with .profile on to see per-kernel timing.".into())
        }
        DotCommand::Timer(toggle) => handle_timer(*toggle, ctx),
        DotCommand::Comparison => handle_comparison(ctx),
        DotCommand::Format(fmt) => handle_format(fmt, ctx),
        DotCommand::Save(path) => handle_save(path, ctx),
        DotCommand::History => handle_history(ctx),
        DotCommand::Clear => DotCommandResult::ClearScreen,
        DotCommand::Help => handle_help(),
        DotCommand::Quit => DotCommandResult::Quit,
    }
}

/// .tables — list loaded tables.
fn handle_tables(ctx: &DotCommandContext) -> DotCommandResult {
    if ctx.tables.is_empty() {
        return DotCommandResult::Output("No tables loaded.".into());
    }

    let mut out = String::from("Tables:\n");
    for (name, format, columns) in ctx.tables {
        out.push_str(&format!("  {} [{}] ({} columns)\n", name, format, columns.len()));
    }
    DotCommandResult::Output(out.trim_end().to_string())
}

/// .schema / .describe — show table schema.
fn handle_schema(table: &str, ctx: &DotCommandContext) -> DotCommandResult {
    if table.is_empty() {
        // Show all tables with schemas
        if ctx.tables.is_empty() {
            return DotCommandResult::Output("No tables loaded.".into());
        }
        let mut out = String::new();
        for (name, format, columns) in ctx.tables {
            out.push_str(&format!("{} [{}]:\n", name, format));
            for (col_name, col_type) in columns {
                out.push_str(&format!("  {} {}\n", col_name, col_type));
            }
            out.push('\n');
        }
        return DotCommandResult::Output(out.trim_end().to_string());
    }

    // Find the specific table
    let table_lower = table.to_lowercase();
    for (name, format, columns) in ctx.tables {
        if name.to_lowercase() == table_lower {
            let mut out = format!("{} [{}]:\n", name, format);
            if columns.is_empty() {
                out.push_str("  (no column info available)\n");
            } else {
                for (col_name, col_type) in columns {
                    out.push_str(&format!("  {} {}\n", col_name, col_type));
                }
            }
            return DotCommandResult::Output(out.trim_end().to_string());
        }
    }

    DotCommandResult::Error(format!("Table '{}' not found.", table))
}

/// .describe <table> — GPU-parallel column statistics.
///
/// Returns a `DescribeTable` result that the caller should intercept and
/// route to `QueryExecutor::execute_describe()` for GPU-parallel stats computation.
/// If no table name is provided, falls back to showing all schemas.
fn handle_describe(table: &str) -> DotCommandResult {
    if table.is_empty() {
        return DotCommandResult::Error("Usage: .describe <table>".to_string());
    }
    DotCommandResult::DescribeTable(table.to_string())
}

/// .gpu — show GPU device info.
fn handle_gpu(ctx: &DotCommandContext) -> DotCommandResult {
    let mem_mb = ctx.gpu_memory_bytes as f64 / (1024.0 * 1024.0);
    let out = format!(
        "GPU Device: {}\nMemory Allocated: {:.1} MB",
        ctx.gpu_device_name, mem_mb
    );
    DotCommandResult::Output(out)
}

/// .profile on|off — toggle profile mode.
fn handle_profile(toggle: Option<bool>, ctx: &DotCommandContext) -> DotCommandResult {
    match toggle {
        Some(on) => DotCommandResult::StateChange(format!(
            "Profile mode: {}",
            if on { "ON" } else { "OFF" }
        )),
        None => DotCommandResult::Output(format!(
            "Profile mode: {}. Use .profile on/off to toggle.",
            if ctx.profile_on { "ON" } else { "OFF" }
        )),
    }
}

/// .timer on|off — toggle query timing display.
fn handle_timer(toggle: Option<bool>, ctx: &DotCommandContext) -> DotCommandResult {
    match toggle {
        Some(on) => DotCommandResult::StateChange(format!(
            "Timer: {}",
            if on { "ON" } else { "OFF" }
        )),
        None => DotCommandResult::Output(format!(
            "Timer: {}. Use .timer on/off to toggle.",
            if ctx.timer_on { "ON" } else { "OFF" }
        )),
    }
}

/// .comparison — show last CPU vs GPU comparison.
fn handle_comparison(ctx: &DotCommandContext) -> DotCommandResult {
    match &ctx.last_comparison {
        Some(info) => DotCommandResult::Output(info.clone()),
        None => DotCommandResult::Output("No comparison data. Run a query first.".into()),
    }
}

/// .format csv|json|table — change output format.
fn handle_format(fmt: &Option<OutputFormat>, ctx: &DotCommandContext) -> DotCommandResult {
    match fmt {
        Some(new_fmt) => DotCommandResult::StateChange(format!("Output format: {}", new_fmt)),
        None => DotCommandResult::Output(format!(
            "Current format: {}. Use .format csv|json|table|jsonl to change.",
            ctx.current_format
        )),
    }
}

/// .save <file> — save last result to file.
fn handle_save(path: &str, ctx: &DotCommandContext) -> DotCommandResult {
    if path.is_empty() {
        return DotCommandResult::Error("Usage: .save <filename>".into());
    }

    let text = match ctx.last_result_text {
        Some(t) => t,
        None => return DotCommandResult::Error("No result to save. Run a query first.".into()),
    };

    // Expand ~ to home directory
    let expanded = if path.starts_with('~') {
        if let Some(home) = dirs_path() {
            path.replacen('~', &home.to_string_lossy(), 1)
        } else {
            path.to_string()
        }
    } else {
        path.to_string()
    };

    match std::fs::write(&expanded, text) {
        Ok(_) => DotCommandResult::StateChange(format!("Saved to {}", expanded)),
        Err(e) => DotCommandResult::Error(format!("Failed to write '{}': {}", expanded, e)),
    }
}

/// .history — show recent queries.
fn handle_history(ctx: &DotCommandContext) -> DotCommandResult {
    if ctx.history.is_empty() {
        return DotCommandResult::Output("No query history.".into());
    }

    let mut out = String::from("Query History:\n");
    // Show most recent first, numbered
    for (i, query) in ctx.history.iter().enumerate() {
        let display = if query.len() > 80 {
            format!("{}...", &query[..77])
        } else {
            query.clone()
        };
        out.push_str(&format!("  {:>3}  {}\n", i + 1, display));
    }
    DotCommandResult::Output(out.trim_end().to_string())
}

/// .help — show all dot commands.
fn handle_help() -> DotCommandResult {
    let help = "\
Dot Commands:
  .tables               List all loaded tables with format and column count
  .schema [table]       Show table schema (column names and types)
  .describe <table>     GPU-parallel column statistics (count, null%, distinct, min, max)
  .gpu                  Show GPU device info and memory usage
  .profile [on|off]     Toggle per-kernel timing output
  .benchmark            Show benchmark info
  .timer [on|off]       Toggle query timing display
  .comparison           Show last CPU vs GPU performance comparison
  .format [csv|json|table|jsonl]  Change output format
  .save <file>          Save last query result to file
  .history              Show recent query history
  .clear                Clear the results display
  .help                 Show this help message
  .quit                 Exit gpu-query";

    DotCommandResult::Output(help.to_string())
}

// ── Query history persistence ──────────────────────────────────────────────

/// Get the config directory path: ~/.config/gpu-query/
fn config_dir() -> Option<PathBuf> {
    dirs_path().map(|home| home.join(".config").join(CONFIG_DIR))
}

/// Get the home directory path.
fn dirs_path() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(PathBuf::from)
}

/// Get the history file path: ~/.config/gpu-query/history
fn history_file_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join(HISTORY_FILE))
}

/// Load query history from the persistent file.
/// Returns an empty Vec if the file doesn't exist or can't be read.
pub fn load_history() -> Vec<String> {
    let path = match history_file_path() {
        Some(p) => p,
        None => return Vec::new(),
    };

    match std::fs::read_to_string(&path) {
        Ok(content) => {
            let entries: Vec<String> = content
                .lines()
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect();
            // Keep only the last MAX_HISTORY_SIZE entries
            if entries.len() > MAX_HISTORY_SIZE {
                entries[entries.len() - MAX_HISTORY_SIZE..].to_vec()
            } else {
                entries
            }
        }
        Err(_) => Vec::new(),
    }
}

/// Save query history to the persistent file.
/// Creates the config directory if it doesn't exist.
/// Keeps only the last MAX_HISTORY_SIZE entries.
pub fn save_history(history: &[String]) -> Result<(), String> {
    let dir = config_dir().ok_or("Cannot determine config directory")?;
    let path = dir.join(HISTORY_FILE);

    // Create config directory if needed
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Failed to create config dir '{}': {}", dir.display(), e))?;

    // Keep only the last MAX_HISTORY_SIZE entries
    let to_save = if history.len() > MAX_HISTORY_SIZE {
        &history[history.len() - MAX_HISTORY_SIZE..]
    } else {
        history
    };

    let content: String = to_save.iter().map(|q| format!("{}\n", q)).collect();
    std::fs::write(&path, &content)
        .map_err(|e| format!("Failed to write history to '{}': {}", path.display(), e))?;

    Ok(())
}

/// Append a single query to the history list (in memory) and persist.
/// Deduplicates: won't add if the same as the last entry.
pub fn append_to_history(history: &mut Vec<String>, query: &str) {
    let trimmed = query.trim().to_string();
    if trimmed.is_empty() {
        return;
    }

    // Don't add if same as last entry
    if history.last().map_or(false, |last| *last == trimmed) {
        return;
    }

    history.push(trimmed);

    // Trim to MAX_HISTORY_SIZE
    if history.len() > MAX_HISTORY_SIZE {
        let excess = history.len() - MAX_HISTORY_SIZE;
        history.drain(0..excess);
    }

    // Persist (best-effort, don't fail on write errors)
    let _ = save_history(history);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Parsing tests ──────────────────────────────────────────────────

    #[test]
    fn test_is_dot_command() {
        assert!(is_dot_command(".tables"));
        assert!(is_dot_command("  .help  "));
        assert!(is_dot_command(".quit"));
        assert!(!is_dot_command("SELECT 1"));
        assert!(!is_dot_command(""));
        assert!(!is_dot_command("tables"));
    }

    #[test]
    fn test_parse_tables() {
        assert_eq!(parse_dot_command(".tables"), Some(DotCommand::Tables));
    }

    #[test]
    fn test_parse_schema_no_arg() {
        assert_eq!(
            parse_dot_command(".schema"),
            Some(DotCommand::Schema(String::new()))
        );
    }

    #[test]
    fn test_parse_schema_with_arg() {
        assert_eq!(
            parse_dot_command(".schema sales"),
            Some(DotCommand::Schema("sales".into()))
        );
    }

    #[test]
    fn test_parse_describe_with_arg() {
        assert_eq!(
            parse_dot_command(".describe users"),
            Some(DotCommand::Describe("users".into()))
        );
    }

    #[test]
    fn test_parse_gpu() {
        assert_eq!(parse_dot_command(".gpu"), Some(DotCommand::Gpu));
    }

    #[test]
    fn test_parse_profile_on() {
        assert_eq!(
            parse_dot_command(".profile on"),
            Some(DotCommand::Profile(Some(true)))
        );
    }

    #[test]
    fn test_parse_profile_off() {
        assert_eq!(
            parse_dot_command(".profile off"),
            Some(DotCommand::Profile(Some(false)))
        );
    }

    #[test]
    fn test_parse_profile_no_arg() {
        assert_eq!(
            parse_dot_command(".profile"),
            Some(DotCommand::Profile(None))
        );
    }

    #[test]
    fn test_parse_timer_on_off() {
        assert_eq!(
            parse_dot_command(".timer on"),
            Some(DotCommand::Timer(Some(true)))
        );
        assert_eq!(
            parse_dot_command(".timer off"),
            Some(DotCommand::Timer(Some(false)))
        );
        assert_eq!(
            parse_dot_command(".timer"),
            Some(DotCommand::Timer(None))
        );
    }

    #[test]
    fn test_parse_format_csv() {
        assert_eq!(
            parse_dot_command(".format csv"),
            Some(DotCommand::Format(Some(OutputFormat::Csv)))
        );
    }

    #[test]
    fn test_parse_format_json() {
        assert_eq!(
            parse_dot_command(".format json"),
            Some(DotCommand::Format(Some(OutputFormat::Json)))
        );
    }

    #[test]
    fn test_parse_format_table() {
        assert_eq!(
            parse_dot_command(".format table"),
            Some(DotCommand::Format(Some(OutputFormat::Table)))
        );
    }

    #[test]
    fn test_parse_format_no_arg() {
        assert_eq!(
            parse_dot_command(".format"),
            Some(DotCommand::Format(None))
        );
    }

    #[test]
    fn test_parse_save() {
        assert_eq!(
            parse_dot_command(".save output.csv"),
            Some(DotCommand::Save("output.csv".into()))
        );
    }

    #[test]
    fn test_parse_save_no_arg() {
        assert_eq!(
            parse_dot_command(".save"),
            Some(DotCommand::Save(String::new()))
        );
    }

    #[test]
    fn test_parse_history() {
        assert_eq!(parse_dot_command(".history"), Some(DotCommand::History));
    }

    #[test]
    fn test_parse_clear() {
        assert_eq!(parse_dot_command(".clear"), Some(DotCommand::Clear));
    }

    #[test]
    fn test_parse_help() {
        assert_eq!(parse_dot_command(".help"), Some(DotCommand::Help));
        assert_eq!(parse_dot_command(".h"), Some(DotCommand::Help));
        assert_eq!(parse_dot_command(".?"), Some(DotCommand::Help));
    }

    #[test]
    fn test_parse_quit() {
        assert_eq!(parse_dot_command(".quit"), Some(DotCommand::Quit));
        assert_eq!(parse_dot_command(".q"), Some(DotCommand::Quit));
        assert_eq!(parse_dot_command(".exit"), Some(DotCommand::Quit));
    }

    #[test]
    fn test_parse_unknown_command() {
        assert_eq!(parse_dot_command(".unknown"), None);
        assert_eq!(parse_dot_command(".foo bar"), None);
    }

    #[test]
    fn test_parse_not_dot_command() {
        assert_eq!(parse_dot_command("SELECT 1"), None);
    }

    #[test]
    fn test_parse_case_insensitive() {
        assert_eq!(parse_dot_command(".TABLES"), Some(DotCommand::Tables));
        assert_eq!(parse_dot_command(".Help"), Some(DotCommand::Help));
        assert_eq!(parse_dot_command(".Quit"), Some(DotCommand::Quit));
    }

    #[test]
    fn test_parse_with_whitespace() {
        assert_eq!(parse_dot_command("  .tables  "), Some(DotCommand::Tables));
        assert_eq!(
            parse_dot_command("  .schema  sales  "),
            Some(DotCommand::Schema("sales".into()))
        );
    }

    // ── Execution tests ────────────────────────────────────────────────

    fn make_ctx<'a>(
        tables: &'a [(String, String, Vec<(String, String)>)],
        history: &'a [String],
    ) -> DotCommandContext<'a> {
        DotCommandContext {
            tables,
            history,
            last_result_text: None,
            current_format: &OutputFormat::Table,
            profile_on: false,
            timer_on: true,
            gpu_device_name: "Apple M4".into(),
            gpu_memory_bytes: 16 * 1024 * 1024, // 16 MB
            last_comparison: None,
        }
    }

    #[test]
    fn test_handle_tables_empty() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Tables, &ctx);
        match result {
            DotCommandResult::Output(s) => assert!(s.contains("No tables")),
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_handle_tables_with_entries() {
        let tables = vec![
            ("sales".into(), "CSV".into(), vec![
                ("id".into(), "INT64".into()),
                ("amount".into(), "FLOAT64".into()),
            ]),
            ("users".into(), "JSON".into(), vec![
                ("uid".into(), "INT64".into()),
            ]),
        ];
        let ctx = make_ctx(&tables, &[]);
        let result = handle_dot_command(&DotCommand::Tables, &ctx);
        match result {
            DotCommandResult::Output(s) => {
                assert!(s.contains("sales"));
                assert!(s.contains("CSV"));
                assert!(s.contains("2 columns"));
                assert!(s.contains("users"));
                assert!(s.contains("JSON"));
            }
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_handle_schema_specific_table() {
        let tables = vec![
            ("sales".into(), "CSV".into(), vec![
                ("id".into(), "INT64".into()),
                ("amount".into(), "FLOAT64".into()),
            ]),
        ];
        let ctx = make_ctx(&tables, &[]);
        let result = handle_dot_command(&DotCommand::Schema("sales".into()), &ctx);
        match result {
            DotCommandResult::Output(s) => {
                assert!(s.contains("sales"));
                assert!(s.contains("id"));
                assert!(s.contains("INT64"));
                assert!(s.contains("amount"));
                assert!(s.contains("FLOAT64"));
            }
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_handle_schema_not_found() {
        let tables = vec![
            ("sales".into(), "CSV".into(), vec![]),
        ];
        let ctx = make_ctx(&tables, &[]);
        let result = handle_dot_command(&DotCommand::Schema("nonexistent".into()), &ctx);
        match result {
            DotCommandResult::Error(s) => assert!(s.contains("not found")),
            _ => panic!("Expected Error"),
        }
    }

    #[test]
    fn test_handle_gpu() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Gpu, &ctx);
        match result {
            DotCommandResult::Output(s) => {
                assert!(s.contains("Apple M4"));
                assert!(s.contains("16.0 MB"));
            }
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_handle_profile_toggle() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Profile(Some(true)), &ctx);
        match result {
            DotCommandResult::StateChange(s) => assert!(s.contains("ON")),
            _ => panic!("Expected StateChange"),
        }
    }

    #[test]
    fn test_handle_timer_query() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Timer(None), &ctx);
        match result {
            DotCommandResult::Output(s) => assert!(s.contains("ON")), // timer_on=true in make_ctx
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_handle_format_change() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Format(Some(OutputFormat::Csv)), &ctx);
        match result {
            DotCommandResult::StateChange(s) => assert!(s.contains("csv")),
            _ => panic!("Expected StateChange"),
        }
    }

    #[test]
    fn test_handle_format_query() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Format(None), &ctx);
        match result {
            DotCommandResult::Output(s) => assert!(s.contains("table")),
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_handle_save_no_path() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Save(String::new()), &ctx);
        match result {
            DotCommandResult::Error(s) => assert!(s.contains("Usage")),
            _ => panic!("Expected Error"),
        }
    }

    #[test]
    fn test_handle_save_no_result() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Save("out.csv".into()), &ctx);
        match result {
            DotCommandResult::Error(s) => assert!(s.contains("No result")),
            _ => panic!("Expected Error"),
        }
    }

    #[test]
    fn test_handle_save_success() {
        let tmp = std::env::temp_dir().join("gpu_query_test_save.txt");
        let tables: Vec<(String, String, Vec<(String, String)>)> = vec![];
        let history: Vec<String> = vec![];
        let ctx = DotCommandContext {
            tables: &tables,
            history: &history,
            last_result_text: Some("id,amount\n1,100\n2,200"),
            current_format: &OutputFormat::Table,
            profile_on: false,
            timer_on: false,
            gpu_device_name: "Test GPU".into(),
            gpu_memory_bytes: 0,
            last_comparison: None,
        };
        let result = handle_dot_command(&DotCommand::Save(tmp.to_string_lossy().to_string()), &ctx);
        match result {
            DotCommandResult::StateChange(s) => assert!(s.contains("Saved")),
            other => panic!("Expected StateChange, got {:?}", other),
        }
        // Verify file contents
        let saved = std::fs::read_to_string(&tmp).unwrap();
        assert_eq!(saved, "id,amount\n1,100\n2,200");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_handle_history_empty() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::History, &ctx);
        match result {
            DotCommandResult::Output(s) => assert!(s.contains("No query history")),
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_handle_history_with_entries() {
        let history = vec![
            "SELECT count(*) FROM sales".to_string(),
            "SELECT * FROM users WHERE id > 10".to_string(),
        ];
        let ctx = make_ctx(&[], &history);
        let result = handle_dot_command(&DotCommand::History, &ctx);
        match result {
            DotCommandResult::Output(s) => {
                assert!(s.contains("SELECT count(*)"));
                assert!(s.contains("SELECT * FROM users"));
                assert!(s.contains("1")); // entry numbers
                assert!(s.contains("2"));
            }
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_handle_clear() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Clear, &ctx);
        assert!(matches!(result, DotCommandResult::ClearScreen));
    }

    #[test]
    fn test_handle_help() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Help, &ctx);
        match result {
            DotCommandResult::Output(s) => {
                assert!(s.contains(".tables"));
                assert!(s.contains(".schema"));
                assert!(s.contains(".gpu"));
                assert!(s.contains(".help"));
                assert!(s.contains(".quit"));
            }
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_handle_quit() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Quit, &ctx);
        assert!(matches!(result, DotCommandResult::Quit));
    }

    #[test]
    fn test_handle_comparison_no_data() {
        let ctx = make_ctx(&[], &[]);
        let result = handle_dot_command(&DotCommand::Comparison, &ctx);
        match result {
            DotCommandResult::Output(s) => assert!(s.contains("No comparison")),
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_handle_comparison_with_data() {
        let tables: Vec<(String, String, Vec<(String, String)>)> = vec![];
        let history: Vec<String> = vec![];
        let ctx = DotCommandContext {
            tables: &tables,
            history: &history,
            last_result_text: None,
            current_format: &OutputFormat::Table,
            profile_on: false,
            timer_on: false,
            gpu_device_name: "Test".into(),
            gpu_memory_bytes: 0,
            last_comparison: Some("GPU 2.3ms vs CPU ~15ms (~6.5x speedup)".into()),
        };
        let result = handle_dot_command(&DotCommand::Comparison, &ctx);
        match result {
            DotCommandResult::Output(s) => assert!(s.contains("6.5x")),
            _ => panic!("Expected Output"),
        }
    }

    // ── History persistence tests ──────────────────────────────────────

    #[test]
    fn test_append_to_history_basic() {
        let mut history = Vec::new();
        append_to_history(&mut history, "SELECT 1");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0], "SELECT 1");
    }

    #[test]
    fn test_append_to_history_dedup() {
        let mut history = vec!["SELECT 1".into()];
        append_to_history(&mut history, "SELECT 1");
        assert_eq!(history.len(), 1); // no duplicate
    }

    #[test]
    fn test_append_to_history_empty_query() {
        let mut history = Vec::new();
        append_to_history(&mut history, "");
        append_to_history(&mut history, "   ");
        assert_eq!(history.len(), 0);
    }

    #[test]
    fn test_append_to_history_max_size() {
        let mut history: Vec<String> = (0..MAX_HISTORY_SIZE)
            .map(|i| format!("query {}", i))
            .collect();
        assert_eq!(history.len(), MAX_HISTORY_SIZE);

        append_to_history(&mut history, "new query");
        assert_eq!(history.len(), MAX_HISTORY_SIZE);
        assert_eq!(history.last().unwrap(), "new query");
        // First entry should have been dropped
        assert_eq!(history[0], "query 1");
    }

    #[test]
    fn test_dot_command_result_display() {
        assert_eq!(DotCommandResult::Output("hello".into()).to_string(), "hello");
        assert_eq!(DotCommandResult::StateChange("changed".into()).to_string(), "changed");
        assert_eq!(DotCommandResult::ClearScreen.to_string(), "(screen cleared)");
        assert_eq!(DotCommandResult::Quit.to_string(), "Goodbye.");
        assert_eq!(DotCommandResult::Error("bad".into()).to_string(), "Error: bad");
    }

    #[test]
    fn test_parse_on_off() {
        assert_eq!(super::parse_on_off("on"), Some(true));
        assert_eq!(super::parse_on_off("ON"), Some(true));
        assert_eq!(super::parse_on_off("off"), Some(false));
        assert_eq!(super::parse_on_off("OFF"), Some(false));
        assert_eq!(super::parse_on_off("1"), Some(true));
        assert_eq!(super::parse_on_off("0"), Some(false));
        assert_eq!(super::parse_on_off("true"), Some(true));
        assert_eq!(super::parse_on_off("false"), Some(false));
        assert_eq!(super::parse_on_off("yes"), Some(true));
        assert_eq!(super::parse_on_off("no"), Some(false));
        assert_eq!(super::parse_on_off("maybe"), None);
        assert_eq!(super::parse_on_off(""), None);
    }

    #[test]
    fn test_history_file_path() {
        // Should return a valid path when HOME is set
        if std::env::var("HOME").is_ok() {
            let path = history_file_path();
            assert!(path.is_some());
            let p = path.unwrap();
            assert!(p.to_string_lossy().contains("gpu-query"));
            assert!(p.to_string_lossy().contains("history"));
        }
    }

    #[test]
    fn test_history_persistence_roundtrip() {
        // Use a temp dir to avoid polluting real config
        let tmp = std::env::temp_dir().join("gpu_query_test_history");
        let _ = std::fs::create_dir_all(&tmp);
        let history_path = tmp.join("history");

        let queries = vec![
            "SELECT count(*) FROM sales".to_string(),
            "SELECT * FROM users WHERE id > 10".to_string(),
        ];

        // Write
        let content: String = queries.iter().map(|q| format!("{}\n", q)).collect();
        std::fs::write(&history_path, &content).unwrap();

        // Read back
        let read_back = std::fs::read_to_string(&history_path).unwrap();
        let loaded: Vec<String> = read_back
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect();

        assert_eq!(loaded, queries);

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
