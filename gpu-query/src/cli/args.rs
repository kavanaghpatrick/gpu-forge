//! Command-line argument parsing via clap derive.
//!
//! Supports non-interactive execution: parse args -> run query -> format output -> exit.
//! Also supports pipe input for SQL queries via stdin.

use clap::Parser;
use std::path::PathBuf;

/// GPU-native local data analytics engine for Apple Silicon.
///
/// Query local files (CSV, Parquet, JSON) at GPU memory bandwidth
/// using Metal compute kernels. Zero-copy via mmap.
#[derive(Parser, Debug)]
#[command(name = "gpu-query", version, about)]
pub struct CliArgs {
    /// Directory containing data files to query.
    #[arg(value_name = "DIRECTORY")]
    pub directory: PathBuf,

    /// SQL query to execute (non-interactive mode).
    #[arg(short = 'e', long = "execute", value_name = "SQL")]
    pub execute: Option<String>,

    /// Input file for SQL query (alternative to -e).
    #[arg(short = 'f', long = "file", value_name = "FILE")]
    pub file: Option<PathBuf>,

    /// Output file path (default: stdout).
    #[arg(short = 'o', long = "output", value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Output format: table, csv, json, jsonl.
    #[arg(long = "format", value_name = "FORMAT", default_value = "table")]
    pub format: OutputFormat,

    /// Disable GPU acceleration (CPU-only mode).
    #[arg(long = "no-gpu")]
    pub no_gpu: bool,

    /// Enable profiling output after query.
    #[arg(long = "profile")]
    pub profile: bool,

    /// Launch interactive dashboard TUI.
    #[arg(long = "dashboard")]
    pub dashboard: bool,

    /// Force cold execution (no caching).
    #[arg(long = "cold")]
    pub cold: bool,

    /// Color theme: thermal, glow, mono.
    #[arg(long = "theme", value_name = "THEME", default_value = "thermal")]
    pub theme: String,
}

/// Output format for non-interactive mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputFormat {
    /// Formatted ASCII table (default).
    Table,
    /// Comma-separated values.
    Csv,
    /// JSON array of objects.
    Json,
    /// Newline-delimited JSON (one object per line).
    Jsonl,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Table => write!(f, "table"),
            OutputFormat::Csv => write!(f, "csv"),
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Jsonl => write!(f, "jsonl"),
        }
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "table" => Ok(OutputFormat::Table),
            "csv" => Ok(OutputFormat::Csv),
            "json" => Ok(OutputFormat::Json),
            "jsonl" | "ndjson" => Ok(OutputFormat::Jsonl),
            _ => Err(format!(
                "unknown format '{}'. Valid: table, csv, json, jsonl",
                s
            )),
        }
    }
}

impl CliArgs {
    /// Parse CLI arguments. Returns parsed args.
    pub fn parse_args() -> Self {
        CliArgs::parse()
    }

    /// Resolve the SQL query from -e flag, -f file, or stdin pipe.
    pub fn resolve_query(&self) -> Result<String, String> {
        if let Some(ref sql) = self.execute {
            return Ok(sql.clone());
        }

        if let Some(ref path) = self.file {
            return std::fs::read_to_string(path)
                .map(|s| s.trim().to_string())
                .map_err(|e| format!("failed to read SQL file '{}': {}", path.display(), e));
        }

        // Try reading from stdin if it's piped (not a TTY)
        if !atty_is_terminal() {
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)
                .map_err(|e| format!("failed to read from stdin: {}", e))?;
            let trimmed = buf.trim().to_string();
            if !trimmed.is_empty() {
                return Ok(trimmed);
            }
        }

        Err("no query provided. Use -e \"SQL\", -f file.sql, or pipe via stdin".to_string())
    }

    /// Check if running in non-interactive (script) mode.
    pub fn is_non_interactive(&self) -> bool {
        self.execute.is_some() || self.file.is_some() || !atty_is_terminal()
    }
}

/// Check if stdin is a terminal (not piped).
fn atty_is_terminal() -> bool {
    unsafe { libc::isatty(libc::STDIN_FILENO) != 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_from_str() {
        assert_eq!(
            "table".parse::<OutputFormat>().unwrap(),
            OutputFormat::Table
        );
        assert_eq!("csv".parse::<OutputFormat>().unwrap(), OutputFormat::Csv);
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!(
            "jsonl".parse::<OutputFormat>().unwrap(),
            OutputFormat::Jsonl
        );
        assert_eq!(
            "ndjson".parse::<OutputFormat>().unwrap(),
            OutputFormat::Jsonl
        );
        assert_eq!("CSV".parse::<OutputFormat>().unwrap(), OutputFormat::Csv);
        assert!("invalid".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Table.to_string(), "table");
        assert_eq!(OutputFormat::Csv.to_string(), "csv");
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::Jsonl.to_string(), "jsonl");
    }
}
