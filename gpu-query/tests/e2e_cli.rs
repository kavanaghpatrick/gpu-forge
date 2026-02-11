//! End-to-end CLI integration tests.
//!
//! Tests the full non-interactive CLI pipeline:
//! gpu-query <dir> -e "SQL" [--format csv|json|jsonl|table] [-o output]
//!
//! Uses cargo binary execution via std::process::Command.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use tempfile::{NamedTempFile, TempDir};

/// Get the path to the compiled gpu-query binary.
fn gpu_query_bin() -> PathBuf {
    // cargo test puts binaries in target/debug/
    let mut path = std::env::current_exe()
        .expect("current_exe")
        .parent()
        .expect("parent")
        .parent()
        .expect("grandparent")
        .to_path_buf();
    path.push("gpu-query");
    path
}

/// Create a temp directory with a simple CSV file.
fn make_test_dir() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("create temp dir");
    let csv_path = dir.path().join("sales.csv");
    std::fs::write(
        &csv_path,
        "id,name,amount,region\n\
         1,Alice,500,East\n\
         2,Bob,200,West\n\
         3,Charlie,800,East\n\
         4,Diana,150,West\n\
         5,Eve,1000,East\n",
    )
    .expect("write csv");
    let dir_path = dir.path().to_path_buf();
    (dir, dir_path)
}

/// Run gpu-query with given args and return (stdout, stderr, exit_code).
fn run_cli(args: &[&str]) -> (String, String, i32) {
    let bin = gpu_query_bin();
    let output = Command::new(&bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?}: {}", bin, e));

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

// ---- Basic query tests ----

#[test]
fn test_cli_count_query() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();
    let (stdout, stderr, code) = run_cli(&[dir_str, "-e", "SELECT count(*) FROM sales"]);

    assert_eq!(code, 0, "exit code should be 0, stderr: {}", stderr);
    assert!(
        stdout.contains("5"),
        "should contain count=5, got: {}",
        stdout
    );
}

#[test]
fn test_cli_filtered_count() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();
    let (stdout, stderr, code) = run_cli(&[
        dir_str,
        "-e",
        "SELECT count(*) FROM sales WHERE amount > 300",
    ]);

    assert_eq!(code, 0, "exit code should be 0, stderr: {}", stderr);
    assert!(
        stdout.contains("3"),
        "should contain count=3, got: {}",
        stdout
    );
}

#[test]
fn test_cli_sum_query() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();
    let (stdout, stderr, code) = run_cli(&[dir_str, "-e", "SELECT sum(amount) FROM sales"]);

    assert_eq!(code, 0, "exit code should be 0, stderr: {}", stderr);
    assert!(
        stdout.contains("2650"),
        "should contain sum=2650, got: {}",
        stdout
    );
}

// ---- Output format tests ----

#[test]
fn test_cli_format_csv() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();
    let (stdout, stderr, code) = run_cli(&[
        dir_str,
        "-e",
        "SELECT count(*) FROM sales",
        "--format",
        "csv",
    ]);

    assert_eq!(code, 0, "exit code should be 0, stderr: {}", stderr);
    let lines: Vec<&str> = stdout.trim().lines().collect();
    assert!(
        lines.len() >= 2,
        "csv should have header + data, got: {}",
        stdout
    );
    // First line should be header
    assert!(
        lines[0].contains("count"),
        "csv header should contain 'count', got: {}",
        lines[0]
    );
}

#[test]
fn test_cli_format_json() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();
    let (stdout, stderr, code) = run_cli(&[
        dir_str,
        "-e",
        "SELECT count(*) FROM sales",
        "--format",
        "json",
    ]);

    assert_eq!(code, 0, "exit code should be 0, stderr: {}", stderr);
    assert!(
        stdout.trim().starts_with('['),
        "json should start with [, got: {}",
        stdout
    );
    assert!(
        stdout.trim().ends_with(']'),
        "json should end with ], got: {}",
        stdout
    );
}

#[test]
fn test_cli_format_jsonl() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();
    let (stdout, stderr, code) = run_cli(&[
        dir_str,
        "-e",
        "SELECT count(*) FROM sales",
        "--format",
        "jsonl",
    ]);

    assert_eq!(code, 0, "exit code should be 0, stderr: {}", stderr);
    let lines: Vec<&str> = stdout.trim().lines().collect();
    assert_eq!(lines.len(), 1, "jsonl should have 1 line for 1 result row");
    assert!(
        lines[0].starts_with('{'),
        "jsonl line should be JSON object, got: {}",
        lines[0]
    );
}

#[test]
fn test_cli_format_table() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();
    let (stdout, stderr, code) = run_cli(&[
        dir_str,
        "-e",
        "SELECT count(*) FROM sales",
        "--format",
        "table",
    ]);

    assert_eq!(code, 0, "exit code should be 0, stderr: {}", stderr);
    assert!(
        stdout.contains("---"),
        "table format should have separator, got: {}",
        stdout
    );
    assert!(
        stdout.contains("row"),
        "table format should have row count, got: {}",
        stdout
    );
}

// ---- Output file tests ----

#[test]
fn test_cli_output_file() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();
    let output_file = tempfile::NamedTempFile::new().expect("create temp output");
    let output_path = output_file.path().to_str().unwrap().to_string();

    let (_, stderr, code) = run_cli(&[
        dir_str,
        "-e",
        "SELECT count(*) FROM sales",
        "--format",
        "csv",
        "-o",
        &output_path,
    ]);

    assert_eq!(code, 0, "exit code should be 0, stderr: {}", stderr);

    let content = std::fs::read_to_string(&output_path).expect("read output file");
    assert!(
        content.contains("count"),
        "output file should contain result, got: {}",
        content
    );
}

// ---- SQL file input tests ----

#[test]
fn test_cli_sql_file_input() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();

    let mut sql_file = NamedTempFile::new().expect("create sql file");
    write!(sql_file, "SELECT count(*) FROM sales").expect("write sql");
    sql_file.flush().expect("flush sql");
    let sql_path = sql_file.path().to_str().unwrap().to_string();

    let (stdout, stderr, code) = run_cli(&[dir_str, "-f", &sql_path]);

    assert_eq!(code, 0, "exit code should be 0, stderr: {}", stderr);
    assert!(
        stdout.contains("5"),
        "should contain count=5, got: {}",
        stdout
    );
}

// ---- Error handling tests ----

#[test]
fn test_cli_invalid_directory() {
    let (_, stderr, code) = run_cli(&["/nonexistent/path", "-e", "SELECT 1"]);
    assert_ne!(code, 0, "should fail on invalid directory");
    assert!(
        stderr.contains("Error") || stderr.contains("error"),
        "should report error, got: {}",
        stderr
    );
}

#[test]
fn test_cli_invalid_sql() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();
    let (_, stderr, code) = run_cli(&[dir_str, "-e", "NOT VALID SQL AT ALL"]);
    assert_ne!(code, 0, "should fail on invalid SQL");
    assert!(
        stderr.contains("error") || stderr.contains("Error"),
        "should report parse error, got: {}",
        stderr
    );
}

#[test]
fn test_cli_no_query_provided() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();
    // No -e or -f flag, stdin is a TTY (in test env it's not piped, but
    // clap will require the directory arg at minimum)
    let (_, _stderr, code) = run_cli(&[dir_str]);
    // Should exit non-zero since no query is provided
    assert_ne!(code, 0, "should fail with no query");
}

// ---- Help and version ----

#[test]
fn test_cli_help() {
    let bin = gpu_query_bin();
    let output = Command::new(&bin)
        .arg("--help")
        .output()
        .expect("run --help");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("gpu-query") || stdout.contains("GPU"),
        "help should mention gpu-query, got: {}",
        stdout
    );
}

#[test]
fn test_cli_version() {
    let bin = gpu_query_bin();
    let output = Command::new(&bin)
        .arg("--version")
        .output()
        .expect("run --version");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("gpu-query"),
        "version should contain name, got: {}",
        stdout
    );
}

// ---- Pipe input test ----

#[test]
fn test_cli_pipe_input() {
    let (_dir, dir_path) = make_test_dir();
    let dir_str = dir_path.to_str().unwrap();

    let bin = gpu_query_bin();
    let mut child = Command::new(&bin)
        .arg(dir_str)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn gpu-query");

    // Write SQL to stdin
    let stdin = child.stdin.as_mut().expect("stdin");
    write!(stdin, "SELECT count(*) FROM sales").expect("write to stdin");
    drop(child.stdin.take()); // Close stdin to signal EOF

    let output = child.wait_with_output().expect("wait for output");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let code = output.status.code().unwrap_or(-1);

    assert_eq!(
        code,
        0,
        "pipe input should succeed, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("5"),
        "pipe query should return count=5, got: {}",
        stdout
    );
}
