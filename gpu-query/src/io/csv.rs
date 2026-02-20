//! CPU-side CSV metadata reader.
//!
//! Parses the header line of a CSV file to extract column names, detect the
//! delimiter (comma, tab, or pipe by frequency analysis), and count columns.
//! This metadata is used by the GPU CSV parser kernel to set up SoA buffers.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Metadata extracted from a CSV file header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsvMetadata {
    /// Column names extracted from the header line.
    pub column_names: Vec<String>,
    /// Detected delimiter character.
    pub delimiter: u8,
    /// Number of columns (== column_names.len()).
    pub column_count: usize,
    /// Path to the source file.
    pub file_path: PathBuf,
}

impl CsvMetadata {
    /// The delimiter as a displayable character.
    pub fn delimiter_char(&self) -> char {
        self.delimiter as char
    }
}

/// Candidate delimiters in priority order.
const CANDIDATES: &[u8] = b",\t|";

/// Detect the most likely delimiter in a header line.
///
/// Counts occurrences of each candidate delimiter and returns the one with
/// the highest frequency. If no candidate appears, defaults to comma.
fn detect_delimiter(line: &str) -> u8 {
    let mut best = b',';
    let mut best_count = 0usize;

    for &delim in CANDIDATES {
        let count = line.bytes().filter(|&b| b == delim).count();
        if count > best_count {
            best_count = count;
            best = delim;
        }
    }

    best
}

/// Parse the header of a CSV file and return metadata.
///
/// Reads the first line, detects the delimiter, splits into column names,
/// and trims whitespace from each name. Column names that are empty after
/// trimming are replaced with `_col_N` (0-indexed).
///
/// # Errors
/// Returns an error if the file cannot be opened, is empty, or has no
/// parseable header line.
pub fn parse_header<P: AsRef<Path>>(path: P) -> io::Result<CsvMetadata> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut header_line = String::new();
    let n = reader.read_line(&mut header_line)?;
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CSV file is empty — no header line",
        ));
    }

    // Strip trailing newline / carriage return
    let line = header_line.trim_end_matches(['\n', '\r']);

    if line.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CSV header line is empty",
        ));
    }

    let delimiter = detect_delimiter(line);

    let column_names: Vec<String> = line
        .split(delimiter as char)
        .enumerate()
        .map(|(i, name)| {
            let trimmed = name.trim().to_string();
            if trimmed.is_empty() {
                format!("_col_{}", i)
            } else {
                trimmed
            }
        })
        .collect();

    let column_count = column_names.len();

    Ok(CsvMetadata {
        column_names,
        delimiter,
        column_count,
        file_path: path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper: create a temp CSV file with given content.
    fn make_csv(content: &str) -> NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(".csv")
            .tempfile()
            .expect("create temp csv");
        f.write_all(content.as_bytes()).expect("write csv");
        f.flush().expect("flush csv");
        f
    }

    #[test]
    fn test_parse_comma_delimited() {
        let tmp = make_csv("id,name,amount\n1,alice,100\n2,bob,200\n");
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.column_names, vec!["id", "name", "amount"]);
        assert_eq!(meta.delimiter, b',');
        assert_eq!(meta.column_count, 3);
    }

    #[test]
    fn test_parse_tab_delimited() {
        let tmp = make_csv("id\tname\tamount\n1\talice\t100\n");
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.column_names, vec!["id", "name", "amount"]);
        assert_eq!(meta.delimiter, b'\t');
        assert_eq!(meta.column_count, 3);
    }

    #[test]
    fn test_parse_pipe_delimited() {
        let tmp = make_csv("id|name|amount\n1|alice|100\n");
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.column_names, vec!["id", "name", "amount"]);
        assert_eq!(meta.delimiter, b'|');
        assert_eq!(meta.column_count, 3);
    }

    #[test]
    fn test_parse_single_column() {
        let tmp = make_csv("value\n42\n99\n");
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.column_names, vec!["value"]);
        assert_eq!(meta.delimiter, b','); // default when no delimiter found
        assert_eq!(meta.column_count, 1);
    }

    #[test]
    fn test_parse_many_columns() {
        let header = (0..20)
            .map(|i| format!("col{}", i))
            .collect::<Vec<_>>()
            .join(",");
        let content = format!(
            "{}\n1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\n",
            header
        );
        let tmp = make_csv(&content);
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.column_count, 20);
        assert_eq!(meta.column_names[0], "col0");
        assert_eq!(meta.column_names[19], "col19");
    }

    #[test]
    fn test_parse_whitespace_in_names() {
        let tmp = make_csv(" id , name , amount \n1,alice,100\n");
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.column_names, vec!["id", "name", "amount"]);
    }

    #[test]
    fn test_parse_empty_column_name() {
        let tmp = make_csv("id,,amount\n1,x,100\n");
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.column_names, vec!["id", "_col_1", "amount"]);
        assert_eq!(meta.column_count, 3);
    }

    #[test]
    fn test_parse_crlf_line_ending() {
        let tmp = make_csv("id,name,amount\r\n1,alice,100\r\n");
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.column_names, vec!["id", "name", "amount"]);
    }

    #[test]
    fn test_parse_header_only_no_data() {
        let tmp = make_csv("x,y,z\n");
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.column_names, vec!["x", "y", "z"]);
        assert_eq!(meta.column_count, 3);
    }

    #[test]
    fn test_parse_empty_file() {
        let tmp = make_csv("");
        let result = parse_header(tmp.path());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_parse_empty_line() {
        // File with only a newline
        let tmp = make_csv("\n");
        let result = parse_header(tmp.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_nonexistent_file() {
        let result = parse_header("/tmp/nonexistent_csv_test_12345.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_delimiter_char() {
        let tmp = make_csv("a\tb\n1\t2\n");
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.delimiter_char(), '\t');
    }

    #[test]
    fn test_file_path_stored() {
        let tmp = make_csv("x,y\n1,2\n");
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.file_path, tmp.path());
    }

    #[test]
    fn test_detect_delimiter_mixed_counts() {
        // More tabs than commas → tab wins
        assert_eq!(detect_delimiter("a\tb\tc,d"), b'\t');
        // More commas than tabs → comma wins
        assert_eq!(detect_delimiter("a,b,c\td"), b',');
        // Equal counts → comma wins (first in candidate list)
        // Actually comma is checked first, so if counts are equal,
        // whichever was set first with > stays. Let's verify:
        // "a,b|c" → comma=1, tab=0, pipe=1 → comma wins (set first)
        assert_eq!(detect_delimiter("a,b|c"), b',');
    }

    #[test]
    fn test_detect_delimiter_no_delimiters() {
        // No delimiters at all → defaults to comma
        assert_eq!(detect_delimiter("hello"), b',');
    }

    #[test]
    fn test_parse_unicode_column_names() {
        let tmp = make_csv("名前,金額,日付\n太郎,100,2024-01-01\n");
        let meta = parse_header(tmp.path()).unwrap();
        assert_eq!(meta.column_names, vec!["名前", "金額", "日付"]);
        assert_eq!(meta.column_count, 3);
    }
}
