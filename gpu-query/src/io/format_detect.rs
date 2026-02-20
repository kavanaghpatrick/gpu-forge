//! File format detection via magic bytes / content inspection.
//!
//! Detects Parquet (PAR1 header), JSON (leading `{` or `[`), and CSV
//! (default fallback) by inspecting the first bytes of a file.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Supported file formats for gpu-query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileFormat {
    /// Comma/tab/pipe separated values.
    Csv,
    /// Apache Parquet columnar format (PAR1 magic header).
    Parquet,
    /// JSON (object or array) / NDJSON.
    Json,
    /// Unrecognized format.
    Unknown,
}

impl std::fmt::Display for FileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileFormat::Csv => write!(f, "CSV"),
            FileFormat::Parquet => write!(f, "Parquet"),
            FileFormat::Json => write!(f, "JSON"),
            FileFormat::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Parquet magic bytes: ASCII "PAR1" at byte offset 0.
const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

/// Maximum number of bytes to read for format detection.
const DETECT_BUF_SIZE: usize = 256;

/// Detect the file format of the file at `path` by inspecting its contents.
///
/// Detection rules (applied in order):
/// 1. First 4 bytes == `PAR1` → [`FileFormat::Parquet`]
/// 2. First non-whitespace byte is `{` or `[` → [`FileFormat::Json`]
/// 3. File has a recognized extension (.parquet, .json, .jsonl, .ndjson, .csv, .tsv) → corresponding format
/// 4. Otherwise → [`FileFormat::Csv`] (default assumption for text data)
///
/// # Errors
/// Returns an I/O error if the file cannot be opened or read.
pub fn detect_format<P: AsRef<Path>>(path: P) -> io::Result<FileFormat> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let mut buf = [0u8; DETECT_BUF_SIZE];
    let n = file.read(&mut buf)?;

    if n == 0 {
        // Empty file — check extension as fallback
        return Ok(format_from_extension(path));
    }

    let bytes = &buf[..n];

    // Rule 1: Parquet magic bytes
    if n >= 4 && &bytes[..4] == PARQUET_MAGIC {
        return Ok(FileFormat::Parquet);
    }

    // Rule 2: JSON — first non-whitespace byte is `{` or `[`
    if let Some(first_non_ws) = bytes.iter().find(|b| !b.is_ascii_whitespace()) {
        if *first_non_ws == b'{' || *first_non_ws == b'[' {
            return Ok(FileFormat::Json);
        }
    }

    // Rule 3: Extension-based fallback
    let ext_format = format_from_extension(path);
    if ext_format != FileFormat::Unknown {
        return Ok(ext_format);
    }

    // Rule 4: Default to CSV for text-like data
    Ok(FileFormat::Csv)
}

/// Attempt format detection from file extension alone.
fn format_from_extension(path: &Path) -> FileFormat {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("parquet") | Some("pq") => FileFormat::Parquet,
        Some("json") | Some("jsonl") | Some("ndjson") => FileFormat::Json,
        Some("csv") | Some("tsv") | Some("txt") | Some("dat") | Some("log") => FileFormat::Csv,
        _ => FileFormat::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper: create a temp file with given content and extension.
    fn make_temp(content: &[u8], suffix: &str) -> NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(suffix)
            .tempfile()
            .expect("create temp file");
        f.write_all(content).expect("write temp");
        f.flush().expect("flush temp");
        f
    }

    #[test]
    fn test_detect_parquet_magic() {
        // PAR1 followed by arbitrary data
        let mut data = b"PAR1".to_vec();
        data.extend_from_slice(&[0u8; 100]);
        let tmp = make_temp(&data, ".bin");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Parquet);
    }

    #[test]
    fn test_detect_parquet_with_extension() {
        // Non-PAR1 content but .parquet extension → extension fallback
        let tmp = make_temp(b"not parquet data", ".parquet");
        // Content doesn't start with PAR1 and doesn't look like JSON,
        // but extension says Parquet → extension wins
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Parquet);
    }

    #[test]
    fn test_detect_json_object() {
        let tmp = make_temp(b"{\"key\": \"value\"}", ".dat");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Json);
    }

    #[test]
    fn test_detect_json_array() {
        let tmp = make_temp(b"[1, 2, 3]", ".dat");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Json);
    }

    #[test]
    fn test_detect_json_with_leading_whitespace() {
        let tmp = make_temp(b"  \n  {\"key\": 1}", ".txt");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Json);
    }

    #[test]
    fn test_detect_json_extension() {
        let tmp = make_temp(b"plain text", ".json");
        // Content is plain text but extension is .json → extension fallback
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Json);
    }

    #[test]
    fn test_detect_ndjson_extension() {
        let tmp = make_temp(b"{\"a\":1}\n{\"a\":2}", ".ndjson");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Json);
    }

    #[test]
    fn test_detect_csv_by_content() {
        let tmp = make_temp(b"id,name,amount\n1,alice,100\n", ".dat");
        // Not PAR1, not JSON → CSV by extension (.dat is recognized)
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Csv);
    }

    #[test]
    fn test_detect_csv_by_extension() {
        let tmp = make_temp(b"a,b,c\n1,2,3\n", ".csv");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Csv);
    }

    #[test]
    fn test_detect_tsv_by_extension() {
        let tmp = make_temp(b"a\tb\tc\n1\t2\t3\n", ".tsv");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Csv);
    }

    #[test]
    fn test_detect_unknown_extension_defaults_csv() {
        // Plain text content with unknown extension → defaults to CSV
        let tmp = make_temp(b"hello world\nfoo bar\n", ".xyz");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Csv);
    }

    #[test]
    fn test_detect_empty_file_with_csv_ext() {
        let tmp = make_temp(b"", ".csv");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Csv);
    }

    #[test]
    fn test_detect_empty_file_unknown_ext() {
        let tmp = make_temp(b"", ".xyz");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Unknown);
    }

    #[test]
    fn test_detect_nonexistent_file() {
        let result = detect_format("/tmp/nonexistent_format_detect_test_12345.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_file_format_display() {
        assert_eq!(format!("{}", FileFormat::Csv), "CSV");
        assert_eq!(format!("{}", FileFormat::Parquet), "Parquet");
        assert_eq!(format!("{}", FileFormat::Json), "JSON");
        assert_eq!(format!("{}", FileFormat::Unknown), "Unknown");
    }

    #[test]
    fn test_detect_parquet_beats_json_extension() {
        // PAR1 magic should win over .json extension
        let mut data = b"PAR1".to_vec();
        data.extend_from_slice(&[0u8; 50]);
        let tmp = make_temp(&data, ".json");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Parquet);
    }

    #[test]
    fn test_detect_json_content_beats_csv_extension() {
        // JSON content should win over .csv extension
        let tmp = make_temp(b"{\"a\": 1}", ".csv");
        assert_eq!(detect_format(tmp.path()).unwrap(), FileFormat::Json);
    }
}
