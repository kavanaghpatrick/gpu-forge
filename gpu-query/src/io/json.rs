//! CPU-side NDJSON metadata reader.
//!
//! Parses the first line of an NDJSON file to extract field names and infer
//! types. NDJSON = one JSON object per line, no wrapping array.
//!
//! Type inference from the first record:
//! - Integer-looking values (digits, optional leading minus) -> Int64
//! - Float-looking values (contain '.') -> Float64
//! - Everything else (strings, booleans, nulls) -> Varchar

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::storage::schema::{ColumnDef, DataType, RuntimeSchema};

/// Metadata extracted from an NDJSON file's first record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdjsonMetadata {
    /// Field names extracted from the first JSON object.
    pub field_names: Vec<String>,
    /// Inferred data types for each field.
    pub field_types: Vec<DataType>,
    /// Number of fields (== field_names.len()).
    pub field_count: usize,
    /// Path to the source file.
    pub file_path: PathBuf,
}

impl NdjsonMetadata {
    /// Convert to a RuntimeSchema for GPU execution.
    pub fn to_schema(&self) -> RuntimeSchema {
        let columns = self
            .field_names
            .iter()
            .zip(self.field_types.iter())
            .map(|(name, dtype)| ColumnDef {
                name: name.clone(),
                data_type: *dtype,
                nullable: false,
            })
            .collect();
        RuntimeSchema::new(columns)
    }
}

/// Infer the DataType from a JSON value string (already stripped of quotes).
fn infer_type(value: &str) -> DataType {
    let trimmed = value.trim();

    // Check for integer: optional minus, then all digits
    if !trimmed.is_empty() {
        let start = if trimmed.starts_with('-') { 1 } else { 0 };
        let digits = &trimmed[start..];
        if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
            return DataType::Int64;
        }
    }

    // Check for float: optional minus, digits, dot, digits
    if trimmed.parse::<f64>().is_ok() && trimmed.contains('.') {
        return DataType::Float64;
    }

    DataType::Varchar
}

/// Parse a simple JSON value from a line starting at `pos`.
/// Returns (value_string, new_pos) where value_string is the raw value.
fn parse_json_value(line: &[u8], start: usize) -> (String, usize) {
    let mut pos = start;

    // Skip whitespace
    while pos < line.len() && (line[pos] == b' ' || line[pos] == b'\t') {
        pos += 1;
    }

    if pos >= line.len() {
        return (String::new(), pos);
    }

    if line[pos] == b'"' {
        // String value - find closing quote
        pos += 1; // skip opening quote
        let val_start = pos;
        while pos < line.len() && line[pos] != b'"' {
            if line[pos] == b'\\' {
                pos += 1; // skip escaped character
            }
            pos += 1;
        }
        let val = String::from_utf8_lossy(&line[val_start..pos]).to_string();
        if pos < line.len() {
            pos += 1; // skip closing quote
        }
        (val, pos)
    } else {
        // Number, boolean, or null
        let val_start = pos;
        while pos < line.len() && line[pos] != b',' && line[pos] != b'}' && line[pos] != b' ' {
            pos += 1;
        }
        let val = String::from_utf8_lossy(&line[val_start..pos]).to_string();
        (val, pos)
    }
}

/// Parse the first line of an NDJSON file to extract field names and types.
///
/// Reads the first line, parses as a JSON object (simple key-value parser,
/// not a full JSON parser), extracts field names in order, and infers types.
///
/// # Errors
/// Returns an error if the file cannot be opened, is empty, or the first
/// line is not a valid JSON object.
pub fn parse_ndjson_header<P: AsRef<Path>>(path: P) -> io::Result<NdjsonMetadata> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut first_line = String::new();
    let n = reader.read_line(&mut first_line)?;
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "NDJSON file is empty",
        ));
    }

    let line = first_line.trim();
    if !line.starts_with('{') || !line.ends_with('}') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "First line is not a JSON object",
        ));
    }

    // Simple JSON object parser: extract key-value pairs
    let bytes = line.as_bytes();
    let mut pos = 1; // skip '{'
    let mut field_names = Vec::new();
    let mut field_types = Vec::new();

    loop {
        // Skip whitespace
        while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\t') {
            pos += 1;
        }

        if pos >= bytes.len() || bytes[pos] == b'}' {
            break;
        }

        // Parse key (must be quoted string)
        if bytes[pos] != b'"' {
            break;
        }
        pos += 1; // skip opening quote
        let key_start = pos;
        while pos < bytes.len() && bytes[pos] != b'"' {
            pos += 1;
        }
        let key = String::from_utf8_lossy(&bytes[key_start..pos]).to_string();
        pos += 1; // skip closing quote

        // Skip whitespace and colon
        while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\t') {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b':' {
            pos += 1;
        }

        // Parse value
        let (value, new_pos) = parse_json_value(bytes, pos);
        pos = new_pos;

        let dtype = infer_type(&value);
        field_names.push(key);
        field_types.push(dtype);

        // Skip whitespace and comma
        while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\t') {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b',' {
            pos += 1;
        }
    }

    if field_names.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No fields found in first JSON object",
        ));
    }

    let field_count = field_names.len();

    Ok(NdjsonMetadata {
        field_names,
        field_types,
        field_count,
        file_path: path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper: create a temp NDJSON file with given content.
    fn make_ndjson(content: &str) -> NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(".ndjson")
            .tempfile()
            .expect("create temp ndjson");
        f.write_all(content.as_bytes()).expect("write ndjson");
        f.flush().expect("flush ndjson");
        f
    }

    #[test]
    fn test_parse_basic_ndjson() {
        let tmp = make_ndjson(
            r#"{"id":1,"amount":100,"name":"alice"}
{"id":2,"amount":200,"name":"bob"}
"#,
        );
        let meta = parse_ndjson_header(tmp.path()).unwrap();
        assert_eq!(meta.field_names, vec!["id", "amount", "name"]);
        assert_eq!(
            meta.field_types,
            vec![DataType::Int64, DataType::Int64, DataType::Varchar]
        );
        assert_eq!(meta.field_count, 3);
    }

    #[test]
    fn test_parse_float_fields() {
        let tmp = make_ndjson(r#"{"x":1.5,"y":2.0,"z":3}"#);
        let meta = parse_ndjson_header(tmp.path()).unwrap();
        assert_eq!(meta.field_names, vec!["x", "y", "z"]);
        assert_eq!(
            meta.field_types,
            vec![DataType::Float64, DataType::Float64, DataType::Int64]
        );
    }

    #[test]
    fn test_parse_negative_int() {
        let tmp = make_ndjson(r#"{"val":-42,"name":"test"}"#);
        let meta = parse_ndjson_header(tmp.path()).unwrap();
        assert_eq!(meta.field_types[0], DataType::Int64);
    }

    #[test]
    fn test_parse_empty_file() {
        let tmp = make_ndjson("");
        let result = parse_ndjson_header(tmp.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_not_json_object() {
        let tmp = make_ndjson("[1, 2, 3]\n");
        let result = parse_ndjson_header(tmp.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_to_schema() {
        let tmp = make_ndjson(r#"{"id":1,"amount":100,"name":"alice"}"#);
        let meta = parse_ndjson_header(tmp.path()).unwrap();
        let schema = meta.to_schema();
        assert_eq!(schema.num_columns(), 3);
        assert_eq!(schema.columns[0].name, "id");
        assert_eq!(schema.columns[0].data_type, DataType::Int64);
        assert_eq!(schema.columns[2].name, "name");
        assert_eq!(schema.columns[2].data_type, DataType::Varchar);
    }

    #[test]
    fn test_file_path_stored() {
        let tmp = make_ndjson(r#"{"x":1}"#);
        let meta = parse_ndjson_header(tmp.path()).unwrap();
        assert_eq!(meta.file_path, tmp.path());
    }

    #[test]
    fn test_parse_spaced_json() {
        let tmp = make_ndjson(r#"{"id": 1, "amount": 200, "name": "bob"}"#);
        let meta = parse_ndjson_header(tmp.path()).unwrap();
        assert_eq!(meta.field_names, vec!["id", "amount", "name"]);
        assert_eq!(
            meta.field_types,
            vec![DataType::Int64, DataType::Int64, DataType::Varchar]
        );
    }
}
