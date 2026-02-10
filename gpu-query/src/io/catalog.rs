//! Directory scanner and data catalog.
//!
//! Walks a directory, detects file formats, and collects metadata for
//! queryable tables. CSV files get header parsing; other formats record
//! path and format only.

use std::io;
use std::path::{Path, PathBuf};

use super::csv::{self, CsvMetadata};
use super::format_detect::{self, FileFormat};

/// An entry in the data catalog representing one queryable table.
#[derive(Debug, Clone)]
pub struct TableEntry {
    /// Table name derived from file stem (e.g. "sales" from "sales.csv").
    pub name: String,
    /// Detected file format.
    pub format: FileFormat,
    /// Absolute or canonical path to the file.
    pub path: PathBuf,
    /// CSV metadata (column names, delimiter) — `Some` only for CSV files.
    pub csv_metadata: Option<CsvMetadata>,
}

/// Scan a directory for data files and return a catalog of table entries.
///
/// Walks the immediate contents of `dir` (non-recursive), detects each file's
/// format, and for CSV files also parses the header. Directories, symlinks to
/// directories, and files with [`FileFormat::Unknown`] are skipped.
///
/// The returned entries are sorted by table name for deterministic output.
///
/// # Errors
/// Returns an I/O error if `dir` cannot be read. Individual file errors
/// (e.g., permission denied on one file) are silently skipped.
pub fn scan_directory<P: AsRef<Path>>(dir: P) -> io::Result<Vec<TableEntry>> {
    let dir = dir.as_ref();
    let mut entries = Vec::new();

    for dir_entry in std::fs::read_dir(dir)? {
        let dir_entry = match dir_entry {
            Ok(e) => e,
            Err(_) => continue, // skip unreadable entries
        };

        let path = dir_entry.path();

        // Skip directories
        if path.is_dir() {
            continue;
        }

        // Detect format
        let format = match format_detect::detect_format(&path) {
            Ok(f) => f,
            Err(_) => continue, // skip unreadable files
        };

        // Skip unknown formats
        if format == FileFormat::Unknown {
            continue;
        }

        // Derive table name from file stem
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Parse CSV metadata if applicable
        let csv_metadata = if format == FileFormat::Csv {
            csv::parse_header(&path).ok()
        } else {
            None
        };

        entries.push(TableEntry {
            name,
            format,
            path,
            csv_metadata,
        });
    }

    // Sort by name for deterministic output
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// Helper: create a file in a temp directory.
    fn write_file(dir: &Path, name: &str, content: &[u8]) {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).expect("create file");
        f.write_all(content).expect("write file");
    }

    #[test]
    fn test_scan_csv_files() {
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "sales.csv", b"id,amount,region\n1,100,US\n2,200,EU\n");
        write_file(tmp.path(), "users.csv", b"uid,name\n1,alice\n");

        let catalog = scan_directory(tmp.path()).unwrap();
        assert_eq!(catalog.len(), 2);

        // Sorted by name
        assert_eq!(catalog[0].name, "sales");
        assert_eq!(catalog[0].format, FileFormat::Csv);
        assert!(catalog[0].csv_metadata.is_some());
        let meta = catalog[0].csv_metadata.as_ref().unwrap();
        assert_eq!(meta.column_names, vec!["id", "amount", "region"]);
        assert_eq!(meta.column_count, 3);

        assert_eq!(catalog[1].name, "users");
        assert_eq!(catalog[1].format, FileFormat::Csv);
        let meta = catalog[1].csv_metadata.as_ref().unwrap();
        assert_eq!(meta.column_names, vec!["uid", "name"]);
    }

    #[test]
    fn test_scan_mixed_formats() {
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "data.csv", b"x,y\n1,2\n");
        write_file(tmp.path(), "records.json", b"{\"a\": 1}\n{\"a\": 2}\n");

        // Write a fake Parquet file (PAR1 magic)
        let mut pq = b"PAR1".to_vec();
        pq.extend_from_slice(&[0u8; 100]);
        write_file(tmp.path(), "table.parquet", &pq);

        let catalog = scan_directory(tmp.path()).unwrap();
        assert_eq!(catalog.len(), 3);

        // Sorted by name: data, records, table
        assert_eq!(catalog[0].name, "data");
        assert_eq!(catalog[0].format, FileFormat::Csv);
        assert!(catalog[0].csv_metadata.is_some());

        assert_eq!(catalog[1].name, "records");
        assert_eq!(catalog[1].format, FileFormat::Json);
        assert!(catalog[1].csv_metadata.is_none());

        assert_eq!(catalog[2].name, "table");
        assert_eq!(catalog[2].format, FileFormat::Parquet);
        assert!(catalog[2].csv_metadata.is_none());
    }

    #[test]
    fn test_scan_empty_directory() {
        let tmp = TempDir::new().unwrap();
        let catalog = scan_directory(tmp.path()).unwrap();
        assert!(catalog.is_empty());
    }

    #[test]
    fn test_scan_skips_subdirectories() {
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "data.csv", b"a,b\n1,2\n");
        std::fs::create_dir(tmp.path().join("subdir")).unwrap();
        write_file(
            &tmp.path().join("subdir"),
            "nested.csv",
            b"x,y\n3,4\n",
        );

        let catalog = scan_directory(tmp.path()).unwrap();
        // Only top-level files
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog[0].name, "data");
    }

    #[test]
    fn test_scan_nonexistent_directory() {
        let result = scan_directory("/tmp/nonexistent_catalog_test_dir_12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_scan_tab_delimited_csv() {
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "tsv_data.tsv", b"col_a\tcol_b\tcol_c\n1\t2\t3\n");

        let catalog = scan_directory(tmp.path()).unwrap();
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog[0].format, FileFormat::Csv);
        let meta = catalog[0].csv_metadata.as_ref().unwrap();
        assert_eq!(meta.delimiter, b'\t');
        assert_eq!(meta.column_names, vec!["col_a", "col_b", "col_c"]);
    }

    #[test]
    fn test_table_entry_path_is_correct() {
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "test.csv", b"a,b\n1,2\n");

        let catalog = scan_directory(tmp.path()).unwrap();
        assert_eq!(catalog[0].path, tmp.path().join("test.csv"));
    }
}
