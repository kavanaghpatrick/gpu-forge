//! CPU-side Parquet metadata reader and column chunk extraction.
//!
//! Uses the `parquet` crate to read the Parquet footer (schema, row groups,
//! column chunks). Extracts raw column bytes for GPU decoding. Supports
//! column pruning: only loads columns referenced in the query.

use std::fs::File;
use std::path::Path;

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::basic::Type as ParquetPhysicalType;
use parquet::file::metadata::RowGroupMetaData;

use crate::storage::schema::{ColumnDef, DataType, RuntimeSchema};

/// Metadata extracted from a Parquet file.
#[derive(Debug, Clone)]
pub struct ParquetMetadata {
    /// Column definitions (name + type).
    pub schema: RuntimeSchema,
    /// Total number of rows across all row groups.
    pub row_count: usize,
    /// Per-row-group metadata.
    pub row_groups: Vec<RowGroupInfo>,
}

/// Metadata for a single row group.
#[derive(Debug, Clone)]
pub struct RowGroupInfo {
    /// Number of rows in this row group.
    pub num_rows: usize,
    /// Per-column chunk info within this row group.
    pub columns: Vec<ColumnChunkInfo>,
}

/// Metadata for a single column chunk within a row group.
#[derive(Debug, Clone)]
pub struct ColumnChunkInfo {
    /// Column name.
    pub name: String,
    /// Our mapped DataType.
    pub data_type: DataType,
    /// Parquet physical type (for choosing the right GPU decode kernel).
    pub physical_type: ParquetPhysicalType,
    /// Column index in the Parquet schema.
    pub column_index: usize,
}

/// Convert a Parquet physical type to our DataType enum.
fn parquet_type_to_data_type(ptype: ParquetPhysicalType) -> DataType {
    match ptype {
        ParquetPhysicalType::INT32 => DataType::Int64,   // widen to i64
        ParquetPhysicalType::INT64 => DataType::Int64,
        ParquetPhysicalType::FLOAT => DataType::Float64,
        ParquetPhysicalType::DOUBLE => DataType::Float64,
        ParquetPhysicalType::BYTE_ARRAY | ParquetPhysicalType::FIXED_LEN_BYTE_ARRAY => {
            DataType::Varchar
        }
        ParquetPhysicalType::BOOLEAN => DataType::Bool,
        ParquetPhysicalType::INT96 => DataType::Int64,   // timestamp fallback
    }
}

/// Read Parquet file metadata (footer, schema, row groups).
///
/// Uses the `parquet` crate's reader to parse the footer and extract
/// schema information. Does NOT read column data -- that's done later
/// with column pruning.
pub fn read_metadata<P: AsRef<Path>>(path: P) -> Result<ParquetMetadata, String> {
    let file = File::open(path.as_ref())
        .map_err(|e| format!("Cannot open Parquet file '{}': {}", path.as_ref().display(), e))?;

    let reader = SerializedFileReader::new(file)
        .map_err(|e| format!("Invalid Parquet file '{}': {}", path.as_ref().display(), e))?;

    let file_meta = reader.metadata();
    let schema_descr = file_meta.file_metadata().schema_descr();
    let num_row_groups = file_meta.num_row_groups();

    // Build our schema from the Parquet schema
    let mut columns = Vec::new();
    for i in 0..schema_descr.num_columns() {
        let col = schema_descr.column(i);
        let name = col.name().to_string();
        let ptype = col.physical_type();
        let data_type = parquet_type_to_data_type(ptype);

        columns.push(ColumnDef {
            name,
            data_type,
            nullable: true,
        });
    }
    let schema = RuntimeSchema::new(columns);

    // Collect row group info
    let mut row_groups = Vec::with_capacity(num_row_groups);
    let mut total_rows = 0usize;

    for rg_idx in 0..num_row_groups {
        let rg_meta: &RowGroupMetaData = file_meta.row_group(rg_idx);
        let num_rows = rg_meta.num_rows() as usize;
        total_rows += num_rows;

        let mut rg_columns = Vec::new();
        for col_idx in 0..schema_descr.num_columns() {
            let col = schema_descr.column(col_idx);
            rg_columns.push(ColumnChunkInfo {
                name: col.name().to_string(),
                data_type: parquet_type_to_data_type(col.physical_type()),
                physical_type: col.physical_type(),
                column_index: col_idx,
            });
        }

        row_groups.push(RowGroupInfo {
            num_rows,
            columns: rg_columns,
        });
    }

    Ok(ParquetMetadata {
        schema,
        row_count: total_rows,
        row_groups,
    })
}

/// Read raw column data from a Parquet file for the specified columns.
///
/// This function reads column data using the `parquet` crate's column reader
/// and returns raw typed values ready for GPU buffer upload.
///
/// Column pruning: only reads columns whose names are in `needed_columns`.
/// If `needed_columns` is None, reads all columns.
///
/// Returns a Vec of (column_name, ColumnData) pairs for each requested column.
pub fn read_columns<P: AsRef<Path>>(
    path: P,
    metadata: &ParquetMetadata,
    needed_columns: Option<&[String]>,
) -> Result<Vec<(String, ColumnData)>, String> {
    use parquet::column::reader::ColumnReader;

    let file = File::open(path.as_ref())
        .map_err(|e| format!("Cannot open Parquet file: {}", e))?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| format!("Invalid Parquet file: {}", e))?;

    let total_rows = metadata.row_count;
    let schema_descr = reader.metadata().file_metadata().schema_descr_ptr();

    // Determine which columns to read
    let schema_cols = &metadata.schema.columns;
    let col_indices: Vec<usize> = schema_cols
        .iter()
        .enumerate()
        .filter(|(_, col)| {
            if let Some(needed) = needed_columns {
                needed.iter().any(|n| n.eq_ignore_ascii_case(&col.name))
            } else {
                true
            }
        })
        .map(|(i, _)| i)
        .collect();

    // Pre-allocate result storage per column
    let mut result: Vec<(String, ColumnData)> = col_indices
        .iter()
        .map(|&ci| {
            let col = &schema_cols[ci];
            let data = match col.data_type {
                DataType::Int64 => ColumnData::Int64(Vec::with_capacity(total_rows)),
                DataType::Float64 => ColumnData::Float64(Vec::with_capacity(total_rows)),
                _ => ColumnData::Int64(Vec::with_capacity(total_rows)), // fallback
            };
            (col.name.clone(), data)
        })
        .collect();

    // Read row groups
    for rg_idx in 0..reader.metadata().num_row_groups() {
        let rg_reader = reader
            .get_row_group(rg_idx)
            .map_err(|e| format!("Cannot read row group {}: {}", rg_idx, e))?;

        let rg_meta = reader.metadata().row_group(rg_idx);
        let num_rows = rg_meta.num_rows() as usize;

        for (result_idx, &col_idx) in col_indices.iter().enumerate() {
            let col_descr = schema_descr.column(col_idx);
            let mut col_reader = rg_reader
                .get_column_reader(col_idx)
                .map_err(|e| format!("Cannot read column {}: {}", col_idx, e))?;

            match &mut result[result_idx].1 {
                ColumnData::Int64(ref mut values) => {
                    match col_reader {
                        ColumnReader::Int64ColumnReader(ref mut r) => {
                            let mut buf = Vec::with_capacity(num_rows);
                            let mut def_levels = Vec::with_capacity(num_rows);
                            let (records_read, _, _) = r
                                .read_records(num_rows, Some(&mut def_levels), None, &mut buf)
                                .map_err(|e| format!("Read INT64 column error: {}", e))?;
                            values.extend_from_slice(&buf[..records_read]);
                        }
                        ColumnReader::Int32ColumnReader(ref mut r) => {
                            let mut buf = Vec::with_capacity(num_rows);
                            let mut def_levels = Vec::with_capacity(num_rows);
                            let (records_read, _, _) = r
                                .read_records(num_rows, Some(&mut def_levels), None, &mut buf)
                                .map_err(|e| format!("Read INT32 column error: {}", e))?;
                            // Widen INT32 to INT64
                            values.extend(buf[..records_read].iter().map(|&v| v as i64));
                        }
                        _ => {
                            return Err(format!(
                                "Unexpected column reader type for INT64 column '{}'",
                                col_descr.name()
                            ));
                        }
                    }
                }
                ColumnData::Float64(ref mut values) => {
                    match col_reader {
                        ColumnReader::FloatColumnReader(ref mut r) => {
                            let mut buf = Vec::with_capacity(num_rows);
                            let mut def_levels = Vec::with_capacity(num_rows);
                            let (records_read, _, _) = r
                                .read_records(num_rows, Some(&mut def_levels), None, &mut buf)
                                .map_err(|e| format!("Read FLOAT column error: {}", e))?;
                            values.extend(buf[..records_read].iter().map(|&v| v as f64));
                        }
                        ColumnReader::DoubleColumnReader(ref mut r) => {
                            let mut buf = Vec::with_capacity(num_rows);
                            let mut def_levels = Vec::with_capacity(num_rows);
                            let (records_read, _, _) = r
                                .read_records(num_rows, Some(&mut def_levels), None, &mut buf)
                                .map_err(|e| format!("Read DOUBLE column error: {}", e))?;
                            values.extend_from_slice(&buf[..records_read]);
                        }
                        _ => {
                            return Err(format!(
                                "Unexpected column reader type for FLOAT64 column '{}'",
                                col_descr.name()
                            ));
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Raw column data read from a Parquet file, typed by our DataType enum.
#[derive(Debug, Clone)]
pub enum ColumnData {
    /// INT64 values (also holds widened INT32).
    Int64(Vec<i64>),
    /// FLOAT64 values (also holds widened FLOAT32).
    Float64(Vec<f64>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;

    /// Helper: create a Parquet file with known INT64 data.
    fn make_test_parquet(path: &Path, ids: &[i64], amounts: &[i64], quantities: &[i64]) {
        let schema_str = "
            message test_schema {
                REQUIRED INT64 id;
                REQUIRED INT64 amount;
                REQUIRED INT64 quantity;
            }
        ";
        let schema = Arc::new(parse_message_type(schema_str).expect("parse schema"));
        let props = Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::UNCOMPRESSED)
                .build(),
        );
        let file = File::create(path).expect("create parquet file");
        let mut writer =
            SerializedFileWriter::new(file, schema, props).expect("create parquet writer");

        let mut rg_writer = writer.next_row_group().expect("next row group");

        // Write id column
        {
            let mut col_writer = rg_writer.next_column().expect("next column").unwrap();
            col_writer
                .typed::<parquet::data_type::Int64Type>()
                .write_batch(ids, None, None)
                .expect("write ids");
            col_writer.close().expect("close column");
        }

        // Write amount column
        {
            let mut col_writer = rg_writer.next_column().expect("next column").unwrap();
            col_writer
                .typed::<parquet::data_type::Int64Type>()
                .write_batch(amounts, None, None)
                .expect("write amounts");
            col_writer.close().expect("close column");
        }

        // Write quantity column
        {
            let mut col_writer = rg_writer.next_column().expect("next column").unwrap();
            col_writer
                .typed::<parquet::data_type::Int64Type>()
                .write_batch(quantities, None, None)
                .expect("write quantities");
            col_writer.close().expect("close column");
        }

        rg_writer.close().expect("close row group");
        writer.close().expect("close writer");
    }

    #[test]
    fn test_read_metadata_basic() {
        let tmp = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();

        let ids = vec![1i64, 2, 3, 4, 5];
        let amounts = vec![100i64, 200, 300, 400, 500];
        let quantities = vec![10i64, 20, 30, 40, 50];
        make_test_parquet(tmp.path(), &ids, &amounts, &quantities);

        let meta = read_metadata(tmp.path()).unwrap();
        assert_eq!(meta.row_count, 5);
        assert_eq!(meta.schema.num_columns(), 3);
        assert_eq!(meta.schema.columns[0].name, "id");
        assert_eq!(meta.schema.columns[0].data_type, DataType::Int64);
        assert_eq!(meta.schema.columns[1].name, "amount");
        assert_eq!(meta.schema.columns[2].name, "quantity");
        assert_eq!(meta.row_groups.len(), 1);
        assert_eq!(meta.row_groups[0].num_rows, 5);
    }

    #[test]
    fn test_read_columns_all() {
        let tmp = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();

        let ids = vec![1i64, 2, 3];
        let amounts = vec![100i64, 200, 300];
        let quantities = vec![10i64, 20, 30];
        make_test_parquet(tmp.path(), &ids, &amounts, &quantities);

        let meta = read_metadata(tmp.path()).unwrap();
        let cols = read_columns(tmp.path(), &meta, None).unwrap();

        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].0, "id");
        if let ColumnData::Int64(ref v) = cols[0].1 {
            assert_eq!(v, &[1, 2, 3]);
        } else {
            panic!("Expected Int64 data");
        }

        assert_eq!(cols[1].0, "amount");
        if let ColumnData::Int64(ref v) = cols[1].1 {
            assert_eq!(v, &[100, 200, 300]);
        } else {
            panic!("Expected Int64 data");
        }
    }

    #[test]
    fn test_read_columns_pruned() {
        let tmp = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();

        let ids = vec![1i64, 2, 3, 4, 5];
        let amounts = vec![100i64, 200, 300, 400, 500];
        let quantities = vec![10i64, 20, 30, 40, 50];
        make_test_parquet(tmp.path(), &ids, &amounts, &quantities);

        let meta = read_metadata(tmp.path()).unwrap();
        // Only request "amount" column -- column pruning
        let needed = vec!["amount".to_string()];
        let cols = read_columns(tmp.path(), &meta, Some(&needed)).unwrap();

        assert_eq!(cols.len(), 1, "Column pruning should only return requested columns");
        assert_eq!(cols[0].0, "amount");
        if let ColumnData::Int64(ref v) = cols[0].1 {
            assert_eq!(v, &[100, 200, 300, 400, 500]);
        } else {
            panic!("Expected Int64 data");
        }
    }
}
