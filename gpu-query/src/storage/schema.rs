//! Runtime schema for parsed data tables.
//!
//! Maps column names to data types at runtime, providing the bridge between
//! CPU-side CSV metadata and GPU-side ColumnSchema structs.

use crate::gpu::types::ColumnSchema;

/// Data types supported by the query engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 64-bit signed integer (MSL: long / int64_t)
    Int64 = 0,
    /// 64-bit IEEE 754 float (MSL: double)
    Float64 = 1,
    /// Variable-length string (dictionary-encoded on GPU)
    Varchar = 2,
    /// Boolean (stored as uint on GPU)
    Bool = 3,
    /// Date (stored as INT64 epoch days on GPU)
    Date = 4,
}

impl DataType {
    /// Convert from the GPU type code (matches ColumnSchema.data_type).
    pub fn from_gpu_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(DataType::Int64),
            1 => Some(DataType::Float64),
            2 => Some(DataType::Varchar),
            3 => Some(DataType::Bool),
            4 => Some(DataType::Date),
            _ => None,
        }
    }

    /// Convert to the GPU type code for ColumnSchema.data_type.
    pub fn to_gpu_code(self) -> u32 {
        self as u32
    }
}

/// A column descriptor with name and type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    /// Column name from CSV header.
    pub name: String,
    /// Data type for this column.
    pub data_type: DataType,
    /// Whether this column can contain nulls.
    pub nullable: bool,
}

/// Runtime schema describing the structure of a parsed table.
#[derive(Debug, Clone)]
pub struct RuntimeSchema {
    /// Ordered list of column definitions.
    pub columns: Vec<ColumnDef>,
}

impl RuntimeSchema {
    /// Create a new schema from column definitions.
    pub fn new(columns: Vec<ColumnDef>) -> Self {
        Self { columns }
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get column index by name (case-insensitive).
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(name))
    }

    /// Count columns of a specific type.
    pub fn count_type(&self, dtype: DataType) -> usize {
        self.columns.iter().filter(|c| c.data_type == dtype).count()
    }

    /// Convert to GPU ColumnSchema array for shader binding.
    pub fn to_gpu_schemas(&self) -> Vec<ColumnSchema> {
        self.columns
            .iter()
            .map(|c| ColumnSchema {
                data_type: c.data_type.to_gpu_code(),
                dict_encoded: 0,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_roundtrip() {
        for code in 0..5u32 {
            let dt = DataType::from_gpu_code(code).unwrap();
            assert_eq!(dt.to_gpu_code(), code);
        }
        assert!(DataType::from_gpu_code(99).is_none());
    }

    #[test]
    fn test_runtime_schema_basics() {
        let schema = RuntimeSchema::new(vec![
            ColumnDef {
                name: "id".to_string(),
                data_type: DataType::Int64,
                nullable: false,
            },
            ColumnDef {
                name: "amount".to_string(),
                data_type: DataType::Float64,
                nullable: true,
            },
            ColumnDef {
                name: "name".to_string(),
                data_type: DataType::Varchar,
                nullable: true,
            },
        ]);

        assert_eq!(schema.num_columns(), 3);
        assert_eq!(schema.column_index("id"), Some(0));
        assert_eq!(schema.column_index("AMOUNT"), Some(1));
        assert_eq!(schema.column_index("missing"), None);
        assert_eq!(schema.count_type(DataType::Int64), 1);
        assert_eq!(schema.count_type(DataType::Float64), 1);
        assert_eq!(schema.count_type(DataType::Varchar), 1);
    }

    #[test]
    fn test_to_gpu_schemas() {
        let schema = RuntimeSchema::new(vec![
            ColumnDef {
                name: "a".to_string(),
                data_type: DataType::Int64,
                nullable: false,
            },
            ColumnDef {
                name: "b".to_string(),
                data_type: DataType::Float64,
                nullable: false,
            },
        ]);

        let gpu = schema.to_gpu_schemas();
        assert_eq!(gpu.len(), 2);
        assert_eq!(gpu[0].data_type, 0); // INT64
        assert_eq!(gpu[1].data_type, 1); // FLOAT64
        assert_eq!(gpu[0].dict_encoded, 0);
    }
}
