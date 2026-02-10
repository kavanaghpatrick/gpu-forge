//! Shared `#[repr(C)]` types matching MSL struct layouts in `shaders/types.h`.
//!
//! All structs must be byte-identical to their Metal counterparts. Any layout
//! change requires updating both this file and `shaders/types.h` in lockstep.

/// Parameters for column filter kernel (WHERE clause).
///
/// Layout: 40 bytes, 8-byte aligned. Matches MSL `FilterParams` in types.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FilterParams {
    /// Threshold for INT64 comparisons (offset 0, 8 bytes)
    pub compare_value_int: i64,
    /// Threshold for FLOAT64 comparisons (offset 8, 8 bytes)
    pub compare_value_float: f64,
    /// Number of rows to process (offset 16, 4 bytes)
    pub row_count: u32,
    /// Bytes between elements for packed types (offset 20, 4 bytes)
    pub column_stride: u32,
    /// 1 if null bitmap is valid (offset 24, 4 bytes)
    pub null_bitmap_present: u32,
    /// Padding to align to 8 bytes (offset 28, 4 bytes)
    pub _pad0: u32,
    /// Upper bound for BETWEEN (offset 32, 8 bytes)
    pub compare_value_int_hi: i64,
}

/// Parameters for aggregation kernel (SUM/COUNT/MIN/MAX/AVG).
///
/// Layout: 16 bytes (4 x u32), 4-byte aligned. Matches MSL `AggParams` in types.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AggParams {
    /// Total rows to aggregate (offset 0, 4 bytes)
    pub row_count: u32,
    /// Number of groups; 0 = no GROUP BY (offset 4, 4 bytes)
    pub group_count: u32,
    /// 0=COUNT, 1=SUM, 2=AVG, 3=MIN, 4=MAX (offset 8, 4 bytes)
    pub agg_function: u32,
    /// Padding (offset 12, 4 bytes)
    pub _pad0: u32,
}

/// Parameters for GPU CSV parser kernels.
///
/// Layout: 24 bytes, 4-byte aligned. Matches MSL `CsvParseParams` in types.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CsvParseParams {
    /// Total bytes in mmap'd file (offset 0, 4 bytes)
    pub file_size: u32,
    /// Number of CSV columns (offset 4, 4 bytes)
    pub num_columns: u32,
    /// Delimiter character as ASCII code (offset 8, 4 bytes)
    pub delimiter: u32,
    /// 1 if first row is header (offset 12, 4 bytes)
    pub has_header: u32,
    /// Maximum rows to parse; 0 = unlimited (offset 16, 4 bytes)
    pub max_rows: u32,
    /// Padding (offset 20, 4 bytes)
    pub _pad0: u32,
}

/// Indirect dispatch arguments for `dispatchThreadgroups(indirectBuffer:)`.
///
/// Layout: 3 x u32 = 12 bytes, matching Metal's indirect dispatch buffer format.
/// Matches MSL `DispatchArgs` in types.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchArgs {
    pub threadgroups_x: u32,
    pub threadgroups_y: u32,
    pub threadgroups_z: u32,
}

/// Column schema descriptor passed to GPU kernels for type-aware parsing.
///
/// Layout: 8 bytes (2 x u32), 4-byte aligned. Matches MSL `ColumnSchema` in types.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ColumnSchema {
    /// 0=INT64, 1=FLOAT64, 2=VARCHAR, 3=BOOL, 4=DATE (offset 0, 4 bytes)
    pub data_type: u32,
    /// 1 if dictionary-encoded string column (offset 4, 4 bytes)
    pub dict_encoded: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    // ---- FilterParams (40 bytes, 8-byte aligned) ----

    #[test]
    fn filter_params_size() {
        assert_eq!(
            mem::size_of::<FilterParams>(),
            40,
            "FilterParams must be 40 bytes"
        );
    }

    #[test]
    fn filter_params_alignment() {
        assert_eq!(
            mem::align_of::<FilterParams>(),
            8,
            "FilterParams must be 8-byte aligned (contains i64/f64)"
        );
    }

    #[test]
    fn filter_params_field_offsets() {
        let fp = FilterParams::default();
        let base = &fp as *const FilterParams as *const u8;
        unsafe {
            // compare_value_int: i64 at offset 0
            let ptr = base as *const i64;
            assert_eq!(*ptr, 0i64, "compare_value_int at offset 0");

            // compare_value_float: f64 at offset 8
            let ptr = base.add(8) as *const f64;
            assert_eq!(*ptr, 0.0f64, "compare_value_float at offset 8");

            // row_count: u32 at offset 16
            let ptr = base.add(16) as *const u32;
            assert_eq!(*ptr, 0u32, "row_count at offset 16");

            // column_stride: u32 at offset 20
            let ptr = base.add(20) as *const u32;
            assert_eq!(*ptr, 0u32, "column_stride at offset 20");

            // null_bitmap_present: u32 at offset 24
            let ptr = base.add(24) as *const u32;
            assert_eq!(*ptr, 0u32, "null_bitmap_present at offset 24");

            // _pad0: u32 at offset 28
            let ptr = base.add(28) as *const u32;
            assert_eq!(*ptr, 0u32, "_pad0 at offset 28");

            // compare_value_int_hi: i64 at offset 32
            let ptr = base.add(32) as *const i64;
            assert_eq!(*ptr, 0i64, "compare_value_int_hi at offset 32");
        }
    }

    #[test]
    fn filter_params_nonzero_values() {
        let fp = FilterParams {
            compare_value_int: 42,
            compare_value_float: 3.14,
            row_count: 1000,
            column_stride: 8,
            null_bitmap_present: 1,
            _pad0: 0,
            compare_value_int_hi: 100,
        };
        let base = &fp as *const FilterParams as *const u8;
        unsafe {
            assert_eq!(*(base as *const i64), 42i64);
            assert_eq!(*(base.add(8) as *const f64), 3.14f64);
            assert_eq!(*(base.add(16) as *const u32), 1000u32);
            assert_eq!(*(base.add(20) as *const u32), 8u32);
            assert_eq!(*(base.add(24) as *const u32), 1u32);
            assert_eq!(*(base.add(32) as *const i64), 100i64);
        }
    }

    // ---- AggParams (16 bytes, 4-byte aligned) ----

    #[test]
    fn agg_params_size() {
        assert_eq!(
            mem::size_of::<AggParams>(),
            16,
            "AggParams must be 16 bytes"
        );
    }

    #[test]
    fn agg_params_alignment() {
        assert_eq!(
            mem::align_of::<AggParams>(),
            4,
            "AggParams must be 4-byte aligned"
        );
    }

    #[test]
    fn agg_params_field_offsets() {
        let ap = AggParams {
            row_count: 5000,
            group_count: 10,
            agg_function: 1, // SUM
            _pad0: 0,
        };
        let base = &ap as *const AggParams as *const u8;
        unsafe {
            assert_eq!(*(base as *const u32), 5000u32, "row_count at offset 0");
            assert_eq!(*(base.add(4) as *const u32), 10u32, "group_count at offset 4");
            assert_eq!(*(base.add(8) as *const u32), 1u32, "agg_function at offset 8");
            assert_eq!(*(base.add(12) as *const u32), 0u32, "_pad0 at offset 12");
        }
    }

    // ---- CsvParseParams (24 bytes, 4-byte aligned) ----

    #[test]
    fn csv_parse_params_size() {
        assert_eq!(
            mem::size_of::<CsvParseParams>(),
            24,
            "CsvParseParams must be 24 bytes"
        );
    }

    #[test]
    fn csv_parse_params_alignment() {
        assert_eq!(
            mem::align_of::<CsvParseParams>(),
            4,
            "CsvParseParams must be 4-byte aligned"
        );
    }

    #[test]
    fn csv_parse_params_field_offsets() {
        let cp = CsvParseParams {
            file_size: 1_000_000,
            num_columns: 5,
            delimiter: b',' as u32,
            has_header: 1,
            max_rows: 0,
            _pad0: 0,
        };
        let base = &cp as *const CsvParseParams as *const u8;
        unsafe {
            assert_eq!(*(base as *const u32), 1_000_000u32, "file_size at offset 0");
            assert_eq!(*(base.add(4) as *const u32), 5u32, "num_columns at offset 4");
            assert_eq!(*(base.add(8) as *const u32), 44u32, "delimiter at offset 8");
            assert_eq!(*(base.add(12) as *const u32), 1u32, "has_header at offset 12");
            assert_eq!(*(base.add(16) as *const u32), 0u32, "max_rows at offset 16");
            assert_eq!(*(base.add(20) as *const u32), 0u32, "_pad0 at offset 20");
        }
    }

    // ---- DispatchArgs (12 bytes, 4-byte aligned) ----

    #[test]
    fn dispatch_args_size() {
        assert_eq!(
            mem::size_of::<DispatchArgs>(),
            12,
            "DispatchArgs must be 12 bytes (3 x u32)"
        );
    }

    #[test]
    fn dispatch_args_alignment() {
        assert_eq!(
            mem::align_of::<DispatchArgs>(),
            4,
            "DispatchArgs must be 4-byte aligned"
        );
    }

    #[test]
    fn dispatch_args_field_offsets() {
        let da = DispatchArgs {
            threadgroups_x: 64,
            threadgroups_y: 1,
            threadgroups_z: 1,
        };
        let base = &da as *const DispatchArgs as *const u8;
        unsafe {
            assert_eq!(*(base as *const u32), 64u32, "threadgroups_x at offset 0");
            assert_eq!(*(base.add(4) as *const u32), 1u32, "threadgroups_y at offset 4");
            assert_eq!(*(base.add(8) as *const u32), 1u32, "threadgroups_z at offset 8");
        }
    }

    // ---- ColumnSchema (8 bytes, 4-byte aligned) ----

    #[test]
    fn column_schema_size() {
        assert_eq!(
            mem::size_of::<ColumnSchema>(),
            8,
            "ColumnSchema must be 8 bytes (2 x u32)"
        );
    }

    #[test]
    fn column_schema_alignment() {
        assert_eq!(
            mem::align_of::<ColumnSchema>(),
            4,
            "ColumnSchema must be 4-byte aligned"
        );
    }

    #[test]
    fn column_schema_field_offsets() {
        let cs = ColumnSchema {
            data_type: 2,     // VARCHAR
            dict_encoded: 1,
        };
        let base = &cs as *const ColumnSchema as *const u8;
        unsafe {
            assert_eq!(*(base as *const u32), 2u32, "data_type at offset 0");
            assert_eq!(*(base.add(4) as *const u32), 1u32, "dict_encoded at offset 4");
        }
    }
}
