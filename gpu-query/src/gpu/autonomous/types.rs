//! Shared `#[repr(C)]` types matching MSL struct layouts in `shaders/autonomous_types.h`.
//!
//! All structs must be byte-identical to their Metal counterparts. Any layout
//! change requires updating both this file and `shaders/autonomous_types.h` in lockstep.

/// Maximum number of filter predicates per query.
pub const MAX_FILTERS: usize = 4;
/// Maximum number of aggregate functions per query.
pub const MAX_AGGS: usize = 5;
/// Maximum number of GROUP BY buckets.
pub const MAX_GROUPS: usize = 256;

/// Filter predicate specification for a single WHERE clause predicate.
///
/// Layout: 48 bytes, 8-byte aligned. Matches MSL `FilterSpec` in autonomous_types.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FilterSpec {
    /// Column index in ColumnMeta array (offset 0, 4 bytes)
    pub column_idx: u32,
    /// Comparison operator: 0=EQ, 1=NE, 2=LT, 3=LE, 4=GT, 5=GE (offset 4, 4 bytes)
    pub compare_op: u32,
    /// Column data type: 0=INT64, 1=FLOAT32 (offset 8, 4 bytes)
    pub column_type: u32,
    /// Padding (offset 12, 4 bytes)
    pub _pad0: u32,
    /// Comparison value for INT64 columns (offset 16, 8 bytes)
    pub value_int: i64,
    /// Comparison value for FLOAT32 columns (IEEE 754 bits) (offset 24, 4 bytes)
    pub value_float_bits: u32,
    /// Padding (offset 28, 4 bytes)
    pub _pad1: u32,
    /// 1 if null check is active (offset 32, 4 bytes)
    pub has_null_check: u32,
    /// Padding to 48 bytes (offset 36, 12 bytes)
    pub _pad2: [u32; 3],
}

/// Aggregate function specification.
///
/// Layout: 16 bytes, 4-byte aligned. Matches MSL `AggSpec` in autonomous_types.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AggSpec {
    /// Aggregate function: 0=COUNT, 1=SUM, 2=AVG, 3=MIN, 4=MAX (offset 0, 4 bytes)
    pub agg_func: u32,
    /// Column index in ColumnMeta array (offset 4, 4 bytes)
    pub column_idx: u32,
    /// Column data type: 0=INT64, 1=FLOAT32 (offset 8, 4 bytes)
    pub column_type: u32,
    /// Padding (offset 12, 4 bytes)
    pub _pad0: u32,
}

/// Work queue slot containing a complete query parameterization.
///
/// Layout: 512 bytes, 8-byte aligned. Matches MSL `QueryParamsSlot` in autonomous_types.h.
/// Three slots form the triple-buffered work queue (3 x 512 = 1536 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QueryParamsSlot {
    /// Monotonically increasing sequence number (offset 0, 4 bytes)
    pub sequence_id: u32,
    /// Padding for 8-byte alignment of query_hash (offset 4, 4 bytes)
    pub _pad_seq: u32,
    /// Hash of query plan structure for JIT cache lookup (offset 8, 8 bytes)
    pub query_hash: u64,
    /// Number of active filter predicates (0..4) (offset 16, 4 bytes)
    pub filter_count: u32,
    /// Padding (offset 20, 4 bytes)
    pub _pad_fc: u32,
    /// Filter predicate specifications (offset 24, 4 * 48 = 192 bytes)
    pub filters: [FilterSpec; MAX_FILTERS],
    /// Number of active aggregate functions (0..5) (offset 216, 4 bytes)
    pub agg_count: u32,
    /// Padding (offset 220, 4 bytes)
    pub _pad_ac: u32,
    /// Aggregate function specifications (offset 224, 5 * 16 = 80 bytes)
    pub aggs: [AggSpec; MAX_AGGS],
    /// Column index for GROUP BY (-1 or column_idx) (offset 304, 4 bytes)
    pub group_by_col: u32,
    /// 1 if GROUP BY is active (offset 308, 4 bytes)
    pub has_group_by: u32,
    /// Total row count in the table (offset 312, 4 bytes)
    pub row_count: u32,
    /// Padding to 512 bytes total (offset 316, 196 bytes)
    pub _padding: [u8; 196],
}

impl Default for QueryParamsSlot {
    fn default() -> Self {
        // Safety: QueryParamsSlot is #[repr(C)] with all fields being plain data.
        // Zeroed memory is a valid representation.
        unsafe { std::mem::zeroed() }
    }
}

/// Column metadata describing a single column in the GPU-resident table.
///
/// Layout: 32 bytes, 8-byte aligned. Matches MSL `ColumnMeta` in autonomous_types.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ColumnMeta {
    /// Byte offset of column data in the data buffer (offset 0, 8 bytes)
    pub offset: u64,
    /// Column data type: 0=INT64, 1=FLOAT32, 2=DICT_U32 (offset 8, 4 bytes)
    pub column_type: u32,
    /// Bytes between consecutive elements (offset 12, 4 bytes)
    pub stride: u32,
    /// Byte offset of null bitmap in the data buffer (offset 16, 8 bytes)
    pub null_offset: u64,
    /// Number of rows (offset 24, 4 bytes)
    pub row_count: u32,
    /// Padding (offset 28, 4 bytes)
    pub _pad: u32,
}

/// Result of a single aggregate computation for one group.
///
/// Layout: 16 bytes, 8-byte aligned. Matches MSL `AggResult` in autonomous_types.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AggResult {
    /// Aggregate result as INT64 (offset 0, 8 bytes)
    pub value_int: i64,
    /// Aggregate result as FLOAT32 (offset 8, 4 bytes)
    pub value_float: f32,
    /// Row count contributing to this aggregate (offset 12, 4 bytes)
    pub count: u32,
}

/// Output buffer written by the GPU kernel with query results.
///
/// Layout: 22560 bytes, 8-byte aligned. Matches MSL `OutputBuffer` in autonomous_types.h.
/// Header (32 bytes) + group_keys (2048 bytes) + agg_results (20480 bytes) = 22560 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OutputBuffer {
    /// 0=pending, 1=ready (offset 0, 4 bytes)
    pub ready_flag: u32,
    /// Echoed from QueryParamsSlot.sequence_id (offset 4, 4 bytes)
    pub sequence_id: u32,
    /// GPU-measured query latency in nanoseconds (offset 8, 8 bytes)
    pub latency_ns: u64,
    /// Number of result rows (groups) (offset 16, 4 bytes)
    pub result_row_count: u32,
    /// Number of result columns (aggs) (offset 20, 4 bytes)
    pub result_col_count: u32,
    /// Error code: 0=success (offset 24, 4 bytes)
    pub error_code: u32,
    /// Padding (offset 28, 4 bytes)
    pub _pad: u32,
    /// GROUP BY key values, one per group (offset 32, 256 * 8 = 2048 bytes)
    pub group_keys: [i64; MAX_GROUPS],
    /// Aggregate results: [group][agg] (offset 2080, 256 * 5 * 16 = 20480 bytes)
    pub agg_results: [[AggResult; MAX_AGGS]; MAX_GROUPS],
}

impl Default for OutputBuffer {
    fn default() -> Self {
        // Safety: OutputBuffer is #[repr(C)] with all fields being plain data.
        // Zeroed memory is a valid representation.
        unsafe { std::mem::zeroed() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{self, offset_of};

    // ================================================================
    // FilterSpec — MSL: 48 bytes, 8-byte aligned
    // ================================================================

    #[test]
    fn filter_spec_size() {
        assert_eq!(
            mem::size_of::<FilterSpec>(),
            48,
            "FilterSpec must be 48 bytes (MSL sizeof)"
        );
    }

    #[test]
    fn filter_spec_alignment() {
        assert_eq!(
            mem::align_of::<FilterSpec>(),
            8,
            "FilterSpec must be 8-byte aligned (contains i64)"
        );
    }

    #[test]
    fn filter_spec_offset_column_idx() {
        assert_eq!(offset_of!(FilterSpec, column_idx), 0, "MSL offset 0");
    }

    #[test]
    fn filter_spec_offset_compare_op() {
        assert_eq!(offset_of!(FilterSpec, compare_op), 4, "MSL offset 4");
    }

    #[test]
    fn filter_spec_offset_column_type() {
        assert_eq!(offset_of!(FilterSpec, column_type), 8, "MSL offset 8");
    }

    #[test]
    fn filter_spec_offset_pad0() {
        assert_eq!(offset_of!(FilterSpec, _pad0), 12, "MSL offset 12");
    }

    #[test]
    fn filter_spec_offset_value_int() {
        assert_eq!(offset_of!(FilterSpec, value_int), 16, "MSL offset 16");
    }

    #[test]
    fn filter_spec_offset_value_float_bits() {
        assert_eq!(
            offset_of!(FilterSpec, value_float_bits),
            24,
            "MSL offset 24"
        );
    }

    #[test]
    fn filter_spec_offset_pad1() {
        assert_eq!(offset_of!(FilterSpec, _pad1), 28, "MSL offset 28");
    }

    #[test]
    fn filter_spec_offset_has_null_check() {
        assert_eq!(
            offset_of!(FilterSpec, has_null_check),
            32,
            "MSL offset 32"
        );
    }

    #[test]
    fn filter_spec_offset_pad2() {
        assert_eq!(offset_of!(FilterSpec, _pad2), 36, "MSL offset 36");
    }

    #[test]
    fn filter_spec_nonzero_round_trip() {
        let fs = FilterSpec {
            column_idx: 3,
            compare_op: 4,  // GT
            column_type: 0, // INT64
            _pad0: 0,
            value_int: 500,
            value_float_bits: 0x41200000, // 10.0f
            _pad1: 0,
            has_null_check: 1,
            _pad2: [0; 3],
        };
        let base = &fs as *const FilterSpec as *const u8;
        unsafe {
            assert_eq!(*(base as *const u32), 3u32, "column_idx at offset 0");
            assert_eq!(*(base.add(4) as *const u32), 4u32, "compare_op at offset 4");
            assert_eq!(
                *(base.add(8) as *const u32),
                0u32,
                "column_type at offset 8"
            );
            assert_eq!(
                *(base.add(16) as *const i64),
                500i64,
                "value_int at offset 16"
            );
            assert_eq!(
                *(base.add(24) as *const u32),
                0x41200000u32,
                "value_float_bits at offset 24"
            );
            assert_eq!(
                *(base.add(32) as *const u32),
                1u32,
                "has_null_check at offset 32"
            );
        }
    }

    // ================================================================
    // AggSpec — MSL: 16 bytes, 4-byte aligned
    // ================================================================

    #[test]
    fn agg_spec_size() {
        assert_eq!(
            mem::size_of::<AggSpec>(),
            16,
            "AggSpec must be 16 bytes (MSL sizeof)"
        );
    }

    #[test]
    fn agg_spec_alignment() {
        assert_eq!(
            mem::align_of::<AggSpec>(),
            4,
            "AggSpec must be 4-byte aligned"
        );
    }

    #[test]
    fn agg_spec_offset_agg_func() {
        assert_eq!(offset_of!(AggSpec, agg_func), 0, "MSL offset 0");
    }

    #[test]
    fn agg_spec_offset_column_idx() {
        assert_eq!(offset_of!(AggSpec, column_idx), 4, "MSL offset 4");
    }

    #[test]
    fn agg_spec_offset_column_type() {
        assert_eq!(offset_of!(AggSpec, column_type), 8, "MSL offset 8");
    }

    #[test]
    fn agg_spec_offset_pad0() {
        assert_eq!(offset_of!(AggSpec, _pad0), 12, "MSL offset 12");
    }

    #[test]
    fn agg_spec_nonzero_round_trip() {
        let ag = AggSpec {
            agg_func: 1,    // SUM
            column_idx: 2,
            column_type: 1, // FLOAT32
            _pad0: 0,
        };
        let base = &ag as *const AggSpec as *const u8;
        unsafe {
            assert_eq!(*(base as *const u32), 1u32, "agg_func at offset 0");
            assert_eq!(*(base.add(4) as *const u32), 2u32, "column_idx at offset 4");
            assert_eq!(
                *(base.add(8) as *const u32),
                1u32,
                "column_type at offset 8"
            );
        }
    }

    // ================================================================
    // QueryParamsSlot — MSL: 512 bytes, 8-byte aligned
    // ================================================================

    #[test]
    fn query_params_slot_size() {
        assert_eq!(
            mem::size_of::<QueryParamsSlot>(),
            512,
            "QueryParamsSlot must be 512 bytes (MSL sizeof)"
        );
    }

    #[test]
    fn query_params_slot_alignment() {
        assert_eq!(
            mem::align_of::<QueryParamsSlot>(),
            8,
            "QueryParamsSlot must be 8-byte aligned (contains u64)"
        );
    }

    #[test]
    fn query_params_slot_offset_sequence_id() {
        assert_eq!(
            offset_of!(QueryParamsSlot, sequence_id),
            0,
            "MSL offset 0"
        );
    }

    #[test]
    fn query_params_slot_offset_pad_seq() {
        assert_eq!(
            offset_of!(QueryParamsSlot, _pad_seq),
            4,
            "MSL offset 4"
        );
    }

    #[test]
    fn query_params_slot_offset_query_hash() {
        assert_eq!(
            offset_of!(QueryParamsSlot, query_hash),
            8,
            "MSL offset 8"
        );
    }

    #[test]
    fn query_params_slot_offset_filter_count() {
        assert_eq!(
            offset_of!(QueryParamsSlot, filter_count),
            16,
            "MSL offset 16"
        );
    }

    #[test]
    fn query_params_slot_offset_pad_fc() {
        assert_eq!(
            offset_of!(QueryParamsSlot, _pad_fc),
            20,
            "MSL offset 20"
        );
    }

    #[test]
    fn query_params_slot_offset_filters() {
        assert_eq!(
            offset_of!(QueryParamsSlot, filters),
            24,
            "MSL offset 24"
        );
    }

    #[test]
    fn query_params_slot_offset_agg_count() {
        assert_eq!(
            offset_of!(QueryParamsSlot, agg_count),
            216,
            "MSL offset 216 (24 + 4*48)"
        );
    }

    #[test]
    fn query_params_slot_offset_pad_ac() {
        assert_eq!(
            offset_of!(QueryParamsSlot, _pad_ac),
            220,
            "MSL offset 220"
        );
    }

    #[test]
    fn query_params_slot_offset_aggs() {
        assert_eq!(
            offset_of!(QueryParamsSlot, aggs),
            224,
            "MSL offset 224"
        );
    }

    #[test]
    fn query_params_slot_offset_group_by_col() {
        assert_eq!(
            offset_of!(QueryParamsSlot, group_by_col),
            304,
            "MSL offset 304 (224 + 5*16)"
        );
    }

    #[test]
    fn query_params_slot_offset_has_group_by() {
        assert_eq!(
            offset_of!(QueryParamsSlot, has_group_by),
            308,
            "MSL offset 308"
        );
    }

    #[test]
    fn query_params_slot_offset_row_count() {
        assert_eq!(
            offset_of!(QueryParamsSlot, row_count),
            312,
            "MSL offset 312"
        );
    }

    #[test]
    fn query_params_slot_offset_padding() {
        assert_eq!(
            offset_of!(QueryParamsSlot, _padding),
            316,
            "MSL offset 316"
        );
    }

    #[test]
    fn query_params_slot_nonzero_round_trip() {
        let mut slot = QueryParamsSlot::default();
        slot.sequence_id = 42;
        slot.query_hash = 0xDEAD_BEEF_CAFE_BABE;
        slot.filter_count = 2;
        slot.filters[0].column_idx = 1;
        slot.filters[0].compare_op = 4; // GT
        slot.filters[0].value_int = 100;
        slot.agg_count = 1;
        slot.aggs[0].agg_func = 0; // COUNT
        slot.group_by_col = 3;
        slot.has_group_by = 1;
        slot.row_count = 1_000_000;

        let base = &slot as *const QueryParamsSlot as *const u8;
        unsafe {
            assert_eq!(*(base as *const u32), 42u32, "sequence_id at offset 0");
            assert_eq!(
                *(base.add(8) as *const u64),
                0xDEAD_BEEF_CAFE_BABEu64,
                "query_hash at offset 8"
            );
            assert_eq!(
                *(base.add(16) as *const u32),
                2u32,
                "filter_count at offset 16"
            );
            // First filter starts at offset 24
            assert_eq!(
                *(base.add(24) as *const u32),
                1u32,
                "filters[0].column_idx at offset 24"
            );
            assert_eq!(
                *(base.add(28) as *const u32),
                4u32,
                "filters[0].compare_op at offset 28"
            );
            assert_eq!(
                *(base.add(40) as *const i64),
                100i64,
                "filters[0].value_int at offset 40"
            );
            assert_eq!(
                *(base.add(216) as *const u32),
                1u32,
                "agg_count at offset 216"
            );
            // First agg starts at offset 224
            assert_eq!(
                *(base.add(224) as *const u32),
                0u32,
                "aggs[0].agg_func at offset 224"
            );
            assert_eq!(
                *(base.add(304) as *const u32),
                3u32,
                "group_by_col at offset 304"
            );
            assert_eq!(
                *(base.add(308) as *const u32),
                1u32,
                "has_group_by at offset 308"
            );
            assert_eq!(
                *(base.add(312) as *const u32),
                1_000_000u32,
                "row_count at offset 312"
            );
        }
    }

    // ================================================================
    // ColumnMeta — MSL: 32 bytes, 8-byte aligned
    // ================================================================

    #[test]
    fn column_meta_size() {
        assert_eq!(
            mem::size_of::<ColumnMeta>(),
            32,
            "ColumnMeta must be 32 bytes (MSL sizeof)"
        );
    }

    #[test]
    fn column_meta_alignment() {
        assert_eq!(
            mem::align_of::<ColumnMeta>(),
            8,
            "ColumnMeta must be 8-byte aligned (contains u64)"
        );
    }

    #[test]
    fn column_meta_offset_offset() {
        assert_eq!(offset_of!(ColumnMeta, offset), 0, "MSL offset 0");
    }

    #[test]
    fn column_meta_offset_column_type() {
        assert_eq!(offset_of!(ColumnMeta, column_type), 8, "MSL offset 8");
    }

    #[test]
    fn column_meta_offset_stride() {
        assert_eq!(offset_of!(ColumnMeta, stride), 12, "MSL offset 12");
    }

    #[test]
    fn column_meta_offset_null_offset() {
        assert_eq!(offset_of!(ColumnMeta, null_offset), 16, "MSL offset 16");
    }

    #[test]
    fn column_meta_offset_row_count() {
        assert_eq!(offset_of!(ColumnMeta, row_count), 24, "MSL offset 24");
    }

    #[test]
    fn column_meta_offset_pad() {
        assert_eq!(offset_of!(ColumnMeta, _pad), 28, "MSL offset 28");
    }

    #[test]
    fn column_meta_nonzero_round_trip() {
        let cm = ColumnMeta {
            offset: 4096,
            column_type: 1, // FLOAT32
            stride: 4,
            null_offset: 8192,
            row_count: 50_000,
            _pad: 0,
        };
        let base = &cm as *const ColumnMeta as *const u8;
        unsafe {
            assert_eq!(*(base as *const u64), 4096u64, "offset at offset 0");
            assert_eq!(
                *(base.add(8) as *const u32),
                1u32,
                "column_type at offset 8"
            );
            assert_eq!(*(base.add(12) as *const u32), 4u32, "stride at offset 12");
            assert_eq!(
                *(base.add(16) as *const u64),
                8192u64,
                "null_offset at offset 16"
            );
            assert_eq!(
                *(base.add(24) as *const u32),
                50_000u32,
                "row_count at offset 24"
            );
        }
    }

    // ================================================================
    // AggResult — MSL: 16 bytes, 8-byte aligned
    // ================================================================

    #[test]
    fn agg_result_size() {
        assert_eq!(
            mem::size_of::<AggResult>(),
            16,
            "AggResult must be 16 bytes (MSL sizeof)"
        );
    }

    #[test]
    fn agg_result_alignment() {
        assert_eq!(
            mem::align_of::<AggResult>(),
            8,
            "AggResult must be 8-byte aligned (contains i64)"
        );
    }

    #[test]
    fn agg_result_offset_value_int() {
        assert_eq!(offset_of!(AggResult, value_int), 0, "MSL offset 0");
    }

    #[test]
    fn agg_result_offset_value_float() {
        assert_eq!(offset_of!(AggResult, value_float), 8, "MSL offset 8");
    }

    #[test]
    fn agg_result_offset_count() {
        assert_eq!(offset_of!(AggResult, count), 12, "MSL offset 12");
    }

    #[test]
    fn agg_result_nonzero_round_trip() {
        let ar = AggResult {
            value_int: 12345,
            value_float: 3.14,
            count: 100,
        };
        let base = &ar as *const AggResult as *const u8;
        unsafe {
            assert_eq!(*(base as *const i64), 12345i64, "value_int at offset 0");
            assert_eq!(
                *(base.add(8) as *const f32),
                3.14f32,
                "value_float at offset 8"
            );
            assert_eq!(*(base.add(12) as *const u32), 100u32, "count at offset 12");
        }
    }

    // ================================================================
    // OutputBuffer — MSL: 22560 bytes, 8-byte aligned
    // ================================================================

    #[test]
    fn output_buffer_size() {
        assert_eq!(
            mem::size_of::<OutputBuffer>(),
            22560,
            "OutputBuffer must be 22560 bytes (32 header + 2048 group_keys + 20480 agg_results)"
        );
    }

    #[test]
    fn output_buffer_alignment() {
        assert_eq!(
            mem::align_of::<OutputBuffer>(),
            8,
            "OutputBuffer must be 8-byte aligned (contains u64/i64)"
        );
    }

    #[test]
    fn output_buffer_offset_ready_flag() {
        assert_eq!(offset_of!(OutputBuffer, ready_flag), 0, "MSL offset 0");
    }

    #[test]
    fn output_buffer_offset_sequence_id() {
        assert_eq!(offset_of!(OutputBuffer, sequence_id), 4, "MSL offset 4");
    }

    #[test]
    fn output_buffer_offset_latency_ns() {
        assert_eq!(offset_of!(OutputBuffer, latency_ns), 8, "MSL offset 8");
    }

    #[test]
    fn output_buffer_offset_result_row_count() {
        assert_eq!(
            offset_of!(OutputBuffer, result_row_count),
            16,
            "MSL offset 16"
        );
    }

    #[test]
    fn output_buffer_offset_result_col_count() {
        assert_eq!(
            offset_of!(OutputBuffer, result_col_count),
            20,
            "MSL offset 20"
        );
    }

    #[test]
    fn output_buffer_offset_error_code() {
        assert_eq!(offset_of!(OutputBuffer, error_code), 24, "MSL offset 24");
    }

    #[test]
    fn output_buffer_offset_pad() {
        assert_eq!(offset_of!(OutputBuffer, _pad), 28, "MSL offset 28");
    }

    #[test]
    fn output_buffer_offset_group_keys() {
        assert_eq!(
            offset_of!(OutputBuffer, group_keys),
            32,
            "MSL offset 32 (after 32-byte header)"
        );
    }

    #[test]
    fn output_buffer_offset_agg_results() {
        assert_eq!(
            offset_of!(OutputBuffer, agg_results),
            2080,
            "MSL offset 2080 (32 + 256*8)"
        );
    }

    #[test]
    fn output_buffer_nonzero_round_trip() {
        let mut ob = OutputBuffer::default();
        ob.ready_flag = 1;
        ob.sequence_id = 99;
        ob.latency_ns = 500_000;
        ob.result_row_count = 5;
        ob.result_col_count = 2;
        ob.error_code = 0;
        ob.group_keys[0] = 42;
        ob.group_keys[255] = -1;
        ob.agg_results[0][0].value_int = 12345;
        ob.agg_results[0][0].value_float = 1.5;
        ob.agg_results[0][0].count = 100;

        let base = &ob as *const OutputBuffer as *const u8;
        unsafe {
            assert_eq!(*(base as *const u32), 1u32, "ready_flag at offset 0");
            assert_eq!(
                *(base.add(4) as *const u32),
                99u32,
                "sequence_id at offset 4"
            );
            assert_eq!(
                *(base.add(8) as *const u64),
                500_000u64,
                "latency_ns at offset 8"
            );
            assert_eq!(
                *(base.add(16) as *const u32),
                5u32,
                "result_row_count at offset 16"
            );
            assert_eq!(
                *(base.add(20) as *const u32),
                2u32,
                "result_col_count at offset 20"
            );
            assert_eq!(
                *(base.add(24) as *const u32),
                0u32,
                "error_code at offset 24"
            );
            // group_keys[0] at offset 32
            assert_eq!(
                *(base.add(32) as *const i64),
                42i64,
                "group_keys[0] at offset 32"
            );
            // group_keys[255] at offset 32 + 255*8 = 2072
            assert_eq!(
                *(base.add(2072) as *const i64),
                -1i64,
                "group_keys[255] at offset 2072"
            );
            // agg_results[0][0] at offset 2080
            assert_eq!(
                *(base.add(2080) as *const i64),
                12345i64,
                "agg_results[0][0].value_int at offset 2080"
            );
            assert_eq!(
                *(base.add(2088) as *const f32),
                1.5f32,
                "agg_results[0][0].value_float at offset 2088"
            );
            assert_eq!(
                *(base.add(2092) as *const u32),
                100u32,
                "agg_results[0][0].count at offset 2092"
            );
        }
    }

    // ================================================================
    // Cross-struct consistency checks
    // ================================================================

    #[test]
    fn filters_array_size_in_query_params() {
        assert_eq!(
            mem::size_of::<[FilterSpec; MAX_FILTERS]>(),
            192,
            "4 FilterSpecs must be 192 bytes (4 * 48)"
        );
    }

    #[test]
    fn aggs_array_size_in_query_params() {
        assert_eq!(
            mem::size_of::<[AggSpec; MAX_AGGS]>(),
            80,
            "5 AggSpecs must be 80 bytes (5 * 16)"
        );
    }

    #[test]
    fn agg_results_array_size_in_output_buffer() {
        assert_eq!(
            mem::size_of::<[[AggResult; MAX_AGGS]; MAX_GROUPS]>(),
            20480,
            "256 * 5 AggResults must be 20480 bytes (256 * 5 * 16)"
        );
    }

    #[test]
    fn group_keys_array_size_in_output_buffer() {
        assert_eq!(
            mem::size_of::<[i64; MAX_GROUPS]>(),
            2048,
            "256 group keys must be 2048 bytes (256 * 8)"
        );
    }
}
