#pragma once

// autonomous_types.h -- Shared struct definitions between Rust host and MSL shaders
// for the GPU-autonomous query engine.
//
// All structs here must match their Rust counterparts in src/gpu/autonomous/types.rs
// byte-for-byte. Any layout change requires updating both files in lockstep.

#include <metal_stdlib>
using namespace metal;

// Query limits (kernel logic constants)
#define MAX_GROUPS  64
#define MAX_AGGS    5
#define MAX_FILTERS 4

// Number of slots in arrays (distinct from query limits above)
#define MAX_GROUP_SLOTS 256

// Column type constants
#define COLUMN_TYPE_INT64   0
#define COLUMN_TYPE_FLOAT32 1
#define COLUMN_TYPE_DICT_U32 2

// Compare operator constants
#define COMPARE_OP_EQ 0
#define COMPARE_OP_NE 1
#define COMPARE_OP_LT 2
#define COMPARE_OP_LE 3
#define COMPARE_OP_GT 4
#define COMPARE_OP_GE 5

// Aggregate function constants
#define AGG_FUNC_COUNT 0
#define AGG_FUNC_SUM   1
#define AGG_FUNC_AVG   2
#define AGG_FUNC_MIN   3
#define AGG_FUNC_MAX   4

// Filter predicate specification for a single WHERE clause predicate.
//
// Layout: 48 bytes, 8-byte aligned. Matches Rust `FilterSpec`.
struct FilterSpec {
    uint     column_idx;       // offset 0  (4 bytes)
    uint     compare_op;       // offset 4  (4 bytes) — 0=EQ,1=NE,2=LT,3=LE,4=GT,5=GE
    uint     column_type;      // offset 8  (4 bytes) — 0=INT64, 1=FLOAT32
    uint     _pad0;            // offset 12 (4 bytes)
    long     value_int;        // offset 16 (8 bytes) — comparison value for INT64 columns
    uint     value_float_bits; // offset 24 (4 bytes) — IEEE 754 bits for FLOAT32
    uint     _pad1;            // offset 28 (4 bytes)
    uint     has_null_check;   // offset 32 (4 bytes) — 1 if null check is active
    uint     _pad2[3];         // offset 36 (12 bytes) — pad to 48
};                              // total: 48 bytes

// Aggregate function specification.
//
// Layout: 16 bytes, 4-byte aligned. Matches Rust `AggSpec`.
struct AggSpec {
    uint     agg_func;         // offset 0  (4 bytes) — 0=COUNT,1=SUM,2=AVG,3=MIN,4=MAX
    uint     column_idx;       // offset 4  (4 bytes)
    uint     column_type;      // offset 8  (4 bytes) — 0=INT64, 1=FLOAT32
    uint     _pad0;            // offset 12 (4 bytes)
};                              // total: 16 bytes

// Work queue slot containing a complete query parameterization.
//
// Layout: 512 bytes, 8-byte aligned. Matches Rust `QueryParamsSlot`.
// Three slots form the triple-buffered work queue (3 x 512 = 1536 bytes).
struct QueryParamsSlot {
    uint       sequence_id;    // offset 0   (4 bytes) — monotonically increasing
    uint       _pad_seq;       // offset 4   (4 bytes) — align query_hash to 8
    ulong      query_hash;     // offset 8   (8 bytes) — plan structure hash for JIT cache
    uint       filter_count;   // offset 16  (4 bytes) — number of active filters (0..4)
    uint       _pad_fc;        // offset 20  (4 bytes)
    FilterSpec filters[4];     // offset 24  (4 * 48 = 192 bytes)
    uint       agg_count;      // offset 216 (4 bytes) — number of active aggs (0..5)
    uint       _pad_ac;        // offset 220 (4 bytes)
    AggSpec    aggs[5];        // offset 224 (5 * 16 = 80 bytes)
    uint       group_by_col;   // offset 304 (4 bytes) — column index for GROUP BY
    uint       has_group_by;   // offset 308 (4 bytes) — 1 if GROUP BY is active
    uint       row_count;      // offset 312 (4 bytes) — total rows in table
    char       _padding[196];  // offset 316 (196 bytes) — pad to 512
};                              // total: 512 bytes

// Column metadata describing a single column in the GPU-resident table.
//
// Layout: 32 bytes, 8-byte aligned. Matches Rust `ColumnMeta`.
struct ColumnMeta {
    ulong    offset;           // offset 0  (8 bytes) — byte offset of column data
    uint     column_type;      // offset 8  (4 bytes) — 0=INT64, 1=FLOAT32, 2=DICT_U32
    uint     stride;           // offset 12 (4 bytes) — bytes between elements
    ulong    null_offset;      // offset 16 (8 bytes) — byte offset of null bitmap
    uint     row_count;        // offset 24 (4 bytes) — number of rows
    uint     _pad;             // offset 28 (4 bytes)
};                              // total: 32 bytes

// Result of a single aggregate computation for one group.
//
// Layout: 16 bytes, 8-byte aligned. Matches Rust `AggResult`.
struct AggResult {
    long     value_int;        // offset 0  (8 bytes) — aggregate result as INT64
    float    value_float;      // offset 8  (4 bytes) — aggregate result as FLOAT32
    uint     count;            // offset 12 (4 bytes) — contributing row count
};                              // total: 16 bytes

// Output buffer written by the GPU kernel with query results.
//
// Layout: 22560 bytes, 8-byte aligned. Matches Rust `OutputBuffer`.
// Header (32 bytes) + group_keys (2048 bytes) + agg_results (20480 bytes) = 22560 bytes.
struct OutputBuffer {
    uint       ready_flag;         // offset 0    (4 bytes) — 0=pending, 1=ready
    uint       sequence_id;        // offset 4    (4 bytes) — echoed from QueryParamsSlot
    ulong      latency_ns;         // offset 8    (8 bytes) — GPU-measured latency (ns)
    uint       result_row_count;   // offset 16   (4 bytes) — number of result rows (groups)
    uint       result_col_count;   // offset 20   (4 bytes) — number of result columns (aggs)
    uint       error_code;         // offset 24   (4 bytes) — 0=success
    uint       _pad;               // offset 28   (4 bytes)
    long       group_keys[256];    // offset 32   (256 * 8 = 2048 bytes)
    AggResult  agg_results[256][5]; // offset 2080 (256 * 5 * 16 = 20480 bytes)
};                                  // total: 22560 bytes
