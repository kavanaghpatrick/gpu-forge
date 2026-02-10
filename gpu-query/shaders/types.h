#ifndef TYPES_H
#define TYPES_H

// types.h -- Shared struct definitions between Rust host and MSL shaders.
//
// All structs here must match their Rust counterparts in src/gpu/types.rs byte-for-byte.
// GPU-query analytics engine: filter, aggregate, sort, parse, dispatch, schema.

#include <metal_stdlib>
using namespace metal;

// Parameters for column filter kernel (WHERE clause).
// Operator and type are set via function constants; this struct carries thresholds.
//
// Layout: 40 bytes (5 x 8-byte fields), 8-byte aligned.
struct FilterParams {
    long     compare_value_int;    // offset 0  (8 bytes) — threshold for INT64 comparisons
    double   compare_value_float;  // offset 8  (8 bytes) — threshold for FLOAT64 comparisons
    uint     row_count;            // offset 16 (4 bytes) — number of rows to process
    uint     column_stride;        // offset 20 (4 bytes) — bytes between elements (for packed types)
    uint     null_bitmap_present;  // offset 24 (4 bytes) — 1 if null bitmap is valid
    uint     _pad0;                // offset 28 (4 bytes) — align to 8
    long     compare_value_int_hi; // offset 32 (8 bytes) — upper bound for BETWEEN
};                                  // total: 40 bytes

// Parameters for aggregation kernel (SUM/COUNT/MIN/MAX/AVG).
//
// Layout: 16 bytes (4 x uint), 4-byte aligned.
struct AggParams {
    uint     row_count;            // offset 0  (4 bytes) — total rows to aggregate
    uint     group_count;          // offset 4  (4 bytes) — number of groups (0 = no GROUP BY)
    uint     agg_function;         // offset 8  (4 bytes) — 0=COUNT,1=SUM,2=AVG,3=MIN,4=MAX
    uint     _pad0;                // offset 12 (4 bytes)
};                                  // total: 16 bytes

// Parameters for radix sort kernel (ORDER BY).
//
// Layout: 16 bytes (4 x uint), 4-byte aligned.
struct SortParams {
    uint     element_count;        // offset 0  (4 bytes) — number of elements to sort
    uint     bit_offset;           // offset 4  (4 bytes) — current 4-bit digit position (0,4,8,...60)
    uint     descending;           // offset 8  (4 bytes) — 0=ASC, 1=DESC (key XOR transform)
    uint     _pad0;                // offset 12 (4 bytes)
};                                  // total: 16 bytes

// Parameters for GPU CSV parser kernels.
//
// Layout: 24 bytes, 4-byte aligned.
struct CsvParseParams {
    uint     file_size;            // offset 0  (4 bytes) — total bytes in mmap'd file
    uint     num_columns;          // offset 4  (4 bytes) — number of CSV columns
    uint     delimiter;            // offset 8  (4 bytes) — delimiter character (ASCII code)
    uint     has_header;           // offset 12 (4 bytes) — 1 if first row is header
    uint     max_rows;             // offset 16 (4 bytes) — maximum rows to parse (0 = unlimited)
    uint     _pad0;                // offset 20 (4 bytes)
};                                  // total: 24 bytes

// Indirect dispatch arguments for dispatchThreadgroups(indirectBuffer:).
// Layout: 3 x uint = 12 bytes, matching Metal's indirect dispatch buffer format.
struct DispatchArgs {
    uint threadgroupsPerGridX;
    uint threadgroupsPerGridY;
    uint threadgroupsPerGridZ;
};

// Column schema descriptor passed to GPU kernels for type-aware parsing.
//
// Layout: 8 bytes (2 x uint), 4-byte aligned.
struct ColumnSchema {
    uint     data_type;            // offset 0 (4 bytes) — 0=INT64,1=FLOAT64,2=VARCHAR,3=BOOL,4=DATE
    uint     dict_encoded;         // offset 4 (4 bytes) — 1 if dictionary-encoded string column
};                                  // total: 8 bytes

#endif // TYPES_H
