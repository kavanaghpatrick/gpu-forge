// csv_parse.metal -- GPU CSV parser kernels.
//
// Two-pass parser:
//   Pass 1 (csv_detect_newlines): Parallel scan for '\n' characters.
//          Each thread checks one byte position for '\n', stores row
//          offsets via atomic counter.
//
//   Pass 2 (csv_parse_fields): Per-row field extraction with type coercion.
//          Each thread processes one row, splits by delimiter, parses
//          integer/float fields, writes to SoA column buffers.
//
// Layout: SoA columns at [col_idx * max_rows + row_idx].
// Float columns use float (32-bit) since Metal lacks double support.
// INT64 columns use long (64-bit).

#include "types.h"

// ---- Pass 1: Newline Detection ----

kernel void csv_detect_newlines(
    device const char*       data         [[buffer(0)]],
    device atomic_uint*      row_count    [[buffer(1)]],
    device uint*             row_offsets  [[buffer(2)]],
    constant CsvParseParams& params       [[buffer(3)]],
    uint tid                              [[thread_position_in_grid]]
) {
    // Each thread checks one byte position.
    if (tid >= params.file_size) {
        return;
    }

    // Check for newline at this position
    if (data[tid] == '\n') {
        // Atomic increment to get a unique slot for this row offset.
        uint row_idx = atomic_fetch_add_explicit(row_count, 1, memory_order_relaxed);
        // Store the byte position AFTER the newline (start of next row).
        row_offsets[row_idx] = tid + 1;
    }
}


// ---- Helper: parse integer from ASCII ----

static inline long parse_int(device const char* data, uint start, uint end) {
    long result = 0;
    bool negative = false;
    uint i = start;

    // Skip leading whitespace
    while (i < end && (data[i] == ' ' || data[i] == '\t')) {
        i++;
    }

    if (i < end && data[i] == '-') {
        negative = true;
        i++;
    } else if (i < end && data[i] == '+') {
        i++;
    }

    while (i < end && data[i] >= '0' && data[i] <= '9') {
        result = result * 10 + (data[i] - '0');
        i++;
    }

    return negative ? -result : result;
}


// ---- Helper: parse float from ASCII (32-bit) ----

static inline float parse_float(device const char* data, uint start, uint end) {
    float result = 0.0f;
    float fraction = 0.0f;
    float divisor = 1.0f;
    bool negative = false;
    bool in_fraction = false;
    uint i = start;

    // Skip leading whitespace
    while (i < end && (data[i] == ' ' || data[i] == '\t')) {
        i++;
    }

    if (i < end && data[i] == '-') {
        negative = true;
        i++;
    } else if (i < end && data[i] == '+') {
        i++;
    }

    while (i < end) {
        char c = data[i];
        if (c >= '0' && c <= '9') {
            if (in_fraction) {
                divisor *= 10.0f;
                fraction += (c - '0') / divisor;
            } else {
                result = result * 10.0f + (c - '0');
            }
        } else if (c == '.' && !in_fraction) {
            in_fraction = true;
        } else {
            break; // stop on non-numeric
        }
        i++;
    }

    result += fraction;
    return negative ? -result : result;
}


// ---- Pass 2: Field Extraction ----
//
// Each thread processes one row. Walks from row start to row end,
// splits by delimiter, parses each field based on column schema,
// writes to SoA column buffers at [col_idx * max_rows + row_idx].
//
// Column schemas are passed in a buffer: array of ColumnSchema structs.
// data_type: 0=INT64, 1=FLOAT64, 2=VARCHAR (skip for now)
//
// Float columns use float (32-bit) since Metal does not support double.

kernel void csv_parse_fields(
    device const char*          data          [[buffer(0)]],
    device const uint*          row_offsets   [[buffer(1)]],
    device const uint*          row_count_buf [[buffer(2)]],
    device long*                int_columns   [[buffer(3)]],
    device float*               float_columns [[buffer(4)]],
    constant CsvParseParams&    params        [[buffer(5)]],
    device const ColumnSchema*  schemas       [[buffer(6)]],
    uint tid                                  [[thread_position_in_grid]]
) {
    uint num_data_rows = row_count_buf[0];
    if (tid >= num_data_rows) {
        return;
    }

    // SoA stride: always use params.max_rows for buffer layout consistency
    // with the host-side ColumnarBatch allocation.
    uint soa_stride = params.max_rows;

    // Determine row start and end.
    uint row_start = row_offsets[tid];
    uint row_end;
    if (tid + 1 < num_data_rows) {
        row_end = row_offsets[tid + 1] - 1;
    } else {
        row_end = params.file_size;
    }

    // Strip trailing \r or \n
    while (row_end > row_start && (data[row_end - 1] == '\n' || data[row_end - 1] == '\r')) {
        row_end--;
    }

    // Walk through fields separated by delimiter
    char delim = (char)params.delimiter;
    uint col = 0;
    uint field_start = row_start;
    uint num_cols = params.num_columns;

    // Track which int/float column index we're at
    uint int_col_idx = 0;
    uint float_col_idx = 0;

    for (uint pos = row_start; pos <= row_end && col < num_cols; pos++) {
        bool is_delim = (pos < row_end) && (data[pos] == delim);
        bool is_end = (pos == row_end);

        if (is_delim || is_end) {
            uint field_end = pos;

            // Parse field based on schema type
            if (col < num_cols) {
                uint dtype = schemas[col].data_type;

                if (dtype == 0) {
                    // INT64
                    long val = parse_int(data, field_start, field_end);
                    int_columns[int_col_idx * soa_stride + tid] = val;
                    int_col_idx++;
                } else if (dtype == 1) {
                    // FLOAT32 (stored as float since Metal lacks double)
                    float val = parse_float(data, field_start, field_end);
                    float_columns[float_col_idx * soa_stride + tid] = val;
                    float_col_idx++;
                }
                // dtype == 2 (VARCHAR): skip for now
            }

            col++;
            field_start = pos + 1;
        }
    }
}
