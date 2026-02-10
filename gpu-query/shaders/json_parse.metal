// json_parse.metal -- GPU NDJSON parser kernels.
//
// Two-pass parser for NDJSON (newline-delimited JSON):
//   Pass 1 (json_structural_index): Parallel scan for structural characters.
//          Each thread scans a chunk of bytes looking for newlines to identify
//          row boundaries. Similar to csv_detect_newlines but for JSON.
//
//   Pass 2 (json_extract_columns): Per-row field extraction.
//          Each thread processes one NDJSON line, walks key-value pairs,
//          extracts integer/float values into SoA column buffers.
//
// NDJSON format: one JSON object per line:
//   {"id":1,"amount":150,"name":"alice"}
//   {"id":2,"amount":200,"name":"bob"}
//
// Assumes fields appear in the same order in every line (POC simplification).
// Layout: SoA columns at [col_idx * max_rows + row_idx].
// Float columns use float (32-bit) since Metal lacks double support.
// INT64 columns use long (64-bit).

#include "types.h"

// Reuse CsvParseParams for JSON parsing:
// - file_size: total bytes
// - num_columns: number of fields to extract
// - delimiter: unused (set to 0)
// - has_header: 0 (NDJSON has no header)
// - max_rows: SoA stride

// ---- Pass 1: Structural Indexing (Newline Detection) ----
//
// Each thread checks one byte position for '\n'.
// Outputs newline offsets (byte positions after '\n') and count.
// This is functionally identical to csv_detect_newlines but named
// for JSON context.

kernel void json_structural_index(
    device const char*       data         [[buffer(0)]],
    device atomic_uint*      row_count    [[buffer(1)]],
    device uint*             row_offsets  [[buffer(2)]],
    constant CsvParseParams& params       [[buffer(3)]],
    uint tid                              [[thread_position_in_grid]]
) {
    if (tid >= params.file_size) {
        return;
    }

    // Detect newline = end of one JSON object / start of next
    if (data[tid] == '\n') {
        uint idx = atomic_fetch_add_explicit(row_count, 1, memory_order_relaxed);
        // Store byte position AFTER the newline (start of next row)
        row_offsets[idx] = tid + 1;
    }
}


// ---- Helper: parse integer from JSON value ----

static inline long json_parse_int(device const char* data, uint start, uint end) {
    long result = 0;
    bool negative = false;
    uint i = start;

    // Skip whitespace
    while (i < end && (data[i] == ' ' || data[i] == '\t')) {
        i++;
    }

    if (i < end && data[i] == '-') {
        negative = true;
        i++;
    }

    while (i < end && data[i] >= '0' && data[i] <= '9') {
        result = result * 10 + (data[i] - '0');
        i++;
    }

    return negative ? -result : result;
}


// ---- Helper: parse float from JSON value (32-bit) ----

static inline float json_parse_float(device const char* data, uint start, uint end) {
    float result = 0.0f;
    float fraction = 0.0f;
    float divisor = 1.0f;
    bool negative = false;
    bool in_fraction = false;
    uint i = start;

    // Skip whitespace
    while (i < end && (data[i] == ' ' || data[i] == '\t')) {
        i++;
    }

    if (i < end && data[i] == '-') {
        negative = true;
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
            break;
        }
        i++;
    }

    result += fraction;
    return negative ? -result : result;
}


// ---- Pass 2: Field Extraction ----
//
// Each thread processes one NDJSON line (one JSON object).
// Walks through key-value pairs in order, matching fields by position
// (assumes all lines have fields in the same order).
//
// For each field, based on the schema type:
// - INT64: parse number, write to int_columns[int_col_idx * max_rows + row_idx]
// - FLOAT64: parse number, write to float_columns[float_col_idx * max_rows + row_idx]
// - VARCHAR: skip (not supported in POC)

kernel void json_extract_columns(
    device const char*          data          [[buffer(0)]],
    device const uint*          row_offsets   [[buffer(1)]],
    device const uint*          row_count_buf [[buffer(2)]],
    device long*                int_columns   [[buffer(3)]],
    device float*               float_columns [[buffer(4)]],
    constant CsvParseParams&    params        [[buffer(5)]],
    device const ColumnSchema*  schemas       [[buffer(6)]],
    uint tid                                  [[thread_position_in_grid]]
) {
    uint num_rows = row_count_buf[0];
    if (tid >= num_rows) {
        return;
    }

    uint soa_stride = params.max_rows;
    uint num_fields = params.num_columns;

    // Determine row start and end
    uint row_start = row_offsets[tid];
    uint row_end;
    if (tid + 1 < num_rows) {
        row_end = row_offsets[tid + 1];
    } else {
        row_end = params.file_size;
    }

    // Strip trailing \r or \n
    while (row_end > row_start && (data[row_end - 1] == '\n' || data[row_end - 1] == '\r')) {
        row_end--;
    }

    // Skip opening '{' and whitespace
    uint pos = row_start;
    while (pos < row_end && (data[pos] == '{' || data[pos] == ' ' || data[pos] == '\t')) {
        pos++;
    }

    uint int_col_idx = 0;
    uint float_col_idx = 0;
    uint field_idx = 0;

    while (pos < row_end && field_idx < num_fields) {
        // Skip whitespace
        while (pos < row_end && (data[pos] == ' ' || data[pos] == '\t')) {
            pos++;
        }

        // Expect '"' for key start
        if (pos >= row_end || data[pos] != '"') {
            break;
        }
        pos++; // skip opening quote of key

        // Skip key (we identify fields by position order)
        while (pos < row_end && data[pos] != '"') {
            if (data[pos] == '\\') pos++; // skip escaped char
            pos++;
        }
        if (pos < row_end) pos++; // skip closing quote of key

        // Skip whitespace and colon
        while (pos < row_end && (data[pos] == ' ' || data[pos] == '\t')) {
            pos++;
        }
        if (pos < row_end && data[pos] == ':') {
            pos++;
        }

        // Skip whitespace before value
        while (pos < row_end && (data[pos] == ' ' || data[pos] == '\t')) {
            pos++;
        }

        // Parse value
        uint val_start = pos;
        uint val_end = pos;

        if (pos < row_end && data[pos] == '"') {
            // String value - skip to closing quote
            pos++; // skip opening quote
            while (pos < row_end && data[pos] != '"') {
                if (data[pos] == '\\') pos++;
                pos++;
            }
            if (pos < row_end) pos++; // skip closing quote
            val_end = pos;
        } else {
            // Number, boolean, or null - scan to delimiter
            while (pos < row_end && data[pos] != ',' && data[pos] != '}' && data[pos] != ' ' && data[pos] != '\t') {
                pos++;
            }
            val_end = pos;
        }

        // Write value to column buffer based on schema type
        if (field_idx < num_fields) {
            uint dtype = schemas[field_idx].data_type;

            if (dtype == 0) {
                // INT64
                long val = json_parse_int(data, val_start, val_end);
                int_columns[int_col_idx * soa_stride + tid] = val;
                int_col_idx++;
            } else if (dtype == 1) {
                // FLOAT64 (stored as float32)
                float val = json_parse_float(data, val_start, val_end);
                float_columns[float_col_idx * soa_stride + tid] = val;
                float_col_idx++;
            }
            // dtype == 2 (VARCHAR): skip
        }

        field_idx++;

        // Skip whitespace and comma
        while (pos < row_end && (data[pos] == ' ' || data[pos] == '\t')) {
            pos++;
        }
        if (pos < row_end && data[pos] == ',') {
            pos++;
        }
    }
}
