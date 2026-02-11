// schema_infer.metal -- GPU schema inference via type voting.
//
// Each thread examines one field value from sample data and votes on
// its type (INT64, FLOAT64, VARCHAR). Atomic counters per column
// accumulate votes; the type with the most votes wins.
//
// For POC: this kernel is a placeholder demonstrating the GPU type
// voting pattern. The CPU-side multi-row inference (sampling up to
// 100 rows) handles schema inference for now. This kernel is ready
// for GPU acceleration when needed for large-scale inference (10K+ rows).

#include <metal_stdlib>
#include "types.h"

using namespace metal;

// Per-column vote counters: [col * 3 + 0] = INT64 votes,
//                           [col * 3 + 1] = FLOAT64 votes,
//                           [col * 3 + 2] = VARCHAR votes
// Plus nullable flag:       [num_columns * 3 + col] = null votes

/// Parameters for schema inference kernel.
struct InferParams {
    uint num_columns;      // number of columns
    uint num_rows;         // number of sample rows
    uint file_size;        // total file size (for bounds checking)
    uint _pad0;
};

// Type codes matching ColumnSchema.data_type
constant uint TYPE_INT64   = 0;
constant uint TYPE_FLOAT64 = 1;
constant uint TYPE_VARCHAR = 2;

/// Check if a byte is a digit character ('0'-'9').
inline bool is_digit(uchar c) {
    return c >= '0' && c <= '9';
}

/// Check if a byte sequence represents a valid integer.
/// Supports optional leading minus sign.
inline bool is_integer(const device uchar* data, uint start, uint len) {
    if (len == 0) return false;
    uint i = 0;
    // Optional leading minus
    if (data[start] == '-') {
        i = 1;
        if (len == 1) return false; // just a minus sign
    }
    for (; i < len; i++) {
        if (!is_digit(data[start + i])) return false;
    }
    return true;
}

/// Check if a byte sequence represents a valid float.
/// Supports optional leading minus, digits, one decimal point.
inline bool is_float(const device uchar* data, uint start, uint len) {
    if (len == 0) return false;
    uint i = 0;
    bool has_dot = false;
    bool has_digit_flag = false;
    // Optional leading minus
    if (data[start] == '-') {
        i = 1;
        if (len == 1) return false;
    }
    for (; i < len; i++) {
        uchar c = data[start + i];
        if (is_digit(c)) {
            has_digit_flag = true;
        } else if (c == '.' && !has_dot) {
            has_dot = true;
        } else if (c == 'e' || c == 'E') {
            // Scientific notation: rest must be optional sign + digits
            i++;
            if (i < len && (data[start + i] == '+' || data[start + i] == '-')) i++;
            for (; i < len; i++) {
                if (!is_digit(data[start + i])) return false;
            }
            return has_digit_flag;
        } else {
            return false;
        }
    }
    return has_digit_flag;
}

/// Schema inference kernel: each thread processes one field value.
///
/// Thread ID = row * num_columns + col
/// Examines the field, votes for INT64, FLOAT64, or VARCHAR.
///
/// Buffers:
///   - data: raw file bytes (mmap'd)
///   - row_offsets: byte offset of each sample row start [num_rows]
///   - field_offsets: byte offset of each field start [num_rows * num_columns]
///   - field_lengths: byte length of each field [num_rows * num_columns]
///   - votes: atomic counters [num_columns * 3] for type votes
///            + [num_columns] for null votes at offset num_columns*3
///   - params: InferParams
kernel void infer_schema(
    const device uchar*   data          [[buffer(0)]],
    const device uint*    field_offsets  [[buffer(1)]],
    const device uint*    field_lengths  [[buffer(2)]],
    device atomic_uint*   votes         [[buffer(3)]],
    constant InferParams& params        [[buffer(4)]],
    uint tid                            [[thread_position_in_grid]]
) {
    uint total_fields = params.num_rows * params.num_columns;
    if (tid >= total_fields) return;

    uint col = tid % params.num_columns;
    uint offset = field_offsets[tid];
    uint len = field_lengths[tid];

    // Empty field = null vote
    if (len == 0) {
        // Vote null for this column (stored after type votes)
        atomic_fetch_add_explicit(&votes[params.num_columns * 3 + col],
                                  1, memory_order_relaxed);
        return;
    }

    // Determine type
    uint vote_idx;
    if (is_integer(data, offset, len)) {
        vote_idx = col * 3 + TYPE_INT64;
    } else if (is_float(data, offset, len)) {
        vote_idx = col * 3 + TYPE_FLOAT64;
    } else {
        vote_idx = col * 3 + TYPE_VARCHAR;
    }

    atomic_fetch_add_explicit(&votes[vote_idx], 1, memory_order_relaxed);
}
