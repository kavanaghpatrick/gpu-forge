// parquet_decode.metal -- GPU Parquet column decoder kernels.
//
// Decodes raw Parquet column data (plain-encoded) into SoA columnar buffers.
// CPU reads Parquet metadata (footer, schema, row groups) and extracts raw
// column chunk bytes. GPU decodes the raw bytes into typed arrays.
//
// Encoding support:
//   - PLAIN INT64: direct memcpy (8 bytes per value)
//   - PLAIN FLOAT/DOUBLE: direct memcpy (4/8 bytes per value)
//   - PLAIN INT32: widen to INT64 (4 bytes in, 8 bytes out)
//   - DICTIONARY: index lookup from dict buffer

#include "types.h"

// ---- Parameters for Parquet decode kernels ----

struct ParquetDecodeParams {
    uint row_count;      // number of values to decode
    uint src_stride;     // bytes per source element
    uint dst_stride;     // bytes per destination element
    uint _pad0;
};

// ---- parquet_decode_plain_int64 ----
//
// Decode plain-encoded INT64 values from raw Parquet column bytes.
// Input: raw bytes (contiguous int64 values), output: SoA int64 array.
// One thread per row.

kernel void parquet_decode_plain_int64(
    device const char*          src_data    [[buffer(0)]],
    device long*                dst_data    [[buffer(1)]],
    constant ParquetDecodeParams& params    [[buffer(2)]],
    uint tid                                [[thread_position_in_grid]]
) {
    if (tid >= params.row_count) {
        return;
    }

    // Plain encoding: values are contiguous int64 (8 bytes each)
    device const long* src = reinterpret_cast<device const long*>(src_data);
    dst_data[tid] = src[tid];
}

// ---- parquet_decode_plain_int32 ----
//
// Decode plain-encoded INT32 values and widen to INT64.
// One thread per row.

kernel void parquet_decode_plain_int32(
    device const char*          src_data    [[buffer(0)]],
    device long*                dst_data    [[buffer(1)]],
    constant ParquetDecodeParams& params    [[buffer(2)]],
    uint tid                                [[thread_position_in_grid]]
) {
    if (tid >= params.row_count) {
        return;
    }

    // Plain INT32: 4 bytes each, widen to int64
    device const int* src = reinterpret_cast<device const int*>(src_data);
    dst_data[tid] = static_cast<long>(src[tid]);
}

// ---- parquet_decode_plain_float ----
//
// Decode plain-encoded FLOAT (32-bit) values from raw Parquet column bytes.
// One thread per row.

kernel void parquet_decode_plain_float(
    device const char*          src_data    [[buffer(0)]],
    device float*               dst_data    [[buffer(1)]],
    constant ParquetDecodeParams& params    [[buffer(2)]],
    uint tid                                [[thread_position_in_grid]]
) {
    if (tid >= params.row_count) {
        return;
    }

    // Plain encoding: contiguous float values (4 bytes each)
    device const float* src = reinterpret_cast<device const float*>(src_data);
    dst_data[tid] = src[tid];
}

// ---- parquet_decode_plain_double ----
//
// Decode plain-encoded DOUBLE (64-bit) values, downcast to float (32-bit)
// since Metal lacks double support.
// One thread per row.

kernel void parquet_decode_plain_double(
    device const char*          src_data    [[buffer(0)]],
    device float*               dst_data    [[buffer(1)]],
    constant ParquetDecodeParams& params    [[buffer(2)]],
    uint tid                                [[thread_position_in_grid]]
) {
    if (tid >= params.row_count) {
        return;
    }

    // Plain DOUBLE: 8 bytes each. Read as two uint32 and reconstruct double bits,
    // then downcast to float. Metal has no native double, but we can bit-cast.
    // Simpler approach: read as two 32-bit halves and do manual conversion.
    device const uint* src = reinterpret_cast<device const uint*>(src_data);
    uint lo = src[tid * 2];
    uint hi = src[tid * 2 + 1];

    // Reconstruct IEEE 754 double from bits
    // Sign: bit 63 of hi
    // Exponent: bits 62-52 (11 bits) in hi
    // Mantissa: bits 51-0 across hi(19:0) and lo(31:0)
    bool sign = (hi >> 31) & 1;
    int exponent = ((hi >> 20) & 0x7FF) - 1023;
    // For conversion to float32: re-bias exponent for float (bias 127)
    // and truncate mantissa from 52 to 23 bits

    // Handle special cases
    if (exponent == 1024) {
        // Inf or NaN
        uint float_bits = (sign ? 0x80000000u : 0u) | 0x7F800000u;
        if ((hi & 0x000FFFFF) != 0 || lo != 0) {
            float_bits |= 0x00400000u; // NaN
        }
        dst_data[tid] = as_type<float>(float_bits);
        return;
    }

    if (exponent == -1023) {
        // Zero or denormal -> zero in float
        dst_data[tid] = sign ? -0.0f : 0.0f;
        return;
    }

    // Normal number: re-bias exponent
    int float_exp = exponent + 127;
    if (float_exp >= 255) {
        // Overflow -> infinity
        uint float_bits = (sign ? 0x80000000u : 0u) | 0x7F800000u;
        dst_data[tid] = as_type<float>(float_bits);
        return;
    }
    if (float_exp <= 0) {
        // Underflow -> zero
        dst_data[tid] = sign ? -0.0f : 0.0f;
        return;
    }

    // Truncate mantissa: take top 23 bits of 52-bit mantissa
    // Double mantissa: hi[19:0] (20 bits) + lo[31:0] (32 bits) = 52 bits
    // Float mantissa: 23 bits = hi[19:0] (20 bits) + lo[31:29] (3 bits)
    uint mantissa = ((hi & 0x000FFFFF) << 3) | (lo >> 29);

    uint float_bits = (sign ? 0x80000000u : 0u)
                    | (static_cast<uint>(float_exp) << 23)
                    | mantissa;
    dst_data[tid] = as_type<float>(float_bits);
}

// ---- parquet_decode_dictionary ----
//
// Dictionary decoding: look up dictionary index -> value.
// Input: dictionary buffer (int64 values), indices buffer (int32 indices),
// Output: decoded int64 values.
// One thread per row.

kernel void parquet_decode_dictionary(
    device const long*          dict_data   [[buffer(0)]],
    device const int*           indices     [[buffer(1)]],
    device long*                dst_data    [[buffer(2)]],
    constant ParquetDecodeParams& params    [[buffer(3)]],
    uint tid                                [[thread_position_in_grid]]
) {
    if (tid >= params.row_count) {
        return;
    }

    int idx = indices[tid];
    dst_data[tid] = dict_data[idx];
}
