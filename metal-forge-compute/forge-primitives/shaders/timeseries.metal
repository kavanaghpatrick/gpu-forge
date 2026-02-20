// timeseries.metal -- Time series analytics kernels.
//
// Each thread computes one output element for a sliding-window operation.
// Window size is specified in params; each thread reads window_size elements
// from the prices/volumes arrays.
//
// Kernels:
//   timeseries_moving_avg: simple moving average of prices
//   timeseries_vwap: volume-weighted average price per window
//
// 256 threads per threadgroup.

#include "types.h"

#define TS_THREADS_PER_TG 256

/// Moving average kernel: output[i] = mean(prices[i-W+1 .. i]) for i >= W-1.
/// For i < W-1, averages over available elements (partial window).
///
/// Buffers:
///   [0] prices:  device const float*       -- input price series
///   [1] output:  device float*             -- moving average output
///   [2] params:  constant TimeSeriesParams& -- tick_count, window_size
kernel void timeseries_moving_avg(
    device const float*         prices  [[buffer(0)]],
    device float*               output  [[buffer(1)]],
    constant TimeSeriesParams&  params  [[buffer(2)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= params.tick_count) {
        return;
    }

    uint window = params.window_size;
    // Start of the window: clamp to 0
    uint start = (gid >= window - 1) ? (gid - window + 1) : 0;
    uint count = gid - start + 1;

    float sum = 0.0f;
    uint i = start;
    // Vectorized window read: process 4 elements at a time
    for (; i + 3 <= gid; i += 4) {
        float4 chunk = float4(prices[i], prices[i+1], prices[i+2], prices[i+3]);
        sum += chunk.x + chunk.y + chunk.z + chunk.w;
    }
    // Scalar remainder
    for (; i <= gid; i++) {
        sum += prices[i];
    }

    output[gid] = sum / float(count);
}

/// VWAP kernel: output[i] = sum(price[j]*volume[j]) / sum(volume[j])
/// for j in [i-W+1 .. i]. Partial windows for i < W-1.
///
/// Buffers:
///   [0] prices:  device const float*       -- input price series
///   [1] volumes: device const float*       -- input volume series
///   [2] output:  device float*             -- VWAP output
///   [3] params:  constant TimeSeriesParams& -- tick_count, window_size
kernel void timeseries_vwap(
    device const float*         prices  [[buffer(0)]],
    device const float*         volumes [[buffer(1)]],
    device float*               output  [[buffer(2)]],
    constant TimeSeriesParams&  params  [[buffer(3)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= params.tick_count) {
        return;
    }

    uint window = params.window_size;
    uint start = (gid >= window - 1) ? (gid - window + 1) : 0;

    float pv_sum = 0.0f;
    float v_sum = 0.0f;
    for (uint i = start; i <= gid; i++) {
        float p = prices[i];
        float v = volumes[i];
        pv_sum += p * v;
        v_sum += v;
    }

    output[gid] = (v_sum > 0.0f) ? (pv_sum / v_sum) : 0.0f;
}
