// spreadsheet.metal -- GPU spreadsheet formula evaluation kernels.
//
// Three kernels simulating spreadsheet operations on a 2D grid (rows x cols):
//   1. spreadsheet_sum: Column SUM -- each thread computes sum of one column
//   2. spreadsheet_average: Column AVERAGE -- SUM / row_count per column
//   3. spreadsheet_vlookup: Binary search in sorted lookup column
//
// Grid layout: row-major float array, grid[row * cols + col].
// All kernels use 256 threads per threadgroup, dispatched 1D over columns (sum/avg)
// or over rows (vlookup).

#include "types.h"

// ============================================================================
// spreadsheet_sum -- Column SUM reduction
// ============================================================================
//
// Each thread sums all rows in one column. Output: one float per column.
//
// Buffer layout:
//   buffer(0): input grid (float array, rows * cols elements, row-major)
//   buffer(1): output sums (float array, cols elements)
//   buffer(2): SpreadsheetParams
//
// Dispatch: ceil(cols / 256) threadgroups of 256 threads.

kernel void spreadsheet_sum(
    device const float*          grid    [[buffer(0)]],
    device float*                output  [[buffer(1)]],
    constant SpreadsheetParams&  params  [[buffer(2)]],
    uint tid                             [[thread_position_in_grid]]
) {
    uint col = tid;
    if (col >= params.cols) return;

    float sum = 0.0f;
    for (uint row = 0; row < params.rows; row++) {
        sum += grid[row * params.cols + col];
    }
    output[col] = sum;
}


// ============================================================================
// spreadsheet_sum_v2 -- 2D row-chunked Column SUM (high parallelism)
// ============================================================================
//
// Each thread sums SS_ROWS_PER_CHUNK rows for one column, writing a partial
// sum to partials[row_chunk * cols + col]. This creates many more threadgroups
// than the 1D version (e.g. 3162/64 * ceil(3162/256) ≈ 600+ TGs vs 13).
//
// Buffer layout:
//   buffer(0): input grid (float array, rows * cols, row-major)
//   buffer(1): partials  (float array, num_row_chunks * cols)
//   buffer(2): SpreadsheetParams
//
// Dispatch 2D: (ceil(cols/TG_X), ceil(rows/SS_ROWS_PER_CHUNK)) threadgroups.

#define SS_ROWS_PER_CHUNK 64

kernel void spreadsheet_sum_v2(
    device const float*          grid     [[buffer(0)]],
    device float*                partials [[buffer(1)]],
    constant SpreadsheetParams&  params   [[buffer(2)]],
    uint2 tid                            [[thread_position_in_grid]]
) {
    uint col = tid.x;
    uint row_chunk = tid.y;
    if (col >= params.cols) return;

    uint row_start = row_chunk * SS_ROWS_PER_CHUNK;
    uint row_end = min(row_start + SS_ROWS_PER_CHUNK, params.rows);
    if (row_start >= params.rows) return;

    float sum = 0.0f;
    for (uint row = row_start; row < row_end; row++) {
        sum += grid[row * params.cols + col];
    }

    uint num_row_chunks = (params.rows + SS_ROWS_PER_CHUNK - 1) / SS_ROWS_PER_CHUNK;
    partials[row_chunk * params.cols + col] = sum;
}


// ============================================================================
// spreadsheet_sum_reduce -- Reduce row-chunk partials into final column sums
// ============================================================================
//
// Each thread reduces all row-chunk partials for one column.
//
// Buffer layout:
//   buffer(0): partials (float array, num_row_chunks * cols)
//   buffer(1): output   (float array, cols)
//   buffer(2): SpreadsheetParams (rows used to compute num_row_chunks)

kernel void spreadsheet_sum_reduce(
    device const float*          partials [[buffer(0)]],
    device float*                output   [[buffer(1)]],
    constant SpreadsheetParams&  params   [[buffer(2)]],
    uint tid                             [[thread_position_in_grid]]
) {
    uint col = tid;
    if (col >= params.cols) return;

    uint num_row_chunks = (params.rows + SS_ROWS_PER_CHUNK - 1) / SS_ROWS_PER_CHUNK;
    float sum = 0.0f;
    for (uint chunk = 0; chunk < num_row_chunks; chunk++) {
        sum += partials[chunk * params.cols + col];
    }
    output[col] = sum;
}


// ============================================================================
// spreadsheet_avg_v2 -- 2D row-chunked Column partial sums (for AVERAGE)
// ============================================================================
// Same as spreadsheet_sum_v2 -- computes partial sums per row chunk.
// The division by rows happens in the reduce step.

kernel void spreadsheet_avg_v2(
    device const float*          grid     [[buffer(0)]],
    device float*                partials [[buffer(1)]],
    constant SpreadsheetParams&  params   [[buffer(2)]],
    uint2 tid                            [[thread_position_in_grid]]
) {
    uint col = tid.x;
    uint row_chunk = tid.y;
    if (col >= params.cols) return;

    uint row_start = row_chunk * SS_ROWS_PER_CHUNK;
    uint row_end = min(row_start + SS_ROWS_PER_CHUNK, params.rows);
    if (row_start >= params.rows) return;

    float sum = 0.0f;
    for (uint row = row_start; row < row_end; row++) {
        sum += grid[row * params.cols + col];
    }

    partials[row_chunk * params.cols + col] = sum;
}


// ============================================================================
// spreadsheet_avg_reduce -- Reduce partials and divide by row count
// ============================================================================

kernel void spreadsheet_avg_reduce(
    device const float*          partials [[buffer(0)]],
    device float*                output   [[buffer(1)]],
    constant SpreadsheetParams&  params   [[buffer(2)]],
    uint tid                             [[thread_position_in_grid]]
) {
    uint col = tid;
    if (col >= params.cols) return;

    uint num_row_chunks = (params.rows + SS_ROWS_PER_CHUNK - 1) / SS_ROWS_PER_CHUNK;
    float sum = 0.0f;
    for (uint chunk = 0; chunk < num_row_chunks; chunk++) {
        sum += partials[chunk * params.cols + col];
    }
    output[col] = sum / float(params.rows);
}


// ============================================================================
// spreadsheet_average -- Column AVERAGE (SUM / COUNT)
// ============================================================================
//
// Each thread computes mean of all rows in one column.
//
// Buffer layout:
//   buffer(0): input grid (float array, rows * cols elements, row-major)
//   buffer(1): output averages (float array, cols elements)
//   buffer(2): SpreadsheetParams
//
// Dispatch: ceil(cols / 256) threadgroups of 256 threads.

kernel void spreadsheet_average(
    device const float*          grid    [[buffer(0)]],
    device float*                output  [[buffer(1)]],
    constant SpreadsheetParams&  params  [[buffer(2)]],
    uint tid                             [[thread_position_in_grid]]
) {
    uint col = tid;
    if (col >= params.cols) return;

    float sum = 0.0f;
    for (uint row = 0; row < params.rows; row++) {
        sum += grid[row * params.cols + col];
    }
    output[col] = sum / float(params.rows);
}


// ============================================================================
// spreadsheet_vlookup -- Binary search in sorted lookup column
// ============================================================================
//
// Each thread has a search key and performs binary search in a sorted array.
// Returns the value from a corresponding value column at the found index
// (or -1.0 if not found).
//
// Buffer layout:
//   buffer(0): sorted lookup keys (float array, rows elements, ascending)
//   buffer(1): lookup values (float array, rows elements)
//   buffer(2): search keys (float array, rows elements -- one per thread)
//   buffer(3): output results (float array, rows elements)
//   buffer(4): SpreadsheetParams (rows = array length, cols unused)
//
// Dispatch: ceil(rows / 256) threadgroups of 256 threads.

kernel void spreadsheet_vlookup(
    device const float*          lookup_keys   [[buffer(0)]],
    device const float*          lookup_vals   [[buffer(1)]],
    device const float*          search_keys   [[buffer(2)]],
    device float*                output        [[buffer(3)]],
    constant SpreadsheetParams&  params        [[buffer(4)]],
    uint tid                                   [[thread_position_in_grid]]
) {
    if (tid >= params.rows) return;

    float key = search_keys[tid];
    uint lo = 0;
    uint hi = params.rows;
    float result = -1.0f;

    // Binary search: find largest index where lookup_keys[idx] <= key
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2;
        if (lookup_keys[mid] <= key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // lo-1 is the last index where lookup_keys[idx] <= key
    if (lo > 0) {
        result = lookup_vals[lo - 1];
    }

    output[tid] = result;
}
