#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 17: MSD+LSD Hybrid Radix Sort (5000+ Mkeys/s target)
//
// Architecture: 1 MSD scatter (bits 24:31) → 256 SLC-resident buckets
// → 3 inner LSD passes per bucket at SLC speed.
// Single encoder, 14 dispatches, zero CPU readback.
// ═══════════════════════════════════════════════════════════════════

#define EXP17_NUM_BINS  256u
#define EXP17_TILE_SIZE 4096u
#define EXP17_ELEMS     16u
#define EXP17_THREADS   256u
#define EXP17_NUM_SGS   8u
#define EXP17_MAX_TPB   17u

struct Exp17Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint pass;
};

struct Exp17InnerParams {
    uint shift;
};

struct BucketDesc {
    uint offset;
    uint count;
    uint tile_count;
    uint tile_base;
};

// ═══════════════════════════════════════════════════════════════════
// Placeholder kernel — trivial copy so the file compiles
// ═══════════════════════════════════════════════════════════════════
kernel void exp17_placeholder(
    device const uint* src [[buffer(0)]],
    device uint*       dst [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    dst[tid] = src[tid];
}
