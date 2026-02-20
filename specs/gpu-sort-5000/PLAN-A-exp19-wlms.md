# Plan A: exp19 — 4-SG WLMS 3-Pass Radix Sort

**Target**: 5000+ Mkeys/s at 16M uint32 on Apple M4 Pro
**Approach**: 3-pass (11+11+10 bit), 128 threads (4 SGs), WLMS ballot ranking, ushort per-SG running counts

## Why This Approach

The fundamental bottleneck in exp18 is the sequential SG scatter (Phase 5, lines 340-371): processing one SG at a time means 87.5% of threads are idle, dropping throughput from the 5487 Mkeys/s scatter ceiling to 1589 Mkeys/s.

With 8 SGs × 2048 bins, per-SG rank arrays need 64KB — exceeding the 32KB TG memory limit. The solution: **use only 4 SGs (128 threads)** so per-SG arrays fit:
- ushort sg_running[4 × 2048] = 16KB (max per-SG count = 32 × 32 = 1024, fits ushort)
- All 4 SGs active simultaneously = 100% utilization of those 4 SGs

## Research Summary (7 agents, 2 rounds of Gemini+Codex)

### Key findings:
1. **WLMS ballot ranking** (unanimous): O(RADIX_BITS) per element via `simd_ballot`, zero TG memory for within-SG rank
2. **8-bit counter overflow** (Codex, critical bug): `uchar sg_counts[8][2048]` overflows because each SG processes up to 512 elements — max count exceeds 255. Need ushort (16-bit) minimum.
3. **Per-iteration prefix needed** (Codex): WLMS gives within-ITERATION rank only. Across 16 elements/lane, need running prefix per digit per SG → this IS the per-SG rank array.
4. **Transposed storage** (Gemini): Layout `[2048][4]` instead of `[4][2048]` enables reading all 4 SG counts in a single uint32 load (4 × 1 byte = 4 bytes contiguous per digit). Not applicable with ushort (4 × 2B = 8B, use uint2 load).
5. **Apple scatter is efficient** (KB #3568): TG reorder buffer provides ZERO benefit on Apple Silicon. Random scatter at 256 bins already achieves ~130 GB/s. No need for coalescing tricks.
6. **4-pass 256-bin may cap at ~4000** (state-of-art agent): 3-pass ceiling is 5487 vs estimated ~4100 for 4-pass. Must use 3 passes to hit 5000+.

### What 128 threads means:
- 4 SGs × 32 lanes = 128 threads per TG
- Tile size = 4096 elements (32 elements per lane, same as exp18)
- 3907 tiles at 16M elements — same tile count as exp18
- tile_status = 3907 × 2048 × 4B = 32MB (borderline SLC, acceptable)
- Memory subsystem saturation: 3907 TGs × 128 threads = 500K total threads — enough to saturate bandwidth

## Architecture

### Constants
```metal
#define EXP19_TILE_SIZE  4096u
#define EXP19_ELEMS      32u      // 32 elements per lane (128 threads × 32 = 4096)
#define EXP19_THREADS    128u
#define EXP19_NUM_SGS    4u
#define EXP19_MAX_BINS   2048u
#define EXP19_RADIX_LOG  11u      // for 2048 bins (10 for 1024)
```

### TG Memory Layout (~28KB)
```
ushort  sg_running[4 * 2048]  = 16,384 bytes  (per-SG running count for ranking)
uint    tile_offset[2048]     =  8,192 bytes  (precomputed: global_prefix + exclusive_pfx)
ushort  tile_hist[2048]       =  4,096 bytes  (tile histogram, max 4096 fits ushort)
─────────────────────────────────────────────
TOTAL                          28,672 bytes (28 KB)
```

### Kernel Structure (same 4 kernels as exp18)
1. `exp19_combined_histogram` — identical to exp18, 128 threads, reads all data once
2. `exp19_global_prefix` — identical to exp18, SG-0 serial prefix sum
3. `exp19_zero_status` — identical to exp18
4. `exp19_partition` — THE CHANGED KERNEL (see below)

### exp19_partition — Phase by Phase

**Phase 1: Load (SG-contiguous, 32 elements/lane)**
```metal
// 4 SGs, each gets contiguous 1024-element block
// SG 0: [base+0..1023], SG 1: [base+1024..2047], etc.
for (uint e = 0; e < 32; e++) {
    uint idx = base + simd_id * (32u * 32u) + e * 32u + simd_lane;
    valid[e] = idx < n;
    keys[e] = valid[e] ? src[idx] : 0xFFFFFFFF;
    digits[e] = valid[e] ? ((keys[e] >> shift) & mask) : mask;
}
```

**Phase 2: Histogram (TG-wide atomics, same as exp18)**
```metal
threadgroup atomic_uint tg_hist_atomic[2048]; // 8KB, REUSED for Phase 5
// All 128 threads atomicAdd their 32 elements → tg_hist
// Then copy to tile_hist (ushort, fits because max=4096)
```

**Phase 3: Publish AGGREGATE + Phase 4: Lookback**
- Identical logic to exp18 but with 128 threads handling 16 bins per thread (bpt=16)
- Each thread handles 2048/128 = 16 bins in the lookback
- After lookback: compute `tile_offset[bin] = global_prefix[bin] + exclusive_pfx[bin]`
- Store tile_offset in the 8KB uint array

**Phase 5: WLMS Ranking + Parallel Scatter (THE KEY CHANGE)**
```metal
// ── Phase 5a: Zero per-SG running counts ──────────────────────
for (uint i = lid; i < 4 * num_bins; i += 128) {
    sg_running[i] = 0;  // ushort
}
barrier();

// ── Phase 5b: WLMS Ranking (iterate through 32 elements) ─────
uint ranks[32];  // within-tile rank for each element

for (uint e = 0; e < 32; e++) {
    if (!valid[e]) continue;
    uint digit = digits[e];

    // WLMS: build peer_mask via bit-decomposition (11 ballots for 11-bit)
    uint peer_mask = 0xFFFFFFFF;
    for (uint k = 0; k < radix_bits; k++) {
        bool my_bit = (digit >> k) & 1;
        uint ballot = (uint)simd_ballot(my_bit);
        peer_mask &= my_bit ? ballot : ~ballot;
    }

    // Within-SG rank for THIS iteration
    uint lane_lt = (1u << simd_lane) - 1u;
    uint within_iter_rank = popcount(peer_mask & lane_lt);

    // Running prefix: read current count for my digit in my SG
    uint running = (uint)sg_running[simd_id * num_bins + digit];
    ranks[e] = running + within_iter_rank;

    // Leader updates running count (non-atomic: exactly 1 leader per (SG,digit))
    bool is_leader = (popcount(peer_mask & lane_lt) == 0); // lowest-set bit
    if (is_leader) {
        sg_running[simd_id * num_bins + digit] = (ushort)(running + popcount(peer_mask));
    }
    // Barrier NOT needed between iterations within same SG
    // (leader writes are visible to same SG's lanes on next iteration)
    // But simd_shuffle is needed to ensure leader's write is visible
}
barrier();

// ── Phase 5c: Cross-SG prefix (all 4 SGs in parallel) ────────
// For each element, compute: cross_rank = sum of sg_running[s][digit] for s < simd_id
// Use transposed layout for vectorized reads

for (uint e = 0; e < 32; e++) {
    if (!valid[e]) continue;
    uint digit = digits[e];

    // Sum counts from preceding SGs
    uint cross_rank = 0;
    for (uint s = 0; s < simd_id; s++) {
        cross_rank += (uint)sg_running[s * num_bins + digit];
    }

    // Final global position
    uint gpos = tile_offset[digit] + cross_rank + ranks[e];
    dst[gpos] = keys[e];
}
```

### Critical Correctness Notes

1. **ushort overflow**: Max per-SG count = 32 lanes × 32 elements = 1024. ushort max = 65535. SAFE.
2. **Stability**: SG-contiguous load ensures SG 0's elements precede SG 1's. Within-SG, WLMS + iteration-order ranking preserves lane order. Cross-SG prefix is deterministic (SG 0 < SG 1 < SG 2 < SG 3).
3. **Leader writes**: After WLMS, exactly one lane per (SG, digit) has `within_iter_rank == 0`. That lane updates sg_running. No race conditions because different digits write different indices, and same-digit writes are from a single leader.
4. **Barrier between iterations**: Within a single SG, the leader's write to `sg_running` needs to be visible to all lanes BEFORE the next iteration reads it. Since all lanes in an SG execute in lockstep, a `threadgroup_barrier` after each iteration would work but is expensive (32 barriers). Alternative: use `simd_broadcast` from the leader to share the updated running count without writing to TG memory at all.

### Optimization: Register-Only Running Count

Instead of writing sg_running to TG memory on each iteration, keep it in registers via simd_broadcast:

```metal
// Each lane maintains running_count for its own digit
// After WLMS, leader broadcasts the new running value to all peers
uint my_running = 0;

for (uint e = 0; e < 32; e++) {
    if (!valid[e]) continue;
    uint digit = digits[e];

    // WLMS ballot...
    uint within_iter_rank = popcount(peer_mask & lane_lt);
    uint peer_count = popcount(peer_mask);

    // Get running count from leader
    uint leader_lane = ctz(peer_mask); // lowest set bit
    uint running_from_leader = simd_shuffle(my_running, leader_lane);
    // Problem: my_running is MY running, but leader has THE DIGIT's running
    // This doesn't work because each lane tracks its OWN digits, not all digits
}
```

This optimization is complex because each lane processes 32 different digits across iterations. The leader for digit X in iteration 5 may not be the leader for digit Y in iteration 6. **Stick with the TG memory approach (sg_running) — it's simpler and correct.**

### Optimization: Barrier-Free Iteration via SG Memory Coherence

Within a single SG, all 32 lanes execute in lockstep. A write by lane L to `sg_running[simd_id * bins + digit]` is visible to all other lanes in the same SG on the NEXT instruction (no barrier needed). This is because SIMD lanes share the same memory interface.

**This means NO barriers needed between the 32 iterations of the ranking loop.** Only one barrier after all 32 iterations before the cross-SG prefix read.

## Dispatch (Rust Side)

```
exp19_combined_histogram:  1 encoder, ceil(16M/4096) = 3907 TGs × 128 threads
exp19_global_prefix:       1 encoder, 1 TG × 32 threads (SG 0 only)

Per pass (3 passes):
  exp19_zero_status:       1 encoder, ceil(3907×2048/128) TGs
  exp19_partition:         1 encoder, 3907 TGs × 128 threads

Total: 1 + 1 + 3×2 = 8 encoders (vs 10 for exp18)
```

### Buffer Sizes
- `buf_a`, `buf_b`: 16M × 4B = 64 MB each (ping-pong)
- `global_hist`: 5120 × 4B = 20 KB
- `tile_status`: 3907 × 2048 × 4B = 32 MB (per pass, reused)

## Expected Performance

- Scatter ceiling (3-pass): 5487 Mkeys/s
- Ranking overhead: WLMS is ~11 ballots per element × 32 elements = 352 ballots/lane. At ~2 cycles each on M4 = 704 cycles. At 1.4 GHz = 0.5 µs per thread. With 3907 TGs: amortized over parallelism.
- Lookback: 16 bins per thread (vs 8 in exp18). Higher bpt but same mechanism.
- **Estimated: 3500-4500 Mkeys/s** (depends on whether 128 threads saturate memory BW)
- **Risk**: If 128 threads don't saturate memory subsystem, could be slower than exp16 (3003)

## Build & Run

```bash
cd metal-gpu-experiments
rm -rf target/release/build/metal-gpu-experiments-* 2>/dev/null
cargo build --release -p metal-gpu-experiments
cargo run --release -p metal-gpu-experiments
```

## Files to Create/Modify

| File | Action |
|------|--------|
| `shaders/exp19_wlms.metal` | CREATE — all 4 kernels |
| `src/exp19_wlms.rs` | CREATE — Rust dispatch + benchmark |
| `src/main.rs` | MODIFY — add `mod exp19_wlms` |
| `build.rs` | VERIFY — auto-discovers .metal files |

## Verification

1. Correctness: `std_sorted == gpu_sorted` at 62.5K, 250K, 1M, 4M, 16M
2. Performance: Mkeys/s at 16M, compare vs exp16 (3003) and exp18 (1589)
3. Stability: Sort `[(i << 11) | digit for i in 0..N]` and verify relative order preserved
