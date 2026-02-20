# Plan B: Improved exp17 — MSD+LSD Hybrid with WLMS Inner Passes

**Target**: 5000+ Mkeys/s at 16M uint32 on Apple M4 Pro
**Approach**: Improve exp17 (currently 3461 Mkeys/s) by applying WLMS ballot ranking + larger tiles to inner LSD passes

## Why This Approach

exp17 is already our best result at 3461 Mkeys/s. It uses a fundamentally different architecture:
1. **MSD scatter** (bits 24:31): 1 pass creates 256 buckets of ~62K elements each (~250KB)
2. **Inner LSD** (3 passes × 8-bit): Sort each bucket independently

Key advantage: inner passes operate on **SLC-resident data** (250KB << 24MB SLC). The SLC delivers 469 GB/s vs 245 GB/s DRAM — a 1.91x bandwidth advantage.

The gap from 3461 to 5000+ is in the **inner pass overhead**. The inner passes use:
- Serial prefix sum over tile_hists (not decoupled lookback — simpler but serialized)
- Per-SG atomic histogram (same as exp16)
- Per-SG atomic ranking (same as exp16)

These can be optimized without changing the overall MSD+LSD structure.

## Research Summary (relevant to this approach)

1. **WLMS for 8-bit is optimal** (all agents): 8 `simd_ballot` calls for 256 bins, zero TG memory for within-SG rank. Replaces O(31) shared_digits scan.
2. **Inner passes should use decoupled lookback** (Codex): Serial prefix scan over tile_hists is O(tiles_per_bucket). With up to 17 tiles per bucket, this is fine, but decoupled lookback would enable larger buckets.
3. **256-bin ranking FITS in TG memory** (all agents): 8 SGs × 256 bins × 4B = 8KB. No TG memory pressure at 256 bins. All 8 SGs can rank in parallel.
4. **Apple scatter at 256 bins is efficient** (KB #3568): ~130 GB/s at DRAM scale, even better at SLC scale.
5. **Larger tiles reduce dispatch overhead** (Codex, Gemini): exp17 has `MAX_TPB=17` tiles per bucket. With larger tiles (8192), only 9 tiles per bucket → less overhead.

## Current exp17 Architecture (3461 Mkeys/s)

### Kernels (14 dispatches total in single command buffer)
1. `exp17_msd_histogram` — histogram for MSD byte (bits 24:31)
2. `exp17_compute_bucket_descs` — derive offset/count/tile_count per bucket
3. `exp17_msd_global_prefix` — exclusive prefix sum on MSD histogram
4. `exp17_msd_partition` — scatter elements into 256 buckets
5. **Per inner pass (3 passes, each has 2-3 dispatches):**
   a. `exp17_inner_zero` — zero tile_hists buffer
   b. `exp17_inner_histogram` — per-tile per-bucket histogram (4352 TGs)
   c. `exp17_inner_scan_scatter` — serial prefix + rank + scatter (4352 TGs)

### Inner pass structure (exp17_inner_scan_scatter)
- 4352 TGs dispatched (256 buckets × 17 tiles each)
- Arithmetic mapping: `bucket_id = gid / 17`, `tile_in_bucket = gid % 17`
- **Serial prefix sum**: Thread `lid` scans `tile_hists[bucket_id * 17 * 256 + t * 256 + lid]` for t=0..tile_in_bucket-1
- **Per-SG atomic ranking**: Same as exp16 (8KB sg_hist, all 8 SGs active)
- **TG memory**: 20KB (sg_hist_or_rank 8KB + sg_prefix 8KB + tile_hist 1KB + exclusive_pfx 1KB + global_digit_pfx 1KB + chunk_totals 32B)

## Optimization Plan

### Opt 1: WLMS Ballot Ranking in Inner Passes (Est. +10-20%)

Replace the per-SG atomic histogram + rank with WLMS ballot ranking in `exp17_inner_scan_scatter`:

**Current** (exp17_inner_scan_scatter Phase 2+5):
```metal
// Phase 2: Per-SG atomic histogram
atomic_fetch_add(&sg_hist_or_rank[simd_id * 256 + digit], 1u, ...);
// Phase 5: Per-SG atomic rank (reuse sg_hist_or_rank as rank counter)
rank = atomic_fetch_add(&sg_hist_or_rank[simd_id * 256 + digit], 1u, ...);
```

**Proposed**:
```metal
// WLMS ballot ranking (8 ballots for 8-bit digit)
uint peer_mask = 0xFFFFFFFF;
for (uint k = 0; k < 8; k++) {
    bool my_bit = (digit >> k) & 1;
    uint ballot = (uint)simd_ballot(my_bit);
    peer_mask &= my_bit ? ballot : ~ballot;
}
uint within_iter_rank = popcount(peer_mask & lane_lt);
uint peer_count = popcount(peer_mask);

// Leader updates running count per SG
// For 256 bins with 8 SGs, per-SG rank arrays fit: 8 × 256 × 4B = 8KB
// So we can use the SAME sg_hist_or_rank array but with non-atomic leader writes
if (within_iter_rank == 0) { // leader
    uint running = sg_hist_or_rank[simd_id * 256 + digit];
    ranks[e] = running + within_iter_rank; // always 0 for leader
    sg_hist_or_rank[simd_id * 256 + digit] = running + peer_count;
} else {
    // Non-leaders need the running value from leader
    // Use simd_shuffle to get it
    uint leader_lane = ctz(peer_mask);
    uint running = simd_shuffle(/* leader's running value */, leader_lane);
    ranks[e] = running + within_iter_rank;
}
```

**Simpler approach**: Keep the per-SG atomic_fetch_add for ranking (it already works and is fast at 256 bins), but add WLMS for the HISTOGRAM phase to eliminate the separate histogram pass entirely.

Actually, for 256 bins with 8 SGs, the current approach already has all 8 SGs active and ranking in parallel. WLMS would mainly help by:
- Eliminating atomic contention (leader-only writes)
- Enabling histogram to be computed simultaneously with ranking (fused)

### Opt 2: Fuse Histogram + Scatter Pass (Est. +15-25%)

Currently the inner sort does 2 dispatches per pass:
1. `exp17_inner_histogram` — builds tile_hists
2. `exp17_inner_scan_scatter` — reads tile_hists, computes prefix, ranks, scatters

**Proposed**: Fuse into a single `exp17_inner_fused` kernel:
- Each TG computes its own histogram
- Uses decoupled lookback (like exp18) instead of serial prefix scan over tile_hists
- Ranks and scatters in one pass

This eliminates:
- The inner_histogram kernel dispatch
- The inner_zero kernel dispatch (no tile_hists buffer needed)
- The serial prefix scan over tile_hists

Per inner pass: 1 dispatch instead of 2-3. Total: MSD(4) + 3×1 = 7 dispatches instead of 14.

**BUT**: Decoupled lookback needs tile_status buffer. At 256 bins × 17 tiles per bucket × 256 buckets = 1.1M entries × 4B = 4.4MB. Fits in SLC!

**Alternative** (simpler): Keep the serial prefix scan but fuse histogram+scatter into one kernel (compute histogram, barrier, scan tile_hists inline, then rank+scatter). This avoids the tile_status buffer.

### Opt 3: Larger Inner Tiles — 8192 Elements (Est. +5-10%)

Change inner tile size from 4096 to 8192:
- `EXP17_TILE_SIZE_LARGE = 8192`, `EXP17_ELEMS_LARGE = 32`
- `EXP17_MAX_TPB_V2 = 9` (already defined in exp17_hybrid.metal!)
- Half as many tiles per bucket → half the serial prefix work
- 256 × 9 = 2304 TGs (instead of 4352)

TG memory impact: 32 elements per thread. Register array: 32 × 4B = 128B keys + 128B digits + 32B valid = 288B per thread. With 256 threads = 72KB register usage. Might reduce occupancy.

### Opt 4: Reduce Inner Passes from 3 to 2 (Est. +33%)

If inner passes sort with 12+12 bits instead of 8+8+8:
- 2 inner passes × 4096 bins each
- BUT: 4096 bins per-SG ranking needs 4096 × 8 × 4B = 128KB TG memory — IMPOSSIBLE

Alternative: 10+7+7 (1024 + 128 + 128 bins):
- First inner: 1024 bins, per-SG = 8 × 1024 × 4B = 32KB — barely fits but no room for anything else
- Second/third inner: 128 bins, trivial

Or 8+8+8 (stay at 3 passes, already working).

**Verdict**: Reducing inner passes is not viable without hitting TG memory limits. Stick with 3 × 8-bit.

### Opt 5: Improve MSD Scatter (Est. +5%)

The MSD scatter pass (exp17_msd_partition, not shown) operates on the full 16M dataset at DRAM bandwidth. Apply the same optimizations as exp16:
- WLMS ballot for MSD histogram
- All 8 SGs scatter in parallel (already the case for 256 bins)

This is probably already close to optimal since 256-bin scatter on Apple Silicon is efficient.

### Opt 6: Better Bucket Load Balancing (Est. +5-10%)

Current issue: buckets are dispatched with `MAX_TPB` tiles per bucket regardless of actual bucket size. Empty tiles early-exit but waste TG slots.

**Proposed**: Use indirect dispatch or compute actual TG count. With `exp17_compute_bucket_descs` already computing tile_count per bucket, we could use a prefix sum to compute exact TG-to-bucket mapping.

## Implementation Order

1. **Opt 3** (larger tiles) — Simplest, already has defines. Benchmark improvement.
2. **Opt 2** (fuse histogram+scatter) — Moderate complexity, biggest single win.
3. **Opt 1** (WLMS inner ranking) — Apply WLMS to inner passes.
4. **Opt 5** (improve MSD scatter) — Minor gains.
5. **Opt 6** (load balancing) — If needed to close the gap.

## Estimated Combined Performance

| Optimization | Individual | Cumulative |
|---|---|---|
| Baseline (exp17) | 3461 Mkeys/s | 3461 |
| Opt 3: 8192 tiles | +5-10% | 3634-3807 |
| Opt 2: Fused inner passes | +15-25% | 4179-4759 |
| Opt 1: WLMS inner ranking | +10-20% | 4597-5711 |
| Opt 5: MSD scatter | +5% | 4827-5997 |

**Conservative estimate**: 4500-5000 Mkeys/s
**Optimistic estimate**: 5000-6000 Mkeys/s

## Files to Create/Modify

| File | Action |
|------|--------|
| `shaders/exp17_hybrid.metal` | MODIFY — add WLMS ranking to inner kernels, add fused kernel |
| `src/exp17_hybrid.rs` | MODIFY — add benchmarks for optimized variants |
| `src/main.rs` | VERIFY — already has `mod exp17_hybrid` |

## Build & Run

```bash
cd metal-gpu-experiments
rm -rf target/release/build/metal-gpu-experiments-* 2>/dev/null
cargo build --release -p metal-gpu-experiments
cargo run --release -p metal-gpu-experiments
```

## Verification

1. Correctness: `std_sorted == gpu_sorted` at 62.5K, 250K, 1M, 4M, 16M
2. Performance: Mkeys/s at 16M, compare vs current exp17 (3461) and exp16 (3003)
3. Per-phase timing breakdown to identify remaining bottlenecks
4. SLC residency verification: inner pass working set should be < 24MB

## Key Advantage Over Plan A

- **Lower risk**: Already at 3461 Mkeys/s, each optimization is incremental
- **Inner passes at SLC speed**: 469 GB/s vs 245 GB/s DRAM
- **256 bins fits perfectly**: No TG memory pressure, all 8 SGs active
- **Proven architecture**: exp17 already works correctly

## Key Risk

- **MSD scatter is inherently random**: The first pass scatters to 256 buckets across the full 64MB dataset. This is at DRAM speed, not SLC.
- **Total data movement**: MSD(2N) + 3×inner(2N per bucket) = 2N + 6N = 8N. Same as 4-pass LSD. The benefit comes from inner passes being at SLC speed, NOT from reduced data movement.
- **Bucket imbalance**: Worst-case bucket sizes can be 256× larger than average, destroying SLC residency for those buckets.
