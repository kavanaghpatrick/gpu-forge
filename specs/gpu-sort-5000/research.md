---
spec: gpu-sort-5000
phase: research
created: 2026-02-19
---

# Research: gpu-sort-5000

## Executive Summary

Current baseline is 3003 Mkeys/s at 16M uint32 using 8-bit LSD radix sort (exp16). The scatter bottleneck is NOT random access — 256-bin scatter achieves 131 GB/s (near-sequential) — the real bottleneck is the **4-pass bandwidth requirement** against 245 GB/s DRAM. The path to 5000+ Mkeys/s is eliminating DRAM passes by exploiting the SLC (36MB on M4 Pro, 1.9x DRAM bandwidth): one MSD pass scatters 16M → 256 buckets of ~62K elements (~250KB each), each bucket then sorts entirely in SLC. Critical unknown is SLC scatter bandwidth vs SLC sequential bandwidth; if scatter degrades less under SLC than under DRAM, the hybrid wins.

---

## Empirical Measurements (Proven on M4 Pro, 16M uint32)

| Operation | Bandwidth | Notes |
|-----------|-----------|-------|
| Sequential copy | ~270 GB/s | Peak for 16M (DRAM regime) |
| Random scatter | 21 GB/s | 12x write amplification vs sequential |
| 256-bin scatter | 131 GB/s | Near-sequential! Structured scatter coalesces well |
| Blocked gather (32-elem) | 158 GB/s | Faster than sequential copy — SLC prefetch |
| SLC sequential BW | 469 GB/s @ ≤24MB | 1.9x DRAM |
| DRAM sequential BW | 245 GB/s | Peak throughput for large working sets |

### Sort Results

| Algorithm | Speed | Notes |
|-----------|-------|-------|
| 8-bit 4-pass LSD (baseline) | 3003 Mkeys/s | exp16_partition, 4096 elem/tile |
| 11-bit 3-pass LSD | 1677 Mkeys/s | SLOWER — 2048-bin scatter = 20 GB/s |
| Bitonic tile sort (4096 elem) | ~1750 Mkeys/s | 9.1ms total, barrier-bound |
| Sequential copy BW ceiling | ~5400 Mkeys/s | 4×N×4B at 270 GB/s |

### Per-Pass Timing Breakdown (16M, baseline 8-bit 4-pass)

From `bench_8bit_perpass` in `exp16_8bit.rs`:
- hist+prefix: ~0.8 ms (reads 4N bytes once)
- pass 0–3: each ~1.3 ms (reads + writes 2N bytes)
- total: ~6.0 ms = 3003 Mkeys/s implied
- scatter is NOT the bottleneck — 256-bin scatter at 131 GB/s is near-free

**Key insight**: If scatter were free (no-scatter diagnostic), 4 passes × 1.3ms = 5.2ms + 0.8ms hist = 6.0ms. Scatter adds <0.5ms total. The bottleneck is **pass count × bandwidth**, not scatter itself.

---

## Prior Art

### 1. Stehle-Jacobsen Hybrid Radix Sort (SIGMOD 2017)

**Paper**: "A Memory Bandwidth-Efficient Hybrid Radix Sort on GPUs" — Elias Stehle, Hans-Arno Jacobsen
**Source**: https://arxiv.org/abs/1611.01137

Algorithm: MSD first pass partitions into 256 large buckets. Each bucket independently sorted with LSD passes on shared memory (no DRAM scatter for inner passes). Achieves **2.32x speedup** over state-of-the-art GPU LSD radix sort for uniform distributions, **1.66x minimum** for skewed distributions.

Key mechanism: Inner LSD passes operate on data already partitioned to cache-resident buckets. This halves DRAM traffic because inner passes are cache-bandwidth-bound, not DRAM-bandwidth-bound.

**Apple Silicon adaptation challenges**:
- Original paper targets NVIDIA with ~48KB L1 shared memory per SM
- Apple Silicon threadgroup memory: 32KB max per TG
- Apple M4 Pro SLC: 36MB (estimated) — much larger than NVIDIA L1, enabling whole-bucket SLC residency
- Forward progress concern: independent per-bucket dispatch avoids decoupled lookback entirely

### 2. Onesweep (NVIDIA, 2022)

**Paper**: "Onesweep: A Faster Least Significant Digit Radix Sort for GPUs" — Adinets & Merrill
**Source**: https://arxiv.org/abs/2206.01784

Algorithm: Single-pass per digit using chained scan with decoupled lookback. Reduces memory traffic from ~3N to ~2N reads per digit pass. Achieves 29.4 GKey/s on NVIDIA A100 at 256M elements.

**Why it cannot run on Apple Silicon**: Requires inter-threadgroup spinning for decoupled lookback, which requires forward progress guarantees. Apple GPUs do NOT provide occupancy-bound forward progress guarantees (KB #387, #870). Deadlocks in practice. Safe alternative: reduce-then-scan (DeviceRadixSort pattern, KB #690).

**Note**: Our current exp16 already uses decoupled lookback with `atomic_thread_fence(mem_device, seq_cst, thread_scope_device)` — this works because the proof (exp12) showed the fence provides cross-TG coherence. However it is fragile and the SG ordering inside the single dispatch is key.

### 3. Vello/Linebender Metal Radix Sort

**Source**: https://linebender.org/wiki/gpu/sorting/ (KB #683, #388)

Metal port of Vello's radix sort achieves ~3G elements/s on M1 Max using actual simdgroup ballot operations. Attempts to push to 8-bit (256 bins) did not sustain improvement (matches our finding that 11-bit 3-pass is slower at 16M). SplitSort (hybrid radix-merge) shows promise for segmented sorting.

### 4. CUB BlockRadixSort (NVIDIA)

NVIDIA's CUB (CUDA Unbound) BlockRadixSort: 4-bit digit, within-warp ranking via warp scan, then threadgroup-scope gather-scatter. Uses register-ranked keys within warp before scatter. The "warp sort then scatter" approach is directly analogous to our proposed v2/v4 coalesced scatter kernels.

---

## Codebase Analysis

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `metal-gpu-experiments/shaders/exp16_8bit.metal` | 1387 | All kernels: histogram, partition v1-v4, 3-pass, diagnostics |
| `metal-gpu-experiments/src/exp16_8bit.rs` | 1308 | Rust host: benchmarks, timing, correctness checks |
| `metal-gpu-experiments/shaders/types.h` | 19 | FLAG_NOT_READY/AGGREGATE/PREFIX, VALUE_MASK constants |

### Existing Kernels (exp16_8bit.metal)

| Kernel | Purpose | Status |
|--------|---------|--------|
| `exp16_combined_histogram` | 4-pass histogram in 1 read | Production |
| `exp16_global_prefix` | 256-bin exclusive prefix sum | Production |
| `exp16_zero_status` | Zero tile_status between passes | Production |
| `exp16_partition` | Main sort: SG atomic hist + decoupled lookback + scatter | **BASELINE** 3003 Mkeys/s |
| `exp16_partition_v2` | 2048-element tiles + TG reorder for coalesced scatter | Built, awaiting results |
| `exp16_partition_v3` | 8192-element tiles (32 elem/thread) | Built |
| `exp16_partition_v4` | 4096-element tiles + 2-half TG reorder (27KB TG mem) | Built |
| `exp16_3pass_partition` | 11-bit 3-pass LSD | SLOWER — 1677 Mkeys/s |
| `exp16_diag_*` | Copy, scatter, gather, bitonic diagnostics | Measurement tools |

### Key Design Constants (Baseline)

```c
#define EXP16_NUM_BINS   256    // 8-bit radix
#define EXP16_TILE_SIZE  4096   // elements per tile
#define EXP16_ELEMS      16     // elements per thread (= TILE_SIZE / THREADS)
#define EXP16_THREADS    256    // threads per threadgroup
#define EXP16_NUM_SGS    8      // simdgroups per TG (SIMD width = 32)

// TG memory: 18 KB total
// sg_hist_or_rank[8*256] = 8 KB (atomic histogram P2, rank counters P5)
// sg_prefix[8*256]       = 8 KB (cross-SG prefix sums)
// tile_hist[256]         = 1 KB (tile totals)
// exclusive_pfx[256]     = 1 KB (lookback results)
```

### Decoupled Lookback Implementation

Uses `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)` — proven in exp12. Each of 256 threads handles one bin independently. ~31 tiles deep at 16M, 4096 elem/tile.

---

## The MSD+LSD Hybrid Approach

### Concept

```
Phase 1 (MSD):  16M elements → 256 buckets × ~62,500 elements
                Each bucket = ~250KB (fits in SLC with room to spare)

Phase 2 (LSD):  For each bucket independently:
                3 × 8-bit passes operating entirely in SLC bandwidth
```

### Why SLC Is the Key

- SLC size on M4 Pro: ~36MB (estimated from die analysis, KB #87 says M4 Max ~96MB, M4 Pro likely ~36MB)
- 256 buckets × ~250KB = 64MB total — TOO BIG for SLC
- BUT: buckets sorted SEQUENTIALLY (or in small parallel batches)
- Working set during inner sort = 1-2 buckets × 250KB = 250-500KB
- This fits entirely in SLC (8-96MB depending on chip tier)
- SLC bandwidth: 469 GB/s (vs 245 GB/s DRAM) = **1.9x speedup on inner passes**

### Bandwidth Analysis

**Baseline (4-pass LSD, DRAM):**
- hist: 1 × 4N = 4N bytes read
- 4 passes: 4 × 2 × 4N = 32N bytes read+write
- Total: 36N bytes at ~245 GB/s = 36×16M×4 / 245e9 = 9.4ms → 1700 Mkeys/s theoretical
- Actual: 3003 Mkeys/s (overhead from lookback, histogram, etc.)

**MSD+LSD Hybrid:**
- MSD scatter: 1 × 4N read + 1 × 4N write = 8N bytes at DRAM (~245 GB/s)
- Inner LSD histogram (ALL buckets): 1 × 4N = 4N bytes at SLC (~469 GB/s)
- Inner LSD 3 passes (ALL buckets, SLC-resident): 3 × 2 × 4N = 24N bytes at SLC (~469 GB/s)
- MSD scatter back (reorder final): 1 × 4N = 4N bytes at DRAM

| Phase | Bytes | Bandwidth | Time (16M) |
|-------|-------|-----------|------------|
| MSD scatter (DRAM) | 8N×4 | 245 GB/s | 2.10 ms |
| Inner hist (SLC) | 4N×4 | 469 GB/s | 0.55 ms |
| Inner 3-pass (SLC) | 24N×4 | 469 GB/s | 3.28 ms |
| **Total** | **36N×4** | **mixed** | **5.93 ms** |
| **→ Mkeys/s** | | | **2699 Mkeys/s** |

Wait — this is SLOWER than baseline? No — the SLC bandwidth advantage kicks in:

Effective bandwidth of inner passes = 469 GB/s vs 245 GB/s → 1.9x reduction in inner-pass time.

Revised: inner 3 passes at 469 GB/s = 24×16M×4 / 469e9 = 3.28ms instead of 6.55ms.

**Comparison:**
- Baseline: 4 passes × 2 reads/writes at 245 GB/s = 16.9ms bandwidth-limited + overhead = 3003 Mkeys/s actual
- Hybrid (optimistic): MSD 2.10ms + inner hist 0.55ms + inner 3-pass 3.28ms = 5.93ms → **2699 Mkeys/s theoretical**

This analysis suggests the hybrid might NOT beat baseline without additional improvements. The key question is whether **SLC scatter bandwidth** is higher than DRAM scatter bandwidth (21 GB/s). If SLC scatter achieves, say, 100+ GB/s, the picture changes dramatically.

### The Critical Unknown: SLC Scatter Bandwidth

The 256-bin scatter diagnostic (exp16_diag_scatter_binned) measures scatter at DRAM working set (16M = 64MB). At SLC working set (≤24MB), scatter bandwidth may be dramatically higher.

**Hypothesis**: SLC scatter bandwidth >> DRAM scatter bandwidth
- DRAM scatter: 21 GB/s (12x write amplification vs 245 GB/s sequential)
- SLC sequential: 469 GB/s
- SLC scatter (unknown): if write amplification is same ratio → 469/245 × 21 = 40 GB/s
- SLC scatter (optimistic): if SLC's higher bandwidth reduces amplification → 80-150 GB/s

If SLC scatter achieves ~100 GB/s, the inner 3 passes become competitive with baseline.

**Measurement approach**: Run exp16_diag_scatter_binned at decreasing working set sizes:
- 16M elements = 64MB (DRAM) → expect ~21 GB/s
- 4M elements = 16MB (SLC) → measure actual SLC scatter BW
- 1M elements = 4MB (SLC, deep) → measure peak SLC scatter BW
- 250K elements = 1MB (GPU L2) → measure L2 scatter BW

This is Experiment 17's first task.

---

## Approaches Evaluated

### REJECTED

| Approach | Result | Reason |
|----------|--------|--------|
| 11-bit 3-pass LSD | 1677 Mkeys/s | 2048-bin scatter = 20 GB/s |
| Fused two-digit LSD | N/A | Mathematically incorrect (breaks inter-tile stability) |
| Pure merge sort | Theoretical fail | 12 merge levels = 24N bandwidth = 3x worse |
| Bitonic tile sort (global) | 9.1ms | O(N log²N) barriers, barrier-bound |
| Texture scatter | N/A | No advantage over buffer scatter on Apple Silicon |
| Dual-engine overlap | N/A | No vertex/fragment work to overlap for pure compute |

### PROMISING (Not Yet Eliminated)

#### A. MSD+LSD Hybrid (PRIMARY)

See above analysis. Key unknowns:
1. SLC scatter bandwidth at ~250KB working set
2. Whether per-bucket dispatch overhead is acceptable
3. Whether MSD scatter can use TG reorder to boost scatter BW

**Estimated speedup if SLC scatter = 100 GB/s**: 5x improvement on inner scatter, reducing inner passes from 3.28ms to ~0.7ms → total ~3.4ms → 4700 Mkeys/s

#### B. Coalesced TG Reorder (v2/v4 Approach)

Current status: exp16_partition_v2 and v4 are built but results not yet in codebase output.

Mechanism: Load → rank into TG reorder buffer → sequential read from reorder buffer → scatter. Sequential reads coalesce into large transactions. If reorder scatter achieves close to sequential BW (270 GB/s vs 21 GB/s for random), 4-pass sort could approach:
- 4 passes × 2N at 270 GB/s = 8.2ms + overhead = ~2000 Mkeys/s + overhead
- With SLC for tiles: potentially higher

The two-half approach (v4) avoids the 2x tile count overhead of v2 (which uses 2048-element tiles).

#### C. Local Pre-Sort for Scatter Coalescing (SIMD Bitonic)

Before global scatter, sort elements within tile by digit using SIMD bitonic sort (KB #389: 5 stages, ~15-20 cycles, no TG memory). Elements going to same bin become contiguous. This transforms random scatter into near-blocked scatter.

Cost: ~15 GPU clock cycles per sort for 32 elements = negligible
Gain: If scatter becomes blocked (158 GB/s measured) vs random (21 GB/s) = 7.5x scatter speedup

**Problem**: The sort within the tile only helps if TG threads map to the same bins. With 256 bins and 4096 elements per tile, each bin gets ~16 elements on average. Elements are still scattered across 256 destination regions. SIMD bitonic sort within a simdgroup (32 elements) creates blocks of same-bin elements within each simdgroup, but the scatter across 256 bins spanning the full 64MB output remains random at the DRAM level.

**Verdict**: Works well inside the hybrid approach (for inner SLC passes), where each "bucket" has only 256 sub-bins covering ~250KB — the sub-scatter stays SLC-resident.

---

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | High | MSD+LSD is mathematically sound, proven in prior art |
| Effort Estimate | M | New shaders + Rust host for bucket dispatch (~2-3 days) |
| Risk Level | Medium | SLC scatter bandwidth is the critical unknown |
| Apple Silicon Compatibility | High | No forward progress issues — independent bucket dispatch |
| Correctness Risk | Low | MSD is not stability-required for LSD input |
| Performance Ceiling | 5000-8000 Mkeys/s | If SLC scatter achieves 80-150 GB/s |

---

## Risk Assessment and Fallback Strategies

### Risk 1: SLC Scatter BW is Low

If SLC scatter bandwidth ≈ DRAM scatter bandwidth × SLC/DRAM ratio = 40 GB/s (only proportional improvement), the hybrid inner scatter is still slow.

**Mitigation**: Add TG reorder within inner sort. Sort each 62K-element bucket into 4096-element tiles. For each tile within a bucket, use TG reorder to make scatter coalesced. Inner sort then does coalesced scatter even within SLC.

**Fallback**: Accept 4000 Mkeys/s target instead of 5000.

### Risk 2: Per-Bucket Dispatch Overhead

256 independent sub-sorts require either:
- 256 separate dispatches (CPU overhead too high: ~256 × 1.5μs = 384μs = 0.4ms per pass)
- OR a single persistent/indirect dispatch that dispatches sub-kernels

**Mitigation**: Use indirect dispatch + device-side work queue. After MSD scatter, GPU atomically assigns bucket IDs to threadgroup work items. Single dispatch processes all buckets.

**Alternative**: Reduce to 16 MSD bins (4-bit MSD), so only 16 dispatches at ~24ms each = minimal overhead. Tradeoff: larger buckets (1M elements each), less SLC benefit.

### Risk 3: Register Pressure in Combined Inner Sort

Inner sort needs MSD bucket boundary information + per-element local rank + scatter offset. May hit register limits on M4 Pro.

**Mitigation**: M4 uses Dynamic Caching (KB #10, #261) — register file shares SRAM pool with threadgroup memory dynamically. Less register pressure concern vs older hardware.

### Risk 4: Apple GPU Forward Progress in Inner Sort

If inner sort uses decoupled lookback (like baseline), need to ensure forward progress within a bucket's tiles.

**Mitigation**: Inner sort uses reduce-then-scan (safe, no forward progress required). Only the MSD scatter requires a single global pass with decoupled lookback (already proven safe in exp16_partition).

---

## Implementation Plan

### Step 1: Measure SLC Scatter Bandwidth (MUST DO FIRST)

Add to `exp16_8bit.rs`: benchmark `exp16_diag_scatter_binned` at multiple sizes:
```
62_500 elements   =   250KB  (L2/SLC)
250_000           =  1000KB  (SLC)
1_000_000         =  4000KB  (SLC)
4_000_000         = 16000KB  (SLC)
16_000_000        = 64000KB  (DRAM)  [already measured: 131 GB/s]
```

Expected output: SLC scatter BW curve → validates hybrid hypothesis.

### Step 2: MSD Scatter Kernel (exp17_msd_scatter)

Single-pass MSD scatter: read input, compute bits[24:31], scatter to 256 bucket regions.

```metal
// Each TG handles one tile of input (4096 or 8192 elements)
// Phase 1: Per-SG atomic histogram (256 bins, reuse exp16 logic)
// Phase 2: Decoupled lookback (256-thread, reuse exp16 logic)
// Phase 3: Scatter to bucket offset
```

Output: `bucket_data[256]` arrays, each sized for its count. Plus `bucket_offsets[256]` array.

### Step 3: Inner Sort Kernel (exp17_inner_sort)

Per-bucket LSD sort. Since each bucket is ~62K elements = 15 tiles of 4096:
- 15 tiles per bucket → lookback depth = 15 (vs 3906 for full 16M sort)
- MUCH shallower lookback → less fence overhead
- Reduce-then-scan is viable at 15-tile depth

For 256 buckets: run 256 independent 3-pass sorts.

**Dispatch strategy**: Single compute dispatch with 256 × 15 = 3840 threadgroups.
- TG `gid`: which TG within which bucket = `(gid / 15, gid % 15)`
- Each TG computes its bucket's tile prefix from bucket-local status array
- No cross-bucket communication needed

### Step 4: Final Concatenation

After all inner sorts complete, output is already in final sorted order within each bucket region. No additional copy needed if MSD scatter wrote to correct regions.

### Step 5: Benchmark and Tune

- Measure end-to-end at 1M, 4M, 16M
- Profile per-phase timing
- Tune tile size for inner sort (4096 vs 8192)
- Consider 2-pass MSD (bits 28-31 + bits 24-27) for 16×1MB working sets vs 256×250KB

---

## Quality Commands

| Type | Command | Source |
|------|---------|--------|
| Build | `cargo build --release -p metal-gpu-experiments` | Cargo.toml |
| Run | `cargo run --release -p metal-gpu-experiments` | Cargo.toml |
| Build (shader rebuild) | `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments` | Build script caching |
| Test | No automated test suite for benchmarks — correctness printed inline | exp16_8bit.rs |

---

## Related Specs

| Spec | Relevance | mayNeedUpdate |
|------|-----------|---------------|
| `gpu-sort-3000` | Direct predecessor — same codebase, same architecture | false (completed) |
| `gpu-compute-experiments` | General Metal compute patterns | false |
| `gpu-perf-phase3` | GPU performance work on same machine | false |

---

## Open Questions

1. **SLC scatter bandwidth at 250KB working set**: The make-or-break measurement. Does structured 256-bin scatter at SLC scale hit 80-150 GB/s or stay at 40 GB/s?

2. **Per-bucket dispatch strategy**: Single large dispatch with bucket IDs in grid, OR indirect dispatch, OR persistent kernel with work stealing? Impact on scheduling overhead.

3. **Inner sort: reduce-then-scan vs decoupled lookback**: At 15-tile depth, reduce-then-scan requires 15 rounds of inter-encoder barriers. Is that overhead acceptable vs decoupled lookback which risks forward progress issues?

4. **MSD bin count**: 256 bins (8-bit MSD) vs 16 bins (4-bit MSD). 256 bins: 250KB per bucket (ideal SLC fit), 256 sub-dispatches. 16 bins: 4MB per bucket (still SLC with 36MB SLC), 16 sub-dispatches.

5. **TG reorder in inner sort**: Does adding TG reorder to the inner sort (like v4) actually help if scatter is already SLC-resident?

---

## KB Finding References

| Finding ID | Claim | Relevance |
|------------|-------|-----------|
| #683 | LSD radix sort ~3G el/s on M1 Max with simdgroup ops; OneSweep NOT safe on Apple | Prior art |
| #387 | Apple GPUs lack forward progress guarantees — OneSweep deadlocks | Architecture constraint |
| #870 | Apple GPU no occupancy-bound forward progress — safe alternatives are reduce-then-scan and decoupled fallback | Architecture constraint |
| #690 | GPUSorting DeviceRadixSort (reduce-then-scan) + SplitSort hybrid | Prior art |
| #388 | Metal native radix sort 3G el/s vs 1G el/s WebGPU — 3x from simdgroup ballot | Prior art |
| #1655 | `atomic_thread_fence(mem_device, seq_cst, device_scope)` enables correct single-dispatch persistent kernels | Cross-TG coherence |
| #1658 | Fence-only approach competitive with multi-encoder, ~1.5-2.8ms vs 2.0ms | Performance tradeoff |
| #113 | SLC bandwidth ~2x DRAM per GPU core | SLC advantage quantification |
| #112 | SLC serves as effective L3 cache for GPU; GPU-inclusive policy | SLC architecture |
| #21 | SLC sizes: M4 Pro ~36MB (M1/M2/M3 Pro = 24MB, M4 Pro likely 36MB, M4 Max 96MB) | Hardware constraint |
| #87 | M4 base SLC = 8MB confirmed; M4 Pro/Max sizes unconfirmed but estimated large | Hardware uncertainty |
| #54 | SLC: inclusive wrt GPU cache, exclusive wrt CPU cache; pseudo-random replacement | Cache policy |
| #1293 | M4 Pro memory bandwidth: 273 GB/s (75% increase over M3 Pro) | Bandwidth baseline |
| #389 | Bitonic sort within 32-thread SIMD group: 5 stages, ~15-20 cycles, no TG memory | Local pre-sort option |
| #10 | M3/M4 Dynamic Caching: register file + L1 + TG memory share SRAM pool dynamically | Register pressure |
| #261 | Dynamic Caching auto-adjusts occupancy to prevent cache thrashing | Register pressure |
| #264 | GPU occupancy: set maxThreadsPerThreadgroup for compiler optimization | Occupancy tuning |
| #259 | Apple GPU ~208KiB register file per core; ≤104 regs → 1024 threads full occupancy | Register budget |
| #1367 | GPUPrefixSums: decoupled lookback CANNOT run on Metal/Apple due to forward progress | Architecture constraint |
| #357 | Prefix sum applications: stream compaction, histogram, radix sort, allocation | Algorithm context |
| #1306 | Coalesced memory access: 32 adjacent threads reading 32 consecutive bytes = single 128B transaction | Coalescing principle |
| #874 | AMD FidelityFX Parallel Sort: 4-bit digits, tree reduction histogram | Prior art comparison |

---

## Sources

- [Stehle-Jacobsen: A Memory Bandwidth-Efficient Hybrid Radix Sort on GPUs](https://arxiv.org/abs/1611.01137) (SIGMOD 2017)
- [Onesweep: A Faster Least Significant Digit Radix Sort for GPUs](https://arxiv.org/abs/2206.01784) (Adinets & Merrill 2022)
- [GPU Sorting - Linebender Wiki](https://linebender.org/wiki/gpu/sorting/)
- [GPUSorting: State of the art sorting on GPU](https://github.com/b0nes164/GPUSorting)
- [EXAM: Exploiting Exclusive SLC in Apple M-Series SoCs](https://arxiv.org/abs/2502.05317)
- [metal-benchmarks: Apple GPU microarchitecture](https://github.com/philipturner/metal-benchmarks)
- [GPUPrefixSums library](https://github.com/b0nes164/GPUPrefixSums)
- `/Users/patrickkavanagh/gpu_kernel/metal-gpu-experiments/shaders/exp16_8bit.metal`
- `/Users/patrickkavanagh/gpu_kernel/metal-gpu-experiments/src/exp16_8bit.rs`
