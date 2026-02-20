# GPU Sort 5000+ Research Synthesis

## Performance Landscape (Measured)

| Experiment | Mkeys/s | Architecture | Bottleneck |
|---|---|---|---|
| exp16 | 3003 | 4-pass 8-bit LSD, 256 bins, 256 threads | Pass count (4 reads+writes) |
| exp17 | 3461 | MSD+3×LSD hybrid, 256 bins inner | Inner pass overhead |
| exp18 | 1589 | 3-pass 11-bit LSD, 2048 bins, 256 threads | Sequential SG scatter (87.5% idle) |
| 3-pass scatter ceiling | 5487 | Pre-computed offsets, pure write | Theoretical max |
| 4-pass estimated ceiling | ~4100 | Extrapolated from scatter ceiling | Hard wall for 4-pass |

## Agent Findings (7 agents, 2 rounds Gemini+Codex)

### Unanimous Agreement
- WLMS ballot ranking is the correct within-SG primitive
- 32KB TG memory is THE binding constraint for 2048 bins
- Sequential SG scatter is THE bottleneck in exp18
- Random scatter at 256 bins is efficient on Apple Silicon (no coalescing tricks needed)
- 3-pass is required to hit 5000+ (4-pass caps ~4100)

### Critical Bugs Found (Codex)
1. **uchar overflow**: `uchar sg_counts[8][2048]` overflows — max per-SG count = 512 > 255
2. **Per-iteration prefix missing**: WLMS gives within-ITERATION rank only. Need running count across 16+ iterations → requires per-SG rank array in TG memory
3. **Invalid lane masking**: Must mask sentinel digits in simd_ballot to avoid corrupt ranks

### Key KB Findings
| ID | Finding | Impact |
|----|---------|--------|
| #3568 | TG reorder does NOT help on Apple Silicon | Rules out coalescing tricks |
| #3348 | 8-round ballot WLMS for Metal radix ranking | Core technique |
| #3228 | Atomic contention 103x range, plateau at 1024 targets | Confirms histogram atomics OK |
| #3572 | 3-pass optimal, scatter ceiling 5000-5500 | Sets target envelope |
| #3573 | 2048-bin per-SG histogram exceeds 32KB TG | Constrains architecture |
| #3352 | No severe uncoalesced write penalty on Apple Silicon | De-risks random scatter |
| #3485 | M4 dynamic register allocation helps fused kernels | De-risks high register pressure |
| #3414 | Pipeline overlap between passes impossible (data dependency) | Rules out inter-pass overlap |
| #3534 | SLC improves scatter BW by 1.71x | Validates MSD+LSD SLC advantage |

### Gemini Insight (Round 2)
**Transposed storage `[2048][8]`**: All 8 SG counts for a digit are contiguous → single `uint2` load (8 bytes) instead of 7 scattered reads. Applicable to cross-SG prefix computation.

### Codex Insight (Round 2)
**Cache per-lane unique digits**: Each lane has ≤16 unique digits across 16 elements. Compute cross-SG prefix once per unique digit, not per element, to reduce redundant reads.

## Corrected Architecture Options

### Option A: 4-SG WLMS (128 threads, exp19)
- ushort sg_running[4 × 2048] = 16KB per-SG rank arrays
- WLMS within-SG, all 4 SGs active simultaneously
- 28KB TG memory total, fits in 32KB
- Risk: 128 threads may not saturate memory bandwidth
- Estimated: 3500-4500 Mkeys/s

### Option B: Improved exp17 MSD+LSD
- Inner passes at 256 bins — no TG memory pressure
- WLMS for inner ranking, fused histogram+scatter, 8192 tiles
- Already at 3461, incremental optimizations
- Estimated: 4500-6000 Mkeys/s

### Option C: 8-SG with uchar (tile=1792, exp20 backup)
- uchar sg_running[8 × 2048] = 16KB (max 224, fits 8-bit at 7 elem/lane)
- All 8 SGs active, 256 threads
- BUT: 8929 tiles, tile_status = 73MB (DRAM lookback)
- Estimated: 2500-3500 Mkeys/s

## WLMS Algorithm (Metal)

```metal
// For d-bit digit (d=8 for 256 bins, d=11 for 2048 bins):
inline uint wlms_rank(uint digit, uint simd_lane, uint d) {
    uint peer_mask = 0xFFFFFFFF;
    for (uint k = 0; k < d; k++) {
        bool my_bit = (digit >> k) & 1;
        uint ballot = (uint)simd_ballot(my_bit);
        peer_mask &= my_bit ? ballot : ~ballot;
    }
    return popcount(peer_mask & ((1u << simd_lane) - 1u));
}
```

Cost: d ballot + d AND + d ternary + 1 popcount = ~2d+2 instructions.
- 8-bit: 18 instructions
- 11-bit: 24 instructions
- vs current O(31) shared_digits scan

## Decision Criteria

Implement BOTH plans. Run benchmarks. Pick the winner:
- If Plan A (exp19) > 5000: Ship it (simpler architecture, pure LSD)
- If Plan B (exp17-improved) > 5000: Ship it (proven MSD+LSD base)
- If both < 5000: Combine insights (MSD scatter → WLMS 3-pass inner per bucket)
