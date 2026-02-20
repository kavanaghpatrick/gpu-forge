---
spec: gpu-sort-5000
phase: requirements
created: 2026-02-19
---

# Requirements: GPU Radix Sort 5000+ Mkeys/s

## Goal

Achieve 5000+ Mkeys/s radix sort throughput for 16M uint32 on M4 Pro using an MSD+LSD hybrid approach that exploits SLC bandwidth (469 GB/s vs 245 GB/s DRAM) to reduce effective pass cost. Measurement-first: validate SLC scatter bandwidth before committing to hybrid architecture.

## User Stories

### US-1: Measure SLC Scatter Bandwidth at Multiple Working Set Sizes

**As a** GPU performance researcher
**I want to** measure 256-bin scatter bandwidth at SLC-resident working set sizes (62.5K to 4M elements)
**So that** I can validate whether the MSD+LSD hybrid approach is viable before investing in kernel development

**Acceptance Criteria:**
- [ ] AC-1.1: Scatter bandwidth measured at 5 sizes: 62.5K, 250K, 1M, 4M, 16M elements
- [ ] AC-1.2: Each measurement uses existing `exp16_diag_scatter_binned` pattern (256-bin structured scatter)
- [ ] AC-1.3: Results printed with p5/p50/p95 timing and GB/s throughput (same format as exp16)
- [ ] AC-1.4: 16M result validates against known baseline (~131 GB/s)
- [ ] AC-1.5: Go/no-go decision documented: hybrid viable if SLC scatter >= 80 GB/s at 250K elements

### US-2: MSD Scatter Pass (Global Partition by Top 8 Bits)

**As a** GPU performance researcher
**I want to** partition 16M elements into 256 buckets by bits[24:31] in a single global pass
**So that** each bucket (~62K elements, ~250KB) fits in SLC for fast inner sorting

**Acceptance Criteria:**
- [ ] AC-2.1: MSD scatter kernel produces 256 buckets with correct element counts (sum = N)
- [ ] AC-2.2: Every element appears in exactly the correct bucket (bits[24:31] match bucket index)
- [ ] AC-2.3: Bucket offsets array provided for inner sort to locate each bucket's data
- [ ] AC-2.4: MSD scatter completes in <= 2.5 ms for 16M elements (131 GB/s scatter baseline)
- [ ] AC-2.5: Correctness verified for uniform random uint32 input

### US-3: Inner Per-Bucket LSD Sort (SLC-Resident)

**As a** GPU performance researcher
**I want to** sort each bucket independently using 3-pass 8-bit LSD radix sort
**So that** inner passes run at SLC bandwidth (~469 GB/s) instead of DRAM bandwidth (~245 GB/s)

**Acceptance Criteria:**
- [ ] AC-3.1: Each bucket sorted correctly in ascending order after 3 LSD passes (bits 0-23)
- [ ] AC-3.2: All 256 buckets dispatched in a single compute dispatch (NOT 256 separate dispatches)
- [ ] AC-3.3: Bucket ID and tile-within-bucket derived from threadgroup grid ID
- [ ] AC-3.4: Inner sort uses reduce-then-scan (safe on Apple Silicon, no forward progress risk)
- [ ] AC-3.5: Per-bucket correctness verified: sorted output within each bucket matches CPU reference

### US-4: End-to-End Hybrid Sort Benchmark

**As a** GPU performance researcher
**I want to** run the complete MSD+LSD hybrid pipeline and measure throughput
**So that** I can determine if 5000+ Mkeys/s is achievable

**Acceptance Criteria:**
- [ ] AC-4.1: End-to-end sort of 16M random uint32 produces correct sorted output (matches `expected.sort()`)
- [ ] AC-4.2: Throughput measured over 50 runs with 5 warmup iterations (matching exp16 pattern)
- [ ] AC-4.3: p5/p50/p95 timing and Mkeys/s printed in standard format
- [ ] AC-4.4: Per-phase timing breakdown printed (MSD scatter, inner hist, inner passes)
- [ ] AC-4.5: Target: p50 >= 5000 Mkeys/s (3.2 ms or less for 16M elements)
- [ ] AC-4.6: Correctness verified at 1M, 4M, and 16M element counts

### US-5: Fallback Evidence if Target Not Met

**As a** GPU performance researcher
**I want to** document measured bottlenecks and bandwidth ceilings if 5000 Mkeys/s is not reached
**So that** I have evidence-based conclusions about M4 Pro sort limits

**Acceptance Criteria:**
- [ ] AC-5.1: If target not met, per-phase bandwidth utilization reported (actual vs theoretical)
- [ ] AC-5.2: Bottleneck identified with measurement (e.g., "MSD scatter limited to X GB/s")
- [ ] AC-5.3: Theoretical ceiling computed from measured phase bandwidths
- [ ] AC-5.4: Comparison printed: baseline vs hybrid at same element count

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | SLC scatter bandwidth benchmark at 5 working set sizes | P0 | AC-1.1 through AC-1.5 |
| FR-2 | Go/no-go gate: abort hybrid if SLC scatter < 80 GB/s at 250K | P0 | AC-1.5 |
| FR-3 | MSD scatter kernel: partition 16M by bits[24:31] into 256 buckets | P0 | AC-2.1 through AC-2.5 |
| FR-4 | Bucket offset computation (exclusive prefix sum of bucket counts) | P0 | AC-2.3 |
| FR-5 | Inner LSD sort kernel: 3-pass 8-bit sort per bucket | P0 | AC-3.1 through AC-3.5 |
| FR-6 | Single-dispatch inner sort (bucket IDs in threadgroup grid) | P0 | AC-3.2, AC-3.3 |
| FR-7 | Reduce-then-scan prefix for inner sort (no decoupled lookback) | P1 | AC-3.4 |
| FR-8 | End-to-end benchmark with per-phase timing breakdown | P0 | AC-4.1 through AC-4.6 |
| FR-9 | Correctness check: GPU output == CPU `sort()` at 1M, 4M, 16M | P0 | AC-4.1, AC-4.6 |
| FR-10 | Fallback analysis if target not met | P1 | AC-5.1 through AC-5.4 |
| FR-11 | Reuse exp16 patterns: per-SG atomic histogram, same buffer alloc, same timing infra | P1 | Code review |
| FR-12 | Tune inner sort tile size (4096 vs 8192 elements/tile) | P2 | Measured comparison |
| FR-13 | Optional: 4-bit MSD (16 bins, ~1M/bucket) as fallback if 8-bit MSD scatter degrades | P2 | Measured comparison |

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Sort throughput | Mkeys/s @ 16M uint32, p50 | >= 5000 |
| NFR-2 | Correctness | Bit-exact match vs CPU sort | 100% across all sizes |
| NFR-3 | MSD scatter latency | ms @ 16M elements | <= 2.5 |
| NFR-4 | Inner sort SLC bandwidth utilization | GB/s during inner passes | >= 350 (75% of 469) |
| NFR-5 | Dispatch overhead | Total GPU idle time from dispatch gaps | <= 0.2 ms |
| NFR-6 | Memory usage | Peak GPU buffer allocation | <= 512 MB (4x input = 256MB + scratch) |
| NFR-7 | Build time | Incremental Rust + Metal shader compile | < 30s |

## Glossary

- **MSD**: Most Significant Digit — sort/partition by highest-order bits first
- **LSD**: Least Significant Digit — sort by lowest-order bits first, each pass stable
- **SLC**: System Level Cache — Apple Silicon shared L3 cache (36MB on M4 Pro), 469 GB/s bandwidth
- **DRAM**: Main device memory, 245 GB/s bandwidth on M4 Pro
- **Decoupled lookback**: Single-pass prefix sum using inter-TG atomic signaling (Merrill-Garland)
- **Reduce-then-scan**: Two-pass prefix sum — first reduce per-tile, then propagate. Safe on Apple Silicon (no forward progress requirement)
- **Scatter**: Writing elements to non-contiguous output positions based on computed offsets
- **Mkeys/s**: Million keys sorted per second — throughput metric (N / time_seconds / 1e6)
- **TG**: Threadgroup — Apple GPU execution unit (equivalent to CUDA block)
- **SG**: Simdgroup — 32-thread SIMD unit within a TG (equivalent to CUDA warp)
- **Bucket**: Partition of input data by MSD digit value (256 buckets for 8-bit MSD)

## Out of Scope

- Key-value pair sorting (uint32 keys only)
- Non-uniform distributions (skewed, sorted, nearly-sorted inputs) — optimization for uniform random only
- Multi-chip or distributed sorting
- Integration with gpu-search or any other crate — standalone experiment only
- Automated test suite — correctness verified inline per exp16 pattern
- Persistent kernel approach for inner sort (rejected: forward progress risk)
- 11-bit radix (rejected: 2048-bin scatter degrades to 20 GB/s)
- Fused two-digit LSD (rejected: mathematically incorrect for inter-tile stability)

## Dependencies

- **Hardware**: Apple M4 Pro (testing machine)
- **Codebase**: `metal-gpu-experiments` crate, specifically `exp16_8bit.metal` and `exp16_8bit.rs` as reference
- **Metal shader compiler**: Requires `-std=metal3.2` for `atomic_thread_fence` with `thread_scope_device`
- **Proven primitives**: Per-SG atomic histogram, decoupled lookback (exp16), device-scope fence (exp12)
- **Build**: `cargo build --release -p metal-gpu-experiments`
- **Run**: `cargo run --release -p metal-gpu-experiments`
- **Shader rebuild**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments`

## Success Criteria

1. **Primary**: p50 throughput >= 5000 Mkeys/s for 16M random uint32 on M4 Pro
2. **Secondary**: Correctness verified at 1M, 4M, 16M — bit-exact match with CPU sort
3. **Fallback**: If 5000 not achievable, documented evidence of M4 Pro bandwidth ceiling with per-phase analysis showing exactly which phase is the bottleneck and what theoretical maximum is

## Unresolved Questions

1. **SLC scatter bandwidth at 250KB**: The make-or-break measurement. If < 80 GB/s, hybrid approach may not beat baseline. Must measure before writing MSD kernel.
2. **Optimal MSD bin count**: 256 bins (250KB/bucket, ideal SLC) vs 16 bins (4MB/bucket, fewer dispatch concerns). 256 is the research recommendation but 16 is the safe fallback.
3. **Inner sort prefix strategy**: Reduce-then-scan is safe but adds an extra encoder barrier per pass. At 15-tile depth, is the overhead acceptable vs the risk of decoupled lookback?
4. **Bucket size variance**: Uniform random gives ~62K/bucket, but worst-case skew (all same top byte) puts 16M in one bucket. Should the inner sort handle arbitrarily large buckets?
5. **TG reorder in inner sort**: Does adding v4-style TG reorder to inner scatter help when data is already SLC-resident, or is the SLC bandwidth sufficient without reordering?

## Next Steps

1. Approve requirements, then proceed to design phase
2. Design phase should produce kernel signatures, buffer layout, dispatch geometry, and per-phase Metal shader pseudocode
3. Implementation starts with FR-1 (SLC scatter bandwidth measurement) as the gate for all subsequent work
