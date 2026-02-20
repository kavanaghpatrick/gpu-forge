# Product Manager Analysis: forge-sort Test Suite

## Research Findings

### GPU Radix Sort Correctness Testing

Research into GPU radix sort testing reveals several critical areas that the current forge-sort test suite does not adequately cover:

1. **Stability verification is fundamental**: Radix sort is defined as a stable sort — elements with equal keys must preserve their relative input order. The forge-sort implementation sorts `u32` values (not key-value pairs), so stability is not directly observable, but the algorithm's correctness depends on each pass being stable internally. If stability breaks in any pass, the final output can be wrong for specific bit patterns even if it appears correct for random data. ([Wikipedia: Radix Sort](https://en.wikipedia.org/wiki/Radix_sort))

2. **Data distribution matters enormously**: GPU sort implementations should be tested against at minimum: random uniform, reverse-sorted, already-sorted, nearly-sorted, and heavily-skewed distributions. The current suite covers random, pre-sorted, and reverse-sorted but misses nearly-sorted, few-unique-values, and adversarial distributions. ([AMD GPUOpen: Boosting GPU Radix Sort](https://gpuopen.com/learn/boosting_gpu_radix_sort/))

3. **Functional correctness on a test suite is not sufficient** to rule out rare but dangerous behaviors such as data races and out-of-bounds memory accesses. GPU kernels that pass all tests can still fail in production due to non-deterministic thread scheduling. ([Cornell: Verifying GPU Kernels by Test Amplification](https://www.cs.cornell.edu/~lerner/papers/verifying_gpu_kernels_by_test_amplification.pdf))

4. **Bit-pattern sensitivity**: 8-bit radix sort partitions keys by individual bytes. Inputs where all elements share the same byte at a particular position (e.g., all values in `0x00XX0000`) create degenerate cases: one MSD bucket receives all elements while 255 buckets are empty. This exercises the empty-bucket early-exit paths and the single-bucket-with-all-data path. ([GeeksforGeeks: Radix Sort](https://www.geeksforgeeks.org/dsa/radix-sort/))

5. **Metal-specific concerns**: Apple Silicon's unified memory model, register pressure sensitivity, and the `atomic_thread_fence` / `threadgroup_barrier` semantics require targeted testing. Register spilling can cause 250x performance variance (observed in exp16). The `simd_shuffle` instruction has known non-uniform-lane bugs on M4 Pro. ([Apple WWDC: Optimize Metal Performance](https://developer.apple.com/videos/play/wwdc2020/10632/))

6. **Determinism is not guaranteed**: GPU thread scheduling is non-deterministic. The same input can produce different intermediate orderings, and subtle bugs in rank computation or atomic ordering can manifest as non-deterministic output. Multi-run determinism testing is essential. ([UBC: Deterministic Execution on GPU Architectures](https://open.library.ubc.ca/media/stream/pdf/24/1.0074006/1))

### Key Source References

- [AMD GPUOpen: Boosting GPU Radix Sort](https://gpuopen.com/learn/boosting_gpu_radix_sort/) — circular buffer extension to Onesweep
- [Stehle & Satish: Memory Bandwidth-Efficient Hybrid Radix Sort](https://arxiv.org/pdf/1611.01137) — hybrid MSD/LSD approach
- [Cornell: Verifying GPU Kernels by Test Amplification](https://www.cs.cornell.edu/~lerner/papers/verifying_gpu_kernels_by_test_amplification.pdf) — test adequacy for GPU code
- [NVIDIA CUB DeviceRadixSort](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh) — reference implementation
- [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/) — practical GPU sort survey
- [ProofWright: Agentic Formal Verification of CUDA](https://arxiv.org/pdf/2511.12294) — formal correctness for GPU kernels

## Business Context

forge-sort is a **public Rust crate** providing GPU-accelerated sorting for Apple Silicon. Its value proposition is clear: 31x faster than CPU `sort_unstable` at 16M elements. However, this value is destroyed if the library produces incorrect results even once in production. The consequences of an incorrect sort differ by downstream use case:

| Use Case | Impact of Incorrect Sort |
|----------|------------------------|
| Search index building | Silent data corruption, wrong search results served to users |
| Database operations | Data integrity violation, potential cascading corruption |
| Graphics/particle systems | Visual glitches (low severity but visible) |
| ML preprocessing | Model training on corrupted data, silent accuracy degradation |
| Financial data processing | Regulatory and correctness violations |

**The 21 existing tests are a proof-of-concept level of coverage.** They verify the happy path but leave critical algorithm internals unexercised. A production library needs tests that are deliberately adversarial — designed to trigger the specific failure modes of this specific algorithm on this specific hardware.

The forge-sort algorithm has 4 GPU kernels with complex interactions:
1. **sort_msd_histogram** — per-SG atomic histogram with cross-SG reduction
2. **sort_msd_prep** — serial prefix sum (thread 0 only) + parallel bucket descriptor writes
3. **sort_msd_atomic_scatter** — atomic fetch-add scatter with per-SG ranking
4. **sort_inner_fused** — 3-pass LSD with self-computed histograms, buffer ping-pong

Each kernel has distinct failure modes. The test suite must exercise each independently and in combination.

## Test Coverage Requirements

### Category 1: Tile Boundary Precision (CRITICAL)

**Why**: TILE_SIZE=4096 and THREADS_PER_TG=256 create hard boundaries. The last tile of any input is partial. Off-by-one errors in `idx < n` checks or `num_tiles` computation cause silent data loss or buffer overruns.

| Test | Size | What It Exercises |
|------|------|-------------------|
| Exactly 1 tile | 4096 | Single full tile, no partial tile handling |
| 1 tile + 1 | 4097 | Minimal partial tile (1 valid element in tile 2) |
| 1 tile - 1 | 4095 | Partial tile with 4095/4096 valid (255 threads valid, last thread 15th element invalid) |
| 2 tiles exact | 8192 | Two full tiles, no partial |
| N*TILE + THREADS | 4096+256 | Partial tile exactly filling one SIMD group's worth |
| N*TILE + 1 thread | 4096+1 | Single thread active in partial tile |
| Prime-sized | 4099, 8191, 16381 | No alignment to any power-of-two |
| Max tiles boundary | 256*4096=1048576 | 256 tiles (matches fused_grid dispatch width) |
| 256 tiles + 1 | 1048577 | One element beyond the dispatch grid width |

### Category 2: Adversarial Bit Patterns (CRITICAL)

**Why**: The MSD scatter partitions by bits[24:31]. The inner fused sort handles bits[0:7], [8:15], [16:23]. Degenerate distributions in any byte position exercise empty-bucket paths, single-bucket-overflow, and atomic contention extremes.

| Test | Pattern | What It Stresses |
|------|---------|-----------------|
| All same MSD byte | `0xAA_XX_XX_XX` | 1 of 256 MSD buckets gets all elements; 255 empty buckets |
| Two MSD buckets only | `0x00_*` and `0xFF_*` alternating | Only 2 buckets populated; extreme skew |
| All 256 MSD buckets uniform | `i % 256 << 24 \| random` | Perfect distribution; max bucket count per bin |
| Identical inner bytes | `XX_00_00_00` | All 3 inner passes see all-zeros; single bin gets everything |
| All bits set | `0xFFFFFFFF` repeated | All bytes = 0xFF; single bin at every pass |
| Bit 24 boundary | `0x00FFFFFF` and `0x01000000` | Adjacent MSD bins, max inner spread |
| Sequential high bytes | `0x00_*`, `0x01_*`, ... `0xFF_*` | Every MSD bucket gets exactly `n/256` elements |
| Power-of-two values | `1, 2, 4, 8, ...` | Sparse bit patterns; most bytes are 0 |
| Alternating 0/MAX | `0x00000000` / `0xFFFFFFFF` | Two extreme bins at every radix pass |
| Near-duplicates | `v, v+1` repeated | Adjacent values stress ranking correctness |

### Category 3: Determinism and Reproducibility (HIGH)

**Why**: GPU atomic operations have non-deterministic ordering. If the algorithm depends on atomic ordering (it does — `atomic_fetch_add` for scatter positioning), non-determinism could manifest as different but both-sorted outputs (acceptable) or as incorrect sorting (not acceptable). We must verify that the output is always the unique correct sorted sequence.

| Test | Method | What It Verifies |
|------|--------|-----------------|
| 10-run identical output | Sort same input 10x, assert all outputs identical | Deterministic for fixed input |
| Cross-size determinism | Sort subset of larger array, compare with standalone sort of subset | No buffer-size-dependent behavior |
| Seeded random battery | 100 different seeded random arrays, verify all correct | Statistical confidence in correctness |

### Category 4: Scale and Stress (HIGH)

**Why**: Large inputs increase tile counts, which increases atomic contention on global counters and the likelihood of hitting hardware limits (TG memory, buffer sizes, dispatch grid dimensions).

| Test | Size | What It Exercises |
|------|------|-------------------|
| 32M elements | 32,000,000 | 2x the current maximum tested |
| 64M elements | 64,000,000 | 256MB data; approaches unified memory pressure |
| 128M elements | 128,000,000 | 512MB data; exercises buffer reallocation |
| Just under u32::MAX tiles | TILE_SIZE * 65535 | Near-max dispatch grid width |
| Rapid small sorts | 1000x sort of 1K | Buffer reuse under rapid cycling |
| Alternating sizes | 16M, 1K, 16M, 1K, ... | Buffer capacity grows then underutilized |

### Category 5: Buffer Management and Reuse (MEDIUM)

**Why**: `ensure_buffers()` only reallocates when `data_bytes > self.data_buf_capacity`. Sorting a smaller array after a larger one reuses the larger buffer. Leftover data in the buffer beyond `n` elements could leak into results if bounds checks are wrong.

| Test | Scenario | What It Verifies |
|------|----------|-----------------|
| Large then small | Sort 16M, then sort 100 | Old data doesn't leak from oversized buffer |
| Small then large | Sort 100, then sort 16M | Reallocation works correctly |
| Same size different data | Sort 1M of X, then 1M of Y | Previous results don't affect new sort |
| Many reuses | Sort 50 different arrays | No memory accumulation or corruption |
| Growing sequence | Sort 1K, 2K, 4K, ... 16M | Progressive reallocation chain |

### Category 6: Error Handling and Edge Cases (MEDIUM)

**Why**: The library must handle all valid inputs gracefully and report errors clearly for invalid states.

| Test | Input | Expected |
|------|-------|----------|
| Empty slice | `[]` | Returns Ok, slice unchanged |
| Single element | `[42]` | Returns Ok, `[42]` |
| Two elements sorted | `[1, 2]` | Returns Ok, `[1, 2]` |
| Two elements reversed | `[2, 1]` | Returns Ok, `[1, 2]` |
| All u32::MAX | `[0xFFFFFFFF; N]` | Returns Ok, all equal |
| All u32::MIN | `[0; N]` | Returns Ok, all zero |
| u32::MAX and u32::MIN | `[MAX, 0, MAX, 0]` | Correctly sorted |
| GpuSorter creation | `GpuSorter::new()` | Returns Ok on Metal-capable hardware |
| Multiple GpuSorter instances | Create 2+ sorters | Both work independently |

### Category 7: Performance Regression (LOW-MEDIUM)

**Why**: Performance is the entire value proposition. Regressions must be caught before they reach users.

| Test | Metric | Threshold |
|------|--------|-----------|
| 16M random sort time | Wall-clock ms | < 10ms (currently ~5.6ms) |
| 1M random sort time | Wall-clock ms | < 2ms |
| Throughput floor | Mk/s at 16M | > 2000 Mk/s |
| Scaling linearity | 4M vs 16M ratio | Within 1.5x (not 4x) |
| Cold start penalty | First sort vs second | < 3x overhead |

### Category 8: Concurrent and Parallel Usage (LOW)

**Why**: Users may create multiple `GpuSorter` instances or share one across threads. The library should either be thread-safe or clearly document that it is not.

| Test | Scenario | What It Verifies |
|------|----------|-----------------|
| Sequential from multiple sorters | Create 2 GpuSorters, sort alternately | No interference |
| Rapid sequential | 100 sorts in tight loop | No resource exhaustion |

## Risk Assessment

### Critical Risks (Data Corruption)

| Risk | Likelihood | Impact | Current Coverage |
|------|-----------|--------|-----------------|
| Off-by-one in partial tile handling | Medium | Data loss/corruption | 1 test (4097) |
| Atomic scatter rank error under skew | Medium | Incorrect sort | 0 tests |
| MSD bucket overflow when 1 bucket gets all data | Medium | Buffer overrun or wrong result | 0 tests |
| Inner fused pass histogram mismatch | Low-Medium | Silent wrong output | 0 tests |
| Buffer reuse data leak (old data in oversized buffer) | Low | Silent wrong output | 1 test (same-size reuse only) |

### High Risks (Reliability)

| Risk | Likelihood | Impact | Current Coverage |
|------|-----------|--------|-----------------|
| Non-deterministic output across runs | Low | User confusion, test flakiness | 0 tests |
| GPU thermal throttle causing timeout | Medium | Sort returns error under sustained load | 0 tests |
| Memory exhaustion at large sizes | Low | Crash or error | 0 tests |
| Performance regression goes unnoticed | Medium | Value proposition destroyed | 1 test (very loose bound) |

### Medium Risks (Usability)

| Risk | Likelihood | Impact | Current Coverage |
|------|-----------|--------|-----------------|
| Error messages unhelpful | Low | Poor developer experience | 1 test (display only) |
| Multiple GpuSorter instances conflict | Low | Unexpected errors | 0 tests |
| Buffer reallocation thrashing | Low | Performance degradation | 0 tests |

## Success Metrics

The test suite is adequate when:

1. **Correctness confidence**: Every test in categories 1-3 passes, providing >99.99% confidence that the sort is correct for all valid u32 inputs up to 128M elements.

2. **Boundary coverage**: Every tile-boundary condition (exact, +1, -1) is tested at both small and large scales.

3. **Bit pattern coverage**: Every byte position (0, 8, 16, 24) has been tested with degenerate inputs (all-same, all-different, two-value).

4. **Determinism proof**: 10-run identical-output tests pass for at least 5 different input sizes and distributions.

5. **Regression detection**: Performance tests have tight enough bounds to catch a 2x regression within CI.

6. **Test count target**: 60-80 integration tests (up from 18), 5-8 unit tests (up from 3). Total: 65-88 tests.

7. **Test execution time**: Full suite completes in < 60 seconds (GPU sorts are fast; the bottleneck is CPU reference sorting for verification).

## Priority Matrix

Tests are ordered by risk-reduction per implementation effort:

| Priority | Category | Tests | Effort | Risk Reduced |
|----------|----------|-------|--------|-------------|
| **P0** | Adversarial Bit Patterns | 10-12 | Low | Critical — these are the most likely to find real bugs |
| **P0** | Tile Boundary Precision | 8-10 | Low | Critical — off-by-one is the #1 GPU kernel bug class |
| **P1** | Determinism Verification | 3-5 | Low | High — proves the atomics are correct |
| **P1** | Buffer Management | 5-6 | Low | High — reuse bugs are subtle and data-corrupting |
| **P2** | Scale/Stress | 4-6 | Medium | High — catches hardware-limit failures |
| **P2** | Error Handling/Edge Cases | 6-8 | Low | Medium — completes the contract |
| **P3** | Performance Regression | 4-5 | Medium | Medium — protects value proposition |
| **P3** | Concurrent Usage | 2-3 | Low | Low — clarifies threading contract |

### Implementation Recommendation

**Phase 1 (P0, ~20 tests)**: Implement adversarial bit patterns and tile boundary tests first. These have the highest probability of finding actual bugs in the shader code. Estimated effort: 2-3 hours.

**Phase 2 (P1, ~10 tests)**: Add determinism and buffer management tests. These catch subtle correctness issues. Estimated effort: 1-2 hours.

**Phase 3 (P2+P3, ~15 tests)**: Add scale, error handling, performance, and concurrency tests. These round out production readiness. Estimated effort: 2-3 hours.

**Total**: ~45 new tests, bringing the suite from 21 to ~66 tests, achievable in 5-8 hours of implementation.
