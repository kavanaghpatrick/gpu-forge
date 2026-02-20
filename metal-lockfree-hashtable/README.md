# metal-lockfree-hashtable

A lock-free GPU hash table on Apple Silicon Metal. **1499 Mops** lookup at 32M keys (DRAM-resident), **5438 Mops** at 1M keys (SLC-resident), with 100% hit rate.

## Overview

This is a standalone, self-contained implementation of a lock-free hash table that runs entirely on the GPU via Metal compute shaders. It uses open-addressing with linear probing, atomic CAS for insertion, and a 50% load factor (capacity = 2x keys).

The project includes three versions showing progressive optimization from a naive baseline to the final design, plus honest documentation of what worked, what didn't, and why.

**Key result**: At DRAM-resident table sizes (32M keys, 512MB table), the AoS interleaved layout (V3) achieves **1499 Mops lookup** — a **91% improvement** over the SoA baseline (V1: 786 Mops) — by eliminating a second random DRAM access per lookup.

## Design

**Open-addressing, linear probing, atomic CAS insert.**

- **Hash function**: MurmurHash3 finalizer (full avalanche — every input bit affects every output bit)
- **Collision resolution**: Linear probing with stride-1 access (mandatory on Apple Silicon per KB finding 3240: stride-32 causes a 30x bandwidth cliff)
- **Insert**: `atomic_compare_exchange_weak` on key slot, max 64 probes
- **Lookup**: Plain device read (non-atomic) after insert completes — 32-bit aligned reads are naturally atomic on Apple Silicon
- **Sentinel**: `0xFFFFFFFF` marks empty slots
- **Capacity**: Always power-of-2 for bitmask indexing (`slot & (capacity - 1)`)
- **Load factor**: 50% (capacity = 2x number of keys) for short probe chains

## Version History

### V1: SoA + Simple Hash + Atomic Lookup (Baseline)

**Layout**: Two separate arrays — `keys[]` (atomic_uint) and `values[]` (uint).

```
keys:   [k0, k1, k2, k3, ...]
values: [v0, v1, v2, v3, ...]
```

Simple integer hash (`key ^= key >> 16; key *= 0x45d9f3b; key ^= key >> 16`). Lookups use `atomic_load_explicit` — correct but unnecessary overhead since the table is read-only after insert completes.

**Performance**: 786 Mops lookup at 32M keys.

### V2: MurmurHash3 + Non-Atomic Lookup

Two targeted changes:

1. **MurmurHash3 finalizer**: Better avalanche means fewer collisions means shorter probe chains
2. **Non-atomic lookup**: After the insert command buffer completes, the table is coherent in unified memory. Plain `device const uint*` reads are sufficient (and 32-bit aligned reads are naturally atomic anyway)

**Result**: Surprisingly, **no measurable improvement** at these sizes. The probe chains are already short at 50% load factor, so better hashing doesn't help much. Non-atomic reads save instruction overhead but the workload is bandwidth-bound, not instruction-bound.

**Performance**: 783 Mops lookup at 32M keys (≈0% change).

### V3: AoS Interleaved (Key+Value in Same Cache Line)

**Layout**: Interleaved key-value pairs in a single buffer.

```
table: [k0, v0, k1, v1, k2, v2, ...]
       └──────┘ └──────┘ └──────┘
        slot 0   slot 1   slot 2
```

Key at `table[slot * 2]`, value at `table[slot * 2 + 1]`.

One 128-byte cache line now holds **16 complete key-value pairs**. When a lookup finds a matching key, the value is in the same cache line — no second random DRAM fetch needed.

**This is the big win.** At DRAM-resident sizes where random access latency dominates, eliminating one random memory access per found key is transformative.

**Performance**: 1499 Mops lookup at 32M keys (**+91% over V1**).

## What Didn't Work

### V4: Blocked/Chunked Layout with Threadgroup SRAM

**Idea**: Divide the table into blocks. Load each block into threadgroup shared memory (SRAM) for fast local probing. This mirrors NVIDIA-era techniques where shared memory is dramatically faster than global memory.

**Implementation**: Each threadgroup loads a chunk of the table into shared memory, performs probing locally, then writes results back.

**Result**: **Abandoned — slower than V3.** The scatter overhead of gathering sparse hash table entries into contiguous SRAM blocks negated any benefit from faster local memory. Apple Silicon's unified memory hierarchy (with SLC acting as an effective L3) already provides good caching for stride-1 access patterns. The explicit data movement was pure overhead.

**Lesson**: On Apple Silicon, the memory hierarchy is well-optimized enough that manual SRAM management rarely beats the hardware's caching for random-access workloads. This is consistent with our broader finding from 9 GPU exploit experiments: most NVIDIA-era optimization tricks show <5% improvement on Apple Silicon.

### ILP-4 (4 Keys per Thread)

**Idea**: Process 4 keys per thread to exploit instruction-level parallelism and amortize launch overhead.

**Result**: **Hurt performance.** Reduced thread-level parallelism (TLP) by 4x, which matters more than ILP for memory-latency-bound workloads. The GPU hides memory latency through massive parallelism — reducing the thread count defeats this mechanism.

## Benchmark Methodology

- **Timing**: GPU-side timestamps via `MTLCommandBuffer.GPUStartTime()`/`GPUEndTime()` — avoids ~300us CPU→GPU firmware overhead inherent in wall-clock timing
- **Warmup**: 2-3 untimed runs before measurement to warm caches and stabilize GPU clock
- **Runs**: 5-10 timed runs per data point
- **Outlier removal**: IQR method (discard samples below Q1 - 1.5×IQR or above Q3 + 1.5×IQR)
- **Statistic reported**: Median (robust to outliers)
- **Table cleared**: Before each insert run to avoid measuring already-populated tables
- **Hardware**: Apple M4 Pro (12-core CPU, 16-core GPU, 48GB unified memory)
- **macOS**: Sequoia 15.5

## Results

### 1M Keys, 2M Capacity (SLC-Resident, ~16MB table)

| Version | Insert (Mops) | Lookup (Mops) | Hit Rate |
|---------|---------------|---------------|----------|
| V1      | 2916          | 3393          | 100%     |
| V2      | 2984          | 3335          | 100%     |
| V3      | 3970          | 5438          | 100%     |

At SLC-resident sizes, all versions are fast because the system-level cache (16-24MB on M4 Pro) holds the entire table. V3's AoS layout still wins by **60%** on lookup because it halves the number of cache line fetches.

### 32M Keys, 64M Capacity (DRAM-Resident, ~512MB table)

| Version | Insert (Mops) | Lookup (Mops) | Lookup vs V1 |
|---------|---------------|---------------|-------------|
| V1      | 552           | 786           | baseline    |
| V2      | 553           | 783           | -0.4%       |
| V3      | 757           | 1499          | **+91%**    |

At DRAM-resident sizes, V3's advantage becomes dramatic. Every SoA lookup that finds a key needs **two** random DRAM accesses (one for the key array, one for the value array). V3 needs **one** (key and value are in the same cache line). Random DRAM latency is ~100ns on Apple Silicon — eliminating one access per hit nearly doubles throughput.

### CPU Comparison

At 1M keys, `std::collections::HashMap` (insert + lookup) takes ~18ms.
GPU V3 (insert + lookup) takes ~0.44ms. **~40x speedup**.

## Key Findings

1. **AoS beats SoA for random-access lookups.** Conventional wisdom says SoA is always better for GPUs (coalesced access). This is true for *sequential* access patterns, but for *random* access (hash tables), locality between related fields (key→value) matters more than lane-level coalescing.

2. **Non-atomic reads are sufficient after insert.** Apple Silicon unified memory guarantees coherence after a command buffer completes. The `atomic_load` in V1's lookup is pure overhead — plain reads work correctly and are faster.

3. **Hash quality barely matters at 50% load.** MurmurHash3's perfect avalanche vs. a simple multiply-shift hash: no measurable difference at 50% load factor. Probe chains are 1-2 entries on average regardless. Hash quality matters more at high load factors (>70%).

4. **SLC is a game-changer at small sizes.** The system-level cache (16-24MB usable on M4 Pro) turns DRAM-latency-bound workloads into cache-speed workloads when the table fits. 5438 Mops at 1M vs. 1499 Mops at 32M — 3.6x faster just from caching.

5. **NVIDIA-era tricks don't apply.** Threadgroup SRAM pre-loading, ILP-4, and other techniques from the CUDA world showed zero or negative benefit. Apple Silicon's memory hierarchy is well-optimized for the patterns that GPUs naturally produce.

## Limitations

- **uint32 keys and values only** — no string keys, no 64-bit support
- **No delete** — insert-only; clear the entire table to "delete"
- **No resize** — capacity is fixed at creation time
- **50% max load factor** — wastes memory for safety
- **Single GPU** — no multi-device support
- **Apple Silicon only** — requires Metal 3.1 and unified memory
- **Max 64 probes** — pathological hash collisions will silently drop inserts

## Building & Running

```bash
# Build
cargo build --release

# Run tests (20 correctness tests)
cargo test

# Run example (insert/lookup 1M + 32M keys, benchmarks, CPU comparison)
cargo run --release --example basic
```

## Project Structure

```
metal-lockfree-hashtable/
├── Cargo.toml                  # Standalone crate
├── build.rs                    # .metal → .metallib via xcrun
├── shaders/
│   ├── hashtable.metal         # V1, V2, V3 kernels
│   └── types.h                 # HashTableParams struct
├── src/
│   ├── lib.rs                  # Public API re-exports
│   ├── metal_ctx.rs            # Minimal Metal context (device, queue, lib)
│   ├── table.rs                # GpuHashTable struct (insert/lookup/clear)
│   └── bench.rs                # Benchmark harness with stats
├── examples/
│   └── basic.rs                # Insert, lookup, verify, benchmark
└── tests/
    └── correctness.rs          # 20 tests: hit rate, collisions, clear, versions agree
```
