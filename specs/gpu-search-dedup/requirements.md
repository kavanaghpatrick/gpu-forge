---
spec: gpu-search-dedup
phase: requirements
created: 2026-02-18T00:00:00Z
generated: auto
---

# Requirements: gpu-search-dedup

## Summary

Replace CPU HashSet dedup in gpu-search-ui with GPU-side lock-free hash table. After path_search_kernel returns matches, new dedup_kernel marks duplicates via atomic CAS insert. CPU filters to unique paths only. Target: <1ms GPU dedup vs 3.6s CPU (~3600x speedup).

## User Stories

### US-1: GPU Dedup Eliminates CPU Bottleneck
As a gpu-search-ui user searching 50K paths, I want dedup to complete in GPU not CPU so that total search time drops from 3.7s to ~1s (path search + dedup <1ms vs 3.6s CPU dedup).

**Acceptance Criteria**:
- AC-1.1: dedup_kernel processes GpuPathMatchResult array, marks unique matches with flag
- AC-1.2: Atomic CAS insertion into GPU hash table (capacity 100K) tracks seen path hashes
- AC-1.3: GPU execution time for 50K matches is <1ms (measured via GPU timers)
- AC-1.4: CPU receives unique_flags[] output buffer, filters results in O(n) time
- AC-1.5: All 50K paths correctly identified as unique or duplicate (zero false negatives)

### US-2: Path Hashing Uses Efficient GPU Algorithm
As a performance engineer, I want path strings hashed with FNV-1a in GPU kernel so that hash computation stays within 1ms budget for 50K paths.

**Acceptance Criteria**:
- AC-2.1: FNV-1a algorithm implemented inline in dedup_kernel (6 ops per byte: multiply by 0x01000193 + XOR)
- AC-2.2: Hash computed from newline-delimited path bytes in chunks_buffer
- AC-2.3: Hash collision rate <5% at 50K matches into 100K table (measured via profiling)
- AC-2.4: SIMD vectorization opportunity explored (simd_shuffle for byte-parallel hashing) but not required for POC

### US-3: Dedup Integrates with Existing Pipeline
As a maintainer of gpu-search-ui, I want dedup to reuse GpuContentSearch pipelines so integration is minimal (<10 lines net code change).

**Acceptance Criteria**:
- AC-3.1: dedup_kernel compiled into path_library (same library as path_search_kernel)
- AC-3.2: Hash table buffer allocated in GpuContentSearch::new_for_paths()
- AC-3.3: Dedup dispatch invoked immediately after path_search dispatch in same command buffer or sequential buffers
- AC-3.4: CPU filter in app.rs search_thread() updated to read unique_flags[] and skip duplicates
- AC-3.5: No new public API — dedup invisible to UI/UX layers (pure optimization)

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | GPU dedup kernel accepts GpuPathMatchResult[], chunk_data, hash table buffer | Must | US-1, US-2 |
| FR-2 | Dedup kernel computes FNV-1a hash of each path string from chunk_data | Must | US-2 |
| FR-3 | Hash table uses atomic CAS insertion with linear probing (open addressing) | Must | US-1 |
| FR-4 | Output: unique_flags[match_count] where 1=first-seen, 0=duplicate | Must | US-1 |
| FR-5 | Hash table capacity is 100K (power of 2, supports 50K entries at 50% load) | Should | US-1 |
| FR-6 | Dedup kernel dispatches with 256 threads/threadgroup (match GPU search pattern) | Should | US-2 |
| FR-7 | Support case-sensitive and case-insensitive path matching (inherit from search options) | Should | US-3 |
| FR-8 | Measure GPU time for dedup kernel dispatch (return via SearchProfile) | Should | US-1 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | GPU dedup latency <1ms for 50K matches (avg case with linear probing) | Performance |
| NFR-2 | Hash collision rate <5% (linear probing avg probe length <1.5) | Performance |
| NFR-3 | Hash table memory overhead ≤3.2MB for 100K capacity | Memory |
| NFR-4 | No breaking changes to GpuContentSearch public API | Compatibility |
| NFR-5 | Dedup kernel code <50 lines MSL (excluding comments/whitespace) | Maintainability |
| NFR-6 | Integration code <5 lines net new Rust (excluding tests) | Maintainability |
| NFR-7 | Zero CPU synchronization between path_search and dedup (async GPU operations) | Concurrency |

## Out of Scope

- Hierarchical/tiered hash tables (50% load factor sufficient)
- SIMD 4-way parallel hashing (sequential FNV-1a sufficient for POC)
- Persistent hash table across searches (per-search allocation + clear)
- Hash table overflow handling (100K capacity assumes ≤50K unique matches)
- GPU-side sorting by path priority (CPU already does this)
- Dedup for content search (only for path search)

## Dependencies

- metal crate (metal-rs) — Metal device, buffers, compute pipelines already available
- objc2-metal NOT required — gpu-search-ui uses `metal` crate exclusively
- No external hash table library — inline FNV-1a + atomic CAS pattern
