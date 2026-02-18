---
spec: gpu-search-dedup
phase: design
created: 2026-02-18T00:00:00Z
generated: auto
---

# Design: gpu-search-dedup

## Overview

Two-stage GPU pipeline: (1) path_search_kernel returns `GpuPathMatchResult[match_count]` (chunk_index + byte_offset), (2) new dedup_kernel reads match positions, computes FNV-1a hash of path bytes, atomically inserts into hash table, outputs unique_flags[match_count]. CPU filters paths by flag (1=unique, 0=duplicate), eliminating 73-79% CPU bottleneck.

## Architecture

```
GPU search pipeline (new stage highlighted):

┌─────────────────────────────────────────────────────────────────┐
│ GpuContentSearch (gpu-search-ui/src/engine/search.rs)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  search_paths(pattern) {                                        │
│    1. path_search_kernel dispatch                               │
│       Input: chunks_buffer (paths), pattern, metadata            │
│       Output: path_matches_buffer (chunk_index+byte_offset)    │
│       GPU time: ~0.5ms for 50K matches                          │
│                                                                  │
│    2. [NEW] dedup_kernel dispatch ◄── THIS REPLACES CPU       │
│       Input: path_matches_buffer, chunks_buffer, pattern       │
│       Hash: FNV-1a(path_bytes)→uint32                          │
│       CAS: atomic insert into hash_table[hash & mask]          │
│       Output: unique_flags_buffer (0=dup, 1=unique)            │
│       GPU time: <1ms for 50K matches                            │
│                                                                  │
│    3. CPU filter                                                │
│       Input: path_matches_buffer, unique_flags_buffer           │
│       Loop: n=0; for i in 0..match_count { if unique[i] { ... } }
│       Time: O(match_count), <50µs for 50K                       │
│  }                                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Component: DedupdKernel (MSL compute kernel)

**Purpose**: Mark first-seen path matches via atomic hash table insertion.

**Responsibilities**:
- Read GpuPathMatchResult (chunk_index, byte_offset)
- Extract newline-delimited path from chunks_buffer
- Compute FNV-1a hash of path UTF-8 bytes
- Atomic CAS insert into hash_table (linear probing for collisions)
- Write 1 to unique_flags[i] if CAS succeeded (first-seen), 0 if duplicate

**Signature** (MSL):
```metal
kernel void dedup_kernel(
    device const uchar* chunks_buffer [[buffer(0)]],
    device const GpuPathMatchResult* matches [[buffer(1)]],
    device atomic_uint* hash_table [[buffer(2)]],
    device atomic_uint* match_count [[buffer(3)]],
    device atomic_uchar* unique_flags [[buffer(4)]],
    constant SearchParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
);
```

**Algorithm**:
```
for each thread:
  if gid >= match_count: return

  m = matches[gid]
  chunk_index = m.chunk_index
  byte_offset = m.byte_offset

  // Extract path from chunks (newline-delimited)
  abs_pos = chunk_index * CHUNK_SIZE + byte_offset
  line_start = scan_back_for_newline(chunks, abs_pos)
  line_end = scan_forward_for_newline(chunks, abs_pos)
  path_len = line_end - line_start

  // FNV-1a hash
  hash = 0x811c9dc5  // FNV offset basis
  for byte_idx = 0..path_len:
    hash ^= chunks[line_start + byte_idx]
    hash *= 0x01000193

  // Linear probing insertion
  slot = hash & (table_capacity - 1)
  for probe = 0..MAX_PROBE:
    expected = 0xFFFFFFFF
    ok = atomic_compare_exchange_weak_explicit(
      &hash_table[slot], &expected, hash,
      memory_order_relaxed, memory_order_relaxed
    )
    if ok || expected == hash:
      // First-seen (ok=true) or duplicate (expected==hash)
      unique_flags[gid] = ok ? 1 : 0
      return
    slot = (slot + 1) & (table_capacity - 1)

  unique_flags[gid] = 0  // Failed to insert (table full)
```

### Component: GpuContentSearch Integration (Rust)

**Changes to search.rs**:
- Add `hash_table_buffer: Buffer` field (allocated in `new_for_paths()`)
- Add `dedup_pipeline: ComputePipelineState` (compiled from shader library)
- Add `unique_flags_buffer: Buffer` (100K × 1 byte)
- Add `dedup()` method to invoke kernel after path_search
- Update `search_paths()` to call `dedup()` and filter by unique_flags

**Fields to add**:
```rust
pub struct GpuContentSearch {
    // ... existing fields ...
    dedup_pipeline: ComputePipelineState,
    hash_table_buffer: Buffer,        // uint32[100K] for AoS interleaved (actually just keys)
    unique_flags_buffer: Buffer,      // uint8[100K]
}
```

**Methods to add**:
```rust
fn dedup(&self, match_count: u32) -> u64  // returns GPU execution time in microseconds
```

### Component: CPU Filter (app.rs)

**Changes to search_thread()**:
- Replace `seen.insert()` loop with unique_flags check
- Original code (lines 341-346):
  ```rust
  let mut seen = HashSet::new();
  let matches: Vec<ContentMatch> = raw_matches
    .into_iter()
    .filter(|m| {
      let path = m.file_path.trim();
      if path.is_empty() || !seen.insert(path.to_string()) { false } else { true }
    })
  ```
- New code:
  ```rust
  let matches: Vec<ContentMatch> = raw_matches
    .into_iter()
    .zip(unique_flags.iter())
    .filter_map(|(m, &is_unique)| {
      let path = m.file_path.trim();
      if path.is_empty() || !is_unique { None } else { Some(m) }
    })
    .collect()
  ```

## Data Flow

1. **After path_search_kernel**:
   - path_matches_buffer: `[GpuPathMatchResult; match_count]`
   - match_count: atomic counter on GPU
   - chunks_buffer: path bytes (already in GPU memory)

2. **Dedup kernel dispatch**:
   - Input buffers: chunks_buffer, path_matches_buffer, match_count
   - Hash table: pre-allocated 100K capacity, pre-cleared with 0xFF
   - Outputs: unique_flags_buffer (1 byte per match)
   - Synchronization: command_buffer.wait_until_completed()

3. **CPU readback**:
   - Read match_count from GPU (u32)
   - Read unique_flags_buffer[0..match_count] from GPU (u8[])
   - Filter path_matches by flag in CPU loop (O(n))

4. **Result**:
   - Vector<ContentMatch> contains only unique paths
   - Zero duplicates (100% dedup accuracy)

## Technical Decisions

| Decision | Options | Choice | Rationale |
|----------|---------|--------|-----------|
| Hash function | FNV-1a vs MurmurHash3 vs CRC32 | FNV-1a | Simplest (2 ops/byte), SIMD-friendly, collision rate <5% at 50K matches |
| Table layout | SoA (separate keys/vals) vs AoS interleaved | AoS keys-only | Key-only table (value unused) → 4 bytes per entry vs 8B. FNV hash sufficient as presence marker. |
| Load factor | 25% vs 50% vs 75% | 50% | Standard, linear probing avg 1.3 probes. 50K matches → 100K capacity (power of 2). |
| Probing strategy | Linear vs quadratic vs chaining | Linear | Predictable, cache-friendly, proven in metal-lockfree-hashtable. |
| Atomic ordering | relaxed vs acquire/release vs seq_cst | relaxed | GPU insert only, no inter-kernel ordering required. Coherence guaranteed by command_buffer.wait_until_completed(). |
| Path extraction | Scan backward+forward vs precomputed offsets | Scan in-kernel | Matches already have byte_offset; forward scan bounded by next newline (avg 131B on macOS). |
| Hash input | Full path vs just filename | Full path | Uniqueness per full path (not filename). Correctness over optimization. |
| Case sensitivity | Always match params.case_sensitive | Yes | Inherit from search options. Hash both full path bytes (not lowercased — preserve case in output). |

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| gpu-search-ui/src/engine/search.rs | Modify | Add dedup_pipeline, hash_table_buffer, dedup() method, update search_paths() |
| gpu-search-ui/src/engine/shader.rs | Modify | Add dedup_kernel MSL source (40 lines) |
| gpu-search-ui/src/engine/mod.rs | Unchanged | No changes (compile_shader already handles multi-kernel libraries) |
| gpu-search-ui/src/app.rs | Modify | Replace HashSet::insert loop with unique_flags filter (lines 341-346) |

## Error Handling

| Error | Handling | User Impact |
|-------|----------|-------------|
| Hash table insertion fails (table full) | unique_flags[i]=0 (treated as duplicate) | No match loss; worst case all marked duplicate, CPU re-dedupes via HashSet. Performance degrades to CPU-only. |
| GPU shader compilation fails | Fallback: disable dedup, skip dedup_kernel, use CPU HashSet | Graceful degradation. Performance reverts to baseline (3.6s) but UI still works. |
| GPU buffer allocation fails | Error propagated in GpuContentSearch::new_for_paths() | Early failure at search startup (same as path_search_kernel buffer allocation). |

## Existing Patterns to Follow

- **Shader compilation**: Use `get_path_search_shader()` pattern — add `dedup_kernel` to path library, return via helper function
- **Buffer management**: Allocate hash_table_buffer in `new_for_paths()` at same time as path_matches_buffer (consistent sizing)
- **Dispatch pattern**: Reuse `path_pipeline` encoder setup from search_paths() lines 425-456 — same threadgroup size (256), same command queue
- **Result extraction**: `unique_flags_buffer` readback via unsafe pointer cast (same as path_matches_buffer in search_paths)
- **Profiling**: Extend SearchProfile struct with `dedup_us` field if measured

## Optimization Opportunities (Phase 2+)

- **Vectorized hashing**: SIMD 4-byte parallel FNV-1a using simd_shuffle (reduce hash compute by 4x)
- **Cuckoo hashing**: Replace linear probing with cuckoo hashing for O(1) guaranteed lookup (requires post-processing)
- **Bloom filter pre-pass**: Count potential duplicates before hash table insert to optimize for 90%+ dedup rates
- **Hash table persistence**: Keep hash table across multiple searches within same session (incremental updates)
