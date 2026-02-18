// hashtable.metal — Lock-Free GPU Hash Table (V1, V2, V3)
//
// Open-addressing, linear probing, atomic CAS insert.
// Three versions showing progressive optimization:
//
//   V1: SoA layout + simple hash + atomic lookup
//       Baseline. Two separate arrays for keys and values.
//       Uses atomic_load for lookups (unnecessary but safe).
//
//   V2: MurmurHash3 + non-atomic lookup
//       Better hash = fewer collisions = shorter probe chains.
//       After insert completes, table is coherent — plain reads suffice.
//       32-bit aligned reads are naturally atomic on Apple Silicon.
//
//   V3: AoS interleaved layout (key+value in same cache line)
//       Layout: [key0, val0, key1, val1, ...]
//       One 128B cache line holds 16 complete key-value pairs.
//       Eliminates the second random DRAM access for the value array.
//       Best version: 92% lookup improvement at DRAM sizes.

#include "types.h"

// ─── CONSTANTS ──────────────────────────────────────────────────

constant uint EMPTY_KEY = 0xFFFFFFFF;
constant uint MAX_PROBE = 64;

// ─── HASH FUNCTIONS ─────────────────────────────────────────────

// Simple integer hash — reasonable distribution but imperfect avalanche.
inline uint simple_hash(uint key) {
    key ^= key >> 16;
    key *= 0x45d9f3b;
    key ^= key >> 16;
    return key;
}

// MurmurHash3 finalizer — full avalanche (every input bit affects every output bit).
inline uint murmur3_hash(uint key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}


// ═══════════════════════════════════════════════════════════════
// V1: SoA + Simple Hash + Atomic Lookup
// ═══════════════════════════════════════════════════════════════
//
// Separate key[] and value[] arrays (Structure-of-Arrays).
// Simple integer hash for slot computation.
// Atomic loads for lookup — safe but unnecessary overhead.

kernel void ht_insert_v1(
    device atomic_uint* keys         [[buffer(0)]],
    device uint* values              [[buffer(1)]],
    device const uint* input_keys    [[buffer(2)]],
    device const uint* input_values  [[buffer(3)]],
    constant HashTableParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_ops) return;

    uint key = input_keys[tid];
    uint val = input_values[tid];
    if (key == EMPTY_KEY) key = 0xFFFFFFFE;

    uint slot = simple_hash(key) & (params.capacity - 1);

    for (uint probe = 0; probe < MAX_PROBE; probe++) {
        uint expected = EMPTY_KEY;
        bool ok = atomic_compare_exchange_weak_explicit(
            &keys[slot], &expected, key,
            memory_order_relaxed, memory_order_relaxed);

        if (ok || expected == key) {
            values[slot] = val;
            return;
        }
        slot = (slot + 1) & (params.capacity - 1);
    }
}

kernel void ht_lookup_v1(
    device const atomic_uint* keys   [[buffer(0)]],
    device const uint* values        [[buffer(1)]],
    device const uint* query_keys    [[buffer(2)]],
    device uint* output              [[buffer(3)]],
    constant HashTableParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_ops) return;

    uint key = query_keys[tid];
    if (key == EMPTY_KEY) key = 0xFFFFFFFE;

    uint slot = simple_hash(key) & (params.capacity - 1);

    for (uint probe = 0; probe < MAX_PROBE; probe++) {
        uint slot_key = atomic_load_explicit(&keys[slot], memory_order_relaxed);

        if (slot_key == key) {
            output[tid] = values[slot];
            return;
        }
        if (slot_key == EMPTY_KEY) {
            output[tid] = EMPTY_KEY;
            return;
        }
        slot = (slot + 1) & (params.capacity - 1);
    }
    output[tid] = EMPTY_KEY;
}


// ═══════════════════════════════════════════════════════════════
// V2: MurmurHash3 + Non-Atomic Lookup
// ═══════════════════════════════════════════════════════════════
//
// Two targeted optimizations over V1:
//   1. MurmurHash3 finalizer: better avalanche → fewer collisions → shorter probes
//   2. Non-atomic lookup: plain device reads after insert completes
//      (32-bit aligned reads are naturally atomic on Apple Silicon)

kernel void ht_insert_v2(
    device atomic_uint* keys         [[buffer(0)]],
    device uint* values              [[buffer(1)]],
    device const uint* input_keys    [[buffer(2)]],
    device const uint* input_values  [[buffer(3)]],
    constant HashTableParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_ops) return;

    uint key = input_keys[tid];
    uint val = input_values[tid];
    if (key == EMPTY_KEY) key = 0xFFFFFFFE;

    uint slot = murmur3_hash(key) & (params.capacity - 1);

    for (uint probe = 0; probe < MAX_PROBE; probe++) {
        uint expected = EMPTY_KEY;
        bool ok = atomic_compare_exchange_weak_explicit(
            &keys[slot], &expected, key,
            memory_order_relaxed, memory_order_relaxed);

        if (ok || expected == key) {
            values[slot] = val;
            return;
        }
        slot = (slot + 1) & (params.capacity - 1);
    }
}

kernel void ht_lookup_v2(
    device const uint* keys          [[buffer(0)]],  // NOT atomic — read-only after insert
    device const uint* values        [[buffer(1)]],
    device const uint* query_keys    [[buffer(2)]],
    device uint* output              [[buffer(3)]],
    constant HashTableParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_ops) return;

    uint key = query_keys[tid];
    if (key == EMPTY_KEY) key = 0xFFFFFFFE;

    uint slot = murmur3_hash(key) & (params.capacity - 1);

    for (uint probe = 0; probe < MAX_PROBE; probe++) {
        uint slot_key = keys[slot];  // Plain load — no atomic needed

        if (slot_key == key) {
            output[tid] = values[slot];
            return;
        }
        if (slot_key == EMPTY_KEY) {
            output[tid] = EMPTY_KEY;
            return;
        }
        slot = (slot + 1) & (params.capacity - 1);
    }
    output[tid] = EMPTY_KEY;
}


// ═══════════════════════════════════════════════════════════════
// V3: AoS Interleaved (Key+Value in Same Cache Line)
// ═══════════════════════════════════════════════════════════════
//
// Key optimization: interleave key+value as adjacent uint pairs.
// Layout: [key0, val0, key1, val1, key2, val2, ...]
// Key at table[slot*2], value at table[slot*2+1].
//
// One 128B cache line now holds 16 complete key-value pairs.
// Lookup hits both key AND value in a single cache line fetch,
// eliminating the second random DRAM access to a separate value array.
//
// This is the fastest version at DRAM-resident table sizes (>16MB).

kernel void ht_insert_v3(
    device atomic_uint* table        [[buffer(0)]],  // interleaved [k,v,k,v,...]
    device const uint* input_keys    [[buffer(1)]],
    device const uint* input_values  [[buffer(2)]],
    constant HashTableParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_ops) return;

    uint key = input_keys[tid];
    uint val = input_values[tid];
    if (key == EMPTY_KEY) key = 0xFFFFFFFE;

    uint slot = murmur3_hash(key) & (params.capacity - 1);

    for (uint probe = 0; probe < MAX_PROBE; probe++) {
        uint expected = EMPTY_KEY;
        bool ok = atomic_compare_exchange_weak_explicit(
            &table[slot * 2], &expected, key,
            memory_order_relaxed, memory_order_relaxed);

        if (ok || expected == key) {
            atomic_store_explicit(&table[slot * 2 + 1], val, memory_order_relaxed);
            return;
        }
        slot = (slot + 1) & (params.capacity - 1);
    }
}

kernel void ht_lookup_v3(
    device const uint* table         [[buffer(0)]],  // interleaved [k,v,k,v,...]
    device const uint* query_keys    [[buffer(1)]],
    device uint* output              [[buffer(2)]],
    constant HashTableParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_ops) return;

    uint key = query_keys[tid];
    if (key == EMPTY_KEY) key = 0xFFFFFFFE;

    uint slot = murmur3_hash(key) & (params.capacity - 1);

    for (uint probe = 0; probe < MAX_PROBE; probe++) {
        uint slot_key = table[slot * 2];

        if (slot_key == key) {
            output[tid] = table[slot * 2 + 1];
            return;
        }
        if (slot_key == EMPTY_KEY) {
            output[tid] = EMPTY_KEY;
            return;
        }
        slot = (slot + 1) & (params.capacity - 1);
    }
    output[tid] = EMPTY_KEY;
}
