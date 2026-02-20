// hash_join.metal -- GPU hash join kernels for forge-primitives.
//
// Open-addressing hash table with linear probing.
// Build phase: insert (key, index) pairs via atomic CAS.
// Probe phase: lookup probe keys, emit (build_idx, probe_idx) match pairs.
//
// Hash table layout: interleaved [key, value] pairs.
//   table[2*slot]     = key  (UINT_MAX = empty sentinel)
//   table[2*slot + 1] = build index
//
// Dispatch build:  ceil(build_count / 256) threadgroups of 256
// Dispatch probe:  ceil(probe_count / 256) threadgroups of 256

#include "types.h"

#define HASH_EMPTY 0xFFFFFFFF

// Simple hash function (multiply-shift)
inline uint hash_key(uint key, uint table_size) {
    // Knuth multiplicative hash
    return (key * 2654435761u) % table_size;
}

// ============================================================================
// hash_join_build -- Insert build keys into open-addressing hash table
// ============================================================================
//
// Buffer layout:
//   buffer(0): build_keys (uint array, build_count elements)
//   buffer(1): hash_table (uint array, table_size * 2 elements: [key, value] pairs)
//   buffer(2): HashJoinParams
//
// Each thread inserts one key with linear probing and atomic CAS.

kernel void hash_join_build(
    device const uint*       build_keys  [[buffer(0)]],
    device atomic_uint*      hash_table  [[buffer(1)]],
    constant HashJoinParams& params      [[buffer(2)]],
    uint tid                             [[thread_position_in_grid]]
) {
    if (tid >= params.build_count) return;

    uint key = build_keys[tid];
    uint slot = hash_key(key, params.table_size);

    // Linear probe until we find an empty slot
    for (uint i = 0; i < params.table_size; i++) {
        uint key_idx = slot * 2;

        // Attempt to claim this slot via CAS on the key position
        uint expected = HASH_EMPTY;
        bool success = atomic_compare_exchange_weak_explicit(
            &hash_table[key_idx],
            &expected,
            key,
            memory_order_relaxed,
            memory_order_relaxed
        );

        if (success) {
            // We claimed the slot -- store the build index
            atomic_store_explicit(&hash_table[key_idx + 1], tid, memory_order_relaxed);
            return;
        }

        // Slot taken -- if same key, also store (last-writer-wins for duplicates)
        if (expected == key) {
            atomic_store_explicit(&hash_table[key_idx + 1], tid, memory_order_relaxed);
            return;
        }

        // Linear probe: advance to next slot
        slot = (slot + 1) % params.table_size;
    }
    // Table full -- should not happen if load factor < 0.5
}

// ============================================================================
// hash_join_probe -- Probe hash table with probe keys, emit match pairs
// ============================================================================
//
// Buffer layout:
//   buffer(0): probe_keys (uint array, probe_count elements)
//   buffer(1): hash_table (uint array, table_size * 2 elements)
//   buffer(2): HashJoinParams
//   buffer(3): output_pairs (uint2 array: [build_idx, probe_idx])
//   buffer(4): match_count (atomic_uint, single element)
//
// Each thread probes one key. On match, atomically increments match_count
// and writes the (build_idx, probe_idx) pair.

kernel void hash_join_probe(
    device const uint*       probe_keys   [[buffer(0)]],
    device const uint*       hash_table   [[buffer(1)]],
    constant HashJoinParams& params       [[buffer(2)]],
    device uint2*            output_pairs [[buffer(3)]],
    device atomic_uint*      match_count  [[buffer(4)]],
    uint tid                              [[thread_position_in_grid]]
) {
    if (tid >= params.probe_count) return;

    uint key = probe_keys[tid];
    uint slot = hash_key(key, params.table_size);

    // Linear probe until we find the key or an empty slot
    for (uint i = 0; i < params.table_size; i++) {
        uint key_idx = slot * 2;
        uint table_key = hash_table[key_idx];

        if (table_key == HASH_EMPTY) {
            // Empty slot -- key not in table
            return;
        }

        if (table_key == key) {
            // Match found -- emit pair
            uint build_idx = hash_table[key_idx + 1];
            uint pos = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
            output_pairs[pos] = uint2(build_idx, tid);
            return;
        }

        // Linear probe: advance
        slot = (slot + 1) % params.table_size;
    }
}
