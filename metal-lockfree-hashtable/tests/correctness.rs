//! Correctness tests for the GPU hash table.

use metal_lockfree_hashtable::{GpuHashTable, Version};
use rand::Rng;

const EMPTY_KEY: u32 = 0xFFFFFFFF;

/// Generate N unique random keys (no duplicates, no EMPTY_KEY).
fn unique_keys(n: usize) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    let mut keys = std::collections::HashSet::with_capacity(n);
    while keys.len() < n {
        let k: u32 = rng.gen();
        if k != EMPTY_KEY {
            keys.insert(k);
        }
    }
    keys.into_iter().collect()
}

// ─── 100% hit rate after inserting N keys and looking up same N keys ──

fn test_full_hit_rate(version: Version) {
    let n = 10_000;
    let keys = unique_keys(n);
    let values: Vec<u32> = (0..n as u32).collect();

    let table = GpuHashTable::new(version, (n * 2) as u32);
    table.insert(&keys, &values);
    let (results, _ms) = table.lookup(&keys);

    let hits = results.iter().filter(|&&v| v != EMPTY_KEY).count();
    assert_eq!(
        hits, n,
        "{}: Expected 100% hit rate, got {}/{} ({:.1}%)",
        version, hits, n, hits as f64 / n as f64 * 100.0
    );
}

#[test]
fn v1_full_hit_rate() {
    test_full_hit_rate(Version::V1);
}

#[test]
fn v2_full_hit_rate() {
    test_full_hit_rate(Version::V2);
}

#[test]
fn v3_full_hit_rate() {
    test_full_hit_rate(Version::V3);
}

// ─── Empty table returns EMPTY_KEY for all lookups ──

fn test_empty_table(version: Version) {
    let table = GpuHashTable::new(version, 1024);
    let keys = vec![1u32, 2, 3, 42, 999];
    let (results, _ms) = table.lookup(&keys);

    for (i, &val) in results.iter().enumerate() {
        assert_eq!(
            val, EMPTY_KEY,
            "{}: Empty table should return EMPTY_KEY for key {}, got {}",
            version, keys[i], val
        );
    }
}

#[test]
fn v1_empty_table() {
    test_empty_table(Version::V1);
}

#[test]
fn v2_empty_table() {
    test_empty_table(Version::V2);
}

#[test]
fn v3_empty_table() {
    test_empty_table(Version::V3);
}

// ─── Correct values returned ──

fn test_correct_values(version: Version) {
    let keys = vec![100u32, 200, 300, 400, 500];
    let values = vec![10u32, 20, 30, 40, 50];

    let table = GpuHashTable::new(version, 64);
    table.insert(&keys, &values);
    let (results, _ms) = table.lookup(&keys);

    for i in 0..keys.len() {
        assert_eq!(
            results[i], values[i],
            "{}: key {} → expected {}, got {}",
            version, keys[i], values[i], results[i]
        );
    }
}

#[test]
fn v1_correct_values() {
    test_correct_values(Version::V1);
}

#[test]
fn v2_correct_values() {
    test_correct_values(Version::V2);
}

#[test]
fn v3_correct_values() {
    test_correct_values(Version::V3);
}

// ─── Missing keys return EMPTY_KEY ──

fn test_missing_keys(version: Version) {
    let insert_keys = vec![1u32, 2, 3];
    let insert_values = vec![10u32, 20, 30];

    let table = GpuHashTable::new(version, 64);
    table.insert(&insert_keys, &insert_values);

    let query_keys = vec![4u32, 5, 6, 7];
    let (results, _ms) = table.lookup(&query_keys);

    for (i, &val) in results.iter().enumerate() {
        assert_eq!(
            val, EMPTY_KEY,
            "{}: Key {} should be missing, got {}",
            version, query_keys[i], val
        );
    }
}

#[test]
fn v1_missing_keys() {
    test_missing_keys(Version::V1);
}

#[test]
fn v2_missing_keys() {
    test_missing_keys(Version::V2);
}

#[test]
fn v3_missing_keys() {
    test_missing_keys(Version::V3);
}

// ─── Collision handling works ──

fn test_collisions(version: Version) {
    // Insert many keys with the same lower bits (forces probe chains).
    let n = 100;
    let cap = 256u32;
    // Keys that all hash to similar slots (multiples of capacity)
    let keys: Vec<u32> = (1..=n as u32).collect();
    let values: Vec<u32> = keys.iter().map(|k| k * 100).collect();

    let table = GpuHashTable::new(version, cap);
    table.insert(&keys, &values);
    let (results, _ms) = table.lookup(&keys);

    let hits = results.iter().filter(|&&v| v != EMPTY_KEY).count();
    assert_eq!(
        hits, n,
        "{}: Expected all {} keys found after collision handling, got {}",
        version, n, hits
    );

    // Verify correct values
    for i in 0..n {
        assert_eq!(
            results[i], values[i],
            "{}: Collision test key {} → expected {}, got {}",
            version, keys[i], values[i], results[i]
        );
    }
}

#[test]
fn v1_collisions() {
    test_collisions(Version::V1);
}

#[test]
fn v2_collisions() {
    test_collisions(Version::V2);
}

#[test]
fn v3_collisions() {
    test_collisions(Version::V3);
}

// ─── Clear resets the table ──

fn test_clear(version: Version) {
    let keys = vec![1u32, 2, 3];
    let values = vec![10u32, 20, 30];

    let mut table = GpuHashTable::new(version, 64);
    table.insert(&keys, &values);

    // Verify they're present
    let (results, _) = table.lookup(&keys);
    assert_eq!(results.iter().filter(|&&v| v != EMPTY_KEY).count(), 3);

    // Clear and verify they're gone
    table.clear();
    let (results, _) = table.lookup(&keys);
    for &val in &results {
        assert_eq!(val, EMPTY_KEY, "{}: Table should be empty after clear", version);
    }
}

#[test]
fn v1_clear() {
    test_clear(Version::V1);
}

#[test]
fn v2_clear() {
    test_clear(Version::V2);
}

#[test]
fn v3_clear() {
    test_clear(Version::V3);
}

// ─── All versions produce same results ──

#[test]
fn all_versions_agree() {
    let n = 5_000;
    let keys = unique_keys(n);
    let values: Vec<u32> = (0..n as u32).collect();

    let v1 = GpuHashTable::new(Version::V1, (n * 2) as u32);
    let v2 = GpuHashTable::new(Version::V2, (n * 2) as u32);
    let v3 = GpuHashTable::new(Version::V3, (n * 2) as u32);

    v1.insert(&keys, &values);
    v2.insert(&keys, &values);
    v3.insert(&keys, &values);

    let (r1, _) = v1.lookup(&keys);
    let (r2, _) = v2.lookup(&keys);
    let (r3, _) = v3.lookup(&keys);

    for i in 0..n {
        assert_eq!(
            r1[i], r2[i],
            "V1 vs V2 disagree at index {}: key={}, v1={}, v2={}",
            i, keys[i], r1[i], r2[i]
        );
        assert_eq!(
            r2[i], r3[i],
            "V2 vs V3 disagree at index {}: key={}, v2={}, v3={}",
            i, keys[i], r2[i], r3[i]
        );
    }
}

// ─── Large-scale test ──

#[test]
fn v3_large_scale() {
    let n = 100_000;
    let keys = unique_keys(n);
    let values: Vec<u32> = (0..n as u32).collect();

    let table = GpuHashTable::new(Version::V3, (n * 2) as u32);
    table.insert(&keys, &values);
    let (results, _ms) = table.lookup(&keys);

    let hits = results.iter().filter(|&&v| v != EMPTY_KEY).count();
    assert_eq!(
        hits, n,
        "V3 large-scale: Expected 100% hit rate at 100K keys, got {}/{} ({:.1}%)",
        hits, n, hits as f64 / n as f64 * 100.0
    );

    // Verify a sample of values
    for i in (0..n).step_by(1000) {
        assert_eq!(results[i], values[i], "V3 large-scale: wrong value at index {}", i);
    }
}
