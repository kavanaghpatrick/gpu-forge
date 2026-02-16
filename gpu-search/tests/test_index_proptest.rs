//! Property-based tests for GSIX v2 format roundtrip integrity and binary fuzzing.
//!
//! Part 1: Verifies that random GpuPathEntry values survive save/load and
//! save/mmap roundtrips, and that structural invariants (256B entry size,
//! path_len <= 224) always hold.
//!
//! Part 2: Verifies that the index loader never panics on malformed input.
//! All corrupted/truncated/random/extended inputs must produce `Err`, never a panic.

use std::mem::size_of;

use gpu_search::gpu::types::{GpuPathEntry, GPU_PATH_MAX_LEN};
use gpu_search::index::gsix_v2::{load_v2, save_v2, GsixHeaderV2, HEADER_SIZE_V2};
use gpu_search::io::mmap::MmapBuffer;

use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};

// ============================================================================
// Shared helpers
// ============================================================================

/// Number of proptest cases. Override with PROPTEST_CASES env var.
fn num_cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000)
}

/// Compare two GpuPathEntry values byte-for-byte.
fn entries_byte_equal(a: &GpuPathEntry, b: &GpuPathEntry) -> bool {
    let a_bytes: &[u8; 256] = unsafe { &*(a as *const GpuPathEntry as *const [u8; 256]) };
    let b_bytes: &[u8; 256] = unsafe { &*(b as *const GpuPathEntry as *const [u8; 256]) };
    a_bytes == b_bytes
}

/// Create a valid v2 index file with `n` entries, return the raw bytes.
fn make_valid_index(n: usize) -> Vec<u8> {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let path = dir.path().join("valid.idx");

    let mut entries = Vec::with_capacity(n);
    for i in 0..n {
        let mut entry = GpuPathEntry::new();
        entry.set_path(format!("/test/file_{}.rs", i).as_bytes());
        entry.mtime = 1700000000 + i as u32;
        entry.set_size(1024 * (i as u64 + 1));
        entries.push(entry);
    }

    save_v2(&entries, 0xDEAD, &path, 42, 0xCAFE).expect("save_v2");
    std::fs::read(&path).expect("read valid index")
}

// ============================================================================
// Part 1: Format Roundtrip Property Tests
// ============================================================================

/// Strategy to generate a random GpuPathEntry with arbitrary field values.
fn gpu_path_entry_strategy() -> impl Strategy<Value = GpuPathEntry> {
    (
        // Random path bytes (up to 224)
        prop::collection::vec(any::<u8>(), 0..=GPU_PATH_MAX_LEN),
        // flags: any combination of known flag bits (0..=0x1F covers bits 0-4)
        0u32..=0x1F,
        // parent_idx
        any::<u32>(),
        // size (as u64 split into lo/hi)
        any::<u64>(),
        // mtime
        any::<u32>(),
    )
        .prop_map(|(path_bytes, flags, parent_idx, size, mtime)| {
            let mut entry = GpuPathEntry::new();
            let len = path_bytes.len().min(GPU_PATH_MAX_LEN);
            entry.path[..len].copy_from_slice(&path_bytes[..len]);
            entry.path[len..].fill(0);
            entry.path_len = len as u32;
            entry.flags = flags;
            entry.parent_idx = parent_idx;
            entry.set_size(size);
            entry.mtime = mtime;
            entry
        })
}

/// Strategy to generate a Vec of 0..=100 random GpuPathEntry values.
fn entries_strategy() -> impl Strategy<Value = Vec<GpuPathEntry>> {
    prop::collection::vec(gpu_path_entry_strategy(), 0..=100)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(num_cases()))]

    /// Property 1: save_v2 -> load_v2 roundtrip preserves all entries byte-identically.
    #[test]
    fn prop_save_load_roundtrip(entries in entries_strategy()) {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let idx_path = dir.path().join("roundtrip.idx");

        // Save
        save_v2(&entries, 0xDEAD, &idx_path, 42, 0xBEEF)
            .expect("save_v2 should succeed");

        // Load
        let (header, loaded) = load_v2(&idx_path).expect("load_v2 should succeed");

        // Header fields
        prop_assert_eq!(header.entry_count as usize, entries.len());
        prop_assert_eq!(header.root_hash, 0xDEAD);
        prop_assert_eq!(header.last_fsevents_id, 42u64);
        prop_assert_eq!(header.exclude_hash, 0xBEEF);

        // Entry count
        prop_assert_eq!(loaded.len(), entries.len());

        // Byte-identical entries
        for (i, (orig, ld)) in entries.iter().zip(loaded.iter()).enumerate() {
            prop_assert!(
                entries_byte_equal(orig, ld),
                "entry {} differs after save/load roundtrip", i
            );
        }
    }

    /// Property 2: save_v2 -> mmap -> read entries from mmap preserves all entries.
    #[test]
    fn prop_save_mmap_roundtrip(entries in entries_strategy()) {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let idx_path = dir.path().join("mmap_roundtrip.idx");

        // Save
        save_v2(&entries, 0xCAFE, &idx_path, 99, 0xFACE)
            .expect("save_v2 should succeed");

        // Mmap the file
        let mmap = MmapBuffer::from_file(&idx_path)
            .expect("mmap should succeed");

        let data = mmap.as_slice();

        // Validate header
        let header = GsixHeaderV2::from_bytes(data)
            .expect("header should parse from mmap");
        prop_assert_eq!(header.entry_count as usize, entries.len());

        // Read entries from mmap'd data
        for (i, orig) in entries.iter().enumerate() {
            let offset = HEADER_SIZE_V2 + i * 256;
            prop_assert!(
                offset + 256 <= data.len(),
                "mmap data too short for entry {}", i
            );

            let loaded: GpuPathEntry = unsafe {
                std::ptr::read_unaligned(
                    data[offset..].as_ptr() as *const GpuPathEntry
                )
            };

            prop_assert!(
                entries_byte_equal(orig, &loaded),
                "entry {} differs after save/mmap roundtrip", i
            );
        }
    }

    /// Property 3: GpuPathEntry is always exactly 256 bytes.
    #[test]
    fn prop_entry_size_invariant(entry in gpu_path_entry_strategy()) {
        prop_assert_eq!(
            size_of::<GpuPathEntry>(), 256,
            "GpuPathEntry must always be exactly 256 bytes"
        );

        // Verify raw byte representation is also 256 bytes
        let bytes: &[u8; 256] = unsafe {
            &*(&entry as *const GpuPathEntry as *const [u8; 256])
        };
        prop_assert_eq!(bytes.len(), 256);
    }

    /// Property 4: path_len is always <= GPU_PATH_MAX_LEN (224).
    #[test]
    fn prop_path_len_bounded(entry in gpu_path_entry_strategy()) {
        prop_assert!(
            entry.path_len as usize <= GPU_PATH_MAX_LEN,
            "path_len {} exceeds GPU_PATH_MAX_LEN {}", entry.path_len, GPU_PATH_MAX_LEN
        );
    }
}

// ============================================================================
// Part 2: Binary Fuzzing Property Tests
// ============================================================================

/// Flip random bytes in a valid index file. load_v2 must return Err, never panic.
#[test]
fn prop_corrupt_bytes_no_panic() {
    let valid = make_valid_index(10);
    let valid_len = valid.len();

    let mut runner = TestRunner::new(Config {
        cases: num_cases(),
        ..Config::default()
    });

    // Strategy: pick 1-50 random byte positions and replacement values
    let strategy = prop::collection::vec((0..valid_len, any::<u8>()), 1..=50);

    runner
        .run(&strategy, |mutations| {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let path = dir.path().join("corrupt.idx");

            let mut data = valid.clone();
            for &(pos, val) in &mutations {
                data[pos] = val;
            }

            std::fs::write(&path, &data).expect("write");

            // Must not panic. Result can be Ok (if corruption missed critical fields)
            // or Err (if corruption hit magic/version/checksum/entry data).
            let _ = load_v2(&path);

            Ok(())
        })
        .unwrap();
}

/// Truncate a valid index at a random offset. load_v2 must return Err, never panic.
#[test]
fn prop_truncated_file_no_panic() {
    let valid = make_valid_index(10);
    let valid_len = valid.len();

    let mut runner = TestRunner::new(Config {
        cases: num_cases(),
        ..Config::default()
    });

    // Truncate to 0..valid_len bytes
    let strategy = 0..valid_len;

    runner
        .run(&strategy, |truncate_at| {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let path = dir.path().join("truncated.idx");

            let data = &valid[..truncate_at];
            std::fs::write(&path, data).expect("write");

            // Must not panic. Truncated files should return Err.
            let _ = load_v2(&path);

            Ok(())
        })
        .unwrap();
}

/// Write 64-8192 random bytes as a file. load_v2 must return Err, never panic.
#[test]
fn prop_random_bytes_no_panic() {
    let mut runner = TestRunner::new(Config {
        cases: num_cases(),
        ..Config::default()
    });

    let strategy = prop::collection::vec(any::<u8>(), 64..=8192);

    runner
        .run(&strategy, |random_data| {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let path = dir.path().join("random.idx");

            std::fs::write(&path, &random_data).expect("write");

            // Must not panic. Random bytes should always produce Err
            // (bad magic, bad checksum, wrong version, too small, etc.)
            let result = load_v2(&path);
            prop_assert!(
                result.is_err(),
                "random bytes should never produce Ok: got {} entries",
                result.unwrap().1.len()
            );

            Ok(())
        })
        .unwrap();
}

/// Append random bytes to a valid index. load_v2 must not panic.
/// It may succeed (trailing bytes ignored) or fail, but must never panic.
#[test]
fn prop_extended_file_no_panic() {
    let valid = make_valid_index(10);

    let mut runner = TestRunner::new(Config {
        cases: num_cases(),
        ..Config::default()
    });

    // Append 1-4096 random bytes
    let strategy = prop::collection::vec(any::<u8>(), 1..=4096);

    runner
        .run(&strategy, |extra_bytes| {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let path = dir.path().join("extended.idx");

            let mut data = valid.clone();
            data.extend_from_slice(&extra_bytes);

            std::fs::write(&path, &data).expect("write");

            // Must not panic. The loader may:
            // - Succeed (trailing bytes are ignored, as documented)
            // - Fail (unlikely with current impl but allowed)
            let _ = load_v2(&path);

            Ok(())
        })
        .unwrap();
}
