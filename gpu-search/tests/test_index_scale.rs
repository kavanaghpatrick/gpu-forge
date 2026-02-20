//! Scale tests for 1M-entry GSIX v2 index operations.
//!
//! All tests are `#[ignore]` — run with `cargo test --test test_index_scale -- --ignored`.
//! These verify that the index can handle 1M entries without OOM or correctness issues.
//! Expected memory: 1M * 256B = ~256MB for entries + 16KB header.

use std::time::Instant;

use tempfile::TempDir;

use gpu_search::gpu::types::{path_flags, GpuPathEntry, GPU_PATH_MAX_LEN};
use gpu_search::index::gsix_v2::{load_v2, save_v2};
use gpu_search::index::snapshot::IndexSnapshot;

// ---------------------------------------------------------------------------
// Local synthetic entry generator (integration tests can't access #[cfg(test)] modules)
// ---------------------------------------------------------------------------

const DIR_COMPONENTS: &[&str] = &[
    "src", "lib", "tests", "benches", "docs", "config", "build", "target", "scripts",
    "utils", "core", "api", "models", "views", "controllers", "services", "middleware",
    "handlers", "proto", "internal", "pkg", "cmd", "assets", "static", "templates",
];

const EXTENSIONS: &[&str] = &[
    ".rs", ".rs", ".rs",
    ".txt", ".js", ".py", ".md", ".toml", ".json", ".yaml", ".html", ".css", ".ts",
    ".sh", ".c", ".h", ".go", ".swift", ".metal",
];

const ROOT_PREFIXES: &[&str] = &[
    "/Users/dev/project",
    "/Users/dev/workspace",
    "/home/user/code",
    "/opt/builds",
    "/var/lib/app",
    "/usr/local/src",
];

/// Simple deterministic LCG (same as in gsix_v2::test_helpers).
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }

    fn next_range(&mut self, bound: u32) -> u32 {
        (self.next() >> 4) % bound
    }
}

fn generate_synthetic_entries(count: usize) -> Vec<GpuPathEntry> {
    let mut rng = Lcg::new(0xBEEF_CAFE);
    let mut entries = Vec::with_capacity(count);

    for i in 0..count {
        let mut entry = GpuPathEntry::new();

        let root = ROOT_PREFIXES[rng.next_range(ROOT_PREFIXES.len() as u32) as usize];
        let depth = (rng.next_range(8) + 1) as usize;
        let mut path = String::with_capacity(224);
        path.push_str(root);

        for _ in 0..depth {
            path.push('/');
            let comp = DIR_COMPONENTS[rng.next_range(DIR_COMPONENTS.len() as u32) as usize];
            path.push_str(comp);
        }

        let ext = EXTENSIONS[rng.next_range(EXTENSIONS.len() as u32) as usize];
        path.push_str(&format!("/file_{}{}", i, ext));

        if path.len() > GPU_PATH_MAX_LEN {
            path.truncate(GPU_PATH_MAX_LEN);
        }

        entry.set_path(path.as_bytes());

        let flag_roll = rng.next_range(100);
        if flag_roll < 10 {
            entry.flags |= path_flags::IS_DIR;
        }
        if flag_roll < 5 {
            entry.flags |= path_flags::IS_HIDDEN;
        }
        if flag_roll < 2 {
            entry.flags |= path_flags::IS_SYMLINK;
        }

        if entry.flags & path_flags::IS_DIR == 0 {
            let size = (rng.next_range(10_000_000) + 100) as u64;
            entry.set_size(size);
        }

        let base_mtime: u32 = 1_672_531_200;
        let offset = rng.next_range(63_158_400);
        entry.mtime = base_mtime + offset;

        if i > 0 && depth > 1 {
            entry.parent_idx = rng.next_range(i as u32);
        }

        entries.push(entry);
    }

    entries
}

const SCALE: usize = 1_000_000;

// ---------------------------------------------------------------------------
// Scale tests
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_scale_1m_build() {
    println!("Generating {} synthetic entries...", SCALE);
    let t0 = Instant::now();
    let entries = generate_synthetic_entries(SCALE);
    let gen_time = t0.elapsed();
    println!("Generated {} entries in {:.2?}", entries.len(), gen_time);

    assert_eq!(entries.len(), SCALE);

    // Verify memory usage makes sense: 1M * 256B = 256MB
    let expected_bytes = SCALE * std::mem::size_of::<GpuPathEntry>();
    assert_eq!(expected_bytes, 256_000_000, "expected 256MB for 1M entries");

    // Spot-check: all entries have valid path_len
    for (i, entry) in entries.iter().enumerate() {
        assert!(
            entry.path_len > 0 && entry.path_len as usize <= GPU_PATH_MAX_LEN,
            "entry {} has invalid path_len: {}",
            i,
            entry.path_len,
        );
    }

    // Verify determinism: first and last entries are stable
    let first_path = &entries[0].path[..entries[0].path_len as usize];
    assert!(
        first_path.starts_with(b"/"),
        "first entry path should start with /"
    );

    let last = &entries[SCALE - 1];
    assert!(
        last.path_len > 0,
        "last entry should have a non-empty path"
    );

    println!(
        "Scale build OK: {} entries, {:.1} MB, generated in {:.2?}",
        entries.len(),
        expected_bytes as f64 / 1_048_576.0,
        gen_time,
    );
}

#[test]
#[ignore]
fn test_scale_1m_save_load_roundtrip() {
    let entries = generate_synthetic_entries(SCALE);
    let dir = TempDir::new().expect("temp dir");
    let idx_path = dir.path().join("scale_1m.idx");

    // Save
    println!("Saving {} entries to {:?}...", SCALE, idx_path);
    let t0 = Instant::now();
    save_v2(&entries, 0xDEAD_BEEF, &idx_path, 42, 0x1234).expect("save_v2 failed");
    let save_time = t0.elapsed();
    println!("Saved in {:.2?}", save_time);

    // Verify file size
    let file_len = std::fs::metadata(&idx_path).expect("metadata").len();
    let expected_len = 16384 + (SCALE * 256);
    assert_eq!(
        file_len, expected_len as u64,
        "file size mismatch: {} vs expected {}",
        file_len, expected_len,
    );

    // Load
    println!("Loading {} entries via load_v2...", SCALE);
    let t1 = Instant::now();
    let (header, loaded) = load_v2(&idx_path).expect("load_v2 failed");
    let load_time = t1.elapsed();
    println!("Loaded in {:.2?}", load_time);

    // Verify count
    assert_eq!(header.entry_count as usize, SCALE);
    assert_eq!(loaded.len(), SCALE);

    // Verify roundtrip: compare first, middle, and last entries byte-for-byte
    let check_indices = [0, SCALE / 2, SCALE - 1];
    for &idx in &check_indices {
        let orig = &entries[idx];
        let loaded_entry = &loaded[idx];

        let orig_bytes: &[u8; 256] =
            unsafe { &*(orig as *const GpuPathEntry as *const [u8; 256]) };
        let loaded_bytes: &[u8; 256] =
            unsafe { &*(loaded_entry as *const GpuPathEntry as *const [u8; 256]) };
        assert_eq!(
            orig_bytes, loaded_bytes,
            "entry {} mismatch in save/load roundtrip",
            idx,
        );
    }

    // Verify header fields
    assert_eq!(header.root_hash, 0xDEAD_BEEF);
    assert_eq!(header.last_fsevents_id, 42);
    assert_eq!(header.exclude_hash, 0x1234);

    println!(
        "Roundtrip OK: save={:.2?}, load={:.2?}, file={:.1}MB",
        save_time,
        load_time,
        file_len as f64 / 1_048_576.0,
    );
}

#[test]
#[ignore]
fn test_scale_1m_mmap_load() {
    let entries = generate_synthetic_entries(SCALE);
    let dir = TempDir::new().expect("temp dir");
    let idx_path = dir.path().join("scale_1m_mmap.idx");

    save_v2(&entries, 0xDEAD_BEEF, &idx_path, 99, 0xABCD).expect("save_v2 failed");

    // Load via mmap (IndexSnapshot without Metal device)
    println!("Loading {} entries via mmap (IndexSnapshot)...", SCALE);
    let t0 = Instant::now();
    let snapshot = IndexSnapshot::from_file(&idx_path, None).expect("mmap load failed");
    let mmap_time = t0.elapsed();
    println!("Mmap load in {:.2?}", mmap_time);

    // Verify entry count
    assert_eq!(snapshot.entry_count(), SCALE);

    // Access entries through mmap'd slice
    let mmap_entries = snapshot.entries();
    assert_eq!(mmap_entries.len(), SCALE);

    // Spot-check first, middle, last entries match originals
    let check_indices = [0, SCALE / 2, SCALE - 1];
    for &idx in &check_indices {
        let orig = &entries[idx];
        let mmap_entry = &mmap_entries[idx];

        assert_eq!(
            orig.path_len, mmap_entry.path_len,
            "entry {} path_len mismatch",
            idx,
        );
        assert_eq!(
            &orig.path[..orig.path_len as usize],
            &mmap_entry.path[..mmap_entry.path_len as usize],
            "entry {} path mismatch",
            idx,
        );
        assert_eq!(orig.flags, mmap_entry.flags, "entry {} flags mismatch", idx);
        assert_eq!(orig.mtime, mmap_entry.mtime, "entry {} mtime mismatch", idx);
    }

    // Verify header
    let header = snapshot.header();
    assert_eq!(header.entry_count as usize, SCALE);
    assert_eq!(header.root_hash, 0xDEAD_BEEF);
    assert_eq!(header.last_fsevents_id, 99);
    assert_eq!(header.exclude_hash, 0xABCD);

    println!(
        "Mmap load OK: {} entries, mmap_time={:.2?}",
        snapshot.entry_count(),
        mmap_time,
    );
}

#[test]
#[ignore]
fn test_scale_1m_find_by_name() {
    let entries = generate_synthetic_entries(SCALE);
    let dir = TempDir::new().expect("temp dir");
    let idx_path = dir.path().join("scale_1m_find.idx");

    save_v2(&entries, 0xDEAD_BEEF, &idx_path, 0, 0).expect("save_v2 failed");

    // Load via mmap
    let snapshot = IndexSnapshot::from_file(&idx_path, None).expect("mmap load failed");
    let mmap_entries = snapshot.entries();

    // Pick a known entry (entry at index 500000) and extract its path
    let target_idx = 500_000;
    let target = &entries[target_idx];
    let target_path = &target.path[..target.path_len as usize];

    // Scan through mmap'd entries to find it
    println!(
        "Scanning {} mmap'd entries for target at index {}...",
        SCALE, target_idx,
    );
    let t0 = Instant::now();
    let mut found_idx = None;
    for (i, entry) in mmap_entries.iter().enumerate() {
        let path = &entry.path[..entry.path_len as usize];
        if path == target_path {
            found_idx = Some(i);
            break;
        }
    }
    let scan_time = t0.elapsed();

    assert_eq!(
        found_idx,
        Some(target_idx),
        "expected to find target at index {}",
        target_idx,
    );

    // Verify the found entry matches the original
    let found = &mmap_entries[found_idx.unwrap()];
    assert_eq!(found.path_len, target.path_len);
    assert_eq!(found.flags, target.flags);
    assert_eq!(found.mtime, target.mtime);
    assert_eq!(found.size_lo, target.size_lo);
    assert_eq!(found.size_hi, target.size_hi);

    println!(
        "Find OK: found entry {} in {:.2?} (linear scan of {} entries)",
        target_idx, scan_time, SCALE,
    );
}
