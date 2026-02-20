//! Index load/save benchmarks: mmap load, first-entry access, and save performance.
//!
//! Targets (Apple Silicon):
//!   - mmap load 100k entries: <1ms
//!   - mmap load 1M entries: <5ms
//!   - first entry access after mmap: <0.1ms
//!   - save 100k entries: <100ms
//!   - save 1M entries: <1s

use criterion::{criterion_group, criterion_main, Criterion};
use std::path::Path;
use tempfile::TempDir;

use gpu_search::gpu::types::{path_flags, GpuPathEntry, GPU_PATH_MAX_LEN};
use gpu_search::index::gsix_v2::{load_v2, save_v2};
use gpu_search::index::snapshot::IndexSnapshot;

// ---------------------------------------------------------------------------
// Local synthetic entry generator (benches cannot access #[cfg(test)] modules)
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

// ---------------------------------------------------------------------------
// Helper: save entries to a temp file for load benchmarks
// ---------------------------------------------------------------------------

fn save_to_temp(entries: &[GpuPathEntry], dir: &Path) -> std::path::PathBuf {
    let idx_path = dir.join("bench.idx");
    save_v2(entries, 0xDEAD_BEEF, &idx_path, 42, 0).expect("save_v2 failed");
    idx_path
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_mmap_load_100k(c: &mut Criterion) {
    let entries = generate_synthetic_entries(100_000);
    let dir = TempDir::new().expect("temp dir");
    let idx_path = save_to_temp(&entries, dir.path());

    c.bench_function("mmap_load_100k", |b| {
        b.iter(|| {
            let snap = IndexSnapshot::from_file(&idx_path, None).expect("load failed");
            assert_eq!(snap.entry_count(), 100_000);
        });
    });
}

fn bench_mmap_load_1m(c: &mut Criterion) {
    let entries = generate_synthetic_entries(1_000_000);
    let dir = TempDir::new().expect("temp dir");
    let idx_path = save_to_temp(&entries, dir.path());

    c.bench_function("mmap_load_1m", |b| {
        b.iter(|| {
            let snap = IndexSnapshot::from_file(&idx_path, None).expect("load failed");
            assert_eq!(snap.entry_count(), 1_000_000);
        });
    });
}

fn bench_mmap_first_entry_access(c: &mut Criterion) {
    let entries = generate_synthetic_entries(100_000);
    let dir = TempDir::new().expect("temp dir");
    let idx_path = save_to_temp(&entries, dir.path());

    c.bench_function("mmap_first_entry_access", |b| {
        b.iter(|| {
            let snap = IndexSnapshot::from_file(&idx_path, None).expect("load failed");
            let e = &snap.entries()[0];
            // Force the compiler to actually read the entry
            assert!(e.path_len > 0);
        });
    });
}

fn bench_save_index_100k(c: &mut Criterion) {
    let entries = generate_synthetic_entries(100_000);
    let dir = TempDir::new().expect("temp dir");
    let idx_path = dir.path().join("save_bench.idx");

    c.bench_function("save_index_100k", |b| {
        b.iter(|| {
            save_v2(&entries, 0xDEAD_BEEF, &idx_path, 42, 0).expect("save failed");
        });
    });
}

fn bench_save_index_1m(c: &mut Criterion) {
    let entries = generate_synthetic_entries(1_000_000);
    let dir = TempDir::new().expect("temp dir");
    let idx_path = dir.path().join("save_bench.idx");

    c.bench_function("save_index_1m", |b| {
        b.iter(|| {
            save_v2(&entries, 0xDEAD_BEEF, &idx_path, 42, 0).expect("save failed");
        });
    });
}

criterion_group!(
    benches,
    bench_mmap_load_100k,
    bench_mmap_load_1m,
    bench_mmap_first_entry_access,
    bench_save_index_100k,
    bench_save_index_1m,
);
criterion_main!(benches);
