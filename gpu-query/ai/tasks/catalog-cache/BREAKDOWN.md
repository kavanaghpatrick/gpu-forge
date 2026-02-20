---
id: catalog-cache.BREAKDOWN
module: catalog-cache
priority: 2
status: failing
version: 1
origin: spec-workflow
dependsOn: []
tags: [performance, gpu-query, io]
testRequirements:
  unit:
    required: true
    pattern: "tests/**/*.rs"
---
# CatalogCache with Fingerprinting + Schema/Dictionary Caching

## Context

`scan_directory()` is called on every query execution (`tui/ui.rs:345`, `tui/event.rs:215`, `tui/mod.rs:65`). It performs `read_dir()` + `format_detect::detect_format()` (reads magic bytes) + CSV header parsing for every file in the data directory. On a directory with 10 CSV files, this costs ~1-3ms per query -- work that is 100% redundant when files haven't changed.

Additionally, `infer_schema_from_csv()` (executor.rs:3020-3101) re-reads 100 rows and runs type-inference voting on every scan. The schema for a given CSV is deterministic and never changes between queries unless the file is modified.

VARCHAR dictionaries (`build_csv_dictionaries()` at executor.rs:3109-3182) re-read the entire CSV on CPU after the GPU has already parsed it, costing 40-80ms for 1M rows. These dictionaries are also deterministic per file.

This module introduces a `CatalogCache` with `(dir_mtime, per-file size+mtime)` fingerprinting, caches `RuntimeSchema` per table entry, and caches VARCHAR dictionaries per table+column. Cache invalidation uses a single `stat()` per file (~0.1ms total for 10 files).

## Acceptance Criteria

1. `CatalogCache` stores `Vec<TableEntry>` and returns cached entries when directory and file fingerprints are unchanged
2. Cache invalidation triggers automatically when: (a) directory mtime changes (file added/removed), (b) any file's size or mtime changes
3. Cache hit cost is <0.2ms for a directory with 10 files (measured via benchmark)
4. Manual invalidation via `.refresh` dot command calls `cache.invalidate()` and forces a full rescan
5. `RuntimeSchema` is cached per table (keyed by file path + size + mtime) and reused across queries -- `infer_schema_from_csv()` runs at most once per file per session (until file changes)
6. VARCHAR dictionaries are cached with the scan result -- `build_csv_dictionaries()` runs at most once per file per session
7. Cached schema is byte-identical to freshly inferred schema (invariant test)
8. Cached catalog entries match freshly scanned entries (invariant test)
9. All 8 existing catalog unit tests (`src/io/catalog.rs`) pass unchanged
10. All 14 existing schema inference tests pass unchanged
11. Adding a new file to the data directory between queries makes the new table visible on the next query
12. Deleting a file between queries produces a graceful error (not a panic) when querying the removed table

## Technical Notes

- **Reference**: OVERVIEW.md Module Roadmap priority 2; TECH.md Section 4 (Fix B5) + Section 6 (Fix B4 integration); PM.md Section 3.2 BUG #4 and #5; QA.md Section 3 Bottleneck 4-5
- **Files to create**:
  - `gpu-query/src/io/catalog_cache.rs` -- `CatalogCache` struct with `get_or_refresh()`, `is_valid()`, `refresh()`, `invalidate()` methods; `FileFingerprint` struct with `(size: u64, modified: SystemTime)`
- **Files to modify**:
  - `gpu-query/src/io/mod.rs` -- Add `pub mod catalog_cache;`
  - `gpu-query/src/tui/app.rs` -- Add `pub catalog_cache: CatalogCache` field; initialize in `AppState::new()`
  - `gpu-query/src/tui/ui.rs:345` -- Replace `scan_directory(&app.data_dir)` with `app.catalog_cache.get_or_refresh()`
  - `gpu-query/src/tui/event.rs:215` -- Same replacement for DESCRIBE path
  - `gpu-query/src/tui/mod.rs:65` -- Use `catalog_cache` for initial catalog population
  - `gpu-query/src/gpu/executor.rs` -- Extend `TableEntry` with `cached_schema: Option<RuntimeSchema>` or use a separate schema HashMap in the cache
- **Cache key design**:
  - Catalog: `(dir_path, dir_mtime)` for directory-level check; `(file_path, file_size, file_mtime)` per entry
  - Schema: Embedded in catalog entry or keyed by `(file_path, file_size, file_mtime)`
  - Dictionary: Cached within `ColumnarBatch` in the ScanResult (handled by executor-cache module)
- **Test**: `cargo test --all-targets`; new unit tests for cache hit/miss/invalidation scenarios; `cargo bench --bench scan_throughput -- "catalog"` for performance
