---
id: path-utils.BREAKDOWN
module: path-utils
priority: 3
status: failing
version: 1
origin: spec-workflow
dependsOn: []
tags: [utility, path, formatting]
testRequirements:
  unit:
    required: true
    pattern: "src/ui/path_utils.rs::tests"
---
# Path Utils Breakdown

## Context

File paths in gpu-search currently display as full absolute paths like `/Users/patrickkavanagh/Library/Application Support/...`, consuming nearly the entire row width and pushing filenames off-screen. At 720px window width with monospace 12px, only ~97 characters fit, and many paths exceed this entirely. The path abbreviation module provides a shared utility used by both `results_list.rs` (file match rows and group headers) and `status_bar.rs` (search root display).

The module implements a three-step priority algorithm: (1) relative to search root, (2) home directory `~` substitution, (3) middle truncation for paths exceeding 50 characters. Home directory is detected via `$HOME` environment variable (no `dirs` crate dependency) and cached with `OnceCell`.

From UX.md Section 4, TECH.md Section 5, PM P1-1, TECH Q2, TECH Q4.

## Acceptance Criteria

1. New file `src/ui/path_utils.rs` created with `pub fn abbreviate_path(path: &Path, search_root: &Path) -> (String, String)` returning `(dir_display, filename)`
2. `pub mod path_utils;` added to `src/ui/mod.rs`
3. Step 1: Paths under search root become relative (strip root prefix, add trailing `/`)
4. Step 2: Paths under `$HOME` get `~` substitution when not under search root
5. Step 3: Directory portions exceeding 50 chars are middle-truncated with `...`
6. `fn middle_truncate(s: &str, max_len: usize) -> String` helper function implemented
7. Home directory detected via `std::env::var("HOME")` with `OnceCell<PathBuf>` caching (no `dirs` crate)
8. Unit test U-PATH-1 passes: relative to search root (`src/`, `main.rs`)
9. Unit test U-PATH-2 passes: file at root level (dir=`""`, filename=`main.rs`)
10. Unit test U-PATH-3 passes: home substitution for paths outside search root
11. Unit test U-PATH-5 passes: paths outside both root and home use absolute path
12. Unit test U-PATH-6 passes: long directory middle-truncated
13. Unit test U-PATH-8 passes: path with no parent directory
14. Unit test U-PATH-9 passes: Unicode path segments (no panic)
15. Unit test U-PATH-11 passes: reads `$HOME` env var correctly
16. Unit test U-PATH-12 passes: `$HOME` unset falls back to absolute path (no panic)
17. All existing tests pass: `cargo test -p gpu-search`

## Technical Notes

- Reference: [spec/OVERVIEW.md] path-utils is priority 3, no dependencies, used by grouped-results and status-bar
- UX: From UX.md Section 4.1 -- three-step priority algorithm; Section 4.3 -- display format varies by context
- Test: From QA.md Section 2.2 (U-PATH-1..12) -- comprehensive path abbreviation tests including edge cases
- Width budget: directory portion gets 15-30 chars (~108-216px) in the row layout (UX.md Section 4.4)
- The function returns a tuple `(dir, filename)` so the renderer can style them independently
- Tooltip on hover shows full absolute path (no abbreviation); clipboard copy also uses full path
- Status bar root uses this function for `~` substitution: `/Users/pk/gpu-search` -> `~/gpu-search`
