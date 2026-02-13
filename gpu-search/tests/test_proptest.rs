//! Property-based tests for GPU search engine using proptest.
//!
//! Verifies that GPU search results are consistent with CPU reference
//! implementation across randomly generated content and pattern pairs.
//!
//! GPU KERNEL CONSTRAINTS (accounted for in all assertions):
//! 1. **64-byte thread boundary**: Matches crossing a boundary may be missed
//! 2. **MAX_MATCHES_PER_THREAD = 4**: Per-thread cap on matches per window
//! 3. **MAX_MATCHES = 10,000**: Global cap on total match results
//! 4. **MAX_PATTERN_LEN = 64**: Pattern length limited to 64 bytes
//!
//! GPU MATCH RESULT FIELDS:
//! - byte_offset = chunk_index * 4096 + context_start + column (actual match position)
//! - column = offset of match within the line
//! - byte_offset already includes column, so it IS the match position
//! - Results sorted by (file_index, line_number, column)
//!
//! IMPORTANT: Each test creates ONE GPU engine and reuses it across all
//! proptest iterations to avoid exhausting Metal command queue resources.
//! RefCell provides interior mutability since TestRunner::run takes Fn (not FnMut).

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::search::content::{ContentSearchEngine, SearchMode, SearchOptions};
use gpu_search::search::orchestrator::SearchOrchestrator;
use gpu_search::search::types::SearchRequest;

use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::io::Write;

// ============================================================================
// Constants (must match GPU kernel limits)
// ============================================================================

const GPU_MAX_MATCHES_PER_THREAD: usize = 4;
const GPU_MAX_MATCHES: usize = 10000;
const GPU_BYTES_PER_THREAD: usize = 64;
const MAX_PATTERN_LEN: usize = 64;

/// Number of proptest iterations per property test.
const NUM_CASES: u32 = 1000;

// ============================================================================
// CPU Reference Implementation
// ============================================================================

/// CPU reference: returns byte offset positions for all matches.
fn cpu_search_positions(content: &[u8], pattern: &[u8], case_sensitive: bool) -> Vec<usize> {
    if pattern.is_empty() || content.len() < pattern.len() {
        return vec![];
    }

    let pattern_bytes: Vec<u8> = if case_sensitive {
        pattern.to_vec()
    } else {
        pattern.iter().map(|b| b.to_ascii_lowercase()).collect()
    };

    let mut positions = Vec::new();

    if pattern_bytes.len() == 1 {
        let needle = pattern_bytes[0];
        let haystack: Vec<u8> = if case_sensitive {
            content.to_vec()
        } else {
            content.iter().map(|b| b.to_ascii_lowercase()).collect()
        };
        let mut start = 0;
        while let Some(pos) = memchr::memchr(needle, &haystack[start..]) {
            positions.push(start + pos);
            start += pos + 1;
        }
        return positions;
    }

    let end = content.len() - pattern.len() + 1;
    for i in 0..end {
        let mut matched = true;
        for j in 0..pattern_bytes.len() {
            let a = if case_sensitive {
                content[i + j]
            } else {
                content[i + j].to_ascii_lowercase()
            };
            if a != pattern_bytes[j] {
                matched = false;
                break;
            }
        }
        if matched {
            positions.push(i);
        }
    }

    positions
}

/// Check if a match at `position` with `pattern_len` crosses a 64-byte boundary.
fn crosses_64byte_boundary(position: usize, pattern_len: usize) -> bool {
    let thread_start = (position / GPU_BYTES_PER_THREAD) * GPU_BYTES_PER_THREAD;
    let thread_end = thread_start + GPU_BYTES_PER_THREAD;
    position + pattern_len > thread_end
}

/// Filter CPU positions to only those NOT crossing a 64-byte boundary.
fn non_boundary_positions(positions: &[usize], pattern_len: usize) -> Vec<usize> {
    positions
        .iter()
        .copied()
        .filter(|&pos| !crosses_64byte_boundary(pos, pattern_len))
        .collect()
}

/// Compute expected GPU match count: groups by 64-byte window, caps per window.
fn expected_gpu_matches(positions: &[usize], pattern_len: usize) -> usize {
    let non_boundary = non_boundary_positions(positions, pattern_len);
    let mut windows: HashMap<usize, usize> = HashMap::new();
    for pos in &non_boundary {
        let window = pos / GPU_BYTES_PER_THREAD;
        *windows.entry(window).or_insert(0) += 1;
    }
    windows
        .values()
        .map(|&count| count.min(GPU_MAX_MATCHES_PER_THREAD))
        .sum::<usize>()
        .min(GPU_MAX_MATCHES)
}

// ============================================================================
// GPU Engine Helper
// ============================================================================

fn create_engine(max_files: usize) -> (ContentSearchEngine, GpuDevice) {
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);
    let engine = ContentSearchEngine::new(&device.device, &pso_cache, max_files);
    (engine, device)
}

fn default_options() -> SearchOptions {
    SearchOptions {
        case_sensitive: true,
        max_results: GPU_MAX_MATCHES,
        mode: SearchMode::Standard,
    }
}

// ============================================================================
// Proptest Strategies
// ============================================================================

/// Generate random ASCII printable content of 64-4096 bytes.
fn content_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(32u8..=126u8, 64..=4096)
}

/// Generate random ASCII printable patterns of 1-32 bytes.
fn pattern_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(32u8..=126u8, 1..=32)
}

/// Generate (content, pattern) pairs for generic tests.
fn content_pattern_strategy() -> impl Strategy<Value = (Vec<u8>, Vec<u8>)> {
    (content_strategy(), pattern_strategy())
}

/// Generate content that is guaranteed to contain the pattern.
fn content_with_pattern_strategy() -> impl Strategy<Value = (Vec<u8>, Vec<u8>)> {
    pattern_strategy().prop_flat_map(|pattern| {
        let pat_len = pattern.len();
        let prefix = prop::collection::vec(32u8..=126u8, 0..=512);
        let suffix = prop::collection::vec(32u8..=126u8, 0..=512);
        (prefix, Just(pattern), suffix).prop_map(move |(pre, pat, suf)| {
            let mut content = Vec::with_capacity(pre.len() + pat_len + suf.len());
            content.extend_from_slice(&pre);
            content.extend_from_slice(&pat);
            content.extend_from_slice(&suf);
            (content, pat)
        })
    })
}

// ============================================================================
// Property Tests
//
// Each test creates ONE GPU engine wrapped in RefCell and reuses it across
// all iterations. RefCell is needed because TestRunner::run takes Fn, not FnMut.
// ============================================================================

/// Property 1: GPU match count <= CPU match count (no false positives by count)
#[test]
fn prop_gpu_count_leq_cpu_count() {
    let (engine, _dev) = create_engine(10);
    let engine = RefCell::new(engine);
    let mut runner = TestRunner::new(Config {
        cases: NUM_CASES,
        ..Config::default()
    });

    runner
        .run(&content_pattern_strategy(), |(content, pattern)| {
            if pattern.len() > MAX_PATTERN_LEN || content.len() < pattern.len() {
                return Ok(());
            }

            let mut eng = engine.borrow_mut();
            eng.reset();
            eng.load_content(&content, 0);
            let gpu = eng.search(&pattern, &default_options());
            drop(eng);

            let cpu = cpu_search_positions(&content, &pattern, true);

            prop_assert!(
                gpu.len() <= cpu.len(),
                "GPU({}) > CPU({}) -- false positives! pattern={:?}",
                gpu.len(),
                cpu.len(),
                String::from_utf8_lossy(&pattern),
            );
            Ok(())
        })
        .unwrap();
}

/// Property 2: GPU results are sorted by (file_index, line_number, column)
///
/// The engine sorts results by file_index, then line_number, then column.
/// This is the sort order that matters for display/consumption.
#[test]
fn prop_gpu_positions_monotonic() {
    let (engine, _dev) = create_engine(10);
    let engine = RefCell::new(engine);
    let mut runner = TestRunner::new(Config {
        cases: NUM_CASES,
        ..Config::default()
    });

    runner
        .run(&content_pattern_strategy(), |(content, pattern)| {
            if pattern.len() > MAX_PATTERN_LEN || content.len() < pattern.len() {
                return Ok(());
            }

            let mut eng = engine.borrow_mut();
            eng.reset();
            eng.load_content(&content, 0);
            let gpu = eng.search(&pattern, &default_options());
            drop(eng);

            // Results are sorted by (file_index, line_number, column)
            for i in 1..gpu.len() {
                let prev = &gpu[i - 1];
                let curr = &gpu[i];

                let prev_key = (prev.file_index, prev.line_number, prev.column);
                let curr_key = (curr.file_index, curr.line_number, curr.column);

                prop_assert!(
                    curr_key >= prev_key,
                    "Not sorted: match[{}]=({},{},{}) < match[{}]=({},{},{})",
                    i,
                    curr.file_index,
                    curr.line_number,
                    curr.column,
                    i - 1,
                    prev.file_index,
                    prev.line_number,
                    prev.column,
                );
            }
            Ok(())
        })
        .unwrap();
}

/// Property 3: No false positives -- every GPU match corresponds to a real match
///
/// The actual match position is byte_offset (which now includes column).
/// Verify the pattern exists at that position.
#[test]
fn prop_no_false_positives() {
    let (engine, _dev) = create_engine(10);
    let engine = RefCell::new(engine);
    let mut runner = TestRunner::new(Config {
        cases: NUM_CASES,
        ..Config::default()
    });

    runner
        .run(&content_pattern_strategy(), |(content, pattern)| {
            if pattern.len() > MAX_PATTERN_LEN || content.len() < pattern.len() {
                return Ok(());
            }

            let mut eng = engine.borrow_mut();
            eng.reset();
            eng.load_content(&content, 0);
            let gpu = eng.search(&pattern, &default_options());
            drop(eng);

            let cpu_positions = cpu_search_positions(&content, &pattern, true);

            // Every GPU match must correspond to a real CPU match position.
            // byte_offset now includes column, so it IS the actual match position.
            for m in &gpu {
                let match_pos = m.byte_offset as usize;

                // Verify this position is in the CPU match set
                prop_assert!(
                    cpu_positions.contains(&match_pos),
                    "False positive: GPU match at byte_offset={} not in CPU positions ({} total)",
                    m.byte_offset,
                    cpu_positions.len(),
                );
            }
            Ok(())
        })
        .unwrap();
}

/// Property 4: Idempotent -- same input produces same output twice
///
/// Compares match count and sorted (file_index, line_number, column) tuples
/// since byte_offset ordering may vary with GPU thread scheduling.
#[test]
fn prop_idempotent() {
    let (engine, _dev) = create_engine(10);
    let engine = RefCell::new(engine);
    let mut runner = TestRunner::new(Config {
        cases: NUM_CASES,
        ..Config::default()
    });

    runner
        .run(&content_pattern_strategy(), |(content, pattern)| {
            if pattern.len() > MAX_PATTERN_LEN || content.len() < pattern.len() {
                return Ok(());
            }

            let mut eng = engine.borrow_mut();
            eng.reset();
            eng.load_content(&content, 0);
            let opts = default_options();
            let gpu1 = eng.search(&pattern, &opts);
            let gpu2 = eng.search(&pattern, &opts);
            drop(eng);

            prop_assert_eq!(
                gpu1.len(),
                gpu2.len(),
                "Idempotency: run1={}, run2={}",
                gpu1.len(),
                gpu2.len(),
            );

            // Compare sorted tuples (results are already sorted by engine)
            for (i, (m1, m2)) in gpu1.iter().zip(gpu2.iter()).enumerate() {
                let k1 = (m1.file_index, m1.line_number, m1.column);
                let k2 = (m2.file_index, m2.line_number, m2.column);
                prop_assert_eq!(
                    k1,
                    k2,
                    "Idempotency: match[{}] ({:?}) != ({:?})",
                    i,
                    k1,
                    k2,
                );
            }
            Ok(())
        })
        .unwrap();
}

/// Property 5: GPU finds at least as many as expected (no worse than predicted)
#[test]
fn prop_gpu_geq_expected() {
    let (engine, _dev) = create_engine(10);
    let engine = RefCell::new(engine);
    let mut runner = TestRunner::new(Config {
        cases: NUM_CASES,
        ..Config::default()
    });

    runner
        .run(&content_pattern_strategy(), |(content, pattern)| {
            if pattern.len() > MAX_PATTERN_LEN || content.len() < pattern.len() {
                return Ok(());
            }

            let mut eng = engine.borrow_mut();
            eng.reset();
            eng.load_content(&content, 0);
            let gpu = eng.search(&pattern, &default_options());
            drop(eng);

            let cpu_positions = cpu_search_positions(&content, &pattern, true);
            let expected = expected_gpu_matches(&cpu_positions, pattern.len());

            prop_assert!(
                gpu.len() >= expected,
                "GPU({}) < expected({}) -- worse than predicted! CPU={}, pattern={:?}",
                gpu.len(),
                expected,
                cpu_positions.len(),
                String::from_utf8_lossy(&pattern),
            );
            Ok(())
        })
        .unwrap();
}

/// Property 6: No false negatives for non-boundary matches
/// (content guaranteed to contain the pattern)
#[test]
fn prop_no_false_negatives_guaranteed_match() {
    let (engine, _dev) = create_engine(10);
    let engine = RefCell::new(engine);
    let mut runner = TestRunner::new(Config {
        cases: NUM_CASES,
        ..Config::default()
    });

    runner
        .run(
            &content_with_pattern_strategy(),
            |(content, pattern)| {
                if pattern.len() > MAX_PATTERN_LEN || content.len() < pattern.len() {
                    return Ok(());
                }

                let mut eng = engine.borrow_mut();
                eng.reset();
                eng.load_content(&content, 0);
                let gpu = eng.search(&pattern, &default_options());
                drop(eng);

                let cpu_positions = cpu_search_positions(&content, &pattern, true);

                // CPU must find at least 1 match (we embedded the pattern)
                prop_assert!(
                    !cpu_positions.is_empty(),
                    "CPU should find at least 1 match in content with embedded pattern"
                );

                // If any CPU match does NOT cross a 64-byte boundary, GPU must find it
                let non_boundary = non_boundary_positions(&cpu_positions, pattern.len());
                if !non_boundary.is_empty() {
                    prop_assert!(
                        !gpu.is_empty(),
                        "GPU found 0 but {} non-boundary CPU matches exist, pattern={:?}",
                        non_boundary.len(),
                        String::from_utf8_lossy(&pattern),
                    );
                }
                Ok(())
            },
        )
        .unwrap();
}

// ============================================================================
// Property 7: Pipeline-level accuracy via SearchOrchestrator
//
// For any (corpus, pattern) pair where the pattern is a substring drawn from
// one of the corpus files, every ContentMatch.line_content must contain the
// pattern. This tests the full pipeline: walk -> GPU dispatch -> resolve_match.
// ============================================================================

/// Strategy: generate 1-10 files of random ASCII content (64-4096 bytes each),
/// then pick a 2-16 byte substring from one file as the search pattern.
fn corpus_with_substring_pattern() -> impl Strategy<Value = (Vec<Vec<u8>>, String)> {
    // Generate 1-10 files of 64-4096 bytes of printable ASCII, then pick a
    // 2-16 byte substring from one file as the search pattern.
    prop::collection::vec(64usize..=4096, 1..=10).prop_flat_map(|sizes| {
        // Generate actual file contents matching the requested sizes
        let file_strats: Vec<_> = sizes
            .iter()
            .map(|&sz| prop::collection::vec(32u8..=126u8, sz..=sz))
            .collect();
        file_strats
            .prop_flat_map(|files: Vec<Vec<u8>>| {
                let num_files = files.len();
                let file_idx = 0..num_files;
                let pat_len = 2usize..=16;
                (Just(files), file_idx, pat_len)
            })
            .prop_flat_map(|(files, fidx, plen)| {
                let file_len = files[fidx].len();
                let max_start = if file_len > plen { file_len - plen } else { 0 };
                (Just(files), Just(fidx), Just(plen), 0..=max_start)
            })
            .prop_map(|(files, fidx, plen, start)| {
                let pattern_bytes = &files[fidx][start..start + plen];
                let pattern = String::from_utf8_lossy(pattern_bytes).to_string();
                (files, pattern)
            })
    })
}

#[test]
fn prop_pipeline_accuracy_no_false_positives() {
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);
    let orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
        .expect("Failed to create SearchOrchestrator");
    let orchestrator = RefCell::new(orchestrator);

    let mut runner = TestRunner::new(Config {
        cases: 100,
        ..Config::default()
    });

    runner
        .run(&corpus_with_substring_pattern(), |(files, pattern)| {
            // Skip empty or whitespace-only patterns
            if pattern.trim().is_empty() || pattern.len() < 2 {
                return Ok(());
            }

            // Create temp directory with corpus files
            let dir = tempfile::TempDir::new().expect("create temp dir");
            for (i, content) in files.iter().enumerate() {
                let path = dir.path().join(format!("file_{:03}.txt", i));
                let mut f = fs::File::create(&path).expect("create file");
                f.write_all(content).expect("write file");
            }

            let request = SearchRequest {
                pattern: pattern.clone(),
                root: dir.path().to_path_buf(),
                file_types: None,
                case_sensitive: false,
                respect_gitignore: false,
                include_binary: false,
                max_results: 10_000,
            };

            let mut orch = orchestrator.borrow_mut();
            let response = orch.search(request);
            drop(orch);

            // Core property: every ContentMatch.line_content contains the pattern
            let pat_lower = pattern.to_lowercase();
            for cm in &response.content_matches {
                let line_lower = cm.line_content.to_lowercase();
                prop_assert!(
                    line_lower.contains(&pat_lower),
                    "FALSE POSITIVE: pattern='{}' file={:?} line={} content='{}'",
                    pattern,
                    cm.path.file_name().unwrap_or_default().to_string_lossy(),
                    cm.line_number,
                    cm.line_content.trim(),
                );
            }

            Ok(())
        })
        .unwrap();
}
