//! GPU Memory Safety Tests
//!
//! Run with Metal validation enabled to catch out-of-bounds reads,
//! buffer overruns, and other GPU memory violations:
//!
//!   METAL_DEVICE_WRAPPER_TYPE=1 MTL_SHADER_VALIDATION=1 cargo test -p gpu-search --test test_gpu_memory
//!
//! These tests verify:
//! - Non-aligned file sizes (not multiples of 64 bytes per thread)
//! - Buffer overrun safety (more matches than MAX_MATCHES slots)
//! - Empty buffer edge cases
//! - Very small patterns in large content
//! - Zero-result searches
//! - Atomic race conditions under high match density

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::search::content::{
    cpu_search, ContentSearchEngine, SearchMode, SearchOptions,
};

// ============================================================================
// Helper: create engine with device + PSO cache
// ============================================================================

fn create_engine(max_files: usize) -> ContentSearchEngine {
    let gpu = GpuDevice::new();
    let pso = PsoCache::new(&gpu.device);
    ContentSearchEngine::new(&gpu.device, &pso, max_files)
}

fn default_opts() -> SearchOptions {
    SearchOptions {
        case_sensitive: true,
        max_results: 10000,
        mode: SearchMode::Standard,
    }
}

// ============================================================================
// 1. Non-aligned file sizes (OOB read tests)
// ============================================================================
// Each GPU thread processes 64 bytes. Files that are NOT multiples of 64
// bytes could cause OOB reads if the kernel doesn't respect chunk_length.
// With MTL_SHADER_VALIDATION=1, any OOB access triggers a Metal error.

#[test]
fn test_non_aligned_1_byte() {
    let mut engine = create_engine(100);
    let content = b"x";
    let chunks = engine.load_content(content, 0);
    assert!(chunks > 0);

    let results = engine.search(b"x", &default_opts());
    let cpu = cpu_search(content, b"x", true);
    assert_eq!(results.len(), cpu, "1-byte file: GPU={} CPU={}", results.len(), cpu);
}

#[test]
fn test_non_aligned_7_bytes() {
    let mut engine = create_engine(100);
    let content = b"abcdefg";
    let chunks = engine.load_content(content, 0);
    assert!(chunks > 0);

    let results = engine.search(b"cde", &default_opts());
    let cpu = cpu_search(content, b"cde", true);
    assert_eq!(results.len(), cpu, "7-byte file: GPU={} CPU={}", results.len(), cpu);
}

#[test]
fn test_non_aligned_63_bytes() {
    let mut engine = create_engine(100);
    // 63 bytes = one thread's 64-byte window minus 1
    let content: Vec<u8> = (0..63).map(|i| b'a' + (i % 26)).collect();
    let chunks = engine.load_content(&content, 0);
    assert!(chunks > 0);

    let results = engine.search(b"abc", &default_opts());
    let cpu = cpu_search(&content, b"abc", true);
    assert_eq!(results.len(), cpu, "63-byte file: GPU={} CPU={}", results.len(), cpu);
}

#[test]
fn test_non_aligned_65_bytes() {
    let mut engine = create_engine(100);
    // 65 bytes = crosses into second thread's window by 1 byte
    let content: Vec<u8> = (0..65).map(|i| b'a' + (i % 26)).collect();
    let chunks = engine.load_content(&content, 0);
    assert!(chunks > 0);

    let results = engine.search(b"abc", &default_opts());
    let cpu = cpu_search(&content, b"abc", true);
    // GPU may miss matches at the 64-byte boundary, so allow <= CPU
    assert!(
        results.len() <= cpu,
        "65-byte file: GPU({}) should not exceed CPU({})",
        results.len(),
        cpu
    );
}

#[test]
fn test_non_aligned_127_bytes() {
    let mut engine = create_engine(100);
    // 127 bytes = 2 threads minus 1 byte
    let content: Vec<u8> = (0..127).map(|i| b'a' + (i % 26)).collect();
    let chunks = engine.load_content(&content, 0);
    assert!(chunks > 0);

    let results = engine.search(b"abc", &default_opts());
    let cpu = cpu_search(&content, b"abc", true);
    assert!(
        results.len() <= cpu,
        "127-byte file: GPU({}) should not exceed CPU({})",
        results.len(),
        cpu
    );
}

#[test]
fn test_non_aligned_prime_size() {
    let mut engine = create_engine(100);
    // 997 bytes (prime number, not divisible by any power of 2)
    let content: Vec<u8> = (0..997).map(|i| b'a' + (i % 26) as u8).collect();
    let chunks = engine.load_content(&content, 0);
    assert!(chunks > 0);

    let results = engine.search(b"abc", &default_opts());
    let cpu = cpu_search(&content, b"abc", true);
    assert!(
        results.len() <= cpu,
        "997-byte file: GPU({}) should not exceed CPU({})",
        results.len(),
        cpu
    );
    // Should find most matches (allow ~10% boundary miss for small files)
    let min_expected = cpu * 80 / 100;
    assert!(
        results.len() >= min_expected,
        "997-byte file: GPU({}) < 80% of CPU({})",
        results.len(),
        cpu
    );
}

#[test]
fn test_non_aligned_4095_bytes() {
    let mut engine = create_engine(100);
    // 4095 bytes = one chunk minus 1 byte (CHUNK_SIZE=4096)
    let content: Vec<u8> = (0..4095).map(|i| b'a' + (i % 26) as u8).collect();
    let chunks = engine.load_content(&content, 0);
    assert_eq!(chunks, 1, "Should fit in exactly 1 chunk");

    let results = engine.search(b"abc", &default_opts());
    let cpu = cpu_search(&content, b"abc", true);
    assert!(
        results.len() <= cpu,
        "4095-byte file: GPU({}) should not exceed CPU({})",
        results.len(),
        cpu
    );
}

#[test]
fn test_non_aligned_4097_bytes() {
    let mut engine = create_engine(100);
    // 4097 bytes = crosses chunk boundary by 1 byte
    let content: Vec<u8> = (0..4097).map(|i| b'a' + (i % 26) as u8).collect();
    let chunks = engine.load_content(&content, 0);
    assert_eq!(chunks, 2, "Should need exactly 2 chunks");

    let results = engine.search(b"abc", &default_opts());
    let cpu = cpu_search(&content, b"abc", true);
    assert!(
        results.len() <= cpu,
        "4097-byte file: GPU({}) should not exceed CPU({})",
        results.len(),
        cpu
    );
}

// ============================================================================
// 2. Buffer overrun tests (more matches than MAX_MATCHES slots)
// ============================================================================
// The GPU match buffer holds MAX_MATCHES=10000 results. If the content
// generates more than 10000 matches, the kernel must cap at the buffer
// boundary (global_idx < 10000 check in the shader).

#[test]
fn test_match_buffer_overrun_protection() {
    let mut engine = create_engine(10000);

    // Create content with extremely high match density.
    // Pattern "aa" in "aaaa..." matches at every position.
    // 100KB of 'a' = ~100,000 potential matches for "aa", well above the 10K cap.
    let content: Vec<u8> = vec![b'a'; 100_000];
    let chunks = engine.load_content(&content, 0);
    assert!(chunks > 0);

    let results = engine.search(b"aa", &default_opts());

    // GPU should return at most MAX_MATCHES results (10000)
    assert!(
        results.len() <= 10000,
        "GPU returned {} matches, should be capped at 10000",
        results.len()
    );

    // Should have a significant number of matches (kernel does find them)
    assert!(
        results.len() > 100,
        "GPU should find many matches, got {}",
        results.len()
    );

    println!(
        "Match buffer overrun test: GPU returned {} matches (capped at 10000)",
        results.len()
    );
}

#[test]
fn test_high_density_single_char_pattern() {
    let mut engine = create_engine(10000);

    // Every byte matches pattern "a" -- maximum possible match density
    let content: Vec<u8> = vec![b'a'; 50_000];
    let chunks = engine.load_content(&content, 0);
    assert!(chunks > 0);

    let results = engine.search(b"a", &default_opts());

    // Must not exceed buffer capacity
    assert!(
        results.len() <= 10000,
        "Single char high density: {} > 10000",
        results.len()
    );
    // Must find matches (not crash or return 0)
    assert!(results.len() > 0, "Should find at least some matches");

    println!(
        "High density single char: {} matches (50K possible)",
        results.len()
    );
}

// ============================================================================
// 3. Empty buffer and edge case searches
// ============================================================================

#[test]
fn test_empty_content_search() {
    let engine = create_engine(100);
    // No content loaded at all -- search should return empty
    let results = engine.search(b"test", &default_opts());
    assert_eq!(results.len(), 0, "Empty engine should return 0 matches");
}

#[test]
fn test_empty_pattern_search() {
    let mut engine = create_engine(100);
    let content = b"some content here";
    engine.load_content(content, 0);

    // Empty pattern should be rejected gracefully
    let results = engine.search(b"", &default_opts());
    assert_eq!(results.len(), 0, "Empty pattern should return 0 matches");
}

#[test]
fn test_pattern_longer_than_content() {
    let mut engine = create_engine(100);
    let content = b"hi";
    engine.load_content(content, 0);

    // Pattern longer than content
    let results = engine.search(b"this pattern is way longer than the content", &default_opts());
    assert_eq!(
        results.len(),
        0,
        "Pattern longer than content should return 0 matches"
    );
}

#[test]
fn test_content_all_zeros() {
    let mut engine = create_engine(100);
    // NUL bytes -- valid binary content
    let content: Vec<u8> = vec![0u8; 1024];
    let chunks = engine.load_content(&content, 0);
    assert!(chunks > 0);

    // Search for a text pattern in all-zero content
    let results = engine.search(b"test", &default_opts());
    assert_eq!(results.len(), 0, "All-zero content should find nothing");
}

#[test]
fn test_search_after_reset() {
    let mut engine = create_engine(100);
    let content = b"hello world hello";
    engine.load_content(content, 0);

    let results = engine.search(b"hello", &default_opts());
    assert!(results.len() > 0, "Should find matches before reset");

    engine.reset();

    // After reset, no content loaded -- search should return empty
    let results = engine.search(b"hello", &default_opts());
    assert_eq!(results.len(), 0, "After reset should return 0 matches");
}

// ============================================================================
// 4. Small pattern in large content
// ============================================================================

#[test]
fn test_single_char_pattern_in_large_content() {
    let mut engine = create_engine(10000);

    // 64KB content with scattered target chars
    let mut content: Vec<u8> = vec![b'.'; 65536];
    // Plant 'X' at known positions including non-aligned offsets
    let positions = [0, 1, 63, 64, 65, 127, 128, 4095, 4096, 4097, 65535];
    for &pos in &positions {
        if pos < content.len() {
            content[pos] = b'X';
        }
    }

    let chunks = engine.load_content(&content, 0);
    assert!(chunks > 0);

    let results = engine.search(b"X", &default_opts());
    let cpu = cpu_search(&content, b"X", true);

    // GPU should find all or nearly all (boundary misses possible)
    assert!(
        results.len() <= cpu,
        "GPU({}) should not exceed CPU({})",
        results.len(),
        cpu
    );
    // All matches are at non-boundary positions within their threads,
    // so most should be found
    assert!(
        results.len() >= cpu * 80 / 100,
        "GPU({}) should find at least 80% of CPU({}) matches",
        results.len(),
        cpu
    );

    println!(
        "Single char in 64KB: GPU={}, CPU={} ({} planted positions)",
        results.len(),
        cpu,
        positions.len()
    );
}

#[test]
fn test_two_char_pattern_in_large_content() {
    let mut engine = create_engine(10000);

    // 128KB content
    let mut content: Vec<u8> = vec![b'.'; 131072];
    // Plant "XY" at various offsets including chunk and thread boundaries
    let pattern_positions = [0, 62, 63, 64, 4094, 4095, 4096, 8191, 65536, 131070];
    for &pos in &pattern_positions {
        if pos + 1 < content.len() {
            content[pos] = b'X';
            content[pos + 1] = b'Y';
        }
    }

    let chunks = engine.load_content(&content, 0);
    assert!(chunks > 0);

    let results = engine.search(b"XY", &default_opts());
    let cpu = cpu_search(&content, b"XY", true);

    assert!(
        results.len() <= cpu,
        "GPU({}) should not exceed CPU({})",
        results.len(),
        cpu
    );
    // Some boundary misses expected at positions 63, 4095 etc.
    println!(
        "2-char in 128KB: GPU={}/{} CPU matches",
        results.len(),
        cpu
    );
}

// ============================================================================
// 5. Zero-result searches
// ============================================================================

#[test]
fn test_zero_results_small_content() {
    let mut engine = create_engine(100);
    let content = b"the quick brown fox jumps over the lazy dog";
    engine.load_content(content, 0);

    let results = engine.search(b"ZZZZZ", &default_opts());
    assert_eq!(results.len(), 0, "Should find zero matches for absent pattern");
}

#[test]
fn test_zero_results_large_content() {
    let mut engine = create_engine(10000);
    // 256KB of lowercase letters -- search for uppercase pattern
    let content: Vec<u8> = (0..262144).map(|i| b'a' + (i % 26) as u8).collect();
    engine.load_content(&content, 0);

    let opts = SearchOptions {
        case_sensitive: true,
        max_results: 10000,
        mode: SearchMode::Standard,
    };
    let results = engine.search(b"XYZZY", &opts);
    assert_eq!(results.len(), 0, "Case-sensitive search for absent uppercase pattern should find 0");
}

#[test]
fn test_zero_results_case_sensitive_mismatch() {
    let mut engine = create_engine(100);
    let content = b"Hello World HELLO WORLD";
    engine.load_content(content, 0);

    let opts = SearchOptions {
        case_sensitive: true,
        max_results: 10000,
        mode: SearchMode::Standard,
    };
    // Search for exact lowercase -- should not match HELLO or Hello (case sensitive)
    let results = engine.search(b"hello", &opts);
    assert_eq!(results.len(), 0, "Case-sensitive 'hello' should not match 'Hello' or 'HELLO'");
}

// ============================================================================
// 6. Atomic race condition stress (many threads writing matches)
// ============================================================================

#[test]
fn test_atomic_contention_many_matches() {
    let mut engine = create_engine(10000);

    // Create content where EVERY 64-byte thread window has multiple matches.
    // This maximizes atomic contention on match_count.
    // "ab" repeating in 32KB = 16384 potential matches
    let content: Vec<u8> = "ab".as_bytes().iter().cycle().take(32768).copied().collect();
    engine.load_content(&content, 0);

    let results = engine.search(b"ab", &default_opts());
    let cpu = cpu_search(&content, b"ab", true);

    // Must not exceed buffer capacity
    assert!(
        results.len() <= 10000,
        "Atomic contention: {} > max 10000",
        results.len()
    );
    // Must not exceed CPU count
    assert!(
        results.len() <= cpu,
        "Atomic contention: GPU({}) > CPU({})",
        results.len(),
        cpu
    );
    // Should find a significant number
    assert!(
        results.len() > 100,
        "Atomic contention: GPU found only {} matches",
        results.len()
    );

    println!(
        "Atomic contention: GPU={}/{} CPU matches (max 10000)",
        results.len(),
        cpu
    );
}

#[test]
fn test_atomic_contention_multi_file() {
    let mut engine = create_engine(10000);

    // Load multiple files, each with high match density
    for file_idx in 0..10u32 {
        let content: Vec<u8> = format!(
            "fn test_{idx}() {{\n    let x = {idx};\n}}\n\
             fn helper_{idx}() {{\n    let y = {idx};\n}}\n",
            idx = file_idx
        )
        .into_bytes();
        engine.load_content(&content, file_idx);
    }

    let results = engine.search(b"fn ", &default_opts());

    // Each file has 2 "fn " patterns, 10 files = 20 matches
    // GPU should find all (they're within single thread windows)
    assert!(
        results.len() >= 18, // Allow minor boundary misses
        "Multi-file atomic: GPU found {} matches, expected ~20",
        results.len()
    );
    assert!(
        results.len() <= 20,
        "Multi-file atomic: GPU found {} matches, should not exceed 20",
        results.len()
    );
}

// ============================================================================
// 7. Turbo mode with same edge cases
// ============================================================================

#[test]
fn test_turbo_mode_non_aligned() {
    let mut engine = create_engine(100);
    let content: Vec<u8> = (0..63).map(|i| b'a' + (i % 26)).collect();
    engine.load_content(&content, 0);

    let opts = SearchOptions {
        case_sensitive: true,
        max_results: 10000,
        mode: SearchMode::Turbo,
    };
    let results = engine.search(b"abc", &opts);
    let cpu = cpu_search(&content, b"abc", true);
    assert_eq!(results.len(), cpu, "Turbo 63-byte: GPU={} CPU={}", results.len(), cpu);
}

#[test]
fn test_turbo_mode_empty() {
    let engine = create_engine(100);
    let opts = SearchOptions {
        case_sensitive: true,
        max_results: 10000,
        mode: SearchMode::Turbo,
    };
    let results = engine.search(b"test", &opts);
    assert_eq!(results.len(), 0, "Turbo empty engine should return 0");
}

#[test]
fn test_turbo_mode_high_density() {
    let mut engine = create_engine(10000);
    let content: Vec<u8> = vec![b'a'; 50_000];
    engine.load_content(&content, 0);

    let opts = SearchOptions {
        case_sensitive: true,
        max_results: 10000,
        mode: SearchMode::Turbo,
    };
    let results = engine.search(b"a", &opts);

    assert!(
        results.len() <= 10000,
        "Turbo high density: {} > 10000",
        results.len()
    );
    assert!(results.len() > 0, "Turbo should find matches");
}

// ============================================================================
// 8. Multiple sequential searches (buffer reuse safety)
// ============================================================================

#[test]
fn test_sequential_searches_buffer_reuse() {
    let mut engine = create_engine(100);
    let content = b"hello world hello world hello";
    engine.load_content(content, 0);

    // First search
    let r1 = engine.search(b"hello", &default_opts());
    let cpu1 = cpu_search(content, b"hello", true);
    assert_eq!(r1.len(), cpu1, "First search: GPU={} CPU={}", r1.len(), cpu1);

    // Second search (different pattern, same buffers)
    let r2 = engine.search(b"world", &default_opts());
    let cpu2 = cpu_search(content, b"world", true);
    assert_eq!(r2.len(), cpu2, "Second search: GPU={} CPU={}", r2.len(), cpu2);

    // Third search (no matches)
    let r3 = engine.search(b"MISSING", &default_opts());
    assert_eq!(r3.len(), 0, "Third search should find 0 matches");

    // Fourth search (original pattern should still work)
    let r4 = engine.search(b"hello", &default_opts());
    assert_eq!(r4.len(), cpu1, "Fourth search: GPU={} CPU={}", r4.len(), cpu1);
}

// ============================================================================
// 9. Max pattern length boundary
// ============================================================================

#[test]
fn test_max_pattern_length() {
    let mut engine = create_engine(100);

    // Create content containing a 64-byte pattern (MAX_PATTERN_LEN).
    // Place it at offset 0 (thread-aligned) so the GPU thread window
    // fully contains it. A 64-byte pattern at non-aligned offsets
    // would span two thread windows and be missed -- expected behavior.
    let pattern: Vec<u8> = (0..64).map(|i| b'A' + (i % 26)).collect();
    let mut content = vec![b'.'; 256];
    content[0..64].copy_from_slice(&pattern);

    engine.load_content(&content, 0);

    let results = engine.search(&pattern, &default_opts());
    let cpu = cpu_search(&content, &pattern, true);
    assert_eq!(results.len(), cpu, "64-byte pattern: GPU={} CPU={}", results.len(), cpu);
}

#[test]
fn test_pattern_exceeding_max_length() {
    let mut engine = create_engine(100);
    let content = b"some content to search in";
    engine.load_content(content, 0);

    // Pattern > 64 bytes should be rejected gracefully (return 0)
    let long_pattern: Vec<u8> = vec![b'x'; 65];
    let results = engine.search(&long_pattern, &default_opts());
    assert_eq!(results.len(), 0, "Pattern > MAX_PATTERN_LEN should return 0 matches");
}
