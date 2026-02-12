//! GPU-CPU dual verification tests.
//!
//! Every GPU search result is verified against a CPU reference implementation.
//! The CPU reference uses memchr for single-byte fast paths and manual
//! byte-by-byte scanning for multi-byte patterns.
//!
//! KNOWN GPU KERNEL LIMITATIONS:
//! 1. **64-byte thread boundary**: Each GPU thread processes 64 bytes. Matches
//!    that span across a boundary will be missed. This is a fundamental
//!    throughput trade-off (79-110 GB/s on M4 Pro).
//! 2. **MAX_MATCHES_PER_THREAD = 4**: Each thread can report at most 4 matches
//!    within its 64-byte window. Dense match patterns (e.g., single-char search
//!    in repetitive content) will be capped per-thread.
//! 3. **MAX_MATCHES = 10,000**: Global match buffer caps total results at 10K.
//!    Tests with very large files account for this cap.
//!
//! Tests are structured to verify:
//! - Exact match when constraints are not violated (few matches per window)
//! - No false positives (GPU never reports matches CPU doesn't find)
//! - Bounded miss rate for real-world content

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::search::content::{
    cpu_search, ContentSearchEngine, SearchMode, SearchOptions,
};

/// Max matches per GPU thread (must match search_types.h MAX_MATCHES_PER_THREAD).
const GPU_MAX_MATCHES_PER_THREAD: usize = 4;

/// Max global matches (must match content.rs MAX_MATCHES).
const GPU_MAX_MATCHES: usize = 10000;

/// GPU thread window size in bytes.
const GPU_BYTES_PER_THREAD: usize = 64;

// ============================================================================
// CPU Reference Implementation (position-aware)
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

    // Use memchr for single-byte patterns (fast path)
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

    // Multi-byte: manual scan
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

/// Checks whether a match at `position` with `pattern_len` crosses a 64-byte
/// thread boundary.
fn crosses_64byte_boundary(position: usize, pattern_len: usize) -> bool {
    let thread_start = (position / GPU_BYTES_PER_THREAD) * GPU_BYTES_PER_THREAD;
    let thread_end = thread_start + GPU_BYTES_PER_THREAD;
    position + pattern_len > thread_end
}

/// Filter CPU positions to only those that do NOT cross a 64-byte boundary.
fn non_boundary_positions(positions: &[usize], pattern_len: usize) -> Vec<usize> {
    positions
        .iter()
        .copied()
        .filter(|&pos| !crosses_64byte_boundary(pos, pattern_len))
        .collect()
}

/// Compute the maximum GPU matches expected given per-thread limits.
/// Groups matches by 64-byte window and caps each window at MAX_MATCHES_PER_THREAD.
fn expected_gpu_matches(positions: &[usize], pattern_len: usize) -> usize {
    // Only count non-boundary matches, grouped by thread window, capped at 4 per window
    let non_boundary = non_boundary_positions(positions, pattern_len);
    let mut windows: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
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
// Helper: initialize GPU engine
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

fn case_insensitive_options() -> SearchOptions {
    SearchOptions {
        case_sensitive: false,
        max_results: GPU_MAX_MATCHES,
        mode: SearchMode::Standard,
    }
}

// ============================================================================
// Test Matrix: Literal Search -- Single Match
// ============================================================================

#[test]
fn test_single_match_at_start() {
    let (mut engine, _dev) = create_engine(10);
    let content = b"hello world, this is a test string with some padding bytes";
    engine.load_content(content, 0);

    let gpu = engine.search(b"hello", &default_options());
    let cpu = cpu_search_positions(content, b"hello", true);

    assert_eq!(gpu.len(), cpu.len(), "GPU({}) != CPU({})", gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 1);
}

#[test]
fn test_single_match_at_end() {
    let (mut engine, _dev) = create_engine(10);
    // Place the match within the last 64-byte window
    let mut content = vec![b'x'; 50];
    content.extend_from_slice(b"FOUND");
    engine.load_content(&content, 0);

    let gpu = engine.search(b"FOUND", &default_options());
    let cpu = cpu_search_positions(&content, b"FOUND", true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 1);
}

#[test]
fn test_single_match_mid_content() {
    let (mut engine, _dev) = create_engine(10);
    let content = b"aaaaaa TARGET bbbbbb";
    engine.load_content(content, 0);

    let gpu = engine.search(b"TARGET", &default_options());
    let cpu = cpu_search_positions(content, b"TARGET", true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 1);
}

// ============================================================================
// Test Matrix: Literal Search -- Multiple Matches
// ============================================================================

#[test]
fn test_multiple_matches_few_per_window() {
    let (mut engine, _dev) = create_engine(10);
    // 4 matches within first 64-byte window (at or below per-thread limit)
    let content = b"ab cd ab cd ab cd ab cd xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
    engine.load_content(content, 0);

    let gpu = engine.search(b"ab", &default_options());
    let cpu = cpu_search_positions(content, b"ab", true);
    let expected = expected_gpu_matches(&cpu, 2);

    assert_eq!(
        gpu.len(),
        expected,
        "GPU({}) != expected({}) [CPU={}]",
        gpu.len(),
        expected,
        cpu.len()
    );
}

#[test]
fn test_multiple_matches_across_windows() {
    let (mut engine, _dev) = create_engine(10);

    // Place exactly 1 match per 64-byte window (well below per-thread limit)
    let mut content = Vec::new();
    for i in 0..10 {
        let mut block = vec![b' '; 64];
        block[0] = b'M';
        block[1] = b'A';
        block[2] = b'R';
        block[3] = b'K';
        block[5] = b'0' + (i as u8);
        content.extend_from_slice(&block);
    }
    engine.load_content(&content, 0);

    let gpu = engine.search(b"MARK", &default_options());
    let cpu = cpu_search_positions(&content, b"MARK", true);
    let expected = expected_gpu_matches(&cpu, 4);

    assert_eq!(gpu.len(), expected);
    assert_eq!(gpu.len(), 10);
}

#[test]
fn test_multiple_matches_multi_file() {
    let (mut engine, _dev) = create_engine(100);

    let file0 = b"fn main() { println!(\"ok\"); }\n";
    let file1 = b"fn test_a() { }\nfn test_b() { }\n";
    let file2 = b"no functions here\n";
    engine.load_content(file0, 0);
    engine.load_content(file1, 1);
    engine.load_content(file2, 2);

    let gpu = engine.search(b"fn ", &default_options());
    let cpu0 = cpu_search(file0, b"fn ", true);
    let cpu1 = cpu_search(file1, b"fn ", true);
    let cpu2 = cpu_search(file2, b"fn ", true);
    let total_cpu = cpu0 + cpu1 + cpu2;

    assert_eq!(gpu.len(), total_cpu);
    assert_eq!(gpu.len(), 3);

    // Verify file_index assignment
    let f0: Vec<_> = gpu.iter().filter(|m| m.file_index == 0).collect();
    let f1: Vec<_> = gpu.iter().filter(|m| m.file_index == 1).collect();
    let f2: Vec<_> = gpu.iter().filter(|m| m.file_index == 2).collect();
    assert_eq!(f0.len(), 1, "file 0 should have 1 match");
    assert_eq!(f1.len(), 2, "file 1 should have 2 matches");
    assert_eq!(f2.len(), 0, "file 2 should have 0 matches");
}

// ============================================================================
// Test Matrix: Overlapping Matches
// ============================================================================

#[test]
fn test_overlapping_pattern_aa_in_aaa() {
    let (mut engine, _dev) = create_engine(10);
    let content = b"aaa";
    engine.load_content(content, 0);

    let gpu = engine.search(b"aa", &default_options());
    let cpu = cpu_search_positions(content, b"aa", true);

    assert_eq!(cpu.len(), 2, "CPU should find 2 overlapping 'aa' in 'aaa'");
    // Both matches within one 64-byte window, count <= MAX_MATCHES_PER_THREAD(4)
    assert_eq!(gpu.len(), cpu.len());
}

#[test]
fn test_overlapping_pattern_aba_in_ababa() {
    let (mut engine, _dev) = create_engine(10);
    let content = b"ababa";
    engine.load_content(content, 0);

    let gpu = engine.search(b"aba", &default_options());
    let cpu = cpu_search_positions(content, b"aba", true);

    assert_eq!(cpu.len(), 2, "CPU should find 2 overlapping 'aba' in 'ababa'");
    assert_eq!(gpu.len(), cpu.len());
}

#[test]
fn test_overlapping_capped_by_per_thread_limit() {
    let (mut engine, _dev) = create_engine(10);
    // Many overlapping matches in one 64-byte window -- will be capped at
    // MAX_MATCHES_PER_THREAD (4) by the GPU kernel
    let content = vec![b'a'; 32];
    engine.load_content(&content, 0);

    let gpu = engine.search(b"aa", &default_options());
    let cpu = cpu_search_positions(&content, b"aa", true);
    let expected = expected_gpu_matches(&cpu, 2);

    assert_eq!(cpu.len(), 31, "CPU should find 31 overlapping 'aa' in 32 a's");
    // GPU is capped at MAX_MATCHES_PER_THREAD per window
    assert_eq!(
        gpu.len(),
        expected,
        "GPU({}) != expected({}) [CPU={}]",
        gpu.len(),
        expected,
        cpu.len()
    );
    // No false positives
    assert!(gpu.len() <= cpu.len());
    println!(
        "Overlapping long run: GPU={} (capped), CPU={} (uncapped)",
        gpu.len(),
        cpu.len()
    );
}

// ============================================================================
// Test Matrix: Case Insensitive
// ============================================================================

#[test]
fn test_case_insensitive_mixed() {
    let (mut engine, _dev) = create_engine(10);
    // Spread matches across separate 64-byte windows to avoid per-thread cap
    // "Hello" = 5 bytes. Place one per window with padding.
    let mut content = Vec::new();
    for variant in &[&b"Hello"[..], &b"HELLO"[..], &b"hello"[..], &b"hElLo"[..]] {
        let mut window = vec![b' '; 64];
        window[0..5].copy_from_slice(variant);
        content.extend_from_slice(&window);
    }
    engine.load_content(&content, 0);

    let gpu = engine.search(b"hello", &case_insensitive_options());
    let cpu = cpu_search_positions(&content, b"hello", false);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 4);
}

#[test]
fn test_case_insensitive_uppercase_pattern() {
    let (mut engine, _dev) = create_engine(10);
    let content = b"test TEST Test tEsT";
    engine.load_content(content, 0);

    let gpu = engine.search(b"TEST", &case_insensitive_options());
    let cpu = cpu_search_positions(content, b"TEST", false);
    let expected = expected_gpu_matches(&cpu, 4);

    assert_eq!(gpu.len(), expected);
    assert_eq!(gpu.len(), 4);
}

#[test]
fn test_case_sensitive_no_match() {
    let (mut engine, _dev) = create_engine(10);
    let content = b"Hello World";
    engine.load_content(content, 0);

    let gpu = engine.search(b"hello", &default_options());
    let cpu = cpu_search_positions(content, b"hello", true);

    assert_eq!(gpu.len(), 0);
    assert_eq!(cpu.len(), 0);
}

#[test]
fn test_case_insensitive_single_char() {
    let (mut engine, _dev) = create_engine(10);
    let content = b"aAbBcCaA";
    engine.load_content(content, 0);

    let gpu = engine.search(b"a", &case_insensitive_options());
    let cpu = cpu_search_positions(content, b"a", false);
    let expected = expected_gpu_matches(&cpu, 1);

    assert_eq!(gpu.len(), expected);
    assert_eq!(gpu.len(), 4); // a, A, a, A
}

// ============================================================================
// Test Matrix: Unicode UTF-8 Multibyte
// ============================================================================

#[test]
fn test_utf8_ascii_pattern_in_utf8_content() {
    let (mut engine, _dev) = create_engine(10);
    let content = "hello wörld hello".as_bytes();
    engine.load_content(content, 0);

    let gpu = engine.search(b"hello", &default_options());
    let cpu = cpu_search_positions(content, b"hello", true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 2);
}

#[test]
fn test_utf8_multibyte_pattern() {
    let (mut engine, _dev) = create_engine(10);
    let content = "café café café".as_bytes();
    let pattern = "café".as_bytes(); // 'é' is 2 bytes in UTF-8
    engine.load_content(content, 0);

    let gpu = engine.search(pattern, &default_options());
    let cpu = cpu_search_positions(content, pattern, true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 3);
}

#[test]
fn test_utf8_cjk_content() {
    let (mut engine, _dev) = create_engine(10);
    let content = "Hello 世界 World 世界 End".as_bytes();
    let pattern = "世界".as_bytes(); // 6 bytes in UTF-8
    engine.load_content(content, 0);

    let gpu = engine.search(pattern, &default_options());
    let cpu = cpu_search_positions(content, pattern, true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 2);
}

#[test]
fn test_utf8_emoji() {
    let (mut engine, _dev) = create_engine(10);
    let content = "test 🔥 data 🔥 end".as_bytes();
    let pattern = "🔥".as_bytes(); // 4 bytes
    engine.load_content(content, 0);

    let gpu = engine.search(pattern, &default_options());
    let cpu = cpu_search_positions(content, pattern, true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 2);
}

// ============================================================================
// Test Matrix: Binary / NUL Detection
// ============================================================================

#[test]
fn test_binary_nul_bytes() {
    let (mut engine, _dev) = create_engine(10);
    let mut content = Vec::new();
    content.extend_from_slice(b"hello\x00world\x00hello");
    engine.load_content(&content, 0);

    let gpu = engine.search(b"hello", &default_options());
    let cpu = cpu_search_positions(&content, b"hello", true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 2);
}

#[test]
fn test_search_for_nul_pattern() {
    let (mut engine, _dev) = create_engine(10);
    let mut content = vec![b'a'; 16];
    content[5] = 0;
    content[10] = 0;
    engine.load_content(&content, 0);

    let gpu = engine.search(&[0u8], &default_options());
    let cpu = cpu_search_positions(&content, &[0u8], true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 2);
}

#[test]
fn test_all_nul_content() {
    let (mut engine, _dev) = create_engine(10);
    let content = vec![0u8; 128];
    engine.load_content(&content, 0);

    let gpu = engine.search(b"test", &default_options());
    let cpu = cpu_search_positions(&content, b"test", true);

    assert_eq!(gpu.len(), 0);
    assert_eq!(cpu.len(), 0);
}

#[test]
fn test_binary_mixed_with_text() {
    let (mut engine, _dev) = create_engine(10);
    let mut content = Vec::new();
    content.extend_from_slice(&[0x7F, 0x45, 0x4C, 0x46]); // ELF magic
    content.extend_from_slice(&[0; 12]);
    content.extend_from_slice(b"needle in binary haystack");
    engine.load_content(&content, 0);

    let gpu = engine.search(b"needle", &default_options());
    let cpu = cpu_search_positions(&content, b"needle", true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 1);
}

// ============================================================================
// Test Matrix: Boundary Crossing (64-byte thread window)
// ============================================================================

#[test]
fn test_boundary_crossing_documented() {
    let (mut engine, _dev) = create_engine(10);

    // Place a match at byte 62-65, crossing the 64-byte boundary.
    // Thread 0 sees bytes 0-63, Thread 1 sees bytes 64-127.
    let mut content = vec![b' '; 128];
    content[62] = b't';
    content[63] = b'e';
    content[64] = b's';
    content[65] = b't';
    engine.load_content(&content, 0);

    let gpu = engine.search(b"test", &default_options());
    let cpu = cpu_search_positions(&content, b"test", true);

    assert_eq!(cpu.len(), 1, "CPU should find the boundary-crossing match");
    // GPU is expected to MISS this -- known 64-byte boundary limitation
    assert!(
        gpu.len() <= cpu.len(),
        "GPU should not produce false positives"
    );
    let missed = cpu.len() - gpu.len();
    println!(
        "Boundary crossing test: CPU={}, GPU={}, missed={}",
        cpu.len(),
        gpu.len(),
        missed
    );
}

#[test]
fn test_no_boundary_crossing() {
    let (mut engine, _dev) = create_engine(10);

    // Place 1 match per 64-byte window (no boundary crossing)
    let mut content = vec![b' '; 256];
    content[10..14].copy_from_slice(b"test");
    content[70..74].copy_from_slice(b"test");
    content[130..134].copy_from_slice(b"test");
    content[200..204].copy_from_slice(b"test");
    engine.load_content(&content, 0);

    let gpu = engine.search(b"test", &default_options());
    let cpu = cpu_search_positions(&content, b"test", true);

    assert_eq!(
        gpu.len(),
        cpu.len(),
        "Non-boundary matches: GPU({}) != CPU({})",
        gpu.len(),
        cpu.len()
    );
    assert_eq!(gpu.len(), 4);
}

#[test]
fn test_boundary_crossing_miss_rate_within_tolerance() {
    let (mut engine, _dev) = create_engine(100);

    // Repeating line with known spacing, some matches may cross boundaries
    let mut content = Vec::new();
    let line = b"fn test_function() { return 42; }\n"; // 34 bytes per line
    for _ in 0..100 {
        content.extend_from_slice(line);
    }
    engine.load_content(&content, 0);

    let gpu = engine.search(b"fn ", &default_options());
    let cpu_positions = cpu_search_positions(&content, b"fn ", true);
    let expected = expected_gpu_matches(&cpu_positions, 3);

    // GPU should find at least the expected count (accounting for per-thread cap)
    assert!(
        gpu.len() >= expected,
        "GPU({}) < expected({})",
        gpu.len(),
        expected
    );

    // GPU should never produce false positives
    assert!(
        gpu.len() <= cpu_positions.len(),
        "GPU({}) > CPU({})",
        gpu.len(),
        cpu_positions.len()
    );

    let miss_rate =
        (cpu_positions.len() as f64 - gpu.len() as f64) / cpu_positions.len() as f64 * 100.0;
    println!(
        "Miss rate: {:.1}% (GPU={}, CPU={}, expected_min={})",
        miss_rate,
        gpu.len(),
        cpu_positions.len(),
        expected
    );
    assert!(
        miss_rate < 20.0,
        "Miss rate {:.1}% exceeds 20% tolerance",
        miss_rate
    );
}

// ============================================================================
// Test Matrix: Empty Files / No Content
// ============================================================================

#[test]
fn test_empty_content() {
    let (mut engine, _dev) = create_engine(10);

    let chunks = engine.load_content(b"", 0);
    assert_eq!(chunks, 0);

    let gpu = engine.search(b"test", &default_options());
    let cpu = cpu_search(b"", b"test", true);

    assert_eq!(gpu.len(), 0);
    assert_eq!(cpu, 0);
}

#[test]
fn test_empty_pattern() {
    let (mut engine, _dev) = create_engine(10);
    let content = b"some content here";
    engine.load_content(content, 0);

    let gpu = engine.search(b"", &default_options());
    assert_eq!(gpu.len(), 0);
}

#[test]
fn test_pattern_longer_than_content() {
    let (mut engine, _dev) = create_engine(10);
    let content = b"hi";
    engine.load_content(content, 0);

    let gpu = engine.search(b"this is a very long pattern", &default_options());
    let cpu = cpu_search(content, b"this is a very long pattern", true);

    assert_eq!(gpu.len(), 0);
    assert_eq!(cpu, 0);
}

#[test]
fn test_single_byte_content() {
    let (mut engine, _dev) = create_engine(10);
    let content = b"x";
    engine.load_content(content, 0);

    let gpu = engine.search(b"x", &default_options());
    let cpu = cpu_search_positions(content, b"x", true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 1);
}

#[test]
fn test_no_content_loaded() {
    let (engine, _dev) = create_engine(10);

    let gpu = engine.search(b"test", &default_options());
    assert_eq!(gpu.len(), 0);
}

// ============================================================================
// Test Matrix: Dense Matches (per-thread cap verification)
// ============================================================================

#[test]
fn test_dense_single_char_per_thread_cap() {
    let (mut engine, _dev) = create_engine(10);

    // 60 'a' chars all in one 64-byte window. CPU finds 60, GPU caps at 4.
    let content = vec![b'a'; 60];
    engine.load_content(&content, 0);

    let gpu = engine.search(b"a", &default_options());
    let cpu = cpu_search_positions(&content, b"a", true);
    let expected = expected_gpu_matches(&cpu, 1);

    assert_eq!(cpu.len(), 60);
    assert_eq!(
        gpu.len(),
        expected,
        "GPU({}) != expected({}) -- per-thread cap should apply",
        gpu.len(),
        expected
    );
    // No false positives
    assert!(gpu.len() <= cpu.len());
    println!(
        "Dense single char: GPU={} (per-thread capped), CPU={}",
        gpu.len(),
        cpu.len()
    );
}

#[test]
fn test_dense_newlines_per_thread_cap() {
    let (mut engine, _dev) = create_engine(10);

    // 50 newlines across multiple 64-byte windows
    let content = vec![b'\n'; 50];
    engine.load_content(&content, 0);

    let gpu = engine.search(b"\n", &default_options());
    let cpu = cpu_search_positions(&content, b"\n", true);
    let expected = expected_gpu_matches(&cpu, 1);

    assert_eq!(cpu.len(), 50);
    assert_eq!(
        gpu.len(),
        expected,
        "GPU({}) != expected({})",
        gpu.len(),
        expected
    );
    // No false positives
    assert!(gpu.len() <= cpu.len());
}

#[test]
fn test_dense_matches_spread_across_windows() {
    let (mut engine, _dev) = create_engine(10);

    // Place exactly 3 matches per 64-byte window (below per-thread cap of 4)
    let mut content = Vec::new();
    for _ in 0..8 {
        let mut window = vec![b' '; 64];
        // 3 matches at positions 0, 20, 40
        window[0..4].copy_from_slice(b"FIND");
        window[20..24].copy_from_slice(b"FIND");
        window[40..44].copy_from_slice(b"FIND");
        content.extend_from_slice(&window);
    }
    engine.load_content(&content, 0);

    let gpu = engine.search(b"FIND", &default_options());
    let cpu = cpu_search_positions(&content, b"FIND", true);

    // 3 per window * 8 windows = 24, all below per-thread cap
    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 24);
}

// ============================================================================
// Test Matrix: Large Files
// ============================================================================

#[test]
fn test_large_file_1mb() {
    let (mut engine, _dev) = create_engine(1000);

    // 1MB content. "fox" is 3 bytes, one match per 45-byte line.
    // ~23K matches exceeds MAX_MATCHES (10K), so GPU result is capped.
    let line = b"the quick brown fox jumps over the lazy dog\n"; // 45 bytes
    let repeat_count = 1024 * 1024 / line.len();
    let mut content = Vec::with_capacity(repeat_count * line.len());
    for _ in 0..repeat_count {
        content.extend_from_slice(line);
    }
    engine.load_content(&content, 0);

    let gpu = engine.search(b"fox", &default_options());
    let cpu_positions = cpu_search_positions(&content, b"fox", true);

    // No false positives
    assert!(
        gpu.len() <= cpu_positions.len(),
        "GPU({}) > CPU({})",
        gpu.len(),
        cpu_positions.len()
    );

    // With ~23K CPU matches, GPU is capped at MAX_MATCHES (10K)
    if cpu_positions.len() > GPU_MAX_MATCHES {
        assert_eq!(
            gpu.len(),
            GPU_MAX_MATCHES,
            "GPU should hit MAX_MATCHES cap: got {}",
            gpu.len()
        );
        println!(
            "1MB test: GPU={} (capped at MAX_MATCHES), CPU={}",
            gpu.len(),
            cpu_positions.len()
        );
    } else {
        let miss_rate =
            (cpu_positions.len() as f64 - gpu.len() as f64) / cpu_positions.len() as f64 * 100.0;
        println!(
            "1MB test: GPU={}, CPU={}, miss_rate={:.1}%",
            gpu.len(),
            cpu_positions.len(),
            miss_rate
        );
        assert!(miss_rate < 15.0, "Miss rate {:.1}% too high", miss_rate);
    }
}

#[test]
fn test_large_file_10mb() {
    let (mut engine, _dev) = create_engine(10000);

    // 10MB content. Long pattern "process_data" (12 bytes) has at most 1 match per
    // 58-byte line, so per-thread cap is not an issue. But total matches (~180K)
    // exceed MAX_MATCHES (10K), so GPU result is capped.
    let line = b"fn process_data(input: &[u8]) -> Result<Vec<u8>, Error> {\n"; // 58 bytes
    let target_size = 10 * 1024 * 1024;
    let repeat_count = target_size / line.len();
    let mut content = Vec::with_capacity(repeat_count * line.len());
    for _ in 0..repeat_count {
        content.extend_from_slice(line);
    }
    engine.load_content(&content, 0);

    let gpu = engine.search(b"process_data", &default_options());
    let cpu_positions = cpu_search_positions(&content, b"process_data", true);

    // No false positives
    assert!(
        gpu.len() <= cpu_positions.len(),
        "GPU({}) > CPU({})",
        gpu.len(),
        cpu_positions.len()
    );

    // With ~180K CPU matches, GPU is capped at MAX_MATCHES (10K)
    if cpu_positions.len() > GPU_MAX_MATCHES {
        assert_eq!(
            gpu.len(),
            GPU_MAX_MATCHES,
            "GPU should hit MAX_MATCHES cap: got {}",
            gpu.len()
        );
        println!(
            "10MB test: GPU={} (capped at MAX_MATCHES), CPU={}",
            gpu.len(),
            cpu_positions.len()
        );
    } else {
        let miss_rate = (cpu_positions.len() as f64 - gpu.len() as f64)
            / cpu_positions.len() as f64
            * 100.0;
        println!(
            "10MB test: GPU={}, CPU={}, miss_rate={:.1}%",
            gpu.len(),
            cpu_positions.len(),
            miss_rate
        );
        assert!(miss_rate < 20.0, "Miss rate {:.1}% too high", miss_rate);
    }
}

#[test]
fn test_large_file_sparse_matches() {
    let (mut engine, _dev) = create_engine(1000);

    // ~1.8MB content with very sparse matches (1 per ~100 lines).
    // Short 3-byte pattern "XYZ" avoids frequent boundary crossing.
    // Filler line is 60 bytes, needle line is 40 bytes -- both well within
    // 64-byte windows, so "XYZ" at position 19 never crosses a boundary.
    let mut content = Vec::new();
    let filler = b"this is a normal line of source code without the magic word\n"; // 60 bytes
    let needle_line = b"this line has XYZ in it here\n"; // 28 bytes
    for i in 0..30000 {
        if i % 100 == 0 {
            content.extend_from_slice(needle_line);
        } else {
            content.extend_from_slice(filler);
        }
    }
    engine.load_content(&content, 0);

    let gpu = engine.search(b"XYZ", &default_options());
    let cpu_positions = cpu_search_positions(&content, b"XYZ", true);

    // No false positives
    assert!(
        gpu.len() <= cpu_positions.len(),
        "GPU({}) > CPU({})",
        gpu.len(),
        cpu_positions.len()
    );

    // Sparse matches with short pattern: most should be found
    let miss_rate = if cpu_positions.is_empty() {
        0.0
    } else {
        (cpu_positions.len() as f64 - gpu.len() as f64) / cpu_positions.len() as f64 * 100.0
    };
    println!(
        "Large sparse test: GPU={}, CPU={}, miss_rate={:.1}%",
        gpu.len(),
        cpu_positions.len(),
        miss_rate
    );
    // With a short pattern placed early in each line, miss rate should be minimal
    assert!(miss_rate < 20.0, "Miss rate {:.1}% too high", miss_rate);
}

// ============================================================================
// Test Matrix: Turbo Mode Consistency
// ============================================================================

#[test]
fn test_turbo_mode_matches_standard() {
    let (mut engine, _dev) = create_engine(100);

    let content = b"hello world hello world hello world";
    engine.load_content(content, 0);

    let standard_opts = SearchOptions {
        case_sensitive: true,
        max_results: GPU_MAX_MATCHES,
        mode: SearchMode::Standard,
    };
    let turbo_opts = SearchOptions {
        case_sensitive: true,
        max_results: GPU_MAX_MATCHES,
        mode: SearchMode::Turbo,
    };

    let standard = engine.search(b"hello", &standard_opts);
    engine.reset();
    engine.load_content(content, 0);
    let turbo = engine.search(b"hello", &turbo_opts);
    let cpu = cpu_search(content, b"hello", true);

    assert_eq!(
        standard.len(),
        turbo.len(),
        "Standard({}) != Turbo({})",
        standard.len(),
        turbo.len()
    );
    assert_eq!(standard.len(), cpu);
}

#[test]
fn test_turbo_mode_case_insensitive() {
    let (mut engine, _dev) = create_engine(100);

    let content = b"Rust RUST rust rUsT";
    engine.load_content(content, 0);

    let turbo_opts = SearchOptions {
        case_sensitive: false,
        max_results: GPU_MAX_MATCHES,
        mode: SearchMode::Turbo,
    };
    let gpu = engine.search(b"rust", &turbo_opts);
    let cpu = cpu_search(content, b"rust", false);

    assert_eq!(gpu.len(), cpu);
    assert_eq!(gpu.len(), 4);
}

// ============================================================================
// Test Matrix: Edge Cases
// ============================================================================

#[test]
fn test_max_pattern_length() {
    let (mut engine, _dev) = create_engine(10);

    // 64-byte pattern (maximum allowed)
    let pattern = vec![b'X'; 64];
    let mut content = vec![b' '; 128];
    content[0..64].copy_from_slice(&pattern);
    engine.load_content(&content, 0);

    let gpu = engine.search(&pattern, &default_options());
    let cpu = cpu_search_positions(&content, &pattern, true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 1);
}

#[test]
fn test_pattern_too_long() {
    let (mut engine, _dev) = create_engine(10);

    let pattern = vec![b'X'; 65];
    let content = vec![b'X'; 128];
    engine.load_content(&content, 0);

    let gpu = engine.search(&pattern, &default_options());
    assert_eq!(gpu.len(), 0, "Pattern > 64 bytes should return no results");
}

#[test]
fn test_pattern_with_special_chars() {
    let (mut engine, _dev) = create_engine(10);

    let content = b"line1\tline2\rline3\nline4";
    engine.load_content(content, 0);

    let gpu = engine.search(b"\t", &default_options());
    let cpu = cpu_search_positions(content, b"\t", true);
    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 1);
}

#[test]
fn test_consecutive_searches_with_reset() {
    let (mut engine, _dev) = create_engine(100);

    // First search
    let content1 = b"alpha beta gamma";
    engine.load_content(content1, 0);
    let gpu1 = engine.search(b"beta", &default_options());
    assert_eq!(gpu1.len(), 1);

    // Reset and search different content
    engine.reset();
    let content2 = b"delta epsilon zeta";
    engine.load_content(content2, 0);
    let gpu2 = engine.search(b"epsilon", &default_options());
    assert_eq!(gpu2.len(), 1);

    // Old pattern should not be found
    let gpu3 = engine.search(b"beta", &default_options());
    assert_eq!(gpu3.len(), 0);
}

// ============================================================================
// Test Matrix: No False Positives Verification
// ============================================================================

#[test]
fn test_no_false_positives_varied_content() {
    let (mut engine, _dev) = create_engine(100);

    // "test" appears as substring in: testing, tests, tested, tester, test_case, test
    // CPU finds 7 occurrences. GPU may cap at 4 (per-thread limit if all in one window).
    let content = b"testing tests tested tester test_case unittest test";
    engine.load_content(content, 0);

    let gpu = engine.search(b"test", &default_options());
    let cpu = cpu_search_positions(content, b"test", true);
    let expected = expected_gpu_matches(&cpu, 4);

    // No false positives
    assert!(
        gpu.len() <= cpu.len(),
        "GPU({}) > CPU({}) -- false positives!",
        gpu.len(),
        cpu.len()
    );
    assert_eq!(
        gpu.len(),
        expected,
        "GPU({}) != expected({})",
        gpu.len(),
        expected
    );
}

#[test]
fn test_no_false_positives_similar_patterns() {
    let (mut engine, _dev) = create_engine(10);

    let content = b"abcd abce abcf abcg";
    engine.load_content(content, 0);

    let gpu = engine.search(b"abcd", &default_options());
    let cpu = cpu_search_positions(content, b"abcd", true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 1);
}

#[test]
fn test_no_false_positives_off_by_one() {
    let (mut engine, _dev) = create_engine(10);

    // Pattern at exact end of valid data within a window
    let mut content = vec![b' '; 64];
    content[59..63].copy_from_slice(b"DONE");
    engine.load_content(&content, 0);

    let gpu = engine.search(b"DONE", &default_options());
    let cpu = cpu_search_positions(&content, b"DONE", true);

    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), 1);
}

// ============================================================================
// CPU Reference Self-Tests
// ============================================================================

#[test]
fn test_cpu_reference_correctness() {
    let positions = cpu_search_positions(b"hello world", b"hello", true);
    assert_eq!(positions, vec![0]);

    let positions = cpu_search_positions(b"hello world hello", b"hello", true);
    assert_eq!(positions, vec![0, 12]);

    // Overlapping
    let positions = cpu_search_positions(b"aaa", b"aa", true);
    assert_eq!(positions, vec![0, 1]);

    // Case insensitive
    let positions = cpu_search_positions(b"Hello HELLO", b"hello", false);
    assert_eq!(positions, vec![0, 6]);

    // No match
    let positions = cpu_search_positions(b"hello", b"world", true);
    assert!(positions.is_empty());

    // Empty
    let positions = cpu_search_positions(b"", b"test", true);
    assert!(positions.is_empty());
    let positions = cpu_search_positions(b"test", b"", true);
    assert!(positions.is_empty());

    // Single byte with memchr fast path
    let positions = cpu_search_positions(b"abcabc", b"a", true);
    assert_eq!(positions, vec![0, 3]);
}

#[test]
fn test_boundary_crossing_helper() {
    assert!(crosses_64byte_boundary(60, 5));
    assert!(!crosses_64byte_boundary(60, 4));
    assert!(!crosses_64byte_boundary(0, 64));
    assert!(crosses_64byte_boundary(63, 2));
    assert!(!crosses_64byte_boundary(64, 4));
}

#[test]
fn test_expected_gpu_matches_helper() {
    // 10 matches in one window, pattern_len=2
    // All non-boundary, but capped at 4 per window
    let positions: Vec<usize> = (0..10).collect();
    assert_eq!(expected_gpu_matches(&positions, 2), 4);

    // 3 matches in one window (below cap)
    let positions = vec![0, 10, 20];
    assert_eq!(expected_gpu_matches(&positions, 4), 3);

    // Matches in 2 windows, 3 each (below cap)
    let positions = vec![0, 10, 20, 64, 74, 84];
    assert_eq!(expected_gpu_matches(&positions, 4), 6);

    // Match crossing boundary filtered out
    let positions = vec![62]; // 62 + 4 = 66 > 64, crosses boundary
    assert_eq!(expected_gpu_matches(&positions, 4), 0);
}
