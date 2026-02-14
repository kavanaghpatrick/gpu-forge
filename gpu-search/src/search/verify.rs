//! CPU verification layer for GPU search results.
//!
//! Uses `memchr::memmem` to find all pattern occurrences in file content,
//! then compares against GPU-reported byte offsets to detect false positives
//! and missed matches.

use memchr::memmem;

/// Controls when CPU verification runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyMode {
    /// No verification.
    Off,
    /// Verify a random sample of matches (low overhead).
    Sample,
    /// Verify every match against CPU ground truth.
    Full,
}

impl VerifyMode {
    /// Parse from the `GPU_SEARCH_VERIFY` environment variable.
    ///
    /// Values: "off", "sample" (default), "full".
    pub fn from_env() -> Self {
        match std::env::var("GPU_SEARCH_VERIFY").as_deref() {
            Ok("full") => VerifyMode::Full,
            Ok("off") => VerifyMode::Off,
            _ => VerifyMode::Sample,
        }
    }

    /// Return the effective verify mode given the result count.
    ///
    /// When mode is `Sample` and result count is below 100, upgrades to `Full`
    /// (cheap enough to verify everything). Otherwise returns `self` unchanged.
    pub fn effective(self, result_count: usize) -> VerifyMode {
        if self == VerifyMode::Sample && result_count < 100 {
            VerifyMode::Full
        } else {
            self
        }
    }
}

/// Result of CPU verification against GPU match results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationResult {
    /// Number of GPU byte offsets confirmed by CPU.
    pub confirmed: u32,
    /// GPU byte offsets that do NOT correspond to a real pattern occurrence.
    pub false_positives: u32,
    /// CPU-found occurrences that GPU did not report.
    pub missed: u32,
}

/// Verify GPU-reported match byte offsets against CPU ground truth.
///
/// Finds all occurrences of `pattern` in `content` using `memchr::memmem`,
/// then checks each `gpu_byte_offsets` entry against the CPU results.
///
/// When `case_sensitive` is false, both `content` and `pattern` are lowered
/// before comparison (allocation cost is acceptable for verification).
pub fn cpu_verify_matches(
    content: &[u8],
    pattern: &[u8],
    gpu_byte_offsets: &[u32],
    case_sensitive: bool,
) -> VerificationResult {
    // Build CPU ground-truth set of byte offsets
    let cpu_offsets: Vec<usize> = if case_sensitive {
        let finder = memmem::Finder::new(pattern);
        finder.find_iter(content).collect()
    } else {
        let lower_content: Vec<u8> = content.iter().map(|b| b.to_ascii_lowercase()).collect();
        let lower_pattern: Vec<u8> = pattern.iter().map(|b| b.to_ascii_lowercase()).collect();
        let finder = memmem::Finder::new(&lower_pattern);
        finder.find_iter(&lower_content).collect()
    };

    let mut confirmed: u32 = 0;
    let mut false_positives: u32 = 0;

    // Check each GPU offset against CPU ground truth
    for &gpu_offset in gpu_byte_offsets {
        if cpu_offsets.contains(&(gpu_offset as usize)) {
            confirmed += 1;
        } else {
            false_positives += 1;
        }
    }

    // Missed = CPU found but GPU did not report
    let gpu_set: std::collections::HashSet<usize> =
        gpu_byte_offsets.iter().map(|&o| o as usize).collect();
    let missed = cpu_offsets
        .iter()
        .filter(|o| !gpu_set.contains(o))
        .count() as u32;

    VerificationResult {
        confirmed,
        false_positives,
        missed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match_single_occurrence() {
        let content = b"hello world";
        let pattern = b"world";
        let gpu_offsets = vec![6]; // correct: "world" starts at byte 6
        let result = cpu_verify_matches(content, pattern, &gpu_offsets, true);
        assert_eq!(result.confirmed, 1);
        assert_eq!(result.false_positives, 0);
        assert_eq!(result.missed, 0);
    }

    #[test]
    fn test_false_positive_detected() {
        let content = b"hello world";
        let pattern = b"world";
        let gpu_offsets = vec![3]; // wrong offset
        let result = cpu_verify_matches(content, pattern, &gpu_offsets, true);
        assert_eq!(result.confirmed, 0);
        assert_eq!(result.false_positives, 1);
        assert_eq!(result.missed, 1); // CPU found at 6, GPU didn't
    }

    #[test]
    fn test_multiple_occurrences() {
        let content = b"abc abc abc";
        let pattern = b"abc";
        // Correct offsets: 0, 4, 8
        let gpu_offsets = vec![0, 4, 8];
        let result = cpu_verify_matches(content, pattern, &gpu_offsets, true);
        assert_eq!(result.confirmed, 3);
        assert_eq!(result.false_positives, 0);
        assert_eq!(result.missed, 0);
    }

    #[test]
    fn test_missed_matches() {
        let content = b"abc abc abc";
        let pattern = b"abc";
        // GPU only found first occurrence
        let gpu_offsets = vec![0];
        let result = cpu_verify_matches(content, pattern, &gpu_offsets, true);
        assert_eq!(result.confirmed, 1);
        assert_eq!(result.false_positives, 0);
        assert_eq!(result.missed, 2); // missed at 4 and 8
    }

    #[test]
    fn test_case_insensitive() {
        let content = b"Hello HELLO hello";
        let pattern = b"hello";
        // Correct offsets: 0, 6, 12
        let gpu_offsets = vec![0, 6, 12];
        let result = cpu_verify_matches(content, pattern, &gpu_offsets, false);
        assert_eq!(result.confirmed, 3);
        assert_eq!(result.false_positives, 0);
        assert_eq!(result.missed, 0);
    }

    #[test]
    fn test_case_sensitive_misses_different_case() {
        let content = b"Hello HELLO hello";
        let pattern = b"hello";
        // Only the last "hello" at byte 12 is a case-sensitive match
        let gpu_offsets = vec![0, 6, 12];
        let result = cpu_verify_matches(content, pattern, &gpu_offsets, true);
        assert_eq!(result.confirmed, 1); // only byte 12
        assert_eq!(result.false_positives, 2); // 0 and 6 are wrong
        assert_eq!(result.missed, 0);
    }

    #[test]
    fn test_no_matches_in_content() {
        let content = b"hello world";
        let pattern = b"xyz";
        let gpu_offsets = vec![2]; // false positive
        let result = cpu_verify_matches(content, pattern, &gpu_offsets, true);
        assert_eq!(result.confirmed, 0);
        assert_eq!(result.false_positives, 1);
        assert_eq!(result.missed, 0);
    }

    #[test]
    fn test_empty_gpu_offsets() {
        let content = b"abc abc";
        let pattern = b"abc";
        let gpu_offsets: Vec<u32> = vec![];
        let result = cpu_verify_matches(content, pattern, &gpu_offsets, true);
        assert_eq!(result.confirmed, 0);
        assert_eq!(result.false_positives, 0);
        assert_eq!(result.missed, 2);
    }

    #[test]
    fn test_empty_content() {
        let content = b"";
        let pattern = b"abc";
        let gpu_offsets: Vec<u32> = vec![];
        let result = cpu_verify_matches(content, pattern, &gpu_offsets, true);
        assert_eq!(result.confirmed, 0);
        assert_eq!(result.false_positives, 0);
        assert_eq!(result.missed, 0);
    }

    #[test]
    fn test_verify_mode_from_env() {
        // Default (no env var) should be Sample
        std::env::remove_var("GPU_SEARCH_VERIFY");
        assert_eq!(VerifyMode::from_env(), VerifyMode::Sample);

        std::env::set_var("GPU_SEARCH_VERIFY", "full");
        assert_eq!(VerifyMode::from_env(), VerifyMode::Full);

        std::env::set_var("GPU_SEARCH_VERIFY", "sample");
        assert_eq!(VerifyMode::from_env(), VerifyMode::Sample);

        std::env::set_var("GPU_SEARCH_VERIFY", "off");
        assert_eq!(VerifyMode::from_env(), VerifyMode::Off);

        // Clean up
        std::env::remove_var("GPU_SEARCH_VERIFY");
    }

    #[test]
    fn test_verify_mode_effective() {
        // Sample upgrades to Full below 100 results
        assert_eq!(VerifyMode::Sample.effective(0), VerifyMode::Full);
        assert_eq!(VerifyMode::Sample.effective(50), VerifyMode::Full);
        assert_eq!(VerifyMode::Sample.effective(99), VerifyMode::Full);

        // Sample stays Sample at 100 and above
        assert_eq!(VerifyMode::Sample.effective(100), VerifyMode::Sample);
        assert_eq!(VerifyMode::Sample.effective(500), VerifyMode::Sample);

        // Full ignores count -- stays Full
        assert_eq!(VerifyMode::Full.effective(0), VerifyMode::Full);
        assert_eq!(VerifyMode::Full.effective(200), VerifyMode::Full);

        // Off ignores count -- stays Off
        assert_eq!(VerifyMode::Off.effective(0), VerifyMode::Off);
        assert_eq!(VerifyMode::Off.effective(50), VerifyMode::Off);
        assert_eq!(VerifyMode::Off.effective(200), VerifyMode::Off);
    }

    // ---- Adaptive VerifyMode unit tests (U-VFY-1 through U-VFY-10) ----

    /// U-VFY-1: Default mode (no env var) is Sample.
    #[test]
    fn u_vfy_1_default_is_sample() {
        std::env::remove_var("GPU_SEARCH_VERIFY");
        assert_eq!(VerifyMode::from_env(), VerifyMode::Sample);
    }

    /// U-VFY-2: Env "off" → Off.
    #[test]
    fn u_vfy_2_env_off() {
        std::env::set_var("GPU_SEARCH_VERIFY", "off");
        assert_eq!(VerifyMode::from_env(), VerifyMode::Off);
        std::env::remove_var("GPU_SEARCH_VERIFY");
    }

    /// U-VFY-3: Env "full" → Full.
    #[test]
    fn u_vfy_3_env_full() {
        std::env::set_var("GPU_SEARCH_VERIFY", "full");
        assert_eq!(VerifyMode::from_env(), VerifyMode::Full);
        std::env::remove_var("GPU_SEARCH_VERIFY");
    }

    /// U-VFY-4: effective() upgrades Sample to Full when count < 100.
    #[test]
    fn u_vfy_4_effective_upgrades_below_100() {
        assert_eq!(VerifyMode::Sample.effective(50), VerifyMode::Full);
        assert_eq!(VerifyMode::Sample.effective(1), VerifyMode::Full);
        assert_eq!(VerifyMode::Sample.effective(10), VerifyMode::Full);
    }

    /// U-VFY-5: effective() keeps Sample at exactly 100.
    #[test]
    fn u_vfy_5_effective_stays_at_100() {
        assert_eq!(VerifyMode::Sample.effective(100), VerifyMode::Sample);
    }

    /// U-VFY-6: effective() keeps Sample above 100.
    #[test]
    fn u_vfy_6_effective_stays_above_100() {
        assert_eq!(VerifyMode::Sample.effective(101), VerifyMode::Sample);
        assert_eq!(VerifyMode::Sample.effective(500), VerifyMode::Sample);
        assert_eq!(VerifyMode::Sample.effective(10_000), VerifyMode::Sample);
    }

    /// U-VFY-7: Full ignores result count — stays Full always.
    #[test]
    fn u_vfy_7_full_ignores_count() {
        assert_eq!(VerifyMode::Full.effective(0), VerifyMode::Full);
        assert_eq!(VerifyMode::Full.effective(50), VerifyMode::Full);
        assert_eq!(VerifyMode::Full.effective(99), VerifyMode::Full);
        assert_eq!(VerifyMode::Full.effective(100), VerifyMode::Full);
        assert_eq!(VerifyMode::Full.effective(1_000), VerifyMode::Full);
    }

    /// U-VFY-8: Off ignores result count — stays Off always.
    #[test]
    fn u_vfy_8_off_ignores_count() {
        assert_eq!(VerifyMode::Off.effective(0), VerifyMode::Off);
        assert_eq!(VerifyMode::Off.effective(50), VerifyMode::Off);
        assert_eq!(VerifyMode::Off.effective(99), VerifyMode::Off);
        assert_eq!(VerifyMode::Off.effective(100), VerifyMode::Off);
        assert_eq!(VerifyMode::Off.effective(1_000), VerifyMode::Off);
    }

    /// U-VFY-9: Boundary 99 — Sample upgrades to Full at 99.
    #[test]
    fn u_vfy_9_boundary_99() {
        assert_eq!(VerifyMode::Sample.effective(99), VerifyMode::Full);
        // And 100 does NOT upgrade
        assert_eq!(VerifyMode::Sample.effective(100), VerifyMode::Sample);
    }

    /// U-VFY-10: Zero results — Sample upgrades to Full.
    #[test]
    fn u_vfy_10_zero_results() {
        assert_eq!(VerifyMode::Sample.effective(0), VerifyMode::Full);
    }

    #[test]
    fn test_overlapping_pattern() {
        let content = b"aaaa";
        let pattern = b"aa";
        // memchr::memmem finds non-overlapping: 0, 2
        let gpu_offsets = vec![0, 2];
        let result = cpu_verify_matches(content, pattern, &gpu_offsets, true);
        assert_eq!(result.confirmed, 2);
        assert_eq!(result.false_positives, 0);
        assert_eq!(result.missed, 0);
    }

    #[test]
    fn test_mixed_confirmed_and_false_positives() {
        let content = b"fn main() { fn helper() }";
        let pattern = b"fn ";
        // CPU finds: 0, 12 ("fn " at start and after "{ ")
        let gpu_offsets = vec![0, 7, 12]; // 7 is a false positive
        let result = cpu_verify_matches(content, pattern, &gpu_offsets, true);
        assert_eq!(result.confirmed, 2);
        assert_eq!(result.false_positives, 1);
        assert_eq!(result.missed, 0);
    }
}
