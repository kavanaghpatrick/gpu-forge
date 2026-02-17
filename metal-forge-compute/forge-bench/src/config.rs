use serde::{Deserialize, Serialize};

/// A benchmark profile with preset sizes, runs, and warmup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchProfile {
    pub name: String,
    pub sizes: Vec<usize>,
    pub runs: u32,
    pub warmup: u32,
}

/// Returns the "quick" profile: 1M / 3 runs / 1 warmup.
pub fn quick_profile() -> BenchProfile {
    BenchProfile {
        name: "quick".to_string(),
        sizes: vec![1_000_000],
        runs: 3,
        warmup: 1,
    }
}

/// Returns the "standard" profile: 1M+10M / 10 runs / 3 warmup.
pub fn standard_profile() -> BenchProfile {
    BenchProfile {
        name: "standard".to_string(),
        sizes: vec![1_000_000, 10_000_000],
        runs: 10,
        warmup: 3,
    }
}

/// Returns the "thorough" profile: 1M+10M+100M / 30 runs / 3 warmup.
pub fn thorough_profile() -> BenchProfile {
    BenchProfile {
        name: "thorough".to_string(),
        sizes: vec![1_000_000, 10_000_000, 100_000_000],
        runs: 30,
        warmup: 3,
    }
}

/// Lookup a profile by name.
pub fn get_profile(name: &str) -> Option<BenchProfile> {
    match name {
        "quick" => Some(quick_profile()),
        "standard" => Some(standard_profile()),
        "thorough" => Some(thorough_profile()),
        _ => None,
    }
}

/// Parse a human-readable size string to a usize.
///
/// Supports:
/// - "1M" or "1m" -> 1_000_000
/// - "10M" -> 10_000_000
/// - "100M" -> 100_000_000
/// - "100K" or "100k" -> 100_000
/// - "1_000_000" -> 1_000_000
/// - "1000000" -> 1_000_000
pub fn parse_size(s: &str) -> Result<usize, String> {
    let s = s.trim();

    // Handle suffix multipliers
    if let Some(prefix) = s.strip_suffix('M').or_else(|| s.strip_suffix('m')) {
        let num: f64 = prefix
            .replace('_', "")
            .parse()
            .map_err(|e| format!("Invalid size '{}': {}", s, e))?;
        return Ok((num * 1_000_000.0) as usize);
    }

    if let Some(prefix) = s.strip_suffix('K').or_else(|| s.strip_suffix('k')) {
        let num: f64 = prefix
            .replace('_', "")
            .parse()
            .map_err(|e| format!("Invalid size '{}': {}", s, e))?;
        return Ok((num * 1_000.0) as usize);
    }

    // Raw number (possibly with underscores)
    s.replace('_', "")
        .parse::<usize>()
        .map_err(|e| format!("Invalid size '{}': {}", s, e))
}

/// Parse a comma-separated list of size strings.
pub fn parse_sizes(raw: &[String]) -> Result<Vec<usize>, String> {
    raw.iter().map(|s| parse_size(s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size_millions() {
        assert_eq!(parse_size("1M").unwrap(), 1_000_000);
        assert_eq!(parse_size("10M").unwrap(), 10_000_000);
        assert_eq!(parse_size("100M").unwrap(), 100_000_000);
        assert_eq!(parse_size("1m").unwrap(), 1_000_000);
    }

    #[test]
    fn test_parse_size_thousands() {
        assert_eq!(parse_size("100K").unwrap(), 100_000);
        assert_eq!(parse_size("100k").unwrap(), 100_000);
        assert_eq!(parse_size("1K").unwrap(), 1_000);
    }

    #[test]
    fn test_parse_size_raw() {
        assert_eq!(parse_size("1000000").unwrap(), 1_000_000);
        assert_eq!(parse_size("1_000_000").unwrap(), 1_000_000);
    }

    #[test]
    fn test_parse_size_invalid() {
        assert!(parse_size("abc").is_err());
        assert!(parse_size("").is_err());
    }

    #[test]
    fn test_profiles() {
        let q = quick_profile();
        assert_eq!(q.sizes, vec![1_000_000]);
        assert_eq!(q.runs, 3);
        assert_eq!(q.warmup, 1);

        let s = standard_profile();
        assert_eq!(s.sizes, vec![1_000_000, 10_000_000]);
        assert_eq!(s.runs, 10);
        assert_eq!(s.warmup, 3);

        let t = thorough_profile();
        assert_eq!(t.sizes, vec![1_000_000, 10_000_000, 100_000_000]);
        assert_eq!(t.runs, 30);
        assert_eq!(t.warmup, 3);
    }

    #[test]
    fn test_get_profile() {
        assert!(get_profile("quick").is_some());
        assert!(get_profile("standard").is_some());
        assert!(get_profile("thorough").is_some());
        assert!(get_profile("unknown").is_none());
    }

    #[test]
    fn test_parse_size_fractional_millions() {
        assert_eq!(parse_size("0.5M").unwrap(), 500_000);
        assert_eq!(parse_size("2.5M").unwrap(), 2_500_000);
    }

    #[test]
    fn test_parse_size_fractional_thousands() {
        assert_eq!(parse_size("1.5K").unwrap(), 1_500);
        assert_eq!(parse_size("0.5k").unwrap(), 500);
    }

    #[test]
    fn test_parse_size_whitespace_trimmed() {
        assert_eq!(parse_size("  1M  ").unwrap(), 1_000_000);
        assert_eq!(parse_size(" 100K ").unwrap(), 100_000);
    }

    #[test]
    fn test_parse_sizes_multiple() {
        let input = vec!["1M".to_string(), "10M".to_string(), "100K".to_string()];
        let result = parse_sizes(&input).unwrap();
        assert_eq!(result, vec![1_000_000, 10_000_000, 100_000]);
    }

    #[test]
    fn test_parse_sizes_empty() {
        let input: Vec<String> = vec![];
        let result = parse_sizes(&input).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_sizes_with_invalid() {
        let input = vec!["1M".to_string(), "bad".to_string()];
        assert!(parse_sizes(&input).is_err());
    }

    #[test]
    fn test_profile_sizes_are_ascending() {
        let s = standard_profile();
        for w in s.sizes.windows(2) {
            assert!(w[0] < w[1], "Profile sizes should be ascending");
        }
        let t = thorough_profile();
        for w in t.sizes.windows(2) {
            assert!(w[0] < w[1], "Profile sizes should be ascending");
        }
    }

    #[test]
    fn test_profile_names_match() {
        assert_eq!(quick_profile().name, "quick");
        assert_eq!(standard_profile().name, "standard");
        assert_eq!(thorough_profile().name, "thorough");
    }

    #[test]
    fn test_parse_size_underscored_with_suffix() {
        assert_eq!(parse_size("1_000K").unwrap(), 1_000_000);
    }
}
