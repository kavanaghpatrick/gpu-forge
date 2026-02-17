//! Hardware detection and bandwidth lookup for Apple Silicon chips.
//!
//! Detects the chip name via Metal device.name() and looks up
//! the theoretical peak memory bandwidth (GB/s).

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

/// Hardware information for the current GPU.
pub struct HardwareInfo {
    /// Chip name (e.g., "Apple M4 Pro").
    pub chip_name: String,
    /// Theoretical peak memory bandwidth in GB/s.
    pub bandwidth_gbs: f64,
    /// Number of GPU cores (if detectable).
    pub gpu_cores: Option<u32>,
}

impl HardwareInfo {
    /// Detect hardware from a Metal device.
    pub fn detect(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let chip_name = device.name().to_string();
        let bandwidth_gbs = lookup_bandwidth(&chip_name);

        Self {
            chip_name,
            bandwidth_gbs,
            gpu_cores: None,
        }
    }

    /// Bandwidth utilization percentage for a measured throughput.
    pub fn bandwidth_utilization(&self, measured_gbs: f64) -> f64 {
        if self.bandwidth_gbs > 0.0 {
            (measured_gbs / self.bandwidth_gbs) * 100.0
        } else {
            0.0
        }
    }
}

/// Lookup theoretical peak memory bandwidth (GB/s) by chip name.
fn lookup_bandwidth(chip_name: &str) -> f64 {
    let name = chip_name.to_lowercase();

    if name.contains("m4 max") {
        546.0
    } else if name.contains("m4 pro") {
        273.0
    } else if name.contains("m4") {
        120.0
    } else if name.contains("m3 max") {
        400.0
    } else if name.contains("m3 pro") {
        200.0
    } else if name.contains("m3") {
        100.0
    } else if name.contains("m2 ultra") {
        800.0
    } else if name.contains("m2 max") {
        400.0
    } else if name.contains("m2 pro") {
        200.0
    } else if name.contains("m2") {
        100.0
    } else if name.contains("m1 ultra") {
        800.0
    } else if name.contains("m1 max") {
        400.0
    } else if name.contains("m1 pro") {
        200.0
    } else if name.contains("m1") {
        68.25
    } else {
        // Unknown chip -- return 0 to indicate unknown.
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_lookup() {
        assert_eq!(lookup_bandwidth("Apple M4 Pro"), 273.0);
        assert_eq!(lookup_bandwidth("Apple M4 Max"), 546.0);
        assert_eq!(lookup_bandwidth("Apple M4"), 120.0);
        assert_eq!(lookup_bandwidth("Apple M1"), 68.25);
        assert_eq!(lookup_bandwidth("Unknown GPU"), 0.0);
    }

    #[test]
    fn test_bandwidth_utilization() {
        let info = HardwareInfo {
            chip_name: "Apple M4 Pro".to_string(),
            bandwidth_gbs: 273.0,
            gpu_cores: None,
        };
        let util = info.bandwidth_utilization(136.5);
        assert!((util - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_bandwidth_utilization_zero_bandwidth() {
        let info = HardwareInfo {
            chip_name: "Unknown".to_string(),
            bandwidth_gbs: 0.0,
            gpu_cores: None,
        };
        assert_eq!(info.bandwidth_utilization(100.0), 0.0);
    }

    #[test]
    fn test_detect_chip_returns_non_empty() {
        // Create a real Metal device and verify chip detection returns a non-empty string
        let device = objc2_metal::MTLCreateSystemDefaultDevice()
            .expect("No Metal device available");
        let info = HardwareInfo::detect(&device);
        assert!(
            !info.chip_name.is_empty(),
            "detect_chip should return a non-empty chip name"
        );
        // On any Apple Silicon, bandwidth should be known (> 0)
        assert!(
            info.bandwidth_gbs > 0.0,
            "bandwidth should be > 0 for known Apple Silicon chips"
        );
    }

    #[test]
    fn test_bandwidth_lookup_case_insensitive_via_lowercase() {
        // lookup_bandwidth lowercases the chip name, so mixed case should work
        assert_eq!(lookup_bandwidth("APPLE M4 PRO"), 273.0);
        assert_eq!(lookup_bandwidth("apple m4 pro"), 273.0);
    }
}
