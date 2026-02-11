//! Metal device initialization and capability queries.
//!
//! Ported from gpu-query's `gpu/device.rs` pattern, adapted for
//! gpu-search compute workloads.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

/// GPU device wrapper with capability info.
pub struct GpuDevice {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub device_name: String,
    pub max_threadgroup_size: (usize, usize, usize),
    pub max_buffer_length: usize,
    pub max_threadgroup_memory: usize,
}

impl Default for GpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDevice {
    /// Initialize Metal device and query capabilities.
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice()
            .expect("Failed to get default Metal device -- no Apple GPU available");

        let device_name = device.name().to_string();

        // Validate we got an Apple GPU (not software renderer)
        assert!(
            !device_name.is_empty(),
            "Metal device returned empty name"
        );

        let max_threads = device.maxThreadsPerThreadgroup();
        let max_threadgroup_size = (
            max_threads.width as usize,
            max_threads.height as usize,
            max_threads.depth as usize,
        );

        let max_buffer_length = device.maxBufferLength() as usize;
        let max_threadgroup_memory = device.maxThreadgroupMemoryLength() as usize;

        Self {
            device,
            device_name,
            max_threadgroup_size,
            max_buffer_length,
            max_threadgroup_memory,
        }
    }

    /// Print device capabilities summary.
    pub fn print_info(&self) {
        println!("GPU Device: {}", self.device_name);
        println!(
            "  Max threadgroup size: {}x{}x{}",
            self.max_threadgroup_size.0,
            self.max_threadgroup_size.1,
            self.max_threadgroup_size.2,
        );
        println!(
            "  Max buffer length: {} MB",
            self.max_buffer_length / (1024 * 1024)
        );
        println!(
            "  Max threadgroup memory: {} KB",
            self.max_threadgroup_memory / 1024
        );
    }
}
