use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::vis::{texture_dim, DemoParams};

#[derive(Clone, Copy, PartialEq)]
pub enum SortAlgorithm {
    Hybrid,
    EightBit,
}

pub const SCALES: &[usize] = &[1_000_000, 4_000_000, 16_000_000, 64_000_000];

#[repr(C)]
pub struct Exp17Params {
    pub element_count: u32,
    pub num_tiles: u32,
    pub shift: u32,
    pub pass: u32,
}

#[repr(C)]
pub struct Exp16Params {
    pub element_count: u32,
    pub num_tiles: u32,
    pub num_tgs: u32,
    pub shift: u32,
    pub pass: u32,
}

#[repr(C)]
#[allow(dead_code)]
pub struct BucketDesc {
    pub offset: u32,
    pub count: u32,
    pub tile_count: u32,
    pub tile_base: u32,
}

pub struct SortEngine {
    // exp17 PSOs
    pso_msd_histogram: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_msd_prep: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_msd_scatter: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_precompute: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_fused_v3: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    // exp16 PSOs
    pso_combined_hist: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_global_prefix: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_zero_status: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_partition: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    // Shuffle PSO
    pso_random_fill: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    // Shared buffers
    pub buf_a: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub buf_b: Retained<ProtocolObject<dyn MTLBuffer>>,

    // exp17 scratch
    buf_msd_hist: Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_counters_17: Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_bucket_descs: Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_inner_hists: Retained<ProtocolObject<dyn MTLBuffer>>,

    // exp16 scratch
    buf_global_hist: Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_tile_status: Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_counters_16: Retained<ProtocolObject<dyn MTLBuffer>>,

    // State
    pub algorithm: SortAlgorithm,
    pub element_count: usize,
    pub last_sort_ms: f64,
}

impl SortEngine {
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Self {
        let pso = |name: &str| -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
            let fn_name = objc2_foundation::NSString::from_str(name);
            let function = library
                .newFunctionWithName(&fn_name)
                .unwrap_or_else(|| panic!("Failed to find '{}'", name));
            #[allow(deprecated)]
            device
                .newComputePipelineStateWithFunction_error(&function)
                .unwrap_or_else(|e| panic!("Failed to create PSO '{}': {:?}", name, e))
        };

        let buf = |size: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            device
                .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
                .unwrap_or_else(|| panic!("Failed to allocate buffer of {} bytes", size))
        };

        // Max 64M elements
        let max_n: usize = 64_000_000;
        let max_tiles = max_n.div_ceil(4096);

        Self {
            pso_msd_histogram: pso("exp17_msd_histogram"),
            pso_msd_prep: pso("exp17_msd_prep"),
            pso_msd_scatter: pso("exp17_msd_atomic_scatter"),
            pso_precompute: pso("exp17_inner_precompute_hists"),
            pso_fused_v3: pso("exp17_inner_fused_v3"),
            pso_combined_hist: pso("exp16_combined_histogram"),
            pso_global_prefix: pso("exp16_global_prefix"),
            pso_zero_status: pso("exp16_zero_status"),
            pso_partition: pso("exp16_partition"),
            pso_random_fill: pso("gpu_random_fill"),

            buf_a: buf(max_n * 4),            // 256 MB
            buf_b: buf(max_n * 4),            // 256 MB
            buf_msd_hist: buf(256 * 4),       // 1 KB
            buf_counters_17: buf(256 * 4),    // 1 KB
            buf_bucket_descs: buf(256 * 16),  // 4 KB
            buf_inner_hists: buf(256 * 3 * 256 * 4), // 786 KB
            buf_global_hist: buf(4 * 256 * 4), // 4 KB
            buf_tile_status: buf(max_tiles * 256 * 4), // 16 MB
            buf_counters_16: buf(4),          // 4 bytes

            algorithm: SortAlgorithm::Hybrid,
            element_count: 16_000_000,
            last_sort_ms: 0.0,
        }
    }

    pub fn encode_shuffle(&self, cmd: &ProtocolObject<dyn MTLCommandBuffer>) {
        let encoder = cmd
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        let dim = texture_dim(self.element_count);
        let params = DemoParams {
            element_count: self.element_count as u32,
            texture_width: dim,
            texture_height: dim,
            max_value: 0xFFFFFFFF,
        };

        encoder.setComputePipelineState(&self.pso_random_fill);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(self.buf_a.as_ref()), 0, 0);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const DemoParams as *mut DemoParams)
                    .unwrap()
                    .cast(),
                std::mem::size_of::<DemoParams>(),
                1,
            );
        }
        let tg = MTLSize { width: 256, height: 1, depth: 1 };
        let grid = MTLSize {
            width: self.element_count.div_ceil(256),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();
    }

    pub fn encode_sort(&self, cmd: &ProtocolObject<dyn MTLCommandBuffer>) {
        match self.algorithm {
            SortAlgorithm::Hybrid => self.encode_exp17(cmd),
            SortAlgorithm::EightBit => self.encode_exp16(cmd),
        }
    }

    fn encode_exp17(&self, cmd: &ProtocolObject<dyn MTLCommandBuffer>) {
        let n = self.element_count;
        let num_tiles = n.div_ceil(4096);

        // CPU-zero buf_msd_hist
        unsafe {
            std::ptr::write_bytes(
                self.buf_msd_hist.contents().as_ptr() as *mut u8,
                0,
                256 * 4,
            );
        }

        let encoder = cmd
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        let tg256 = MTLSize { width: 256, height: 1, depth: 1 };

        // D1: exp17_msd_histogram
        {
            let params = Exp17Params {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                shift: 24,
                pass: 0,
            };
            encoder.setComputePipelineState(&self.pso_msd_histogram);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.buf_a.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.buf_msd_hist.as_ref()), 0, 1);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const Exp17Params as *mut Exp17Params)
                        .unwrap()
                        .cast(),
                    std::mem::size_of::<Exp17Params>(),
                    2,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tiles, height: 1, depth: 1 },
                tg256,
            );
        }

        // D2: exp17_msd_prep
        {
            let tile_size: u32 = 4096;
            encoder.setComputePipelineState(&self.pso_msd_prep);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.buf_msd_hist.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.buf_counters_17.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.buf_bucket_descs.as_ref()), 0, 2);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&tile_size as *const u32 as *mut u32)
                        .unwrap()
                        .cast(),
                    std::mem::size_of::<u32>(),
                    3,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                tg256,
            );
        }

        // D3: exp17_msd_atomic_scatter
        {
            let params = Exp17Params {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                shift: 24,
                pass: 0,
            };
            encoder.setComputePipelineState(&self.pso_msd_scatter);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.buf_a.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.buf_b.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.buf_counters_17.as_ref()), 0, 2);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const Exp17Params as *mut Exp17Params)
                        .unwrap()
                        .cast(),
                    std::mem::size_of::<Exp17Params>(),
                    3,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tiles, height: 1, depth: 1 },
                tg256,
            );
        }

        // D4: exp17_inner_precompute_hists
        {
            encoder.setComputePipelineState(&self.pso_precompute);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.buf_b.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.buf_inner_hists.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.buf_bucket_descs.as_ref()), 0, 2);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 256, height: 1, depth: 1 },
                tg256,
            );
        }

        // D5: exp17_inner_fused_v3
        {
            encoder.setComputePipelineState(&self.pso_fused_v3);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.buf_a.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.buf_b.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.buf_bucket_descs.as_ref()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(self.buf_inner_hists.as_ref()), 0, 3);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 256, height: 1, depth: 1 },
                tg256,
            );
        }

        encoder.endEncoding();
    }

    fn encode_exp16(&self, cmd: &ProtocolObject<dyn MTLCommandBuffer>) {
        let n = self.element_count;
        let num_tiles = n.div_ceil(4096);

        // CPU-zero buf_global_hist
        unsafe {
            std::ptr::write_bytes(
                self.buf_global_hist.contents().as_ptr() as *mut u8,
                0,
                4 * 256 * 4,
            );
        }

        let tg256 = MTLSize { width: 256, height: 1, depth: 1 };

        // Encoder 1: exp16_combined_histogram
        {
            let encoder = cmd
                .computeCommandEncoder()
                .expect("Failed to create encoder");
            let params = Exp16Params {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                num_tgs: num_tiles as u32,
                shift: 0,
                pass: 0,
            };
            encoder.setComputePipelineState(&self.pso_combined_hist);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.buf_a.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.buf_global_hist.as_ref()), 0, 1);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const Exp16Params as *mut Exp16Params)
                        .unwrap()
                        .cast(),
                    std::mem::size_of::<Exp16Params>(),
                    2,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tiles, height: 1, depth: 1 },
                tg256,
            );
            encoder.endEncoding();
        }

        // Encoder 2: exp16_global_prefix
        {
            let encoder = cmd
                .computeCommandEncoder()
                .expect("Failed to create encoder");
            encoder.setComputePipelineState(&self.pso_global_prefix);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.buf_global_hist.as_ref()), 0, 0);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                tg256,
            );
            encoder.endEncoding();
        }

        // 4 passes: each has zero_status + partition encoder
        for pass in 0u32..4 {
            let (src, dst) = if pass % 2 == 0 {
                (self.buf_a.as_ref(), self.buf_b.as_ref())
            } else {
                (self.buf_b.as_ref(), self.buf_a.as_ref())
            };
            let shift = pass * 8;
            let params = Exp16Params {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                num_tgs: num_tiles as u32,
                shift,
                pass,
            };

            // Zero status encoder
            {
                let encoder = cmd
                    .computeCommandEncoder()
                    .expect("Failed to create encoder");
                encoder.setComputePipelineState(&self.pso_zero_status);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(self.buf_tile_status.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(self.buf_counters_16.as_ref()), 0, 1);
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::new(&params as *const Exp16Params as *mut Exp16Params)
                            .unwrap()
                            .cast(),
                        std::mem::size_of::<Exp16Params>(),
                        2,
                    );
                }
                let zero_threads = num_tiles * 256;
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: zero_threads.div_ceil(256), height: 1, depth: 1 },
                    tg256,
                );
                encoder.endEncoding();
            }

            // Partition encoder
            {
                let encoder = cmd
                    .computeCommandEncoder()
                    .expect("Failed to create encoder");
                encoder.setComputePipelineState(&self.pso_partition);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(self.buf_tile_status.as_ref()), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(self.buf_counters_16.as_ref()), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(self.buf_global_hist.as_ref()), 0, 4);
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::new(&params as *const Exp16Params as *mut Exp16Params)
                            .unwrap()
                            .cast(),
                        std::mem::size_of::<Exp16Params>(),
                        5,
                    );
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: num_tiles, height: 1, depth: 1 },
                    tg256,
                );
                encoder.endEncoding();
            }
        }
    }

    pub fn set_element_count(&mut self, n: usize) {
        self.element_count = n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn exp17_params_size() {
        assert_eq!(size_of::<Exp17Params>(), 16);
    }

    #[test]
    fn exp16_params_size() {
        assert_eq!(size_of::<Exp16Params>(), 20);
    }

    #[test]
    fn bucket_desc_size() {
        assert_eq!(size_of::<BucketDesc>(), 16);
    }
}
