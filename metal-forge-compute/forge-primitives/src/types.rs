//! Shared types matching Metal shader structs.
//!
//! All param structs are #[repr(C)] and 16-byte aligned to match
//! the Metal Shading Language struct layout.

/// Parameters for reduce kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReduceParams {
    pub element_count: u32,
    pub _pad: [u32; 3],
}

/// Parameters for prefix scan kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ScanParams {
    pub element_count: u32,
    pub pass: u32,
    pub _pad: [u32; 2],
}

/// Parameters for histogram kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HistogramParams {
    pub element_count: u32,
    pub num_bins: u32,
    pub _pad: [u32; 2],
}

/// Parameters for stream compaction kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CompactParams {
    pub element_count: u32,
    pub threshold: u32,
    pub _pad: [u32; 2],
}

/// Parameters for radix sort kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SortParams {
    pub element_count: u32,
    pub bit_offset: u32,
    pub num_threadgroups: u32,
    pub _pad: u32,
}

/// Parameters for filter benchmark kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FilterBenchParams {
    pub element_count: u32,
    pub threshold: u32,
    pub _pad: [u32; 2],
}

/// Parameters for GEMM (General Matrix Multiply) kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GemmParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub _pad: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_params_layout() {
        assert_eq!(
            std::mem::size_of::<ReduceParams>(),
            16,
            "ReduceParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<ReduceParams>(),
            4,
            "ReduceParams must be 4-byte aligned"
        );

        // Verify field offsets
        let p = ReduceParams {
            element_count: 0,
            _pad: [0; 3],
        };
        let base = &p as *const _ as usize;
        let element_count_offset =
            &p.element_count as *const _ as usize - base;
        assert_eq!(element_count_offset, 0, "element_count at offset 0");
    }

    #[test]
    fn test_scan_params_layout() {
        assert_eq!(
            std::mem::size_of::<ScanParams>(),
            16,
            "ScanParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<ScanParams>(),
            4,
            "ScanParams must be 4-byte aligned"
        );

        let p = ScanParams {
            element_count: 0,
            pass: 0,
            _pad: [0; 2],
        };
        let base = &p as *const _ as usize;
        let element_count_offset =
            &p.element_count as *const _ as usize - base;
        let pass_offset = &p.pass as *const _ as usize - base;
        assert_eq!(element_count_offset, 0, "element_count at offset 0");
        assert_eq!(pass_offset, 4, "pass at offset 4");
    }

    #[test]
    fn test_histogram_params_layout() {
        assert_eq!(
            std::mem::size_of::<HistogramParams>(),
            16,
            "HistogramParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<HistogramParams>(),
            4,
            "HistogramParams must be 4-byte aligned"
        );

        let p = HistogramParams {
            element_count: 0,
            num_bins: 0,
            _pad: [0; 2],
        };
        let base = &p as *const _ as usize;
        let element_count_offset =
            &p.element_count as *const _ as usize - base;
        let num_bins_offset = &p.num_bins as *const _ as usize - base;
        assert_eq!(element_count_offset, 0, "element_count at offset 0");
        assert_eq!(num_bins_offset, 4, "num_bins at offset 4");
    }

    #[test]
    fn test_compact_params_layout() {
        assert_eq!(
            std::mem::size_of::<CompactParams>(),
            16,
            "CompactParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<CompactParams>(),
            4,
            "CompactParams must be 4-byte aligned"
        );

        let p = CompactParams {
            element_count: 0,
            threshold: 0,
            _pad: [0; 2],
        };
        let base = &p as *const _ as usize;
        let element_count_offset =
            &p.element_count as *const _ as usize - base;
        let threshold_offset = &p.threshold as *const _ as usize - base;
        assert_eq!(element_count_offset, 0, "element_count at offset 0");
        assert_eq!(threshold_offset, 4, "threshold at offset 4");
    }

    #[test]
    fn test_filter_bench_params_layout() {
        assert_eq!(
            std::mem::size_of::<FilterBenchParams>(),
            16,
            "FilterBenchParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<FilterBenchParams>(),
            4,
            "FilterBenchParams must be 4-byte aligned"
        );

        let p = FilterBenchParams {
            element_count: 0,
            threshold: 0,
            _pad: [0; 2],
        };
        let base = &p as *const _ as usize;
        let element_count_offset =
            &p.element_count as *const _ as usize - base;
        let threshold_offset = &p.threshold as *const _ as usize - base;
        assert_eq!(element_count_offset, 0, "element_count at offset 0");
        assert_eq!(threshold_offset, 4, "threshold at offset 4");
    }

    #[test]
    fn test_gemm_params_layout() {
        assert_eq!(
            std::mem::size_of::<GemmParams>(),
            16,
            "GemmParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<GemmParams>(),
            4,
            "GemmParams must be 4-byte aligned"
        );

        let p = GemmParams {
            m: 0,
            n: 0,
            k: 0,
            _pad: 0,
        };
        let base = &p as *const _ as usize;
        let m_offset = &p.m as *const _ as usize - base;
        let n_offset = &p.n as *const _ as usize - base;
        let k_offset = &p.k as *const _ as usize - base;
        assert_eq!(m_offset, 0, "m at offset 0");
        assert_eq!(n_offset, 4, "n at offset 4");
        assert_eq!(k_offset, 8, "k at offset 8");
    }

    #[test]
    fn test_sort_params_layout() {
        assert_eq!(
            std::mem::size_of::<SortParams>(),
            16,
            "SortParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<SortParams>(),
            4,
            "SortParams must be 4-byte aligned"
        );

        let p = SortParams {
            element_count: 0,
            bit_offset: 0,
            num_threadgroups: 0,
            _pad: 0,
        };
        let base = &p as *const _ as usize;
        let element_count_offset =
            &p.element_count as *const _ as usize - base;
        let bit_offset_offset = &p.bit_offset as *const _ as usize - base;
        let num_tg_offset = &p.num_threadgroups as *const _ as usize - base;
        assert_eq!(element_count_offset, 0, "element_count at offset 0");
        assert_eq!(bit_offset_offset, 4, "bit_offset at offset 4");
        assert_eq!(num_tg_offset, 8, "num_threadgroups at offset 8");
    }
}
