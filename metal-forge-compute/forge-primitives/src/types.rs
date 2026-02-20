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

/// Parameters for group-by aggregate kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GroupByParams {
    pub element_count: u32,
    pub num_groups: u32,
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

/// Parameters for spreadsheet formula kernels.
/// formula_type: 0 = SUM, 1 = AVERAGE, 2 = VLOOKUP
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SpreadsheetParams {
    pub rows: u32,
    pub cols: u32,
    pub formula_type: u32,
    pub _pad: u32,
}

/// Parameters for time series analytics kernels.
/// op_type: 0 = moving average, 1 = VWAP, 2 = bollinger (moving avg for POC)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TimeSeriesParams {
    pub tick_count: u32,
    pub window_size: u32,
    pub op_type: u32,
    pub _pad: u32,
}

/// Parameters for hash join kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HashJoinParams {
    pub build_count: u32,
    pub probe_count: u32,
    pub table_size: u32,
    pub _pad: u32,
}

/// Parameters for CSV bench kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CsvBenchParams {
    pub byte_count: u32,
    pub _pad: [u32; 3],
}

/// Parameters for GPU exploit experiments.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExploitParams {
    pub element_count: u32,
    pub num_passes: u32,
    pub mode: u32,
    pub _pad: u32,
}

/// Parameters for GPU OS primitive experiments.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuOsParams {
    pub capacity: u32,
    pub num_ops: u32,
    pub num_queues: u32,
    pub mode: u32,
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
        let element_count_offset = &p.element_count as *const _ as usize - base;
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
        let element_count_offset = &p.element_count as *const _ as usize - base;
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
        let element_count_offset = &p.element_count as *const _ as usize - base;
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
        let element_count_offset = &p.element_count as *const _ as usize - base;
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
        let element_count_offset = &p.element_count as *const _ as usize - base;
        let threshold_offset = &p.threshold as *const _ as usize - base;
        assert_eq!(element_count_offset, 0, "element_count at offset 0");
        assert_eq!(threshold_offset, 4, "threshold at offset 4");
    }

    #[test]
    fn test_groupby_params_layout() {
        assert_eq!(
            std::mem::size_of::<GroupByParams>(),
            16,
            "GroupByParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<GroupByParams>(),
            4,
            "GroupByParams must be 4-byte aligned"
        );

        let p = GroupByParams {
            element_count: 0,
            num_groups: 0,
            _pad: [0; 2],
        };
        let base = &p as *const _ as usize;
        let element_count_offset = &p.element_count as *const _ as usize - base;
        let num_groups_offset = &p.num_groups as *const _ as usize - base;
        assert_eq!(element_count_offset, 0, "element_count at offset 0");
        assert_eq!(num_groups_offset, 4, "num_groups at offset 4");
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
        let element_count_offset = &p.element_count as *const _ as usize - base;
        let bit_offset_offset = &p.bit_offset as *const _ as usize - base;
        let num_tg_offset = &p.num_threadgroups as *const _ as usize - base;
        assert_eq!(element_count_offset, 0, "element_count at offset 0");
        assert_eq!(bit_offset_offset, 4, "bit_offset at offset 4");
        assert_eq!(num_tg_offset, 8, "num_threadgroups at offset 8");
    }

    #[test]
    fn test_spreadsheet_params_layout() {
        assert_eq!(
            std::mem::size_of::<SpreadsheetParams>(),
            16,
            "SpreadsheetParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<SpreadsheetParams>(),
            4,
            "SpreadsheetParams must be 4-byte aligned"
        );

        let p = SpreadsheetParams {
            rows: 0,
            cols: 0,
            formula_type: 0,
            _pad: 0,
        };
        let base = &p as *const _ as usize;
        let rows_offset = &p.rows as *const _ as usize - base;
        let cols_offset = &p.cols as *const _ as usize - base;
        let formula_offset = &p.formula_type as *const _ as usize - base;
        assert_eq!(rows_offset, 0, "rows at offset 0");
        assert_eq!(cols_offset, 4, "cols at offset 4");
        assert_eq!(formula_offset, 8, "formula_type at offset 8");
    }

    #[test]
    fn test_timeseries_params_layout() {
        assert_eq!(
            std::mem::size_of::<TimeSeriesParams>(),
            16,
            "TimeSeriesParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<TimeSeriesParams>(),
            4,
            "TimeSeriesParams must be 4-byte aligned"
        );

        let p = TimeSeriesParams {
            tick_count: 0,
            window_size: 0,
            op_type: 0,
            _pad: 0,
        };
        let base = &p as *const _ as usize;
        let tick_offset = &p.tick_count as *const _ as usize - base;
        let window_offset = &p.window_size as *const _ as usize - base;
        let op_offset = &p.op_type as *const _ as usize - base;
        assert_eq!(tick_offset, 0, "tick_count at offset 0");
        assert_eq!(window_offset, 4, "window_size at offset 4");
        assert_eq!(op_offset, 8, "op_type at offset 8");
    }

    #[test]
    fn test_hash_join_params_layout() {
        assert_eq!(
            std::mem::size_of::<HashJoinParams>(),
            16,
            "HashJoinParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<HashJoinParams>(),
            4,
            "HashJoinParams must be 4-byte aligned"
        );

        let p = HashJoinParams {
            build_count: 0,
            probe_count: 0,
            table_size: 0,
            _pad: 0,
        };
        let base = &p as *const _ as usize;
        let build_offset = &p.build_count as *const _ as usize - base;
        let probe_offset = &p.probe_count as *const _ as usize - base;
        let table_offset = &p.table_size as *const _ as usize - base;
        assert_eq!(build_offset, 0, "build_count at offset 0");
        assert_eq!(probe_offset, 4, "probe_count at offset 4");
        assert_eq!(table_offset, 8, "table_size at offset 8");
    }

    #[test]
    fn test_csv_bench_params_layout() {
        assert_eq!(
            std::mem::size_of::<CsvBenchParams>(),
            16,
            "CsvBenchParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<CsvBenchParams>(),
            4,
            "CsvBenchParams must be 4-byte aligned"
        );

        let p = CsvBenchParams {
            byte_count: 0,
            _pad: [0; 3],
        };
        let base = &p as *const _ as usize;
        let byte_count_offset = &p.byte_count as *const _ as usize - base;
        assert_eq!(byte_count_offset, 0, "byte_count at offset 0");
    }

    #[test]
    fn test_exploit_params_layout() {
        assert_eq!(
            std::mem::size_of::<ExploitParams>(),
            16,
            "ExploitParams must be 16 bytes"
        );

        let p = ExploitParams {
            element_count: 0,
            num_passes: 0,
            mode: 0,
            _pad: 0,
        };
        let base = &p as *const _ as usize;
        let element_count_offset = &p.element_count as *const _ as usize - base;
        let num_passes_offset = &p.num_passes as *const _ as usize - base;
        let mode_offset = &p.mode as *const _ as usize - base;
        assert_eq!(element_count_offset, 0, "element_count at offset 0");
        assert_eq!(num_passes_offset, 4, "num_passes at offset 4");
        assert_eq!(mode_offset, 8, "mode at offset 8");
    }

    #[test]
    fn test_all_params_are_16_bytes() {
        // Verify all param structs are exactly 16 bytes for Metal alignment
        assert_eq!(std::mem::size_of::<ReduceParams>(), 16);
        assert_eq!(std::mem::size_of::<ScanParams>(), 16);
        assert_eq!(std::mem::size_of::<HistogramParams>(), 16);
        assert_eq!(std::mem::size_of::<CompactParams>(), 16);
        assert_eq!(std::mem::size_of::<SortParams>(), 16);
        assert_eq!(std::mem::size_of::<FilterBenchParams>(), 16);
        assert_eq!(std::mem::size_of::<GroupByParams>(), 16);
        assert_eq!(std::mem::size_of::<GemmParams>(), 16);
        assert_eq!(std::mem::size_of::<SpreadsheetParams>(), 16);
        assert_eq!(std::mem::size_of::<TimeSeriesParams>(), 16);
        assert_eq!(std::mem::size_of::<HashJoinParams>(), 16);
        assert_eq!(std::mem::size_of::<CsvBenchParams>(), 16);
        assert_eq!(std::mem::size_of::<ExploitParams>(), 16);
        assert_eq!(std::mem::size_of::<GpuOsParams>(), 16);
    }

    #[test]
    fn test_gpuos_params_layout() {
        assert_eq!(
            std::mem::size_of::<GpuOsParams>(),
            16,
            "GpuOsParams must be 16 bytes"
        );

        let p = GpuOsParams {
            capacity: 0,
            num_ops: 0,
            num_queues: 0,
            mode: 0,
        };
        let base = &p as *const _ as usize;
        let capacity_offset = &p.capacity as *const _ as usize - base;
        let num_ops_offset = &p.num_ops as *const _ as usize - base;
        let num_queues_offset = &p.num_queues as *const _ as usize - base;
        let mode_offset = &p.mode as *const _ as usize - base;
        assert_eq!(capacity_offset, 0, "capacity at offset 0");
        assert_eq!(num_ops_offset, 4, "num_ops at offset 4");
        assert_eq!(num_queues_offset, 8, "num_queues at offset 8");
        assert_eq!(mode_offset, 12, "mode at offset 12");
    }
}
