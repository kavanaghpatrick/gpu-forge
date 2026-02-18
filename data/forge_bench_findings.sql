-- forge-bench kernel rewrite findings
-- Verified patterns from Apple M4 Pro benchmarks (10M elements, 10 runs, 3 warmup)
-- All speedups vs optimized CPU baselines (Accelerate BLAS, hand-tuned HashMap, etc.)

-- ============================================================================
-- CATEGORY 1: GPU-Side Timing (metal-compute, skill_id=3)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3086, 3, 'gpu-side timing',
'GPU-side timing via GPUStartTime()/GPUEndTime() on MTLCommandBuffer eliminates ~300us firmware scheduling overhead from benchmarks. Wall-clock BenchTimer measures dispatch+scheduling+execution; GpuTimer measures only GPU execution. At 10M elements the difference is 5-15%; at 100K it can be 50%+.',
'Implemented GpuTimer in forge-bench: after commit()+waitUntilCompleted(), read cmd_buf.GPUStartTime() and cmd_buf.GPUEndTime() (both f64 seconds). Returns None if timestamps are 0.0 (buffer not completed). Applied to all 15 experiments. hash_join went from 35.5x to 88.3x at 10M when switching from wall-clock to GPU-side timing, proving firmware overhead was dominating the measurement at that dispatch pattern.',
'forge-bench/forge-primitives/src/timing.rs', 'forge-bench GpuTimer implementation', 'empirical_test',
'high', '2026-02-17',
'gpu-timing,metal,benchmark,GPUStartTime,GPUEndTime,MTLCommandBuffer,firmware-overhead',
'Pattern: let start = cmd_buf.GPUStartTime(); let end = cmd_buf.GPUEndTime(); return (end - start) * 1000.0. MUST call after waitUntilCompleted(). For multi-cmdbuf pipelines, wall-clock is still appropriate since GpuTimer only measures one command buffer.',
'm4', 'current');

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3087, 3, 'multi-cmdbuf timing',
'Multi-command-buffer GPU pipelines (compact, groupby, pipeline, duckdb) cannot use GPU-side timing from a single command buffer. When GPU work is split across multiple commit()+waitUntilCompleted() calls with CPU work between them, wall-clock timing is the only way to capture total pipeline time including CPU-GPU synchronization overhead.',
'Tested on forge-bench: compact (3 cmd_bufs for scan), groupby (sort loop + aggregation), pipeline (filter+scan+scatter), duckdb (filter+scan+scatter+groupby). GpuTimer on just the last cmd_buf would miss 60-80% of actual GPU work. These experiments kept BenchTimer for run_gpu() while single-cmdbuf experiments (hash_join, filter, reduce, etc.) switched to GpuTimer.',
'forge-bench experiments', 'forge-bench multi-cmdbuf analysis', 'empirical_test',
'high', '2026-02-17',
'multi-cmdbuf,wall-clock,timing,pipeline,benchmark',
'Consolidating multi-cmdbuf experiments into single command buffer is the right fix (not changing the timer). The multi-cmdbuf overhead itself is the performance problem.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 2: Vectorized Loads Pattern (msl-kernels, skill_id=4)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3088, 4, 'uint4 vectorized load pattern',
'The load_uint4_safe() / load_float4_safe() pattern loads 4 elements per thread with bounds checking, achieving 2-4x throughput vs scalar loads. Each thread processes 4 elements, reducing dispatch size by 4x while maintaining full GPU occupancy. Pattern: uint4 vals = load_uint4_safe(input, tid * 4, element_count); uint sum = vals.x + vals.y + vals.z + vals.w.',
'Applied to reduce, histogram, filter, scan kernels in forge-bench. Reduce v2: 4 elements/thread with load_uint4_safe reduced threadgroup count 4x. Filter: 7.0x GPU vs CPU at 10M (up from ~2x with scalar). Histogram: 4.5x at 10M. The helper function zero-initializes out-of-bounds elements, avoiding branching in the main kernel body.',
'forge-primitives/shaders/types.h', 'forge-bench vectorized load implementation', 'empirical_test',
'high', '2026-02-17',
'uint4,float4,vectorized-load,bounds-check,throughput,msl',
'Code in types.h: inline uint4 load_uint4_safe(device const uint* data, uint base_idx, uint element_count) { uint4 result = uint4(0); if (base_idx < element_count) result.x = data[base_idx]; if (base_idx+1 < element_count) result.y = data[base_idx+1]; ... return result; } Dispatch: ceil(N / (TG_SIZE * 4)) threadgroups.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 3: SIMD Reduction Pattern (simd-wave, skill_id=7)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3089, 7, '3-level SIMD reduction pattern',
'The canonical Apple Silicon reduction is 3 levels: (1) simd_sum/simd_min/simd_max within simdgroup (32 lanes, zero-cost hardware), (2) threadgroup shared memory with barrier for cross-SIMD aggregation (8 SIMD groups × 256 threads), (3) global atomic or partials array. This pattern achieves 45% bandwidth utilization (123 GB/s on M4 Pro 273 GB/s) at 10M u32 elements.',
'Implemented in forge-bench reduce.metal: Level 1 uses simd_sum() (single-cycle hardware reduction). Level 2 writes simd_lane==0 to tg_partials[simd_group_id], barriers, then simd_group_id==0 does simd_sum of partials. Level 3 options: atomic_fetch_add (v1, contention at high TG count) or per-TG partials array (v2, atomic-free). v2 with vectorized loads achieves 0.325ms for 10M u32 sum.',
'forge-primitives/shaders/reduce.metal', 'forge-bench reduce implementation', 'empirical_test',
'high', '2026-02-17',
'simd_sum,simd_min,simd_max,reduction,3-level,threadgroup,shared-memory,barrier',
'Key insight: the tg_partials shared memory array only needs MAX_THREADS/32 entries (8 for 256 threads). The second simd_sum in Level 2 operates on just 8 values loaded from shared memory. This is more efficient than a log-step Blelloch tree in shared memory.',
'm4', 'current');

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3090, 7, 'simd_prefix_exclusive_sum for scan and radix sort',
'simd_prefix_exclusive_sum() is the key building block for both prefix scan and radix sort on Apple Silicon. For scan: each thread computes prefix within its SIMD group, SIMD group totals are stored to shared memory, a cross-SIMD scan is performed, then base offsets are combined. For radix sort scatter: simd_prefix_exclusive_sum(match) per digit gives stable intra-SIMD rank without shared memory atomic contention.',
'Scan at 10M: 5.0x GPU vs CPU (0.662ms GPU, 3.338ms CPU). Sort at 10M: 8.2x GPU vs CPU (12.170ms GPU, 99.873ms CPU). Scan processes 1024 elements per TG (256 threads × 4 elements). The cross-SIMD step uses only 8 threads (one per SIMD group) doing simd_prefix_exclusive_sum on shared memory totals. Sort scatter replaced quadratic O(N²) local ranking loop with O(16) SIMD prefix passes.',
'forge-primitives/shaders/scan.metal, radix_sort.metal', 'forge-bench scan + sort implementation', 'empirical_test',
'high', '2026-02-17',
'simd_prefix_exclusive_sum,prefix-scan,radix-sort,exclusive-scan,3-pass,reduce-then-scan',
'CRITICAL: Decoupled lookback (single-pass scan) DEADLOCKS on Apple Silicon. Must use reduce-then-scan (3-pass) approach. The 3 passes are: (1) scan_local - per-TG SIMD scan + write TG total to partials, (2) scan_partials - single TG scans the partials array, (3) scan_add_offsets - add scanned partials to each element.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 4: simdgroup_matrix GEMM (simd-wave, skill_id=7)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3091, 7, 'simdgroup_matrix_multiply_accumulate GEMM',
'simdgroup_float8x8 with simdgroup_multiply_accumulate() achieves 1.7x vs Apple Accelerate BLAS at 1024x1024 FP32 GEMM on M4 Pro. Uses 8 SIMD groups per TG (256 threads) arranged 2×4, each computing two stacked 8×8 output tiles (16×8 per SIMD group, 32×32 per TG). Bank-conflict padding (+4 columns) in shared memory tiles is essential.',
'gemm_simdgroup_f32 kernel: BM=BN=BK=32, PAD=4. Cooperative loading: 256 threads load 32×32 tile (4 batches of 8 rows). Inner loop: 4 MAC steps (kk += 8) per BK=32 tile, loading simdgroup_float8x8 from tileA and tileB with padded stride. Output: simdgroup_store of two 8×8 accumulators per SIMD group. Result: 0.651ms GPU vs 1.079ms Accelerate at 1024. At 4096: 47.85ms GPU vs 46.63ms Accelerate (1.0x, matching Apple''s hand-tuned BLAS).',
'forge-primitives/shaders/gemm.metal', 'forge-bench GEMM simdgroup implementation', 'empirical_test',
'high', '2026-02-17',
'simdgroup_matrix,simdgroup_float8x8,simdgroup_multiply_accumulate,GEMM,matrix-multiply,bank-conflict,shared-memory-padding',
'Layout: 8 SIMD groups: simd_row = simd_id >> 2 (0..1), simd_col = simd_id & 3 (0..3). Each SIMD group owns acc0 and acc1 (two 8x8 tiles stacked vertically = 16x8). This gives 2 SIMD rows × 4 SIMD cols × 8 wide = 32×32 per TG. Bank conflict avoidance: tileA[32][32+4], tileB[32][32+4]. Dispatch: (ceil(N/32), ceil(M/32)) TGs of 256 threads.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 5: Histogram Bitmask Optimization (msl-kernels, skill_id=4)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3092, 4, 'bitmask binning for power-of-2 histogram',
'Replacing modulo (value % num_bins) with bitmask (value & (num_bins - 1)) for power-of-2 bin counts eliminates integer division entirely. Integer division on Apple GPU costs 20-40 cycles; bitwise AND is 1 cycle. Combined with uint4 vectorized loads and threadgroup-local histogram (256 bins = 1KB shared memory), achieves 4.5x GPU vs CPU at 10M elements.',
'histogram_256 kernel: is_power_of_2 = (num_bins & (num_bins - 1)) == 0; mask = num_bins - 1. For each of 4 loaded elements: bin = is_power_of_2 ? (val & mask) : (val % num_bins). TG-local atomic histogram avoids global atomic contention. Final merge: threads cooperatively write local_hist bins to global_hist via atomic_fetch_add. Measured 0.525ms GPU at 10M (76.2 GB/s, 28% bandwidth utilization).',
'forge-primitives/shaders/histogram.metal', 'forge-bench histogram implementation', 'empirical_test',
'high', '2026-02-17',
'histogram,bitmask,modulo,integer-division,threadgroup-atomic,shared-memory,vectorized-load',
'The threadgroup-local histogram pattern: (1) Initialize 256-bin shared memory histogram to zero. (2) Each thread processes 4 elements via load_uint4_safe, atomically increments local_hist[bin]. (3) Barrier. (4) Cooperative merge: threads stripe over bins writing local counts to global histogram. TG atomics are cheap (~1-2 cycles for uncontended access in shared memory) vs global atomics.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 6: 2D Dispatch for Coalesced Access (gpu-perf, skill_id=6)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3093, 6, '2D dispatch for coalesced column access',
'Row-major 2D grids with column-wise operations (SUM, AVERAGE) should use 2D dispatch (tid.x=column, tid.y=row_chunk) so adjacent threads read adjacent columns (coalesced). The naive approach (one thread per column iterating rows) causes strided memory access. The 2D row-chunked approach partitions rows into chunks of 64, creating many more threadgroups and enabling parallel row processing. Result: 4.2x GPU vs CPU at 10M (up from 0.1x with strided access).',
'spreadsheet_sum_v2: 2D dispatch with SS_ROWS_PER_CHUNK=64. Each thread sums rows [row_chunk*64 .. (row_chunk+1)*64) for one column. Writes partial to partials[row_chunk * cols + col]. A separate reduce kernel sums partials per column. Dispatch: (ceil(cols/TG_X), ceil(rows/64)) TGs. This creates ~50x more TGs than 1D (e.g. 600+ vs 13), fully utilizing the GPU. Memory pattern: grid[row * cols + col] with tid.x = col means adjacent threads in same simdgroup read adjacent float values = perfect coalescing.',
'forge-primitives/shaders/spreadsheet.metal', 'forge-bench spreadsheet 2D dispatch', 'empirical_test',
'high', '2026-02-17',
'2d-dispatch,coalesced-access,column-reduction,row-chunking,memory-coalescing,strided-access',
'The key insight: 1D dispatch (one thread per column) gives only ceil(cols/256) TGs. With 3162 cols and 3162 rows, that is just 13 TGs — not enough to saturate a 20-core GPU. 2D dispatch: ceil(3162/256) × ceil(3162/64) = 13 × 50 = 650 TGs. The reduce step adds negligible overhead (ceil(rows/64) partial sums per column). Achieved 87.6% bandwidth utilization (239.2 GB/s on 273 GB/s M4 Pro).',
'm4', 'current');

-- ============================================================================
-- CATEGORY 7: Radix Sort Architecture (msl-kernels, skill_id=4)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3094, 4, 'reduce-then-scan radix sort on Apple Silicon',
'4-bit radix sort with reduce-then-scan (NOT OneSweep/decoupled lookback) achieves 8.2x vs std::sort at 10M u32 elements on M4 Pro. Architecture: 8 passes × 3 dispatches = 24 dispatches total (histogram + prefix scan + scatter per pass). Digit-major histogram layout: histogram[digit * num_tg + tg_idx] enables the prefix scan to produce correct global scatter positions.',
'radix_sort.metal + sort.rs: 8 passes (32 bits / 4 bits), each pass: (1) radix_histogram: per-TG 16-bin histogram using shared memory atomics, output in digit-major order. (2) prefix scan: reuse existing scan_local + scan_partials + scan_add_offsets on histogram array of size 16 * num_tg. (3) radix_scatter: SIMD prefix rank for stable local ordering + global scatter. Double-buffer ping-pong between keys_a and keys_b. 12.170ms at 10M = 822K elements/ms.',
'forge-primitives/shaders/radix_sort.metal', 'forge-bench radix sort implementation', 'empirical_test',
'high', '2026-02-17',
'radix-sort,4-bit,reduce-then-scan,digit-major,histogram,scatter,double-buffer,stable-sort',
'CRITICAL: OneSweep (single-pass radix sort) uses decoupled lookback which DEADLOCKS on Apple Silicon GPUs. Apple''s GPU does not guarantee forward progress across threadgroups — a threadgroup waiting for a previous threadgroup''s output will never unblock if that previous threadgroup hasn''t been scheduled. Must use traditional multi-pass reduce-then-scan approach.',
'm4', 'current');

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3095, 4, 'SIMD prefix rank for stable radix scatter',
'SIMD prefix rank replaces the quadratic O(N²) local ranking loop in radix scatter. For each of 16 digit values: match = (my_digit == d) ? 1 : 0; prefix = simd_prefix_exclusive_sum(match); count = simd_sum(match). This gives each thread its intra-SIMD rank for its digit in 16 passes of O(32) = O(512) total work vs O(256²) = O(65536) for the naive approach. Cross-SIMD offset via simd_digit_counts[8][16] shared memory array.',
'radix_scatter kernel: Phase 1 iterates 16 digit values, computing SIMD prefix and storing per-SIMD-group digit counts. Phase 2 sums digit counts from all SIMD groups before current one (8 groups max). Phase 3 scatters: global_pos = scanned_hist[digit * num_tg + tg_idx] + cross_offset + my_rank_in_simd. This is a stable sort — elements with the same digit preserve their relative order.',
'forge-primitives/shaders/radix_sort.metal', 'forge-bench radix scatter implementation', 'empirical_test',
'high', '2026-02-17',
'simd-prefix,radix-scatter,stable-sort,local-rank,simd_prefix_exclusive_sum,digit-count',
'The simd_digit_counts[NUM_SIMD_GROUPS][RADIX_BINS] (8×16 = 128 uints) shared memory array enables cross-SIMD offset computation without atomics. Each SIMD group''s last lane (simd_lane==31) writes its digit counts. After barrier, each thread sums counts from all SIMD groups with index < its own. Total shared memory: 128 × 4 = 512 bytes.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 8: Prefix Scan Architecture (msl-kernels, skill_id=4)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3096, 4, '3-pass SIMD prefix scan at 120 GB/s',
'3-pass exclusive prefix scan using SIMD intrinsics processes 1024 elements per threadgroup (256 threads × 4 elements each). Pass 1 (scan_local): per-TG SIMD prefix scan + write TG total to partials. Pass 2 (scan_partials): single TG scans the partials array (handles up to 1024 partials = 1M elements). Pass 3 (scan_add_offsets): add scanned partials to each element. Achieves 5.0x vs CPU at 10M (120.8 GB/s, 44.3% bandwidth).',
'scan.metal implementation: Step 1: uint4 load via load_uint4_safe. Step 2: thread_total = sum of 4 values. Step 3: simd_prefix = simd_prefix_exclusive_sum(thread_total). Step 4: simd_lane==31 writes simd_prefix+thread_total to simd_totals[simd_idx] shared memory. Step 5: first 8 threads do simd_prefix_exclusive_sum on simd_totals. Step 6: base_offset = simd_prefix + simd_totals[simd_idx]. Step 7: write 4 running prefix sums per thread.',
'forge-primitives/shaders/scan.metal', 'forge-bench scan implementation', 'empirical_test',
'high', '2026-02-17',
'prefix-scan,exclusive-scan,3-pass,simd-prefix,1024-elements-per-tg,reduce-then-scan',
'Maximum supported size with 2-level scan: 1024 partials × 1024 elements/partial = 1,048,576 elements (~1M). For larger inputs, need 3-level scan (scan of scan partials) or CPU fallback for the partials scan. The scan_partials kernel is identical in structure to scan_local but operates on the smaller partials array. scan_add_offsets is a simple broadcast: each thread adds partials[tg_idx] to its 4 output elements.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 9: Hash Join Pattern (msl-kernels, skill_id=4)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3097, 4, 'GPU open-addressing hash join achieves 88x vs CPU HashMap',
'Open-addressing hash table with atomic CAS build + linear probe achieves 88.3x vs Rust std::HashMap at 10M keys on M4 Pro. Build phase: atomic_compare_exchange_weak on [key, value] pairs (last-writer-wins for duplicates). Probe phase: linear probe until key match or EMPTY sentinel. Single command buffer with two encoders (build → probe). Hash table sized at next_power_of_two(2 × build_count) for load factor < 0.5.',
'hash_join.metal: Build kernel inserts keys via atomic CAS into hash_table[slot * 2] (key) and hash_table[slot * 2 + 1] (value). Linear probing on collision. Probe kernel: for each probe key, linear probe until match or empty slot. On match: atomic output pair count + write (build_idx, probe_idx) pair. 12.551ms for 10M build + 10M probe = 20M total keys. CPU HashMap: 1107.655ms (dominated by random access cache misses).',
'forge-primitives/shaders/hash_join.metal', 'forge-bench hash join implementation', 'empirical_test',
'high', '2026-02-17',
'hash-join,open-addressing,atomic-cas,linear-probe,database,gpu-join',
'Why GPU wins so dramatically: CPU HashMap lookup is dominated by random cache misses (L2/L3 miss for each probe). GPU hides latency via massive parallelism — thousands of in-flight probes mask individual memory access latency. The 88.3x speedup is the largest in the entire forge-bench suite. Validation: GPU uses last-writer-wins (dedup) vs CPU multimap, so validation compares against dedup CPU baseline.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 10: Timeseries Float4 Window (msl-kernels, skill_id=4)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3098, 4, 'float4 vectorized window reads for moving average',
'Moving average kernel with float4 vectorized inner loop achieves 10.7x vs CPU at 10M elements (1.067ms GPU, 11.381ms CPU) with 787 GB/s effective bandwidth (288% of theoretical — indicates strong SLC cache reuse). Inner loop reads 4 contiguous floats at a time: float4 chunk = float4(prices[i], prices[i+1], prices[i+2], prices[i+3]); sum += chunk.x + chunk.y + chunk.z + chunk.w. Scalar remainder loop for last 0-3 elements.',
'timeseries.metal: timeseries_moving_avg kernel. Each thread computes one output element''s moving average over a window of configurable size. The window sum loop processes 4 elements per iteration via float4 construction (not float4 load — the data is strided per output element). Cache reuse: adjacent threads have overlapping windows, so data loaded by one SIMD group is reused by neighboring threads in the SLC.',
'forge-primitives/shaders/timeseries.metal', 'forge-bench timeseries implementation', 'empirical_test',
'high', '2026-02-17',
'float4,moving-average,timeseries,window-function,cache-reuse,vectorized-loop',
'The >100% bandwidth utilization indicates SLC cache is servicing most reads. With window_size=20 and 10M elements, each element is read by ~20 threads (overlap). The M4 Pro SLC is 48MB — at 40MB data (10M × 4 bytes), nearly all data fits in SLC after first pass. This is a best-case scenario for GPU: embarrassingly parallel with high cache reuse.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 11: Filter with SIMD Reduce (msl-kernels, skill_id=4)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3099, 4, 'filter count with uint4 loads + simd_sum reduction',
'Columnar filter (count elements > threshold) using uint4 vectorized loads + simd_sum SIMD reduction achieves 7.0x vs CPU at 10M elements. Pattern: load 4 elements via load_uint4_safe, evaluate predicate on each component, sum matches (0-4), simd_sum across SIMD group, lane 0 atomically adds to global count. This reduces global atomic operations by 128x (32 lanes × 4 elements per thread).',
'filter_count_gt kernel: uint4 vals = load_uint4_safe(input, gid*4, element_count); uint match_val = uint(vals.x > threshold) + uint(vals.y > threshold) + uint(vals.z > threshold) + uint(vals.w > threshold); uint simd_matches = simd_sum(match_val); if (simd_lane == 0 && simd_matches > 0) atomic_fetch_add(output, simd_matches). 0.546ms at 10M, 73.3 GB/s.',
'forge-primitives/shaders/filter_bench.metal', 'forge-bench filter implementation', 'empirical_test',
'high', '2026-02-17',
'filter,predicate,uint4,simd_sum,atomic-reduction,columnar',
'The guard (simd_matches > 0) before atomic_fetch_add eliminates no-op atomic operations when entire SIMD groups have zero matches. For selective predicates (few matches), this significantly reduces atomic contention. For non-selective predicates (most match), the SIMD reduction still reduces atomics by 32×.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 12: Benchmark Results Summary (gpu-perf, skill_id=6)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3100, 6, 'forge-bench complete benchmark results M4 Pro',
'Complete forge-bench results on Apple M4 Pro (20-core GPU, 273 GB/s LPDDR5X) at 10M elements, 10 runs, 3 warmup: hash_join 88.3x DOMINANT, timeseries 10.7x DOMINANT, sort 8.2x STRONG, filter 7.0x STRONG, scan 5.0x STRONG, histogram 4.5x SOLID, spreadsheet 4.2x SOLID, json_parse 2.9x SOLID, gemm@1024 1.7x MARGINAL, duckdb 1.5x MARGINAL, compact 1.2x MARGINAL, gemv@4096 1.0x MARGINAL, reduce 0.9x SLOWER, groupby 0.6x SLOWER, pipeline 0.6x SLOWER.',
'15 GPU compute experiments benchmarked against optimized CPU baselines: Accelerate cblas_sgemm for GEMM/GEMV, std::sort for sort, std::HashMap for hash_join, hand-optimized scalar code for others. 11 of 15 experiments GPU >= 1.0x. The 3 slower experiments (reduce, groupby, pipeline) are limited by multi-command-buffer dispatch overhead or competing against Apple''s hardware-accelerated Accelerate framework.',
'forge-bench full suite', 'forge-bench M4 Pro benchmark results', 'benchmark',
'high', '2026-02-17',
'benchmark,m4-pro,speedup,forge-bench,gpu-vs-cpu,apple-silicon',
'Key takeaways: (1) GPU dominates random-access patterns (hash_join 88x — CPU cache misses killed by GPU latency hiding). (2) GPU strong for parallel reductions with SIMD intrinsics (sort 8.2x, filter 7.0x, scan 5.0x). (3) GPU struggles vs Accelerate BLAS (GEMM 1.7x at 1024, 1.0x at 4096 — Accelerate uses AMX). (4) Multi-cmdbuf overhead kills composite experiments (groupby 0.6x). (5) Simple reduction (reduce 0.9x) limited by Accelerate vDSP speed.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 13: Anti-Patterns and Lessons (gpu-perf, skill_id=6)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3101, 6, 'decoupled lookback deadlocks on Apple Silicon',
'Decoupled lookback (used in OneSweep single-pass radix sort and CUB-style single-pass prefix scan) DEADLOCKS on Apple Silicon GPUs. The algorithm requires that threadgroup N waits for threadgroup N-1''s output before proceeding. Apple''s GPU scheduler does not guarantee forward progress for threadgroups already dispatched — if threadgroup N-1 hasn''t started, threadgroup N spins forever.',
'Verified during forge-bench scan and sort kernel development. Initial attempt to implement CUB-style decoupled lookback scan resulted in GPU timeout (metal command buffer timeout). Switching to traditional 3-pass reduce-then-scan approach resolved the issue. The 3-pass approach has 3x dispatch overhead but each pass has no inter-threadgroup dependencies, avoiding the deadlock.',
'forge-bench development', 'forge-bench deadlock investigation', 'empirical_test',
'high', '2026-02-17',
'deadlock,decoupled-lookback,onesweep,forward-progress,apple-silicon,anti-pattern',
'This is a fundamental architectural difference between Apple GPUs and NVIDIA/AMD GPUs. NVIDIA guarantees independent thread scheduling (since Volta). AMD guarantees wave-level forward progress. Apple provides NO such guarantee — all threadgroups in a dispatch are peers with no ordering guarantee. Any algorithm relying on threadgroup ordering will deadlock.',
'm4', 'current');

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3102, 6, 'multi-command-buffer overhead kills composite GPU pipelines',
'Multiple commit()+waitUntilCompleted() calls per frame cause 0.3-1.0ms overhead PER synchronization point. Composite experiments (groupby: sort+aggregate, pipeline: filter+scan+scatter, compact: predicate+scan+scatter) that use multiple command buffers with CPU work between them lose 50-80% of their time to synchronization overhead. Groupby: 148.6ms GPU vs 95.3ms CPU (0.6x). Pipeline: 79.4ms GPU vs 45.5ms CPU (0.6x).',
'Measured in forge-bench: Each commit()+waitUntilCompleted() round-trip adds ~300us of firmware scheduling overhead. A composite pipeline with 6 dispatches × 2 sync points = ~1.8ms of pure overhead. When the actual GPU compute is only 2-5ms, synchronization dominates. The fix is consolidating into a single command buffer with multiple compute encoders — demonstrated by sort (single cmdbuf, 8.2x) vs pipeline (multi cmdbuf, 0.6x).',
'forge-bench composite experiments', 'forge-bench synchronization overhead analysis', 'empirical_test',
'high', '2026-02-17',
'multi-cmdbuf,synchronization,overhead,commit,waitUntilCompleted,anti-pattern,single-cmdbuf',
'RULE: Use a single command buffer with multiple compute command encoders for multi-kernel pipelines. Create encoder, dispatch, end encoding, create next encoder, dispatch, end encoding — all before calling commit() once. This eliminates CPU-GPU round trips. sort.rs demonstrates this: 8 passes × (histogram + scan × 3 + scatter) = 40 dispatches in ONE command buffer = 8.2x speedup.',
'm4', 'current');

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3103, 6, 'GPU reduce slower than Accelerate vDSP at 10M',
'Simple reduction (sum of 10M u32 values) is 0.9x GPU vs CPU because Apple Accelerate vDSP uses NEON SIMD + AMX co-processor for array summation. The GPU kernel achieves 123 GB/s (45% bandwidth) which is respectable, but vDSP achieves ~140 GB/s on the CPU side by exploiting the AMX''s dedicated accumulate hardware. Two-pass atomic-free reduce achieves 0.325ms GPU vs 0.285ms CPU.',
'Reduce kernel already uses best practices: uint4 vectorized loads, simd_sum within SIMD groups, threadgroup shared memory reduction, and atomic-free two-pass architecture. The bottleneck is NOT the kernel — it''s that the CPU baseline (Accelerate) is extremely optimized. At 100M+ elements where GPU bandwidth advantage grows, the GPU would likely win.',
'forge-bench reduce experiment', 'forge-bench reduce vs Accelerate analysis', 'empirical_test',
'high', '2026-02-17',
'reduce,accelerate,vDSP,AMX,cpu-baseline,bandwidth-bound',
'Lesson: When benchmarking GPU kernels, the CPU baseline matters enormously. Using naive CPU code (for loop) would show 5-10x GPU advantage. Using Accelerate/BLAS shows the true competition. For production use, the decision should be: (1) reduce standalone → use Accelerate, (2) reduce as part of GPU pipeline → keep on GPU to avoid CPU-GPU transfer.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 14: Shared Types and Host-Shader Contract (metal-compute, skill_id=3)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3104, 3, 'shared types.h for Metal-Rust parameter struct alignment',
'A shared types.h header defines parameter structs used by both Metal shaders and Rust host code (via #[repr(C)]). All structs are 16-byte aligned with explicit _pad fields. This eliminates alignment bugs between host and device. Vectorized load helpers (load_uint4_safe, load_float4_safe) are defined inline in types.h for reuse across all kernels.',
'types.h defines 10 parameter structs (ReduceParams, ScanParams, HistogramParams, CompactParams, SortParams, FilterBenchParams, GroupByParams, GemmParams, SpreadsheetParams, TimeSeriesParams, HashJoinParams, CsvBenchParams) plus 2 inline vectorized load functions. Each struct has explicit uint _pad[] members to ensure 16-byte alignment. Rust counterparts in types.rs use #[repr(C)] and identical field ordering.',
'forge-primitives/shaders/types.h', 'forge-bench shared types', 'empirical_test',
'high', '2026-02-17',
'types,alignment,repr-c,16-byte,parameter-struct,metal-rust,shared-header',
'Pattern: struct Params { uint field1; uint field2; uint _pad[2]; }; // 16 bytes. Rust: #[repr(C)] pub struct Params { pub field1: u32, pub field2: u32, pub _pad: [u32; 2] }. The alloc_buffer_with_data(&device, &[params]) Rust function handles buffer creation. This pattern scales to any number of kernels without alignment issues.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 15: Dispatch Patterns (metal-compute, skill_id=3)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3105, 3, 'dispatch_1d and dispatch_2d helper patterns',
'Reusable dispatch helpers simplify kernel launches: dispatch_1d(encoder, pso, buffers, total_threads) computes threadgroups = ceil(total/tg_size) automatically. dispatch_2d(encoder, pso, buffers, (width, height)) dispatches a 2D grid. Buffer binding via slice: &[(buffer, index)] eliminates manual setBuffer calls. Threadgroup size defaults to maxTotalThreadsPerThreadgroup from PSO (typically 256 for 1D, (16,16) for 2D).',
'forge-primitives/src/dispatch.rs: dispatch_1d takes encoder, PSO, buffer slice, and total thread count. Computes tg_size from PSO maxTotalThreadsPerThreadgroup, dispatches ceil(total/tg_size) threadgroups. dispatch_2d takes 2D grid dimensions. Both iterate the buffer slice calling setBuffer with the provided index. This eliminates boilerplate across 15 experiments.',
'forge-primitives/src/dispatch.rs', 'forge-bench dispatch helpers', 'empirical_test',
'high', '2026-02-17',
'dispatch,helper,threadgroup-size,1d,2d,metal,compute-encoder',
'The PSO-driven threadgroup size (maxTotalThreadsPerThreadgroup) is the safest default — it reflects the compiler''s analysis of register pressure and occupancy. Override only when benchmarking shows a different TG size performs better for a specific kernel.',
'm4', 'current');

-- ============================================================================
-- CATEGORY 16: Single Command Buffer Pipeline (metal-compute, skill_id=3)
-- ============================================================================

INSERT INTO findings (id, skill_id, topic, claim, evidence, source_url, source_title, source_type, confidence, date_found, tags, notes, gpu_generation, temporal_status)
VALUES (3106, 3, 'single command buffer with sequential encoders for multi-kernel pipelines',
'Multi-kernel GPU pipelines should use ONE command buffer with sequential compute command encoders, not multiple command buffers. Pattern: create encoder → set buffers → dispatch → endEncoding() → create next encoder → set buffers → dispatch → endEncoding() → repeat → commit() once → waitUntilCompleted() once. This eliminates ~300us per-dispatch firmware overhead. Sort: 40 dispatches in 1 cmdbuf = 8.2x. Hash join: 2 dispatches in 1 cmdbuf = 88.3x.',
'sort.rs uses single command buffer for all 8 radix passes: each pass creates 5 encoders (histogram + 3 scan passes + scatter). Total: 40 encoder create/end cycles, 1 commit, 1 wait. hash_join.rs: build encoder + probe encoder, 1 commit. Metal guarantees sequential execution within a command buffer — no explicit barriers needed between encoders (implicit coherence on Apple unified memory).',
'forge-bench sort.rs, hash_join.rs', 'forge-bench single cmdbuf pattern', 'empirical_test',
'high', '2026-02-17',
'single-cmdbuf,command-buffer,encoder,sequential,implicit-coherence,pipeline',
'Apple unified memory provides implicit coherence between sequential command encoder dispatches within the same command buffer. No memory barriers or synchronization needed. The GPU executes encoders in order, and buffer writes from encoder N are visible to encoder N+1. This is simpler than NVIDIA/AMD where explicit barriers are required.',
'm4', 'current');

-- Verify insertion
-- SELECT COUNT(*) FROM findings WHERE id >= 3086 AND id <= 3106;
