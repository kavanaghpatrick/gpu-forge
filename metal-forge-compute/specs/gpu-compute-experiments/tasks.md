# Tasks: gpu-compute-experiments

## Phase 4: Application-Level Experiments

- [x] 4.4 JSON/CSV parsing experiment
  - **Do**:
    1. Create `forge-bench/src/experiments/json_parse.rs` -- JsonParseExperiment:
       - For this experiment, implement a simplified GPU CSV parser in `shaders/csv_bench.metal`:
         - Parallel newline detection via scan
         - Per-field extraction kernel
       - `run_gpu`: dispatch newline scan -> field extraction
       - `run_cpu`: simple sequential byte scanning for newlines and field counting
       - `validate`: parsed field count matches
       - `metrics`: MB/s throughput, rows/sec
    2. Add `csv_records()` generator to data_gen.rs -- generate random CSV data as a byte buffer
    3. Register json_parse in experiments/mod.rs
  - **Files**: `forge-primitives/shaders/csv_bench.metal`, `forge-bench/src/experiments/json_parse.rs`, `forge-bench/src/data_gen.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench json_parse --sizes 1M --runs 3` reports MB/s
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- json_parse --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add JSON/CSV parsing experiment`

- [x] 4.5 Hash join kernel + experiment
  - **Do**:
    1. Add `HashJoinParams` to types: `{ build_count: u32, probe_count: u32, table_size: u32, _pad: u32 }`
    2. Create `forge-primitives/shaders/hash_join.metal`:
       - `hash_join_build`: hash build keys, insert into open-addressing hash table with atomic CAS
       - `hash_join_probe`: each thread hashes its probe key, linear-probes hash table, writes match pairs
    3. Create `forge-bench/src/experiments/hash_join.rs` -- HashJoinExperiment:
       - `setup`: generate build table (1M) and probe table (1M, 10M) with varying join selectivity
       - `run_gpu`: build -> probe (2 dispatches in 1 cmdbuf)
       - `run_cpu`: HashMap build + probe
       - `validate`: join result set matches
       - `metrics`: joins/sec, build_ms, probe_ms
    4. Register hash_join in experiments/mod.rs
  - **Files**: `forge-primitives/src/types.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/hash_join.metal`, `forge-bench/src/experiments/hash_join.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench hash_join --sizes 1M --runs 3` reports joins/sec
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- hash_join --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add hash join kernel and experiment`
