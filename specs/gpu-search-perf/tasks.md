# Tasks: GPU Search Path Performance (20 GB/s -> 60+ GB/s)

## Phase 1: POC — New Kernel Compiles and Runs

Focus: Get `path_search_kernel` compiling, dispatching, and returning matches. Proves the architectural change works before optimizing.

- [x] 1.1 Add `PATH_SEARCH_SHADER` constant and `get_path_search_shader()` to shader.rs
  - **Do**:
    1. Open `/Users/patrickkavanagh/gpu-search-ui/src/engine/shader.rs`
    2. After the existing `get_content_search_shader()` function (line 706), add the `PATH_SEARCH_SHADER` constant containing the minimal path search kernel MSL source
    3. Add `pub fn get_path_search_shader() -> String` that replaces `{{APP_SHADER_HEADER}}` with `SHADER_HEADER`
    4. The kernel source is from design.md section 4 — includes `PathMatchResult` struct (8 bytes: chunk_index + byte_offset), `int` loop indices, 64-byte `local_data`, SIMD cooperative output, NO `find_line_bounds`, NO `score_file_extension`, NO metadata buffer binding
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/shader.rs` (after line 706)
  - **Code**:
    ```rust
    // After line 706, before #[cfg(test)]

    /// Minimal path search shader — separate from content search for I-cache benefit.
    const PATH_SEARCH_SHADER: &str = r#"
    {{APP_SHADER_HEADER}}

    #include <metal_simdgroup>

    #define CHUNK_SIZE 4096
    #define MAX_PATTERN_LEN 64
    #define THREADGROUP_SIZE 256
    #define BYTES_PER_THREAD 64
    #define MAX_MATCHES_PER_THREAD 4

    struct SearchParams {
        uint chunk_count;
        uint pattern_len;
        uint case_sensitive;
        uint total_bytes;
    };

    struct PathMatchResult {
        uint chunk_index;
        uint byte_offset;
    };

    inline bool char_eq_fast(uchar a, uchar b, bool case_sensitive) {
        if (case_sensitive) return a == b;
        uchar a_lower = (a >= 'A' && a <= 'Z') ? a + 32 : a;
        uchar b_lower = (b >= 'A' && b <= 'Z') ? b + 32 : b;
        return a_lower == b_lower;
    }

    kernel void path_search_kernel(
        device const uchar4* data [[buffer(0)]],
        constant SearchParams& params [[buffer(2)]],
        constant uchar* pattern [[buffer(3)]],
        device PathMatchResult* matches [[buffer(4)]],
        device atomic_uint& match_count [[buffer(5)]],
        uint gid [[thread_position_in_grid]],
        uint simd_lane [[thread_index_in_simdgroup]]
    ) {
        int thread_id = int(gid);
        int byte_base = thread_id * BYTES_PER_THREAD;
        int vec4_base = byte_base / 4;

        if (byte_base >= int(params.total_bytes)) return;

        int chunk_idx = byte_base / CHUNK_SIZE;
        int offset_in_chunk = byte_base % CHUNK_SIZE;

        if (chunk_idx >= int(params.chunk_count)) return;

        uchar local_data[BYTES_PER_THREAD];

        int valid_bytes = min(BYTES_PER_THREAD, CHUNK_SIZE - offset_in_chunk);

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i * 4 < valid_bytes) {
                uchar4 v = data[vec4_base + i];
                local_data[i*4 + 0] = v.x;
                local_data[i*4 + 1] = v.y;
                local_data[i*4 + 2] = v.z;
                local_data[i*4 + 3] = v.w;
            }
        }

        int local_matches_pos[MAX_MATCHES_PER_THREAD];
        int local_match_count = 0;

        bool case_sensitive = params.case_sensitive != 0;
        int pat_len = int(params.pattern_len);
        int search_end = (valid_bytes >= pat_len) ? (valid_bytes - pat_len + 1) : 0;

        for (int pos = 0; pos < search_end && local_match_count < MAX_MATCHES_PER_THREAD; pos++) {
            bool match_found = true;
            for (int j = 0; j < pat_len && match_found; j++) {
                if (!char_eq_fast(local_data[pos + j], pattern[j], case_sensitive)) {
                    match_found = false;
                }
            }
            if (match_found) {
                local_matches_pos[local_match_count++] = pos;
            }
        }

        uint simd_total = simd_sum(uint(local_match_count));
        uint my_offset = simd_prefix_exclusive_sum(uint(local_match_count));

        uint group_base = 0;
        if (simd_lane == 0 && simd_total > 0) {
            group_base = atomic_fetch_add_explicit(&match_count, simd_total, memory_order_relaxed);
        }
        group_base = simd_broadcast_first(group_base);

        for (int i = 0; i < local_match_count; i++) {
            uint global_idx = group_base + my_offset + uint(i);
            if (global_idx < 100000) {
                PathMatchResult result;
                result.chunk_index = uint(chunk_idx);
                result.byte_offset = uint(offset_in_chunk + local_matches_pos[i]);
                matches[global_idx] = result;
            }
        }
    }
    "#;

    /// Get the path search shader with header substituted
    pub fn get_path_search_shader() -> String {
        PATH_SEARCH_SHADER.replace("{{APP_SHADER_HEADER}}", SHADER_HEADER)
    }
    ```
  - **Done when**: `get_path_search_shader()` returns valid MSL source containing `path_search_kernel`, no `{{APP_SHADER_HEADER}}` placeholders remain
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release 2>&1 | tail -5`
  - **Commit**: `feat(shader): add PATH_SEARCH_SHADER with dedicated path_search_kernel`
  - _Requirements: FR-1, FR-2, FR-3, FR-4, FR-5_
  - _Design: Component 1 (New MSL Kernel), Component 4 (Shader Source Management)_

- [x] 1.2 Add `GpuPathMatchResult` struct and pipeline fields to `GpuContentSearch`
  - **Do**:
    1. Open `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`
    2. After `GpuMatchResult` struct (line 71), add `GpuPathMatchResult` (8 bytes, repr(C))
    3. Add `path_pipeline: ComputePipelineState` and `path_matches_buffer: Buffer` fields to `GpuContentSearch` struct (after `direct_pipeline` at line 156)
    4. Add `use super::shader::get_path_search_shader;` import at the top
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs` (lines 7-8, 71, 146-176)
  - **Code** (struct after line 71):
    ```rust
    /// Minimal match result from path_search_kernel (8 bytes, must match MSL)
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    struct GpuPathMatchResult {
        chunk_index: u32,
        byte_offset: u32,
    }
    ```
  - **Code** (fields in GpuContentSearch, after line 156):
    ```rust
    path_pipeline: ComputePipelineState,
    path_matches_buffer: Buffer,
    ```
  - **Done when**: `GpuPathMatchResult` is 8 bytes; `GpuContentSearch` has `path_pipeline` and `path_matches_buffer` fields
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release 2>&1 | tail -5`
  - **Commit**: `feat(search): add GpuPathMatchResult struct and path pipeline fields`
  - _Requirements: FR-1, FR-12_
  - _Design: Component 2a, 2b_

- [x] 1.3 Initialize path pipeline and buffer in `GpuContentSearch::new()`
  - **Do**:
    1. Open `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`
    2. In `new()` after `direct_pipeline` creation (line 192), compile path shader and create pipeline:
       ```rust
       let path_library = compile_shader(device, &get_path_search_shader())?;
       let path_pipeline = create_compute_pipeline(device, &path_library, "path_search_kernel")?;
       let path_matches_buffer = create_buffer(device, MAX_MATCHES * mem::size_of::<GpuPathMatchResult>());
       ```
    3. Add `path_pipeline` and `path_matches_buffer` to the `Ok(Self { ... })` block (after `direct_pipeline,` around line 208)
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs` (lines 192, 202-220)
  - **Done when**: `GpuContentSearch::new()` compiles path shader and creates path_pipeline successfully; struct initializer includes both new fields
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release 2>&1 | tail -5`
  - **Commit**: `feat(search): initialize path_pipeline and path_matches_buffer in new()`
  - _Requirements: FR-1, FR-12_
  - _Design: Component 2c_

- [x] 1.4 Add `score_file_extension_cpu()` function
  - **Do**:
    1. Open `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`
    2. After the `impl GpuContentSearch` block closing brace (after the last method), add the free function `score_file_extension_cpu(path: &str) -> u32`
    3. Must produce identical scores as the GPU `score_file_extension()`: 0=docs, 1=media, 2=code, 3=config, 4=junk, 128=unknown; hidden file penalty +64
    4. Use byte slice matching for extension lookup (see design.md Component 3)
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs` (after the `impl GpuContentSearch` closing brace, before `#[cfg(test)]`)
  - **Code**: See design.md Component 3 for full function body
  - **Done when**: Function compiles and covers all extension categories from the GPU version
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release 2>&1 | tail -5`
  - **Commit**: `feat(search): add CPU-side score_file_extension_cpu() for path ranking`
  - _Requirements: FR-7_
  - _Design: Component 3_

- [x] 1.5 Rewrite `search_paths()` to use new path pipeline with `set_bytes()`
  - **Do**:
    1. Open `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`
    2. Replace the entire `search_paths()` method body (lines 325-438) with the new implementation from design.md Component 2d
    3. Key changes:
       - Use `self.path_pipeline` instead of `self.turbo_pipeline`
       - Use `encoder.set_bytes()` for params (buffer 2) and pattern (buffer 3) instead of `set_buffer()`
       - Use `self.path_matches_buffer` at buffer index 4 instead of `self.matches_buffer`
       - Skip metadata buffer(1) entirely
       - Read `GpuPathMatchResult` (chunk_index + byte_offset) instead of `GpuMatchResult`
       - CPU-side newline scanning to extract full path from `self.chunk_data`
       - Call `score_file_extension_cpu()` for priority scoring
       - `Vec::with_capacity(result_count)` pre-allocation
       - Single String allocation: `std::str::from_utf8()` + `.trim()` + `.to_string()`
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs` (lines 325-438, replace entirely)
  - **Code**: See design.md Component 2d for complete replacement
  - **Done when**: `search_paths()` dispatches `path_search_kernel`, uses `set_bytes()`, does CPU-side path extraction and scoring
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release && cargo test -- --nocapture 2>&1 | tail -30`
  - **Commit**: `feat(search): rewrite search_paths() with path_pipeline, set_bytes, CPU scoring`
  - _Requirements: FR-6, FR-7, FR-8, FR-10, FR-11, FR-12_
  - _Design: Component 2d_

- [x] 1.6 [VERIFY] Quality checkpoint: build + all existing tests pass
  - **Do**:
    1. `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release`
    2. `cd /Users/patrickkavanagh/gpu-search-ui && cargo test -- --nocapture`
    3. Verify all 10 tests in `test_path_search.rs` pass (these are the integration tests that prove correctness)
    4. Verify unit tests in `search.rs` and `shader.rs` pass
    5. If any test fails, diagnose and fix — this is the POC gate
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release && cargo test 2>&1 | tail -20`
  - **Done when**: All tests pass, zero regressions
  - **Commit**: `chore(search): pass quality checkpoint` (only if fixes needed)
  - _Requirements: AC-2.1, AC-3.1, AC-3.2, AC-3.3_

- [x] 1.7 POC Checkpoint: measure throughput improvement
  - **Do**:
    1. Run the 200K-path scale test with timing to validate measurable improvement:
       ```bash
       cd /Users/patrickkavanagh/gpu-search-ui && cargo test test_scale_200k_paths_friendship -- --nocapture 2>&1
       ```
    2. Run the 1M-path scale test:
       ```bash
       cd /Users/patrickkavanagh/gpu-search-ui && cargo test test_scale_1m_long_paths -- --nocapture 2>&1
       ```
    3. Run the real filesystem test:
       ```bash
       cd /Users/patrickkavanagh/gpu-search-ui && cargo test test_real_filesystem_old_friendship -- --nocapture 2>&1
       ```
    4. Verify all tests pass and search produces correct results
    5. Test the actual UI by building and running:
       ```bash
       cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release
       cd / && /Users/patrickkavanagh/gpu-search-ui/target/release/gpu-search-ui &
       sleep 3
       # The app should launch and path search should work
       kill %1 2>/dev/null
       ```
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test test_scale_200k -- --nocapture 2>&1 | grep -E "(test.*ok|FAILED|friendship)" | head -10`
  - **Done when**: All scale tests pass; UI launches without crash; path search returns correct results
  - **Commit**: `feat(search): complete path search performance POC`
  - _Requirements: AC-1.1, AC-2.1, AC-2.2, AC-2.3_

## Phase 2: Optimization — `-Os` Compilation

Focus: Add `-Os` shader compilation for the path kernel (the one optimization not yet applied).

- [ ] 2.1 Add `compile_shader_os()` function to mod.rs
  - **Do**:
    1. Open `/Users/patrickkavanagh/gpu-search-ui/src/engine/mod.rs`
    2. After the existing `compile_shader()` function (line 17), add `compile_shader_os()` that uses raw `msg_send!` to set optimization level to 1 (size/-Os)
    3. The `metal` crate re-exports `objc` as `pub extern crate objc`, so `metal::objc::msg_send!` is available. No additional dependency needed.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/mod.rs` (after line 17)
  - **Code**:
    ```rust
    /// Compile a Metal shader library with size optimization (-Os).
    /// MTLLibraryOptimizationLevel: 0 = Default, 1 = Size
    /// Falls back to default optimization if the -Os API call is unavailable.
    pub fn compile_shader_os(device: &Device, source: &str) -> Result<Library, String> {
        let options = CompileOptions::new();
        unsafe {
            let _: () = metal::objc::msg_send![&*options, setOptimizationLevel: 1u64];
        }
        device
            .new_library_with_source(source, &options)
            .map_err(|e| format!("Failed to compile shaders (-Os): {}", e))
    }
    ```
  - **Done when**: `compile_shader_os()` compiles and is callable
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release 2>&1 | tail -5`
  - **Commit**: `feat(engine): add compile_shader_os() with -Os via msg_send!`
  - _Requirements: FR-9_
  - _Design: Component 5_

- [ ] 2.2 Use `-Os` compilation for path kernel with fallback
  - **Do**:
    1. Open `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`
    2. Add `use super::compile_shader_os;` import
    3. In `new()`, change path library compilation to try `-Os` first, fallback to default:
       ```rust
       let path_library = compile_shader_os(device, &get_path_search_shader())
           .or_else(|_| compile_shader(device, &get_path_search_shader()))?;
       ```
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs` (import + `new()` path compilation)
  - **Done when**: Path kernel compiles with `-Os`; falls back to default if `-Os` fails
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release && cargo test 2>&1 | tail -10`
  - **Commit**: `feat(search): use -Os compilation for path_search_kernel`
  - _Requirements: FR-9_
  - _Design: Component 5_

- [ ] 2.3 [VERIFY] Quality checkpoint: build + tests after -Os
  - **Do**:
    1. `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release`
    2. `cd /Users/patrickkavanagh/gpu-search-ui && cargo test`
    3. Verify no regressions from `-Os` compilation change
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release && cargo test 2>&1 | tail -10`
  - **Done when**: All tests pass
  - **Commit**: `chore(search): pass quality checkpoint` (only if fixes needed)

## Phase 3: Testing

Focus: Add unit tests for new components. Existing integration tests already validate correctness.

- [ ] 3.1 Add shader unit tests for `get_path_search_shader()`
  - **Do**:
    1. Open `/Users/patrickkavanagh/gpu-search-ui/src/engine/shader.rs`
    2. In the existing `#[cfg(test)] mod tests` block (line 708), add tests:
       ```rust
       #[test]
       fn test_path_search_shader_header_substitution() {
           let shader = get_path_search_shader();
           assert!(shader.contains("#include <metal_stdlib>"));
           assert!(shader.contains("using namespace metal;"));
           assert!(!shader.contains("{{APP_SHADER_HEADER}}"));
       }

       #[test]
       fn test_path_search_shader_contains_kernel() {
           let shader = get_path_search_shader();
           assert!(shader.contains("path_search_kernel"));
           assert!(shader.contains("PathMatchResult"));
           // Must NOT contain content search functions
           assert!(!shader.contains("find_line_bounds"));
           assert!(!shader.contains("score_file_extension"));
           assert!(!shader.contains("turbo_search_kernel"));
       }
       ```
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/shader.rs` (inside `mod tests`)
  - **Done when**: Tests validate shader source correctness
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test shader::tests -- --nocapture 2>&1 | tail -10`
  - **Commit**: `test(shader): add unit tests for path_search_shader`
  - _Requirements: AC-3.1_
  - _Design: Test Strategy_

- [ ] 3.2 Add unit tests for `GpuPathMatchResult` and `score_file_extension_cpu()`
  - **Do**:
    1. Open `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`
    2. In the existing `#[cfg(test)] mod tests` block (line 968), add:
       ```rust
       #[test]
       fn test_path_match_result_size() {
           assert_eq!(mem::size_of::<GpuPathMatchResult>(), 8);
       }

       #[test]
       fn test_score_file_extension_cpu_documents() {
           assert_eq!(score_file_extension_cpu("/foo/bar.md"), 0);
           assert_eq!(score_file_extension_cpu("/foo/bar.pdf"), 0);
           assert_eq!(score_file_extension_cpu("/foo/bar.txt"), 0);
           assert_eq!(score_file_extension_cpu("/foo/bar.docx"), 0);
           assert_eq!(score_file_extension_cpu("/foo/bar.csv"), 0);
       }

       #[test]
       fn test_score_file_extension_cpu_media() {
           assert_eq!(score_file_extension_cpu("/foo/bar.png"), 1);
           assert_eq!(score_file_extension_cpu("/foo/bar.jpg"), 1);
           assert_eq!(score_file_extension_cpu("/foo/bar.mp4"), 1);
           assert_eq!(score_file_extension_cpu("/foo/bar.heic"), 1);
       }

       #[test]
       fn test_score_file_extension_cpu_code() {
           assert_eq!(score_file_extension_cpu("/foo/bar.rs"), 2);
           assert_eq!(score_file_extension_cpu("/foo/bar.py"), 2);
           assert_eq!(score_file_extension_cpu("/foo/bar.js"), 2);
           assert_eq!(score_file_extension_cpu("/foo/bar.swift"), 2);
           assert_eq!(score_file_extension_cpu("/foo/bar.c"), 2);
           assert_eq!(score_file_extension_cpu("/foo/bar.h"), 2);
       }

       #[test]
       fn test_score_file_extension_cpu_config() {
           assert_eq!(score_file_extension_cpu("/foo/bar.json"), 3);
           assert_eq!(score_file_extension_cpu("/foo/bar.yaml"), 3);
           assert_eq!(score_file_extension_cpu("/foo/bar.toml"), 3);
       }

       #[test]
       fn test_score_file_extension_cpu_junk() {
           assert_eq!(score_file_extension_cpu("/foo/bar.log"), 4);
           assert_eq!(score_file_extension_cpu("/foo/bar.bak"), 4);
           assert_eq!(score_file_extension_cpu("/foo/bar.tmp"), 4);
           assert_eq!(score_file_extension_cpu("/foo/bar.lock"), 4);
       }

       #[test]
       fn test_score_file_extension_cpu_unknown() {
           assert_eq!(score_file_extension_cpu("/foo/bar.xyz"), 128);
           assert_eq!(score_file_extension_cpu("/foo/bar"), 128);
       }

       #[test]
       fn test_score_file_extension_cpu_hidden_penalty() {
           // Hidden file gets +64
           assert_eq!(score_file_extension_cpu("/foo/.gitignore"), 128 + 64);
           assert_eq!(score_file_extension_cpu("/foo/.hidden.rs"), 2 + 64);
           assert_eq!(score_file_extension_cpu("/foo/.env.json"), 3 + 64);
       }

       #[test]
       fn test_score_file_extension_cpu_case_insensitive() {
           assert_eq!(score_file_extension_cpu("/foo/bar.RS"), 2);
           assert_eq!(score_file_extension_cpu("/foo/bar.Py"), 2);
           assert_eq!(score_file_extension_cpu("/foo/bar.PDF"), 0);
       }
       ```
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs` (inside `mod tests`)
  - **Done when**: All extension scoring tests pass, struct size is 8 bytes
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test search::tests -- --nocapture 2>&1 | tail -20`
  - **Commit**: `test(search): add unit tests for GpuPathMatchResult and score_file_extension_cpu`
  - _Requirements: AC-2.2, AC-2.3_
  - _Design: Test Strategy_

- [ ] 3.3 [VERIFY] Quality checkpoint: all tests pass
  - **Do**:
    1. `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release`
    2. `cd /Users/patrickkavanagh/gpu-search-ui && cargo test`
    3. Confirm ALL unit tests + integration tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release && cargo test 2>&1 | tail -10`
  - **Done when**: All tests green
  - **Commit**: `chore(search): pass quality checkpoint` (only if fixes needed)

## Phase 4: Quality Gates

- [ ] 4.1 [VERIFY] Full local CI: build (release) + all tests
  - **Do**:
    1. `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release`
    2. `cd /Users/patrickkavanagh/gpu-search-ui && cargo test`
    3. `cd /Users/patrickkavanagh/gpu-search-ui && cargo clippy -- -D warnings 2>&1 || true` (check for warnings, fix if any)
    4. Verify no compiler warnings in release build
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release 2>&1 | grep -c "warning\[" && cargo test 2>&1 | tail -5`
  - **Done when**: Clean release build, all tests pass, no clippy warnings
  - **Commit**: `fix(search): address lint/type issues` (if fixes needed)

- [ ] 4.2 Create PR and verify CI
  - **Do**:
    1. Verify current branch: `git branch --show-current` (should be `feat/gpu-query` or a feature branch)
    2. Stage changed files in `/Users/patrickkavanagh/gpu-search-ui/`:
       - `src/engine/shader.rs`
       - `src/engine/search.rs`
       - `src/engine/mod.rs`
    3. Push branch: `git push -u origin <branch-name>`
    4. Create PR using gh CLI with title and summary describing the 6 optimizations
  - **Verify**: `gh pr checks --watch` or `gh pr checks` to poll CI status
  - **Done when**: All CI checks green, PR ready for review

## Phase 5: PR Lifecycle

- [ ] 5.1 Monitor CI and fix failures
  - **Do**:
    1. After PR creation, monitor CI: `gh pr checks`
    2. If any check fails, read failure details, fix locally, push
    3. Re-verify: `gh pr checks --watch`
  - **Verify**: `gh pr checks` shows all passing
  - **Done when**: All CI checks green

- [ ] 5.2 [VERIFY] AC checklist
  - **Do**: Programmatically verify each acceptance criterion:
    1. AC-1.1: search_paths() <5ms — run scale test, check timing
    2. AC-1.2: GPU throughput >= 60 GB/s — run profiling test
    3. AC-2.1: Case-insensitive matching correct — all test_path_search.rs tests pass
    4. AC-2.2: Results sorted by file type priority — test_scale_200k test passes with priority sorting
    5. AC-2.3: Hidden file penalty +64 — unit test for score_file_extension_cpu
    6. AC-3.1: Existing 4 kernels unchanged — `git diff HEAD~N src/engine/shader.rs` shows only additions
    7. AC-3.2: All existing tests pass — `cargo test`
    8. AC-3.3: ContentMatch struct unchanged — grep for struct definition
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test 2>&1 | tail -5`
  - **Done when**: All acceptance criteria confirmed via automated checks

## Notes

### POC shortcuts taken
- No separate benchmark binary — using existing test infrastructure for perf validation
- `-Os` msg_send may silently fail and fall back to default optimization — acceptable

### Production TODOs
- Consider adding a proper benchmark using criterion for ongoing throughput regression testing
- Monitor whether boundary match misses (64-byte window) cause user complaints
- If `CompileOptions` in future metal crate versions exposes `set_optimization_level()`, replace raw `msg_send!`

### Key implementation details
- `set_bytes()` signature: `fn set_bytes(&self, index: NSUInteger, length: NSUInteger, bytes: *const c_void)` — index is buffer binding, NOT byte offset
- `metal` crate re-exports `objc`: `metal::objc::msg_send!` is available — no additional Cargo dependency needed
- MSL `match` is a reserved word in some contexts — use `match_found` as variable name in kernel
- Existing `pattern_buffer` and `params_buffer` kept for content search methods — only `search_paths()` uses `set_bytes()`
- `path_matches_buffer`: 100K * 8B = 800KB (vs 3.2MB for full MatchResult buffer)
