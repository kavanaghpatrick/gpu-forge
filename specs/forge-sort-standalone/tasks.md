---
spec: forge-sort-standalone
phase: tasks
total_tasks: 7
created: 2026-02-20
---

# Tasks: forge-sort Standalone Crate

## Phase 1: Make It Work (POC)

Focus: Inline ~200 lines, swap imports, delete dep line. Zero API changes.

- [x] 1.1 Create metal_helpers.rs with inlined code
  - **Do**:
    1. Create `metal-forge-compute/forge-sort/src/metal_helpers.rs`
    2. Copy `FnConstant` enum from `forge-primitives/src/pso_cache.rs` lines 17-24
    3. Copy `PsoCache` struct + impl from `forge-primitives/src/pso_cache.rs` lines 26-191 (exclude `#[cfg(test)]` block)
    4. Copy `alloc_buffer()` from `forge-primitives/src/dispatch.rs` lines 116-125
    5. Write new `init_device_and_queue()` function (device + queue init from MetalContext::new() lines 22-26, without metallib loading)
    6. Add required imports at top: `std::collections::HashMap`, `std::ptr::NonNull`, `objc2::{rc::Retained, runtime::ProtocolObject}`, `objc2_foundation::NSString`, `objc2_metal::{MTLBuffer, MTLCommandQueue, MTLComputePipelineDescriptor, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary, MTLPipelineOption, MTLResourceOptions}`
    7. All items `pub` (needed for `use` from lib.rs); module declared without `pub` in lib.rs
    8. Add origin comment at top: `//! Inlined from forge-primitives (pso_cache.rs, dispatch.rs, metal_ctx.rs).`
  - **Files**: `metal-forge-compute/forge-sort/src/metal_helpers.rs` (CREATE)
  - **Done when**: File exists with ~210 lines, compiles in isolation
  - **Verify**: `wc -l metal-forge-compute/forge-sort/src/metal_helpers.rs` shows ~200-220 lines
  - **Commit**: `feat(forge-sort): add metal_helpers.rs with inlined PsoCache, alloc_buffer, device init`
  - _Requirements: FR-1, FR-2, FR-3, FR-4_
  - _Design: metal_helpers.rs Content_

- [x] 1.2 Update lib.rs imports and GpuSorter::new()
  - **Do**:
    1. In `forge-sort/src/lib.rs` line 4: delete `use forge_primitives::pso_cache::FnConstant;`
    2. In line 5: delete `use forge_primitives::{alloc_buffer, MetalContext, PsoCache};`
    3. After line 2 (`use std::ptr::NonNull;`), insert:
       ```
       mod metal_helpers;
       use metal_helpers::{alloc_buffer, init_device_and_queue, FnConstant, PsoCache};
       ```
    4. Line 560: replace `let ctx = MetalContext::new();` with `let (device, queue) = init_device_and_queue();`
    5. Line 566-567: replace `ctx.device.newLibraryWithFile_error(...)` with `device.newLibraryWithFile_error(...)`
    6. Lines 620-622: replace `device: ctx.device, queue: ctx.queue,` with `device, queue,`
  - **Files**: `metal-forge-compute/forge-sort/src/lib.rs` (MODIFY)
  - **Done when**: No `forge_primitives` references in lib.rs
  - **Verify**: `grep -c 'forge_primitives' metal-forge-compute/forge-sort/src/lib.rs` outputs `0`
  - **Commit**: `feat(forge-sort): replace forge-primitives imports with local metal_helpers`
  - _Requirements: FR-5, FR-7_
  - _Design: lib.rs Changes_

- [x] 1.3 Remove forge-primitives dependency from Cargo.toml
  - **Do**:
    1. Delete line 8 from `forge-sort/Cargo.toml`: `forge-primitives = { path = "../forge-primitives" }`
  - **Files**: `metal-forge-compute/forge-sort/Cargo.toml` (MODIFY)
  - **Done when**: No forge-primitives in Cargo.toml
  - **Verify**: `grep -c 'forge-primitives' metal-forge-compute/forge-sort/Cargo.toml` outputs `0`
  - **Commit**: `feat(forge-sort): remove forge-primitives dependency`
  - _Requirements: FR-6, AC-1.1_
  - _Design: Cargo.toml Change_

- [ ] 1.4 [VERIFY] Full test suite: forge-sort + workspace
  - **Do**:
    1. Run `cargo test -p forge-sort -- --test-threads=1` from `metal-forge-compute/` — all 165 tests must pass
    2. Run `cargo build -p forge-bench` — workspace member still compiles
    3. Run `cargo build -p forge-primitives` — untouched crate still compiles
    4. Verify no forge-primitives references remain: `grep -r 'forge.primitives' metal-forge-compute/forge-sort/src/`
  - **Verify**: All 4 commands exit 0; grep returns no matches
  - **Done when**: 165 tests pass, forge-bench builds, zero forge-primitives refs in src/
  - **Commit**: `chore(forge-sort): verify standalone build passes all tests` (only if fixes needed)
  - _Requirements: AC-1.2, AC-1.3, AC-2.1, AC-2.2, AC-2.4_

## Phase 2: Quality Gates

- [ ] 2.1 [VERIFY] Full local CI
  - **Do**:
    1. `cargo clippy -p forge-sort -- -D warnings`
    2. `cargo test -p forge-sort -- --test-threads=1`
    3. `cargo build -p forge-bench`
    4. `cargo build -p forge-primitives`
    5. `cargo doc -p forge-sort --no-deps`
  - **Verify**: All commands exit 0
  - **Done when**: Clippy clean, all tests pass, workspace intact, docs build
  - **Commit**: `chore(forge-sort): pass local CI` (only if fixes needed)

- [ ] 2.2 Create PR and verify CI
  - **Do**:
    1. Verify on feature branch: `git branch --show-current`
    2. If on default branch, STOP and alert user
    3. Push: `git push -u origin $(git branch --show-current)`
    4. Create PR: `gh pr create --title "feat(forge-sort): remove forge-primitives dependency" --body "..."`
  - **Verify**: `gh pr checks --watch` — all checks green
  - **Done when**: PR created, CI passes
  - **If CI fails**: fix locally, push, re-verify

- [ ] 2.3 [VERIFY] AC checklist
  - **Do**: Programmatically verify each acceptance criterion:
    1. AC-1.1: `grep -c 'forge-primitives' metal-forge-compute/forge-sort/Cargo.toml` = 0
    2. AC-1.2: `cargo build -p forge-sort` exits 0
    3. AC-1.3: `cargo test -p forge-sort -- --test-threads=1` passes 165 tests
    4. AC-1.4: `cargo bench -p forge-sort --no-run` exits 0
    5. AC-2.1: `cargo build -p forge-primitives` exits 0
    6. AC-2.2: `cargo build -p forge-bench` exits 0
    7. AC-2.3: `grep 'forge-primitives' metal-forge-compute/Cargo.toml` finds workspace member
    8. AC-2.4: No files modified in `forge-primitives/` (git diff)
    9. AC-3.1: Public types unchanged (grep for `pub struct GpuSorter`, `pub struct SortBuffer`, etc.)
    10. AC-3.2: `grep 'pub use.*PsoCache\|pub use.*FnConstant' metal-forge-compute/forge-sort/src/lib.rs` = 0
    11. AC-3.3: No new `pub` items in lib.rs exports
  - **Verify**: All grep/build/test commands produce expected output
  - **Done when**: All 11 acceptance criteria confirmed via automated checks

## Phase 3: PR Lifecycle

- [ ] 3.1 Address review feedback and maintain CI
  - **Do**:
    1. Monitor PR for review comments
    2. Address any feedback with fixup commits
    3. Re-verify CI after each push
  - **Verify**: `gh pr checks` — all green after each push
  - **Done when**: PR approved and merged

## Notes

- **POC shortcuts**: None needed — this is a direct copy + import swap
- **Production TODOs**: None — the refactor IS production-ready
- **Test count**: 165 tests confirmed via `cargo test -p forge-sort -- --list | grep -c ': test$'`
- **Risk**: Near-zero. Code is copied verbatim; only import paths change.
- **forge-sort README.md** mentions forge-primitives (line 218) but that's documentation, not code. Optional cleanup.
