---
id: verify-mode.BREAKDOWN
module: verify-mode
priority: 4
status: failing
version: 1
origin: spec-workflow
dependsOn: []
tags: [p0, verification, correctness]
testRequirements:
  unit:
    required: true
    pattern: "src/search/verify.rs::tests"
---
# Verify Mode Breakdown

## Context

GPU compute kernels can produce false positive matches (bloom filter characteristics). The `VerifyMode` currently defaults to `Off` (`verify.rs:25-29`), requiring the user to set `GPU_SEARCH_VERIFY=full` or `GPU_SEARCH_VERIFY=sample` environment variable to enable CPU verification. This means false positives reach the UI in production, compounding the stale results trust issue.

The fix changes the default from `Off` to `Sample` (statistical sampling at 5%) and adds adaptive upgrade logic: when the result count is below 100, Sample mode automatically upgrades to Full verification since the overhead is negligible for small result sets. This catches 100% of false positives in the common case (small result sets most visible to users) while keeping overhead low for large result sets.

From PM.md P0-2, TECH.md Section 7, PM Q2.

## Acceptance Criteria

1. `VerifyMode::from_env()` default changed from `Off` to `Sample` in `src/search/verify.rs`
2. New method `pub fn effective(self, result_count: usize) -> VerifyMode` added to `VerifyMode` impl
3. `effective()` returns `Full` when mode is `Sample` and `result_count < 100`
4. `effective()` returns `Sample` when mode is `Sample` and `result_count >= 100`
5. `effective()` passes through `Full` and `Off` unchanged regardless of count
6. `GPU_SEARCH_VERIFY=off` env var still works to opt out completely
7. `GPU_SEARCH_VERIFY=full` env var still forces full verification
8. Orchestrator (`src/search/orchestrator.rs`) calls `base_mode.effective(gpu_results.len())` before verification stage
9. Unit test U-VFY-1 passes: default mode is `Sample` (not `Off`)
10. Unit test U-VFY-4 passes: `Sample.effective(50)` returns `Full`
11. Unit test U-VFY-5 passes: `Sample.effective(100)` returns `Sample`
12. Unit test U-VFY-8 passes: `Off.effective(50)` returns `Off`
13. Unit test U-VFY-9 passes: `Sample.effective(99)` returns `Full` (boundary)
14. Unit test U-VFY-10 passes: `Sample.effective(0)` returns `Full`
15. Existing test `test_verify_mode_from_env` updated to expect `Sample` default
16. Verification overhead measured: Sample mode adds <5ms to typical search (<10% of 8-50ms GPU search time)
17. All existing tests pass: `cargo test -p gpu-search`

## Technical Notes

- Reference: [spec/OVERVIEW.md] verify-mode is priority 4, no dependencies
- UX: No direct UX changes; reduces false positive rate in displayed results
- Test: From QA.md Section 2.6 (U-VFY-1..10) -- adaptive mode transitions
- Breaking change consideration: users who relied on verification being off will see slight overhead (TECH.md Section 7.4)
- Measured on M4: memchr::memmem verification for 5% sample of 500 matches costs ~2ms
- For <100 matches (auto-Full), verification costs ~1ms -- within 10% overhead budget
- Users can explicitly set `GPU_SEARCH_VERIFY=off` for the old behavior
