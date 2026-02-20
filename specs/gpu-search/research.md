---
spec: gpu-search
phase: research
created: 2026-02-11
generated: auto
---

# Research: gpu-search

## Executive Summary

GPU-accelerated filesystem search tool combining ripgrep-class throughput (55-80 GB/s via Metal compute) with a polished native GUI (egui/eframe + wgpu Metal backend). Ports ~8,000 lines of proven search kernel code from `rust-experiment/` (deprecated `metal 0.33`) to `objc2-metal 0.3`. Fixed string search for v1. High feasibility -- core algorithms proven, only API layer changes.

## Codebase Analysis

### Source Codebase (rust-experiment)

| File | Lines | Purpose | Port Complexity |
|------|-------|---------|-----------------|
| `gpu_os/content_search.rs` | 1660 | Vectorized uchar4 search kernel (55-80 GB/s) | Medium -- API calls change, MSL unchanged |
| `gpu_os/persistent_search.rs` | 1089 | Persistent kernel, batch processing | Medium -- replace with gpu-query re-dispatch |
| `gpu_os/streaming_search.rs` | 1071 | Quad-buffered I/O + search pipeline | Medium -- streaming logic preserved |
| `gpu_os/gpu_io.rs` | 524 | MTLIOCommandQueue (raw msg_send!) | High -- replace all msg_send! with native bindings |
| `gpu_os/batch_io.rs` | 514 | Batch file loading | Low -- type changes only |
| `gpu_os/gpu_index.rs` | 867 | GPU-resident filesystem index (256B entries) | Low -- type changes only |
| `gpu_os/shared_index.rs` | 921 | Shared index manager, ~/.gpu_os/ cache | Low -- path + type changes |
| `gpu_os/mmap_buffer.rs` | 566 | Zero-copy mmap + newBufferNoCopy | Low -- simplest port |

### Target Patterns (gpu-query)

| Pattern | Source | Reuse Strategy |
|---------|--------|----------------|
| Metal device init | `gpu/device.rs` | Copy directly (already objc2-metal) |
| PSO cache (HashMap) | `gpu/pipeline.rs` | Copy, adapt for search kernel names |
| Triple-buffered work queue | `gpu/autonomous/work_queue.rs` | Adapt for SearchRequest type |
| Completion-handler re-dispatch | `gpu/autonomous/executor.rs` | Adapt for search dispatch loop |
| MTLSharedEvent idle/wake | `gpu/autonomous/executor.rs` | Copy pattern directly |
| build.rs Metal compilation | `build.rs` | Copy, point to gpu-search shaders/ |
| #[repr(C)] layout tests | `gpu/autonomous/types.rs` | Copy pattern for search types |

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| objc2-metal | 0.3 | Metal GPU compute (direct API) |
| objc2 | 0.6 | Objective-C runtime |
| block2 | 0.6 | Completion handlers |
| dispatch2 | 0.3 | GCD dispatch |
| eframe | 0.31 | egui + winit + wgpu (Metal backend) |
| syntect | 5 | Syntax highlighting (Sublime definitions) |
| rayon | 1.10 | Parallel filesystem scanning |
| ignore | 0.4 | .gitignore filtering (ripgrep's crate) |
| criterion | 0.5 | Performance benchmarks |

### Constraints

- Apple Silicon required (Metal GPU, unified memory)
- macOS only (Metal framework)
- No Metal GPU CI runners available -- self-hosted required
- Metal watchdog kills long-running command buffers -- chunked dispatches needed
- egui-wgpu and raw objc2-metal must share same MTLDevice
- MTLIOCommandQueue bindings may be incomplete in objc2-metal -- msg_send! fallback for missing methods

## Technology Evaluation

### objc2-metal 0.3 vs metal 0.33

| Aspect | metal 0.33 | objc2-metal 0.3 |
|--------|------------|-----------------|
| Status | **Deprecated**, archived | **Active**, auto-generated from Apple headers |
| Memory management | Manual retain/release | RAII `Retained<T>`, automatic |
| MTLIOCommandQueue | Manual msg_send! | Native bindings (feature flag) |
| Type safety | Raw pointers | Protocol-based trait objects |
| Ecosystem alignment | Orphaned | winit, wgpu migrating to objc2 |

**Verdict**: objc2-metal is the clear choice. Consistent with gpu-query, actively maintained, safer API.

### egui/eframe vs alternatives

| Framework | Typography | Compute Integration | Verdict |
|-----------|-----------|-------------------|---------|
| egui/eframe | Good (wgpu) | Dual Metal: wgpu for UI, raw objc2-metal for compute | **CHOSEN** -- immediate mode fits search UX |
| iced | Good (wgpu) | Elm architecture, subscription model | Runner-up -- more complex widget system |
| Slint | Excellent (Skia) | Declarative DSL, less clear compute path | Rejected -- DSL learning curve |
| Tauri | Excellent (CSS) | WebKit startup ~200ms too slow | Rejected -- latency conflict |

### Dual Metal Architecture

Both egui-wgpu and objc2-metal reference same physical `MTLDevice` (Apple Silicon has one GPU). Separate `MTLCommandQueue` instances prevent contention. This architecture is proven -- shared GPU device, isolated command submission.

## Competitive Analysis

| Feature | gpu-search | ripgrep | Spotlight | VS Code Search |
|---------|-----------|---------|-----------|---------------|
| Throughput | 55-80 GB/s (GPU) | 5-15 GB/s (CPU SIMD) | ~2 GB/s (index) | ~2 GB/s |
| Content search | Yes | Yes | Metadata only | Yes |
| Syntax highlighting | Yes (syntect) | No | No | Yes |
| Search-as-you-type | <5ms | N/A (CLI) | ~50ms | 100-500ms |
| GUI | Native floating panel | CLI | Native | Electron |
| .gitignore | Yes | Yes | No | Yes |

**Unique value**: ripgrep speed + VS Code presentation + Spotlight interaction model.

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | **High** | Core algorithms proven at 55-80 GB/s. Port is API-level, not algorithmic. |
| Effort Estimate | **L** | ~8K line port + new GUI + new orchestrator + benchmarks. 7 modules, ~59 tasks. |
| Risk Level | **Medium** | Metal watchdog (chunked dispatches mitigate), objc2-metal IO bindings completeness, egui+Metal device sharing |
| Performance Confidence | **High** | MSL kernels unchanged. Same hardware. Same algorithms. Throughput should match. |

## Open Questions

1. **objc2-metal MTLIOCommandQueue completeness** -- feature flag exists, but some methods may need msg_send! fallback
2. **egui virtual scroll at 10K+ results** -- may need custom lazy rendering widget
3. **Global hotkey registration** -- macOS Accessibility permissions required, fallback needed
4. **syntect highlighting latency** -- cache aggressively, highlight only visible context lines

## Recommendations

1. Port in dependency order: device -> types -> mmap -> gpu_io -> batch -> content_search -> streaming
2. Extract MSL shaders from inline strings to standalone .metal files first
3. Use GPU-CPU dual verification for every search kernel test
4. Build POC with core search pipeline before adding UI complexity
5. Accept whatever wgpu version eframe pins -- avoid version conflicts
