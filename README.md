# gpu-forge

**Apple Silicon GPU systems engineering -- knowledge base, compute libraries, and tools.**

gpu-forge is a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) plugin that gives Claude deep expertise in Apple Silicon GPU programming, plus a growing collection of production Metal compute libraries built using that knowledge.

The plugin ships a curated knowledge base of **753 research findings** across **11 GPU computing domains**, backed by **94 academic citations** and **24 structured investigations** drawn from Apple documentation, reverse-engineering projects (Asahi Linux), academic papers, and Metal framework internals.

The compute libraries are real implementations -- a GPU radix sort hitting **4,800+ Mkeys/s**, GPU-accelerated text search, and a real-time sort visualization -- all written in Rust + Metal and tested on Apple Silicon.

## Highlights

| Project | What | Performance |
|---------|------|-------------|
| [forge-sort](metal-forge-compute/forge-sort/) | GPU radix sort library | **31x** faster than `sort_unstable()`, **4,800+ Mk/s** zero-copy |
| [gpu-search](gpu-search/) | GPU-accelerated text search | 110 GB/s content scanning |
| [sort-demo](sort-demo/) | Real-time sort visualization | 60fps interactive rendering |
| gpu-forge plugin | Knowledge base + AI agents | 753 findings, 11 domains |

## forge-sort

The fastest GPU radix sort on Apple Silicon. Supports **6 data types**, **argsort**, and **key-value pair sorting**.

```rust
use forge_sort::GpuSorter;

let mut sorter = GpuSorter::new()?;

// Sort any supported type
sorter.sort_u32(&mut data_u32)?;
sorter.sort_i32(&mut data_i32)?;
sorter.sort_f32(&mut data_f32)?;
sorter.sort_u64(&mut data_u64)?;
sorter.sort_i64(&mut data_i64)?;
sorter.sort_f64(&mut data_f64)?;

// Argsort -- returns index permutation
let indices = sorter.argsort_f32(&data)?;

// Key-value pair sort -- co-sorts values by key order
sorter.sort_pairs_u32(&mut keys, &mut values)?;

// Zero-copy -- data stays in GPU memory, no memcpy
let buf = sorter.alloc_sort_buffer::<u32>(16_000_000);
sorter.sort_buffer(&buf)?;
```

### Benchmarks (M4 Pro, 20-core GPU)

| Size | `sort_unstable` | `rayon par_sort` | **forge-sort** | vs CPU | vs Rayon |
|-----:|----------------:|-----------------:|---------------:|-------:|---------:|
| 1M | 9.05 ms | 2.61 ms | **0.97 ms** | 9.3x | 2.7x |
| 4M | 39.71 ms | 10.55 ms | **1.50 ms** | 26.5x | 7.0x |
| 16M | 172.72 ms | 49.60 ms | **5.67 ms** | 30.5x | 8.7x |

Zero-copy throughput (data already in Metal buffer): **4,800+ Mk/s** at sweet-spot sizes.

**Algorithm**: MSD+fused-inner 8-bit radix sort in 4 GPU dispatches within a single command encoder. Atomic scatter (no spin-wait), fused 3-pass inner LSD, per-simdgroup histograms, ~22 KB threadgroup memory.

165 tests covering u32/i32/f32/u64/i64/f64, argsort, sort_pairs, edge cases, and performance.

See [forge-sort/README.md](metal-forge-compute/forge-sort/README.md) for full API docs and benchmarks.

## Repository Structure

```
gpu-forge/
  metal-forge-compute/           # Rust + Metal compute workspace
    forge-sort/                  #   GPU radix sort library (v0.2.0)
    forge-primitives/            #   Metal device, buffer, PSO utilities
    forge-bench/                 #   Benchmark harness CLI
  gpu-search/                    # GPU-accelerated text search (Rust + Metal)
  sort-demo/                     # Real-time sort visualization (Rust + winit)
  .claude-plugin/plugin.json     # Claude Code plugin manifest
  skills/                        # 11 domain skills (2,043 lines of expertise)
  agents/                        # 3 specialized AI agents
  commands/                      # 6 slash commands
  hooks/                         # Auto-loading context system
  templates/                     # 9 code templates + 5 scaffold specs
  data/gpu_knowledge.db          # SQLite FTS5 knowledge base
  scripts/kb                     # Knowledge base CLI
  tests/                         # 212 BATS tests
  .github/workflows/             # CI/CD (3 workflows)
```

## Numbers

| Metric | Value |
|--------|-------|
| **Compute** | |
| forge-sort throughput | 4,800+ Mk/s zero-copy |
| forge-sort data types | u32, i32, f32, u64, i64, f64 |
| forge-sort tests | 165 (+ 40 forge-primitives) |
| **Knowledge** | |
| Research findings | 753 |
| Verified findings | 448 (59%) |
| Academic citations | 94 |
| Investigations | 24 |
| Domain skills | 11 (5 layers) |
| **Plugin** | |
| Specialized agents | 3 (Haiku, Sonnet, Opus) |
| Slash commands | 6 |
| BATS tests | 212 (100% pass) |
| Knowledge DB size | 1.3 MB |

## Installation

### Plugin only

```bash
git clone https://github.com/kavanaghpatrick/gpu-forge.git
claude --plugin-dir ./gpu-forge
```

Requirements: `sqlite3` (pre-installed on macOS), `bash` 4.0+. Optional: `jq` (hooks), Xcode Command Line Tools (Metal compilation tests).

### Compute libraries

```bash
cd metal-forge-compute
cargo build --workspace --release
cargo test -p forge-sort -- --test-threads=1
```

Requirements: Rust 1.70+, macOS with Apple Silicon (M1+), Xcode Command Line Tools.

## Commands

### `/gpu-forge:ask` -- Query the knowledge base

Ask any question about Apple Silicon GPU computing. gpu-forge detects relevant skill domains from your keywords, searches the FTS5 index with BM25 ranking, and returns findings with confidence levels and source citations.

```
/gpu-forge:ask "What's the difference between device and threadgroup memory in MSL?"
/gpu-forge:ask "How does SLC cache work on M4 Max?" --skill unified-memory
/gpu-forge:ask "simdgroup_matrix performance characteristics"
```

### `/gpu-forge:investigate` -- Deep research

Launch a multi-phase investigation that searches the web, reads sources, extracts findings, and stores everything in the database with citations. Runs in an isolated Opus-powered agent with 100-turn budget.

```
/gpu-forge:investigate gpu-silicon "M5 neural accelerators in GPU cores"
/gpu-forge:investigate metal4-api "cooperative tensor operations"
/gpu-forge:investigate gpu-distributed "RDMA latency over Thunderbolt 5"
```

### `/gpu-forge:knowledge` -- Browse the database

Direct access to the knowledge base. Eight subcommands for exploring what's been researched.

```
/gpu-forge:knowledge stats                    # Overview of all 11 domains
/gpu-forge:knowledge search "occupancy"       # Full-text BM25 search
/gpu-forge:knowledge skill gpu-perf           # All findings for a domain
/gpu-forge:knowledge detail 142               # Full detail on one finding
/gpu-forge:knowledge unverified               # Findings needing validation
/gpu-forge:knowledge export gpu-silicon       # Export as markdown
```

### `/gpu-forge:scaffold` -- Create GPU projects

Interactive wizard that generates complete GPU compute projects. Pick your chip target, compute pattern, and language -- get a working project with shaders, host code, and build configuration.

```
/gpu-forge:scaffold my-reduction              # Interactive wizard
/gpu-forge:scaffold                           # Prompts for everything
```

**Project types:** Swift + Metal, Python + MLX, benchmark harness, full project, minimal skeleton.
**Patterns:** reduction, GEMM, scan, histogram, custom.
**Targets:** M4, M4 Pro, M4 Max, M5.

### `/gpu-forge:template` -- Generate code from patterns

Insert production-ready GPU code from the template library. All Metal templates are verified to compile.

```
/gpu-forge:template reduction --type float --op max
/gpu-forge:template gemm --type half --tile-size 32
/gpu-forge:template scan --type int --op sum
/gpu-forge:template blank --name my_kernel --type float
```

### `/gpu-forge:review` -- Code review against best practices

Review Metal/MSL code against the knowledge base. Checks threadgroup sizing, memory access patterns, SIMD utilization, atomics usage, and common anti-patterns -- with specific finding citations.

```
/gpu-forge:review Sources/Shaders/compute.metal
/gpu-forge:review my_kernel.py
```

## Skills Architecture

gpu-forge organizes GPU expertise into **5 layers** of progressive depth. Each skill is loaded on-demand when Claude detects relevant keywords in your conversation.

```
Layer 4:  gpu-centric-arch (71 findings)
          GPU-as-CPU paradigm, persistent kernels, GPU OS, reverse offloading

Layer 3:  mlx-compute (49)  |  metal4-api (70)  |  gpu-distributed (51)
          MLX custom kernels |  Metal 4 tensors   |  RDMA, Thunderbolt 5

Layer 2:  gpu-io (38)  |  gpu-perf (49)  |  simd-wave (56)
          Fast I/O, mmap |  Profiling, occupancy |  SIMD ops, reductions

Layer 1:  metal-compute (84)  |  msl-kernels (95)
          Metal pipeline, sync |  MSL language, atomics

Layer 0:  gpu-silicon (86)  |  unified-memory (104)
          Hardware architecture |  Memory system, SLC, storage modes
```

**Progressive disclosure**: Claude gets ~100 tokens of metadata per skill at session start. When a relevant topic comes up, the full SKILL.md body loads (~150-200 lines of domain expertise). For deep dives, the `references/` directory provides all 753 individual findings with evidence and citations.

## Agents

**knowledge-retriever** (Haiku, 10 turns) -- Fast knowledge lookup. Parses queries, detects domains, searches FTS5, returns ranked findings. Optimized for low latency.

**investigation-agent** (Opus, 100 turns) -- Deep research in isolated context. Conducts web searches, analyzes sources, extracts findings, stores in DB with proper citations. Used by `/gpu-forge:investigate`.

**architecture-advisor** (Sonnet) -- GPU compute architecture advisor. Preloads 5 core skills (gpu-silicon, unified-memory, metal-compute, gpu-perf, gpu-centric-arch). Cross-references findings across domains for architectural recommendations.

## Metal Templates

All Metal templates produce valid MSL that compiles with `xcrun -sdk macosx metal -c`.

| Template | Pattern | Parameters |
|----------|---------|------------|
| `reduction` | Parallel tree reduction with threadgroup shared memory | `{{TYPE}}`, `{{OP}}`, `{{IDENTITY}}` |
| `gemm` | Tiled matrix multiply with simdgroup memory | `{{TYPE}}`, `{{TILE_SIZE}}` |
| `scan` | Hillis-Steele inclusive prefix scan | `{{TYPE}}`, `{{OP}}`, `{{IDENTITY}}` |
| `histogram` | Two-phase: threadgroup-local bins + atomic global merge | `{{TYPE}}`, `{{BINS}}` |
| `blank` | Minimal compute kernel skeleton | `{{KERNEL_NAME}}`, `{{TYPE}}` |

## Knowledge Database

The SQLite database uses **FTS5 full-text search** with **BM25 column weighting** for relevance ranking:

| Column | Weight | Rationale |
|--------|--------|-----------|
| `claim` | 10.0 | Primary assertion -- highest relevance signal |
| `evidence` | 5.0 | Supporting data and measurements |
| `tags` | 2.0 | Categorization keywords |
| `notes` | 1.0 | Additional context |

Every finding has:
- **Skill domain** (one of 11)
- **Confidence level** (verified, high, medium, low)
- **Source type** (academic_paper, official_docs, technical_blog, reverse_engineering, benchmark, conference_talk)
- **Source URL** and title
- **Confidence ceiling**: blog/forum sources can never be marked "verified"

The database is self-contained at 1.3 MB and ships with the plugin.

## Testing

212 BATS tests across 4 categories:

| Category | Tests | What it covers |
|----------|-------|---------------|
| Unit | 118 | Plugin structure, skills validation, commands, agents, hooks, templates, KB CLI, DB integrity, FTS5 search |
| Golden queries | 72 | FTS5 search relevance -- ~5-10 queries per domain, verifying the right skill surfaces for natural-language questions |
| Integration | 13 | End-to-end workflows, hook execution, SQL injection prevention, concurrent DB access |
| Performance | 9 | Response time thresholds (search < 1s, detail < 500ms), size budgets (DB < 5MB, descriptions < 16KB) |

Run the full suite:

```bash
cd gpu-forge && bats tests/
```

## Knowledge Domains at a Glance

| Domain | Key Topics | Example Findings |
|--------|-----------|-----------------|
| **gpu-silicon** | M4/M5 architecture, SIMD width 32, TBDR, ALU pipelines, GPU core layout | "Apple M4 has 10 GPU cores at ~2.9 TFLOPS FP32" |
| **unified-memory** | SLC cache hierarchy, MTLStorageMode, zero-copy buffers, bandwidth (120-546 GB/s) | "makeBuffer(bytesNoCopy:) enables true zero-copy GPU access to mmap'd files" |
| **metal-compute** | MTLCommandQueue, compute encoders, indirect dispatch, synchronization | "Triple-buffering command buffers prevents CPU-GPU stalls" |
| **msl-kernels** | Address spaces (device/threadgroup/constant), atomics, function constants | "Function constants enable compile-time specialization without recompilation" |
| **gpu-io** | MTLIOCommandQueue, mmap, fast resource loading, SSD-to-GPU streaming | "Metal Fast Resource Loading bypasses CPU for async SSD-to-GPU transfers" |
| **gpu-perf** | Occupancy, profiling, bandwidth optimization, memory coalescing | "Threadgroup size must be multiple of 32 for full SIMD occupancy" |
| **simd-wave** | simd_shuffle, simdgroup_matrix, reductions, prefix scans | "simdgroup_matrix enables hardware-accelerated 8x8 matrix ops" |
| **mlx-compute** | mx.fast.metal_kernel(), lazy evaluation, streams, distributed training | "MLX custom kernels bypass framework overhead for 2-3x speedup" |
| **metal4-api** | MTLTensor, cooperative tensors, unified encoder, residency sets | "Metal 4 replaces command buffer model with unified encoder + command bundles" |
| **gpu-distributed** | RDMA over Thunderbolt 5, MLX distributed, ring allreduce | "Thunderbolt 5 RDMA enables ~10 GB/s inter-Mac GPU communication" |
| **gpu-centric-arch** | Persistent kernels, megakernels, GPU scheduling, reverse offloading | "LithOS (SOSP 2025) demonstrates GPU-centric OS with CPU as I/O coprocessor" |

## CI/CD

Three GitHub Actions workflows on `macos-14` (Apple Silicon) runners:

| Workflow | Triggers | What it does |
|----------|----------|-------------|
| `forge-ci.yml` | Push/PR to `metal-forge-compute/**` | Clippy + check + 205 tests + perf tests (main only) + release build + artifact upload |
| `gpu-search-ci.yml` | Push/PR to `gpu-search/**` | 5-stage pipeline: lint, build, unit, integration, bench |
| `test.yml` | Push/PR | 212 BATS plugin tests |

All workflows run with `MTL_SHADER_VALIDATION=1` and `RUSTFLAGS=-D warnings`.

## Contributing

Grow the knowledge base through investigation:

```
/gpu-forge:investigate <skill-name> "<topic>"
```

1. Check existing knowledge first: `/gpu-forge:knowledge search "<topic>"`
2. Use one of the 11 skill names listed above
3. Apple Silicon focus only (no AMD/NVIDIA unless comparing)
4. Prefer 2024-2026 sources unless foundational
5. Every finding needs a source URL and confidence level
6. Academic papers get full citation records

## Key Sources

The knowledge base draws from:

- **Apple Developer Documentation** -- Metal, Metal Shading Language, Metal Best Practices Guide
- **WWDC Sessions** -- Metal 4 (2025), GPU programming, performance optimization
- **Asahi Linux GPU Driver** -- Reverse-engineered Apple GPU architecture and ISA
- **Philip Turner's metal-benchmarks** -- Apple GPU microarchitecture measurements
- **Philip Turner's metal-usm** -- CPU pointer access from GPU shaders
- **arXiv:2502.05317** -- Apple Silicon HPC benchmark study
- **LithOS (SOSP 2025)** -- GPU-centric operating system research
- **MLX Documentation** -- Apple's ML framework for Apple Silicon

## License

MIT
