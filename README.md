# gpu-forge

A Claude Code plugin that turns Claude into a GPU systems engineer for Apple Silicon. Ships a curated knowledge base of 601 research findings across 11 GPU computing domains, backed by 92 citations from academic papers, Apple documentation, and reverse-engineering projects.

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI
- `sqlite3` (pre-installed on macOS)
- `bash` (4.0+ recommended)
- `jq` (optional, used by hooks)
- Xcode Command Line Tools (optional, for Metal shader compilation tests)

## Installation

**Local directory (recommended for development):**

```bash
git clone https://github.com/patrickkavanagh/gpu-forge.git
claude --plugin-dir ./gpu-forge
```

**Manual setup:**

1. Clone this repository
2. Point Claude Code to the plugin directory:
   ```bash
   claude --plugin-dir /path/to/gpu-forge
   ```

**Marketplace (future):**

```bash
claude plugins install gpu-forge
```

## Quick Start

Ask a question about Apple Silicon GPU computing:

```
/gpu-forge:ask "What is the SIMD width on Apple GPUs?"
```

Check what the knowledge base contains:

```
/gpu-forge:knowledge stats
```

Research a new topic and store findings:

```
/gpu-forge:investigate gpu-silicon "M5 neural accelerators"
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `ask <query> [--skill <name>]` | Query the knowledge base. Detects relevant skill domains from keywords, returns findings ranked by BM25 relevance with confidence levels and citations. |
| `investigate <skill> <topic>` | Launch a deep research investigation. Runs in a forked subagent that conducts web research, extracts findings, and stores them in the database with citations. |
| `knowledge [subcommand]` | Browse the knowledge database. Subcommands: `stats`, `search <query>`, `skill <name>`, `topic <name>`, `detail <id>`, `unverified`, `export [skill]`, `investigations`. Defaults to `stats`. |
| `scaffold [project-name]` | Interactive project scaffolding wizard. Creates complete GPU compute projects with Metal shaders, Swift host code, build configuration, and benchmarks tailored to a target Apple Silicon chip. |
| `template <name> [--type <t>] [--op <o>]` | Generate code from the template library. Applies parameter substitutions to produce production-ready Metal, Swift, or MLX code for common GPU compute patterns. |
| `review <file-path>` | Review Metal/MSL/Swift/MLX code against GPU best practices. Checks threadgroup sizing, memory access patterns, SIMD utilization, and common anti-patterns using knowledge base findings. |

## Skills

gpu-forge organizes GPU computing expertise into 11 domain skills across 5 layers, from hardware fundamentals up to system architecture.

| Skill | Layer | Findings | Domain |
|-------|-------|----------|--------|
| `gpu-silicon` | 0 | 57 | Apple GPU microarchitecture, SIMD, TBDR, ALU pipelines |
| `unified-memory` | 0 | 75 | Unified memory, SLC cache, storage modes, zero-copy |
| `metal-compute` | 1 | 40 | Metal compute pipeline, command buffers, encoders, sync |
| `msl-kernels` | 1 | 49 | Metal Shading Language, address spaces, atomics, intrinsics |
| `gpu-io` | 2 | 38 | GPU I/O, Metal Fast Resource Loading, mmap, SSD streaming |
| `gpu-perf` | 2 | 49 | GPU performance, profiling, occupancy, optimization |
| `simd-wave` | 2 | 56 | SIMD/wave operations, reductions, scans, simdgroup_matrix |
| `mlx-compute` | 3 | 49 | MLX framework, custom kernels, lazy eval, streams |
| `metal4-api` | 3 | 66 | Metal 4 API, MTLTensor, cooperative tensors, unified encoder |
| `gpu-distributed` | 3 | 51 | Distributed GPU, RDMA, Thunderbolt 5, multi-Mac clusters |
| `gpu-centric-arch` | 4 | 71 | GPU-centric architecture, persistent kernels, reverse offloading |

Skills are auto-discovered at plugin load time. Each skill includes a SKILL.md with domain overview, key knowledge areas, query instructions, common Q&A patterns, and cross-references to related skills.

## Agents

gpu-forge includes 3 specialized agents, each tuned for a specific task:

**knowledge-retriever** (Haiku) -- Fast knowledge lookup agent. Parses queries, detects skill domains from keywords, runs FTS5 searches against the database, and returns findings with confidence labels and citations. Optimized for low latency with a 10-turn limit.

**investigation-agent** (Opus) -- Deep research agent that runs in a forked (isolated) context. Conducts multi-phase investigations: setup, web research, finding extraction, citation storage, and summary. Stores all results in the knowledge database with proper source attribution. 100-turn limit for thorough research sessions.

**architecture-advisor** (Sonnet) -- Expert advisor that designs GPU compute pipelines and recommends memory strategies. Preloads gpu-silicon, unified-memory, metal-compute, gpu-perf, and gpu-centric-arch skills. Cross-references findings across domains to provide architectural recommendations backed by evidence.

## Templates

### Metal Templates (5)

| Template | Description |
|----------|-------------|
| `reduction.metal.tmpl` | Parallel reduction (sum, max, min) with simdgroup optimization |
| `gemm.metal.tmpl` | Tiled matrix multiplication using threadgroup memory |
| `scan.metal.tmpl` | Prefix scan (inclusive/exclusive) with work-efficient algorithm |
| `histogram.metal.tmpl` | Parallel histogram with threadgroup-local bins and atomic merge |
| `blank.metal.tmpl` | Minimal Metal compute kernel scaffold |

### Swift Templates (3)

| Template | Description |
|----------|-------------|
| `Package.swift.tmpl` | Swift Package manifest with Metal framework dependency |
| `main.swift.tmpl` | Entry point with Metal device setup, buffer allocation, dispatch |
| `MetalCompute.swift.tmpl` | Metal compute pipeline wrapper class |

### MLX Template (1)

| Template | Description |
|----------|-------------|
| `custom-kernel.py.tmpl` | MLX custom Metal kernel via `mx.fast.metal_kernel()` |

### Scaffold Specs (5)

| Spec | Description |
|------|-------------|
| `reduction-kernel.json` | Parallel reduction project with Swift + Metal |
| `matrix-multiply.json` | Tiled GEMM using simdgroup_matrix |
| `scan-kernel.json` | Prefix sum / stream compaction project |
| `metal-compute-blank.json` | Blank Metal compute project skeleton |
| `mlx-custom-kernel.json` | Python + MLX custom kernel project |

All templates support parameter substitution (`{{TYPE}}`, `{{OP}}`, `{{KERNEL_NAME}}`, `{{TILE_SIZE}}`). Metal templates compile cleanly with `xcrun -sdk macosx metal -c` after parameter substitution.

## Contributing

The primary way to grow the knowledge base is through the investigate command:

```
/gpu-forge:investigate <skill-name> "<topic>"
```

This launches a structured investigation that searches the web, analyzes sources, extracts findings, and stores everything in the SQLite database with proper citations and confidence levels.

**Guidelines:**

1. Check existing knowledge first with `/gpu-forge:knowledge search "<topic>"` to avoid duplicates.
2. Use the correct skill name from the 11 domains listed above.
3. Focus on Apple Silicon only. No AMD/NVIDIA content unless directly comparing.
4. Prefer 2024-2026 sources unless the information is foundational.
5. Every finding needs a source URL and confidence level (verified, high, medium, low).
6. Academic papers get full citation records in the citations table.

After investigating, verify the new findings were stored:

```
/gpu-forge:knowledge search "<your topic>"
```

## License

MIT
