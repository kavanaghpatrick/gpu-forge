---
name: investigate
description: "Launch a deep investigation into a GPU computing topic. Conducts web research, analyzes sources, and stores all findings in the knowledge database with citations."
argument-hint: "<skill-name> <topic>"
context: fork
agent: investigation-agent
disable-model-invocation: true
---

# Investigate Command

This command delegates entirely to the **investigation-agent**, which runs in a forked (isolated subagent) context. The agent conducts structured web research on the given GPU computing topic, analyzes sources, extracts findings, and stores everything in the knowledge database with proper citations and confidence levels.

## Valid Skill Names

- `gpu-silicon` — Apple GPU microarchitecture, SIMD, TBDR, ALU pipelines
- `unified-memory` — Unified memory, SLC cache, storage modes, zero-copy
- `metal-compute` — Metal compute pipeline, command buffers, encoders, sync
- `msl-kernels` — Metal Shading Language, address spaces, atomics, intrinsics
- `gpu-io` — GPU I/O, Metal Fast Resource Loading, mmap, SSD streaming
- `gpu-perf` — GPU performance, profiling, occupancy, optimization
- `simd-wave` — SIMD/wave operations, reductions, scans, simdgroup_matrix
- `mlx-compute` — MLX framework, custom kernels, lazy eval, streams
- `metal4-api` — Metal 4 API, MTLTensor, cooperative tensors, unified encoder
- `gpu-distributed` — Distributed GPU, RDMA, Thunderbolt 5, multi-Mac clusters
- `gpu-centric-arch` — GPU-centric architecture, persistent kernels, reverse offloading

## Example Usage

```
/gpu-forge:investigate gpu-silicon "execution pipeline deep dive"
/gpu-forge:investigate unified-memory "SLC cache behavior under contention"
/gpu-forge:investigate metal4-api "cooperative tensor programming model"
```

## How It Works

1. The command parses `$ARGUMENTS` to extract the skill name and topic
2. Forks an isolated subagent running the investigation-agent
3. The agent executes its 5-phase protocol: setup, research, store findings, citations, summary
4. All findings are stored in the knowledge database at `${CLAUDE_PLUGIN_ROOT}/data/gpu_knowledge.db`
5. Returns a summary of findings added, sources consulted, and coverage gaps

**Note**: This runs in forked context — the investigation agent operates in an isolated subagent that does not share conversation history with the main session.
