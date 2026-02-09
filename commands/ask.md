---
name: ask
description: Query the GPU computing knowledge base. Ask any question about Apple Silicon GPU architecture, Metal APIs, MSL, performance optimization, or related topics.
argument-hint: <query> [--skill <name>] [--limit N]
allowed-tools: [Bash, Read, Grep, Task]
model: sonnet
---

# GPU Knowledge Query Agent

You are querying the GPU computing knowledge database to answer questions about Apple Silicon GPU-centric computing.

## Arguments

Parse `$ARGUMENTS` to extract:
- **Query text**: The main question or search terms
- **--skill <name>**: Optional skill domain filter (gpu-silicon, unified-memory, metal-compute, msl-kernels, gpu-io, gpu-perf, simd-wave, mlx-compute, metal4-api, gpu-distributed, gpu-centric-arch)
- **--limit N**: Optional result limit (default: 10)

## Query Strategy

1. **Detect skill domains from keywords**:
   - GPU cores, SIMD, TBDR, M4, M5 → gpu-silicon
   - Unified memory, SLC, storage modes, zero-copy → unified-memory
   - Metal API, pipeline, command queue, encoder → metal-compute
   - MSL, kernel, threadgroup, address space → msl-kernels
   - GPU I/O, mmap, Metal IO, SSD streaming → gpu-io
   - Performance, occupancy, profiling, optimization → gpu-perf
   - Simdgroup, wave, reduction, scan → simd-wave
   - MLX, custom kernel, lazy eval → mlx-compute
   - Metal 4, MTLTensor, cooperative → metal4-api
   - Distributed, RDMA, Thunderbolt → gpu-distributed
   - GPU-centric, persistent kernel, reverse offload → gpu-centric-arch

2. **Run KB search**:
   ```bash
   ${CLAUDE_PLUGIN_ROOT}/scripts/kb search "<query>" --limit N
   ```

3. **If --skill specified, also filter by domain**:
   ```bash
   ${CLAUDE_PLUGIN_ROOT}/scripts/kb skill <skill-name>
   ```

## Result Formatting

For each finding, display:
- **ID**: Finding ID for reference
- **Domain**: Skill area (e.g., "gpu-silicon")
- **Claim**: The main finding
- **Confidence**: [verified] [high] [medium] [low]
- **Source**: source_title with URL

Format:
```
[<id>] <skill_name>: <claim>
Confidence: <confidence_level>
Source: <source_title> (<source_url>)
```

## Grouping for Cross-Domain Queries

If query spans multiple skills:
1. Group results by primary skill domain
2. Show related findings from other domains under "Related from other domains:"

## No Results Fallback

If search returns nothing:
1. Suggest broadening the query (remove technical jargon, use synonyms)
2. Recommend using `/gpu-forge:investigate <skill> <topic>` to research and add new knowledge
3. List related skills that might contain relevant information

## Examples

**Query**: "How does threadgroup memory work?"
- Detects: msl-kernels, gpu-perf
- Searches: "threadgroup memory"
- Returns: Findings about address spaces, sizing, bank conflicts

**Query**: "M4 GPU SIMD width --skill gpu-silicon --limit 5"
- Detects: --skill flag overrides auto-detection
- Searches: "M4 GPU SIMD width" filtered to gpu-silicon skill
- Returns: Max 5 results

**Query**: "Metal 4 tensor API"
- Detects: metal4-api
- Searches: "Metal 4 tensor API"
- Returns: MTLTensor findings, cooperative tensor operations

## Response Format

Always end with:
```
---
💡 Need more detail? Use /gpu-forge:knowledge detail <id>
🔬 Topic not found? Use /gpu-forge:investigate <skill> "<topic>"
```
