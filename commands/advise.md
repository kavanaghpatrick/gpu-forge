---
name: advise
description: "Get architectural recommendations for GPU compute designs, backed by knowledge database findings."
argument-hint: "<description-of-requirements>"
context: fork
agent: architecture-advisor
disable-model-invocation: true
---

# Advise Command

This command delegates entirely to the **architecture-advisor**, which runs in a forked
(isolated subagent) context. The agent queries the knowledge database across multiple
skill domains, cross-references findings, and provides architectural recommendations
with citations.

## Valid Skill Areas

- `gpu-silicon` -- Apple GPU microarchitecture, SIMD, TBDR, ALU pipelines
- `unified-memory` -- Unified memory, SLC cache, storage modes, zero-copy
- `metal-compute` -- Metal compute pipeline, command buffers, encoders, sync
- `gpu-perf` -- GPU performance, profiling, occupancy, optimization
- `gpu-centric-arch` -- GPU-centric architecture, persistent kernels, reverse offloading

## Example Usage

```
/gpu-forge:advise "Design a high-performance reduction kernel for 10M floats on M4 Max"
/gpu-forge:advise "Memory strategy for streaming 2GB dataset through GPU pipeline"
/gpu-forge:advise "Architecture for persistent kernel radix sort with decoupled lookback"
```

## How It Works

1. The command parses `$ARGUMENTS` to extract the requirement description
2. Forks an isolated subagent running the architecture-advisor
3. The advisor queries KB across 5+ skill domains
4. Cross-references findings to provide evidence-based recommendations
5. Returns structured output: Architecture Overview, Key Decisions, Rationale, Trade-offs

**Note**: This runs in forked context -- the advisor operates in an isolated subagent
that does not share conversation history with the main session.
