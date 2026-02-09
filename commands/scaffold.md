---
name: scaffold
description: "Interactive project scaffolding wizard for GPU compute projects. Creates complete project structure with Metal shaders, Swift host code, build configuration, and benchmarks."
argument-hint: "[project-name]"
disable-model-invocation: true
allowed-tools: [Bash, Read, Write, Edit]
---

# GPU Project Scaffolding Wizard

Create a complete GPU compute project with proper structure, Metal shaders, host code, build configuration, and benchmarks tailored to the target Apple Silicon chip.

## Step 1: Gather Project Configuration

### 1.1 Project Name

If `$ARGUMENTS` contains a project name, use it. Otherwise ask:

```
What is your project name? (lowercase-with-dashes, e.g., "matrix-multiply")
```

Validate: lowercase alphanumeric + dashes only, no spaces.

### 1.2 Target Chip

Ask the user to select a target chip:

```
Target Apple Silicon chip:
  1) M4       — 10 GPU cores, 2.9 TFLOPS, 120 GB/s
  2) M4 Pro   — 20 GPU cores, ~9.2 TFLOPS, 273 GB/s
  3) M4 Max   — 40 GPU cores, ~18.4 TFLOPS, 546 GB/s
  4) M5       — Neural Accelerators in GPU cores, 4x AI compute vs M4

Enter number [1-4]:
```

Map selection to `{{TARGET_CHIP}}`: `m4`, `m4-pro`, `m4-max`, `m5`

### 1.3 Compute Pattern

Ask the user to select a compute pattern:

```
Compute pattern:
  1) reduction   — Parallel sum, min, max, dot product
  2) gemm        — General matrix multiply (tiled, simdgroup_matrix)
  3) scan        — Prefix sum, stream compaction
  4) histogram   — Binning, frequency counting
  5) custom      — Blank kernel, you fill in the logic

Enter number [1-5]:
```

Map selection to `{{COMPUTE_PATTERN}}`: `reduction`, `gemm`, `scan`, `histogram`, `custom`

### 1.4 Language

Ask the user to select a language stack:

```
Language:
  1) Swift + Metal   — Swift Package with .metal compute shaders
  2) Python + MLX    — Python with custom Metal kernels via mx.fast.metal_kernel
  3) Both            — Full project with Swift and Python components

Enter number [1-3]:
```

Map selection to `{{LANGUAGE}}`: `swift-metal`, `mlx-kernel`, `both`

### 1.5 Complexity Level

Ask the user to select project complexity:

```
Complexity:
  1) minimal          — Bare minimum: one kernel + host code
  2) standard         — Kernel + host + basic benchmark + README
  3) benchmark-ready  — Full benchmark harness, multiple input sizes, GPU timing, CSV output

Enter number [1-3]:
```

Map selection to `{{COMPLEXITY}}`: `minimal`, `standard`, `benchmark-ready`

## Step 2: Determine Template Type

Based on configuration, select the template type:

| Language | Complexity | Template Type |
|----------|-----------|---------------|
| swift-metal | minimal | `minimal` |
| swift-metal | standard | `swift-metal` |
| swift-metal | benchmark-ready | `benchmark` |
| mlx-kernel | minimal | `minimal` |
| mlx-kernel | standard | `mlx-kernel` |
| mlx-kernel | benchmark-ready | `benchmark` |
| both | any | `full-project` |

Template types:
- **`minimal`**: Bare minimum compute shader + host code to run one dispatch
- **`swift-metal`**: Swift Package + Metal compute shader + benchmark target
- **`mlx-kernel`**: Python + custom Metal kernel via `mx.fast.metal_kernel()`
- **`benchmark`**: Benchmark harness with GPU timing, multiple sizes, CSV output
- **`full-project`**: Complete project with Swift, Python/MLX, benchmarks, and docs

## Step 3: Read Templates

Read scaffold specification files from the templates directory:

```bash
ls ${CLAUDE_PLUGIN_ROOT}/templates/project/
```

Load relevant templates based on the selected template type:
- For `swift-metal` / `minimal` (Swift): Read from `${CLAUDE_PLUGIN_ROOT}/templates/swift/` and `${CLAUDE_PLUGIN_ROOT}/templates/metal/`
- For `mlx-kernel`: Read from `${CLAUDE_PLUGIN_ROOT}/templates/mlx/`
- For `full-project`: Read from all template directories
- For `benchmark`: Read from `${CLAUDE_PLUGIN_ROOT}/templates/project/`

## Step 4: Template Parameter Substitution

Replace these placeholders in all generated files:

| Placeholder | Value | Example |
|-------------|-------|---------|
| `{{PROJECT_NAME}}` | Project name | `matrix-multiply` |
| `{{KERNEL_NAME}}` | PascalCase kernel name derived from project name | `MatrixMultiply` |
| `{{KERNEL_NAME_LOWER}}` | snake_case kernel function name | `matrix_multiply` |
| `{{TYPE}}` | Template type selected | `swift-metal` |
| `{{TARGET_CHIP}}` | Selected chip | `m4-max` |
| `{{COMPUTE_PATTERN}}` | Selected pattern | `gemm` |
| `{{SIMD_WIDTH}}` | Always 32 for Apple Silicon | `32` |
| `{{MAX_THREADGROUP}}` | Always 1024 for Apple Silicon | `1024` |
| `{{THREADGROUP_MEMORY}}` | Always 32768 (32KB) for Apple Silicon | `32768` |
| `{{DATE}}` | Current date in YYYY-MM-DD format | `2026-02-09` |

## Step 5: Generate Project Structure

### For `minimal` (Swift + Metal)

```
{{PROJECT_NAME}}/
  {{KERNEL_NAME_LOWER}}.metal    — Compute kernel
  main.swift                     — Host code to dispatch kernel
  README.md                      — One-line description
```

### For `minimal` (Python + MLX)

```
{{PROJECT_NAME}}/
  {{KERNEL_NAME_LOWER}}.py       — MLX custom kernel
  README.md                      — One-line description
```

### For `swift-metal`

```
{{PROJECT_NAME}}/
  Package.swift                  — Swift Package manifest
  Sources/
    {{KERNEL_NAME}}/
      {{KERNEL_NAME}}.swift      — Host code (device, pipeline, dispatch)
      Shaders/
        {{KERNEL_NAME_LOWER}}.metal  — Compute kernel
  Benchmarks/
    {{KERNEL_NAME}}Benchmark.swift   — Basic benchmark
  README.md                      — Setup, build, run instructions
```

### For `mlx-kernel`

```
{{PROJECT_NAME}}/
  {{KERNEL_NAME_LOWER}}/
    __init__.py                  — Module init
    kernel.py                    — mx.fast.metal_kernel implementation
    metal/
      {{KERNEL_NAME_LOWER}}.metal    — Raw Metal kernel source
  benchmark.py                   — MLX benchmark with mx.metal.start/stop_capture
  requirements.txt               — mlx, numpy
  README.md                      — Setup, usage, benchmark instructions
```

### For `benchmark`

```
{{PROJECT_NAME}}/
  Package.swift                  — Swift Package manifest (if Swift)
  Sources/
    {{KERNEL_NAME}}/
      {{KERNEL_NAME}}.swift
      Shaders/
        {{KERNEL_NAME_LOWER}}.metal
  Benchmarks/
    {{KERNEL_NAME}}Benchmark.swift   — Full benchmark harness
    BenchmarkConfig.swift            — Sizes, iterations, warmup config
    CSVReporter.swift                — CSV output for results
  results/                           — Output directory for CSV
  README.md
```

### For `full-project`

```
{{PROJECT_NAME}}/
  Package.swift
  Sources/
    {{KERNEL_NAME}}/
      {{KERNEL_NAME}}.swift
      Shaders/
        {{KERNEL_NAME_LOWER}}.metal
  Benchmarks/
    {{KERNEL_NAME}}Benchmark.swift
    BenchmarkConfig.swift
    CSVReporter.swift
  python/
    {{KERNEL_NAME_LOWER}}/
      __init__.py
      kernel.py
      metal/
        {{KERNEL_NAME_LOWER}}.metal
    benchmark.py
    requirements.txt
  results/
  README.md
```

## Step 6: Compute Pattern Content

Generate kernel content based on the selected compute pattern:

### `reduction`
- Metal kernel: parallel reduction using `simdgroup_reduce_add` / threadgroup shared memory
- Two-pass approach: local reduce per threadgroup, then final reduce

### `gemm`
- Metal kernel: tiled matrix multiply using `simdgroup_matrix` types
- Configurable tile sizes based on target chip

### `scan`
- Metal kernel: Blelloch-style prefix sum
- Work-efficient with up-sweep and down-sweep phases

### `histogram`
- Metal kernel: per-threadgroup local histograms with atomic add
- Final merge pass across threadgroups

### `custom`
- Metal kernel: empty kernel body with proper argument table setup
- Comments indicating where to add custom logic

## Step 7: Target Chip Configuration

Set chip-specific parameters in generated code:

| Parameter | M4 | M4 Pro | M4 Max | M5 |
|-----------|-----|---------|---------|-----|
| GPU Cores | 10 | 20 | 40 | TBD |
| FP32 TFLOPS | 2.9 | ~9.2 | ~18.4 | TBD |
| Memory BW (GB/s) | 120 | 273 | 546 | TBD |
| Optimal threadgroup size | 256 | 256 | 256 | 256 |
| Max buffer size hint | 8GB | 24GB | 128GB | TBD |

Add comments in generated code with performance targets based on chip.

## Step 8: Create Files

Use the `Write` tool to create each file in the project directory. After all files are created:

```bash
find {{PROJECT_NAME}}/ -type f | sort
```

Display the created structure to the user.

## Step 9: Post-Scaffold Summary

Print a summary:

```
Project "{{PROJECT_NAME}}" created successfully!

Template:  {{TYPE}}
Pattern:   {{COMPUTE_PATTERN}}
Target:    {{TARGET_CHIP}}
Language:  {{LANGUAGE}}

Files created:
  <list of files>

Next steps:
  cd {{PROJECT_NAME}}
  <build/run commands based on language>
```

For Swift projects:
```
  swift build
  swift run {{KERNEL_NAME}}
  swift run {{KERNEL_NAME}}Benchmark
```

For Python/MLX projects:
```
  pip install -r requirements.txt
  python -m {{KERNEL_NAME_LOWER}}
  python benchmark.py
```
