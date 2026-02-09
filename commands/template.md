---
name: template
description: "Generate code from Metal/Swift/MLX template library. Applies parameter substitutions and inserts production-ready GPU compute patterns."
argument-hint: "<template-name> [--type <data-type>] [--op <operation>]"
allowed-tools: [Read, Write]
---

# GPU Code Template Generator

You generate production-ready GPU compute code from the template library. Templates provide battle-tested patterns for common GPU computing operations on Apple Silicon.

## Arguments

Parse `$ARGUMENTS` to extract:
- **Template name** (required): Name of the template to generate (e.g., reduction, gemm, scan, histogram, blank)
- **--type <data-type>**: Data type substitution (float, half, int, uint) - replaces `{{TYPE}}` in template
- **--op <operation>**: Operation substitution (add, max, min, mul) - replaces `{{OP}}` in template
- **--name <kernel-name>**: Kernel name - replaces `{{KERNEL_NAME}}` in template
- **--tile-size <N>**: Tile size for GEMM template - replaces `{{TILE_SIZE}}` in template

## Available Templates

### Metal Compute Templates (`${CLAUDE_PLUGIN_ROOT}/templates/metal/`)

| Template | Description | Parameters |
|----------|-------------|------------|
| `reduction` | Parallel reduction (sum, max, min) with simdgroup optimization | `--type`, `--op`, `--name` |
| `gemm` | Tiled matrix multiplication using threadgroup memory | `--type`, `--tile-size`, `--name` |
| `scan` | Prefix scan (inclusive/exclusive) with work-efficient algorithm | `--type`, `--op`, `--name` |
| `histogram` | Parallel histogram with threadgroup-local bins | `--type`, `--name` |
| `blank` | Minimal Metal compute kernel scaffold | `--type`, `--name` |

### Swift Templates (`${CLAUDE_PLUGIN_ROOT}/templates/swift/`)

| Template | Description | Parameters |
|----------|-------------|------------|
| `Package.swift` | Swift package manifest with Metal dependencies | `--name` |
| `main.swift` | Entry point with Metal device setup and command queue | `--name` |
| `MetalCompute.swift` | Metal compute pipeline wrapper class | `--name` |

### MLX Templates (`${CLAUDE_PLUGIN_ROOT}/templates/mlx/`)

| Template | Description | Parameters |
|----------|-------------|------------|
| `custom-kernel.py` | MLX custom Metal kernel via `mx.fast.metal_kernel()` | `--type`, `--op`, `--name` |

## Parameter Validation

Before applying substitutions, validate all parameters:

1. **Template name**: Must match one of the available templates listed above. If not found:
   ```
   Error: Unknown template "<name>".
   Available templates: reduction, gemm, scan, histogram, blank, Package.swift, main.swift, MetalCompute.swift, custom-kernel.py
   ```

2. **--type**: Must be one of: `float`, `half`, `int`, `uint`. If invalid:
   ```
   Error: Invalid type "<value>". Supported types: float, half, int, uint
   ```

3. **--op**: Must be one of: `add`, `max`, `min`, `mul`. If invalid:
   ```
   Error: Invalid operation "<value>". Supported operations: add, max, min, mul
   ```

4. **--tile-size**: Must be a power of 2 between 4 and 64. If invalid:
   ```
   Error: Invalid tile size "<value>". Must be a power of 2 between 4 and 64 (e.g., 8, 16, 32)
   ```

5. **--name**: Must be a valid C/Swift identifier (letters, digits, underscores, no leading digit). If invalid:
   ```
   Error: Invalid kernel name "<value>". Must be a valid identifier (letters, digits, underscores).
   ```

## Template Resolution

1. Determine template category from name:
   - `reduction`, `gemm`, `scan`, `histogram`, `blank` -> Metal templates (`.tmpl` extension)
   - `Package.swift`, `main.swift`, `MetalCompute.swift` -> Swift templates (`.tmpl` extension)
   - `custom-kernel.py` -> MLX templates (`.tmpl` extension)

2. Construct template path:
   - Metal: `${CLAUDE_PLUGIN_ROOT}/templates/metal/<name>.tmpl`
   - Swift: `${CLAUDE_PLUGIN_ROOT}/templates/swift/<name>.tmpl`
   - MLX: `${CLAUDE_PLUGIN_ROOT}/templates/mlx/<name>.tmpl`

3. Read the template file content.

4. If the template file does not exist:
   ```
   Error: Template file not found at <path>.
   Run /gpu-forge:template --help to see available templates.
   ```

## Substitution Process

Apply parameter substitutions in order:

1. Replace all `{{TYPE}}` with the `--type` value (default: `float` if not specified)
2. Replace all `{{OP}}` with the `--op` value (default: `add` if not specified)
3. Replace all `{{KERNEL_NAME}}` with the `--name` value (default: derived from template name, e.g., `my_reduction`)
4. Replace all `{{TILE_SIZE}}` with the `--tile-size` value (default: `16` if not specified)

## Output

After successful substitution, present the generated code:

1. Show a summary header:
   ```
   Generated from template: <template-name>
   Parameters: type=<type>, op=<op>, name=<name>, tile-size=<tile-size>
   ```

2. Output the complete generated code in the appropriate fenced code block (```metal, ```swift, or ```python)

3. Suggest next steps:
   ```
   Next steps:
   - Save to a file: Copy the code above into your project
   - Customize: Modify the generated code for your specific use case
   - Profile: Use /gpu-forge:profile to measure kernel performance
   ```

## Examples

**Generate a float reduction kernel**:
```
/gpu-forge:template reduction --type float --op add --name sum_reduce
```
-> Reads `templates/metal/reduction.tmpl`, replaces `{{TYPE}}` with `float`, `{{OP}}` with `add`, `{{KERNEL_NAME}}` with `sum_reduce`

**Generate a GEMM kernel with custom tile size**:
```
/gpu-forge:template gemm --type half --tile-size 32 --name matmul_fp16
```
-> Reads `templates/metal/gemm.tmpl`, replaces `{{TYPE}}` with `half`, `{{TILE_SIZE}}` with `32`, `{{KERNEL_NAME}}` with `matmul_fp16`

**Generate an MLX custom kernel**:
```
/gpu-forge:template custom-kernel.py --type float --op mul --name elementwise_mul
```
-> Reads `templates/mlx/custom-kernel.py.tmpl`, replaces placeholders accordingly

**Generate a Swift project scaffold**:
```
/gpu-forge:template Package.swift --name MyGPUProject
```
-> Reads `templates/swift/Package.swift.tmpl`, replaces `{{KERNEL_NAME}}` with `MyGPUProject`
