---
name: knowledge
description: "Query the GPU computing knowledge database. Supports stats, search, skill listing, topic lookup, finding details, unverified findings, export, and investigation history."
argument-hint: "<subcommand> [args]"
model: haiku
allowed-tools: [Bash, Read]
---

# Knowledge Command

Query and browse the GPU computing knowledge database using the KB CLI.

## Arguments

Parse `$ARGUMENTS` to extract a subcommand and its arguments. If `$ARGUMENTS` is empty or blank, default to `stats`.

## Subcommands

All subcommands delegate to `${CLAUDE_PLUGIN_ROOT}/scripts/kb`.

### `stats` (default when no arguments)

Show an overview of the knowledge base: finding counts per skill, verified vs unverified totals, citation count, investigation count.

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/kb stats
```

### `search <query>`

Full-text BM25-ranked search across all findings. Returns the most relevant results first.

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "<query>"
```

Optional: append `--limit N` to control result count (default 10).

### `skill <name>`

List all findings for a specific skill domain.

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill <name>
```

Valid skill names: `gpu-silicon`, `unified-memory`, `metal-compute`, `msl-kernels`, `gpu-io`, `gpu-perf`, `simd-wave`, `mlx-compute`, `metal4-api`, `gpu-distributed`, `gpu-centric-arch`.

### `topic <name>`

Find findings by topic across all skills.

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/kb topic <name>
```

### `detail <id>`

Show the full detail of a specific finding, including all fields and any associated citations.

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <id>
```

### `unverified`

List all findings that need validation (confidence = "unverified").

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/kb unverified
```

### `export [skill]`

Export all findings as markdown. Optionally filter to a single skill.

```bash
# Export all findings
${CLAUDE_PLUGIN_ROOT}/scripts/kb export

# Export findings for one skill
${CLAUDE_PLUGIN_ROOT}/scripts/kb export <skill-name>
```

### `investigations`

List past investigation sessions with their status, finding counts, and timestamps.

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/kb investigations
```

## Execution

1. Parse `$ARGUMENTS` to extract the subcommand (first word) and remaining arguments
2. If no subcommand is provided, run `stats`
3. Run the appropriate `${CLAUDE_PLUGIN_ROOT}/scripts/kb` command
4. Display the output directly to the user

## Output Formatting

- For tabular output (stats, skill, topic, unverified, investigations), display as-is from the CLI
- For `detail`, show all fields clearly labeled
- For `search`, highlight the confidence level and source for each result
- For `export`, display the full markdown output

## Examples

**Stats overview** (default):
```
/gpu-forge:knowledge
/gpu-forge:knowledge stats
```

**Search for a topic**:
```
/gpu-forge:knowledge search "threadgroup memory"
/gpu-forge:knowledge search "SIMD width" --limit 5
```

**List findings for a skill**:
```
/gpu-forge:knowledge skill gpu-silicon
/gpu-forge:knowledge skill metal-compute
```

**Find findings by topic**:
```
/gpu-forge:knowledge topic "occupancy"
/gpu-forge:knowledge topic "zero-copy"
```

**Get full detail of a finding**:
```
/gpu-forge:knowledge detail 42
```

**List unverified findings**:
```
/gpu-forge:knowledge unverified
```

**Export findings**:
```
/gpu-forge:knowledge export
/gpu-forge:knowledge export gpu-perf
```

**View investigation history**:
```
/gpu-forge:knowledge investigations
```

## Error Handling

- If the subcommand is not recognized, show the list of valid subcommands
- If the KB CLI returns an error, display it and suggest checking that the database exists
- For `detail` with a non-numeric ID, inform the user that the ID must be a number
- For `skill` with an invalid name, list the valid skill names
