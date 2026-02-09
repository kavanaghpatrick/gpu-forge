# Changelog

All notable changes to the gpu-forge plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-09

First stable release of gpu-forge, a Claude Code plugin for Apple Silicon GPU-centric computing expertise.

### Added

#### Knowledge Database
- 601 curated findings across 11 GPU computing domains
- FTS5 full-text search with BM25 column weighting (claim 10x, evidence 5x, tags 2x, notes 1x)
- Portable KB CLI (`scripts/kb`) with 3-level path resolution (env > plugin root > script-relative)
- SQLite database with schema enforcement, FTS5 sync triggers, and confidence ceiling constraints
- DB bootstrap script (`scripts/bootstrap-db.sh`) for fresh installations
- Reference export script (`scripts/export-refs.sh`) generating per-skill finding summaries

#### Skills (11 domains, 4 layers)
- **Layer 0 (Hardware)**: `gpu-silicon` (57 findings), `unified-memory` (75 findings)
- **Layer 1 (API)**: `metal-compute` (40 findings), `msl-kernels` (49 findings)
- **Layer 2 (Optimization)**: `gpu-io` (38 findings), `gpu-perf` (49 findings), `simd-wave` (56 findings)
- **Layer 3 (Frameworks)**: `mlx-compute` (49 findings), `metal4-api` (66 findings), `gpu-distributed` (51 findings)
- **Layer 4 (Architecture)**: `gpu-centric-arch` (71 findings)
- Each skill includes YAML frontmatter, domain overview, key knowledge areas, KB query instructions, Q&A pairs, and cross-references
- All skills under 500 lines with portable `${CLAUDE_PLUGIN_ROOT}` path references
- Layer 3 reference files (`all-findings.md`, `index.md`) auto-generated per skill

#### Agents (3 specialized)
- **knowledge-retriever** (Haiku, 10 turns): Fast KB queries with adaptive result counts and citation formatting
- **investigation-agent** (Opus, 100 turns): Deep research with 5-phase protocol (setup, research, store, citations, summary) using WebSearch/WebFetch
- **architecture-advisor** (Sonnet): Cross-domain architectural guidance with 5 preloaded skills

#### Commands (6)
- `/gpu-forge:ask <query>` -- Knowledge queries with BM25-ranked results and confidence labels (Sonnet)
- `/gpu-forge:investigate <skill> <topic>` -- Deep research sessions in forked context (Opus agent)
- `/gpu-forge:knowledge <subcommand>` -- Direct KB access: stats, search, skill, topic, detail, unverified, export, investigations (Haiku)
- `/gpu-forge:scaffold <type>` -- Interactive project scaffolding with template parameter substitution
- `/gpu-forge:template <name>` -- Insert code templates with type/operation parameter substitution
- `/gpu-forge:review <file>` -- GPU code review checking anti-patterns against KB findings (Sonnet)

#### Hook System
- **SessionStart**: Context loader verifies DB presence and reports finding count
- **PostToolUse**: Metal file validator checks `.metal` files for missing `[[kernel]]` attributes on Edit/Write
- **kb-wrapper.sh**: Bridge script for hook-based KB access with path resolution

#### Template Library (14 templates)
- **Metal (5)**: `reduction.metal.tmpl` (tree-based parallel), `gemm.metal.tmpl` (tiled with simdgroup shared memory), `scan.metal.tmpl` (Hillis-Steele inclusive prefix sum), `histogram.metal.tmpl` (two-phase with atomics), `blank.metal.tmpl` (minimal skeleton)
- **Swift (3)**: `Package.swift.tmpl` (SPM with Metal framework), `main.swift.tmpl` (Metal device + buffer setup), `MetalCompute.swift.tmpl` (pipeline creation + dispatch)
- **MLX (1)**: `custom-kernel.py.tmpl` (mx.fast.metal_kernel boilerplate)
- **Scaffold Specs (5)**: `reduction-kernel.json`, `matrix-multiply.json`, `scan-kernel.json`, `metal-compute-blank.json`, `mlx-custom-kernel.json`
- All Metal templates verified to compile with `xcrun -sdk macosx metal -c`

#### Test Suite (194 BATS tests)
- **Unit tests (118)**: plugin structure, KB CLI, FTS5 search, DB integrity, skills validation, commands, agents, hooks, templates, Metal compilation
- **Golden queries (54)**: ~5 queries per skill verifying FTS5 relevance ranking
- **Integration tests (13)**: end-to-end workflows, hook execution, SQL injection prevention
- **Performance benchmarks (9)**: KB CLI timing (<2s stats, <1s search, <500ms detail), context budget (<16K chars), DB size (<5MB)
- Dual BATS library loading (Homebrew + submodule) for CI and local dev

#### Infrastructure
- Plugin manifest (`.claude-plugin/plugin.json`)
- GitHub Actions CI pipeline (`test.yml`) targeting macOS-latest with BATS
- Plugin state schema (`schemas/plugin-state.schema.json`) for investigation tracking
- MIT License

## [Unreleased]

_No unreleased changes._
