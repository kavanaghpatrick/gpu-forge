---
id: agent-prompts.BREAKDOWN
module: agent-prompts
priority: 5
status: failing
version: 1
origin: spec-workflow
dependsOn: [cli-search.BREAKDOWN, cli-commands.BREAKDOWN]
tags: [gpu-forge, temporal-relevance]
testRequirements:
  unit:
    required: true
    pattern: "tests/unit/temporal.bats"
---

# BREAKDOWN: Agent Prompts -- Temporal Awareness for AI Agents

## Context

Three agent prompt files need surgical additions to integrate temporal awareness into the investigation and retrieval pipeline. The changes are minimal (total ~230 tokens across 3 files) but critical for ensuring new findings get generation tags and supersession links are created automatically.

The investigation agent is the primary write-path agent. It needs generation tagging instructions (Phase 3), supersession checking (Phase 2), and supersession resolution (Phase 5). The knowledge retriever is the primary read-path agent. It needs generation detection from query text. The architecture advisor needs cross-generation consistency awareness.

All prompt changes are additive insertions, not rewrites. Existing prompt structure and instructions are preserved.

Reference: TECH.md Section 4 (Agent Prompt Changes), UX.md Section 2 (Agent Interaction Design)

## Tasks

### T-001: Update investigation-agent.md -- Phase 2 supersession check

Insert after existing Phase 2 step 3 ("Code Analysis"):

```markdown
4. **Supersession check**: Before storing a generation-specific finding, check if the KB
   already has findings on the same skill+topic with a different generation:
   \`\`\`bash
   ${CLAUDE_PLUGIN_ROOT}/scripts/kb search "<topic keywords>" --limit 5
   \`\`\`
   If an older-generation finding covers the same metric, note it for supersession in Phase 5.
```

~40 tokens added.

### T-002: Update investigation-agent.md -- Phase 3 generation tagging

Insert after the "Rules for storing findings" section (after confidence-source cross-validation table):

```markdown
**Generation tagging (required for benchmarks):**
- For `benchmark` or `empirical_test` findings, always set `gpu_generation` to the
  relevant Apple Silicon generation: m1, m2, m3, m4, m5, or universal.
- Add gpu_generation to your INSERT: `..., gpu_generation) VALUES ..., '<gen>')`.
- If generation is unclear, leave NULL -- `kb verify` will flag it for review.
```

~45 tokens added.

### T-003: Update investigation-agent.md -- Phase 5 supersession resolution

Insert after the existing `kb verify` / `kb dedup` step:

```markdown
3. **Resolve supersessions** if you identified older findings during Phase 2:
   \`\`\`bash
   ${CLAUDE_PLUGIN_ROOT}/scripts/kb supersede <old_id> <new_id> "<reason>"
   \`\`\`
```

~20 tokens added.

### T-004: Update knowledge-retriever.md -- Generation detection

Insert after existing step 4 ("Run the query") in Query Strategy section:

```markdown
5. **Detect generation context** from the query:
   - If query mentions "M4", "M4 Pro", "M4 Max": add `--gen m4` to search
   - If query mentions "M5": add `--gen m5` to search
   - If query mentions "M1"/"M2"/"M3": add the corresponding `--gen` flag
   - If query asks about "current" or "latest": add `--gen m4` (or m5 when available)
   - If no generation mentioned: do NOT add `--gen` (return all generations)
```

~50 tokens added.

### T-005: Update knowledge-retriever.md -- Generation-aware result formatting

Insert after existing "Full Format" example in Result Formatting section:

```markdown
### Generation-Aware Format:
When `--gen` was used, note it in the response:
[verified] Finding #123 (gpu-silicon, m4): Apple M4 GPU has 10 cores...
When a finding is from a different generation than requested, flag it:
[verified] Finding #14 (gpu-silicon, m1) -- DIFFERENT GEN: M1 specs, not M4
```

~40 tokens added.

### T-006: Update architecture-advisor.md -- Cross-generation consistency

Insert after existing step 4 ("Apply M4/M5 Hardware Constraints") in Cross-Reference section:

```markdown
5. **Check generation consistency**: When citing findings, verify they apply to the
   target hardware generation. Use `kb detail <id>` to check gpu_generation.
   Flag findings from earlier generations that may not apply to the target hardware.
```

~35 tokens added.

## Acceptance Criteria

1. `investigation-agent.md` contains "gpu_generation" string (generation tagging instruction present)
2. `investigation-agent.md` contains "supersession check" or "supersede" reference
3. `investigation-agent.md` Phase 5 includes `kb supersede` command
4. `knowledge-retriever.md` contains "--gen" string (generation detection instruction present)
5. `knowledge-retriever.md` contains "DIFFERENT GEN" in result formatting
6. `architecture-advisor.md` contains "generation consistency" check
7. Total token overhead across all 3 files: <300 tokens (target: ~230)
8. Existing prompt structure and instructions preserved (insertions only, no rewrites)
9. All prompt changes use `${CLAUDE_PLUGIN_ROOT}` prefix for script references

## Technical Notes

- Reference: spec/TECH.md Section 4.1-4.3 (Agent Prompt Changes)
- Reference: spec/UX.md Section 2.1-2.3 (Agent Interaction Design)
- Test: spec/QA.md Section 6 (Agent Prompt Validation) -- tests AP-1, AP-2
- Token overhead: ~105 (investigation) + ~90 (retriever) + ~35 (advisor) = ~230 total
- PM NFR-5 target was <50 tokens per prompt; actual is slightly higher but necessary for complete coverage
- Behavioral validation is manual (golden agent queries) per QA Section 6.2
- Agent prompt files are in `agents/` directory relative to plugin root
