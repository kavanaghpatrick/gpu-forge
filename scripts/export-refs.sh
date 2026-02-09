#!/bin/bash
# Export all knowledge DB findings as Layer 3 reference files per skill
# Usage: export-refs.sh
# Creates skills/<name>/references/all-findings.md and index.md for each skill

set -euo pipefail

# 3-level DB path resolution (same as kb script):
# 1. GPU_FORGE_DB env var (highest priority)
# 2. ${CLAUDE_PLUGIN_ROOT}/data/gpu_knowledge.db (plugin runtime)
# 3. $(dirname "$0")/../data/gpu_knowledge.db (script-relative fallback)
if [ -n "${GPU_FORGE_DB:-}" ]; then
  DB="$GPU_FORGE_DB"
elif [ -n "${CLAUDE_PLUGIN_ROOT:-}" ]; then
  DB="${CLAUDE_PLUGIN_ROOT}/data/gpu_knowledge.db"
else
  DB="$(dirname "$0")/../data/gpu_knowledge.db"
fi

if [ ! -f "$DB" ]; then
  echo "ERROR: Database not found at $DB" >&2
  exit 1
fi

# Resolve the root of the gpu-forge directory (parent of scripts/)
FORGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKILLS_DIR="${FORGE_ROOT}/skills"

# All 11 skills
SKILLS="gpu-silicon unified-memory metal-compute msl-kernels gpu-io gpu-perf simd-wave mlx-compute metal4-api gpu-distributed gpu-centric-arch"

for skill in $SKILLS; do
  REF_DIR="${SKILLS_DIR}/${skill}/references"
  mkdir -p "$REF_DIR"

  ALL_FILE="${REF_DIR}/all-findings.md"
  INDEX_FILE="${REF_DIR}/index.md"

  # Query all findings for this skill
  count=$(sqlite3 "$DB" "SELECT COUNT(*) FROM findings f JOIN skills s ON s.id = f.skill_id WHERE s.name='${skill}';")

  if [ "$count" -eq 0 ]; then
    echo "# ${skill} — References" > "$ALL_FILE"
    echo "" >> "$ALL_FILE"
    echo "No findings in database." >> "$ALL_FILE"
  else
    echo "# ${skill} — All Findings (${count})" > "$ALL_FILE"
    echo "" >> "$ALL_FILE"

    # Export each finding using tab-separated fields for reliable parsing
    sqlite3 -separator '	' "$DB" \
      "SELECT f.id, f.claim, f.confidence, COALESCE(f.source_title,''), COALESCE(f.evidence,''), COALESCE(f.tags,'')
       FROM findings f
       JOIN skills s ON s.id = f.skill_id
       WHERE s.name='${skill}'
       ORDER BY f.topic, f.id;" | while IFS='	' read -r fid claim confidence source_title evidence tags; do
      echo "## Finding ${fid}: ${claim}" >> "$ALL_FILE"
      echo "**Confidence**: ${confidence}" >> "$ALL_FILE"
      echo "**Source**: ${source_title}" >> "$ALL_FILE"
      echo "**Evidence**: ${evidence}" >> "$ALL_FILE"
      echo "**Tags**: ${tags}" >> "$ALL_FILE"
      echo "" >> "$ALL_FILE"
    done
  fi

  # Create index.md
  echo "# ${skill} — References Index" > "$INDEX_FILE"
  echo "" >> "$INDEX_FILE"
  echo "## Available Reference Files" >> "$INDEX_FILE"
  echo "" >> "$INDEX_FILE"
  echo "- [all-findings.md](all-findings.md) — All ${count} findings from the knowledge database" >> "$INDEX_FILE"
  echo "" >> "$INDEX_FILE"

  echo "  ${skill}: ${count} findings exported"
done

echo ""
echo "Export complete. Reference files written to ${SKILLS_DIR}/*/references/"
