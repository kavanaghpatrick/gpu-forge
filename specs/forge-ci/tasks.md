# Tasks: forge-ci

## Phase 1: Create Files

- [x] 1.1 Create forge-ci.yml workflow
  - **Do**:
    1. Create `.github/workflows/forge-ci.yml` with the exact YAML from design.md
    2. 3 jobs: `checks` (clippy + check + tests), `perf` (main-only, feature-gated), `build` (release + artifact upload)
    3. All jobs: `macos-14`, shared cache key `forge-cargo`, `working-directory: metal-forge-compute`
    4. Global env: `CARGO_TERM_COLOR: always`, `RUSTFLAGS: -D warnings`, `MTL_SHADER_VALIDATION: 1`
  - **Files**: `.github/workflows/forge-ci.yml` (create)
  - **Done when**: File exists with all 3 jobs, correct triggers, env, defaults
  - **Verify**: `grep -c 'runs-on: macos-14' .github/workflows/forge-ci.yml` returns 3; `grep 'MTL_SHADER_VALIDATION' .github/workflows/forge-ci.yml` finds env var; `grep 'working-directory: metal-forge-compute' .github/workflows/forge-ci.yml` finds default
  - **Commit**: `ci(forge): add forge-ci.yml with checks, perf, and build jobs`
  - _Requirements: FR-1, FR-2, FR-3, FR-4_
  - _Design: forge-ci.yml Structure_

- [x] 1.2 Fix test.yml -- remove `|| true` from BATS steps
  - **Do**:
    1. Edit `.github/workflows/test.yml` line 33: `bats tests/unit/ || true` -> `bats tests/unit/`
    2. Edit line 38: `bats tests/integration/ || true` -> `bats tests/integration/`
    3. Edit line 43: `bats tests/performance/ || true` -> `bats tests/performance/`
    4. Do NOT touch line 48 (`bats tests/` -- already correct)
  - **Files**: `.github/workflows/test.yml` (modify)
  - **Done when**: No `|| true` remains in test.yml
  - **Verify**: `grep -c '|| true' .github/workflows/test.yml` returns 0 (exit code 1 = no matches = correct)
  - **Commit**: `fix(ci): remove || true from BATS test steps in test.yml`
  - _Requirements: FR-5.1, FR-5.2, FR-5.3, FR-5.4_
  - _Design: test.yml Changes_

## Phase 2: Verify and Ship

- [ ] 2.1 [VERIFY] Validate both workflow files
  - **Do**:
    1. Verify forge-ci.yml structure:
       - 3 jobs present: `checks`, `perf`, `build`
       - `perf` has `needs: checks` and `if:` gate for main/dispatch
       - `build` has `needs: checks`
       - All 3 jobs use `macos-14`
       - Cache key matches `${{ runner.os }}-forge-cargo-${{ hashFiles('metal-forge-compute/Cargo.lock') }}`
       - `--test-threads=1` present on forge-sort test steps
       - `upload-artifact@v4` present with `forge-bench-macos-arm64`
    2. Verify test.yml: no `|| true` in any BATS step; `bats tests/` step unchanged
    3. YAML syntax: parse both files with `python3 -c "import yaml; yaml.safe_load(open('...'))"` or equivalent
  - **Verify**:
    - `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/forge-ci.yml'))" && echo OK`
    - `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml'))" && echo OK`
    - `grep -c 'needs: checks' .github/workflows/forge-ci.yml` returns 2
    - `grep -c 'test-threads=1' .github/workflows/forge-ci.yml` returns 2
    - `grep -c '|| true' .github/workflows/test.yml` returns 0
  - **Done when**: Both files parse as valid YAML, all structural checks pass
  - **Commit**: None (or `fix(ci): correct YAML issues` if fixes needed)

- [ ] 2.2 Create PR and verify CI
  - **Do**:
    1. Verify on feature branch: `git branch --show-current`
    2. If on default branch, STOP
    3. Push: `git push -u origin <branch>`
    4. Create PR: `gh pr create --title "ci(forge): add forge-ci.yml and fix test.yml" --body "..."`
    5. Wait for CI: `gh pr checks --watch`
  - **Verify**: `gh pr checks` shows all green
  - **Done when**: PR created, CI passes
  - **If CI fails**: Read `gh pr checks`, fix, push, re-verify

- [ ] 2.3 [VERIFY] AC checklist
  - **Do**: Programmatically verify each acceptance criterion:
    - FR-1: triggers, macos-14, working-directory, env vars
    - FR-2: clippy, check, forge-primitives test, forge-sort test with --test-threads=1
    - FR-3: perf job with if-gate, needs checks, release + features perf-test
    - FR-4: build job, needs checks, release build, upload-artifact
    - FR-5: no `|| true` in test.yml
  - **Verify**:
    - `grep 'workflow_dispatch' .github/workflows/forge-ci.yml` (FR-1.1)
    - `grep -c 'macos-14' .github/workflows/forge-ci.yml` returns 3 (FR-1.2)
    - `grep 'working-directory: metal-forge-compute' .github/workflows/forge-ci.yml` (FR-1.3)
    - `grep 'RUSTFLAGS: -D warnings' .github/workflows/forge-ci.yml` (FR-1.4)
    - `grep 'clippy --workspace' .github/workflows/forge-ci.yml` (FR-2.3)
    - `grep 'test -p forge-primitives' .github/workflows/forge-ci.yml` (FR-2.5)
    - `grep 'test -p forge-sort.*test-threads=1' .github/workflows/forge-ci.yml` (FR-2.6)
    - `grep "refs/heads/main" .github/workflows/forge-ci.yml` (FR-3.1)
    - `grep 'features perf-test' .github/workflows/forge-ci.yml` (FR-3.3)
    - `grep 'upload-artifact@v4' .github/workflows/forge-ci.yml` (FR-4.3)
    - `grep -c '|| true' .github/workflows/test.yml` returns 0 (FR-5)
  - **Done when**: All grep checks match expected values
  - **Commit**: None

## Notes

- **No POC phase needed**: Pure config files, not code
- **No tests to write**: CI workflows are tested by running them in CI
- **YAML from design.md**: Task 1.1 is copy-exact from the design; do not improvise
- **test.yml edits**: 3 lines only, preserve everything else
