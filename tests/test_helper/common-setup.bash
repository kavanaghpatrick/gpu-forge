#!/usr/bin/env bash

# Resolve plugin root
PLUGIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PLUGIN_ROOT
export CLAUDE_PLUGIN_ROOT="$PLUGIN_ROOT"

# Load BATS libraries (dual Homebrew/submodule pattern)
if [ -d "/opt/homebrew/lib/bats-support" ]; then
  load "/opt/homebrew/lib/bats-support/load"
  load "/opt/homebrew/lib/bats-assert/load"
elif [ -d "${PLUGIN_ROOT}/tests/test_helper/bats-support" ]; then
  load "${PLUGIN_ROOT}/tests/test_helper/bats-support/load"
  load "${PLUGIN_ROOT}/tests/test_helper/bats-assert/load"
fi
