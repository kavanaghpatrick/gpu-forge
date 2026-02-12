#!/usr/bin/env python3
"""Benchmark regression detection for Criterion.rs output.

Parses Criterion benchmark results and compares against a saved baseline.
Exits with code 1 if any benchmark regressed more than the threshold.

Usage:
    # Check against baseline (Criterion directory format):
    check_bench_regression.py target/criterion --threshold 10

    # Check a standalone JSON file (for CI/testing):
    check_bench_regression.py bench.json --threshold 10

    # Save current results as baseline:
    check_bench_regression.py target/criterion --save-baseline

Criterion directory structure:
    target/criterion/<bench_name>/new/estimates.json
        { "mean": { "point_estimate": <nanoseconds> }, ... }

Standalone JSON format:
    {
        "benchmarks": {
            "<name>": { "mean_ns": <float> },
            ...
        },
        "baseline": {
            "<name>": { "mean_ns": <float> },
            ...
        }
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


BASELINE_FILENAME = ".bench_baseline.json"


def find_baseline_path(bench_path: str) -> Path:
    """Determine where to store/load the baseline file."""
    p = Path(bench_path)
    if p.is_file():
        return p.parent / BASELINE_FILENAME
    return p / BASELINE_FILENAME


def parse_criterion_dir(criterion_dir: str) -> dict[str, float]:
    """Parse Criterion directory structure for benchmark results.

    Looks for: <criterion_dir>/<bench_name>/new/estimates.json
    Each estimates.json contains: { "mean": { "point_estimate": N } }
    where N is in nanoseconds.
    """
    results = {}
    criterion_path = Path(criterion_dir)

    if not criterion_path.is_dir():
        return results

    for bench_dir in sorted(criterion_path.iterdir()):
        if not bench_dir.is_dir():
            continue

        # Skip report directory
        if bench_dir.name == "report":
            continue

        estimates_file = bench_dir / "new" / "estimates.json"
        if estimates_file.exists():
            try:
                with open(estimates_file) as f:
                    data = json.load(f)
                mean_ns = data.get("mean", {}).get("point_estimate")
                if mean_ns is not None:
                    results[bench_dir.name] = float(mean_ns)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"  WARN: Failed to parse {estimates_file}: {e}")
                continue

        # Also check for nested benchmark groups (e.g., group/bench_name)
        for sub_dir in sorted(bench_dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name in ("new", "base", "report"):
                continue
            sub_estimates = sub_dir / "new" / "estimates.json"
            if sub_estimates.exists():
                try:
                    with open(sub_estimates) as f:
                        data = json.load(f)
                    mean_ns = data.get("mean", {}).get("point_estimate")
                    if mean_ns is not None:
                        name = f"{bench_dir.name}/{sub_dir.name}"
                        results[name] = float(mean_ns)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"  WARN: Failed to parse {sub_estimates}: {e}")
                    continue

    return results


def parse_json_file(json_path: str) -> tuple[dict[str, float], Optional[dict[str, float]]]:
    """Parse a standalone JSON benchmark file.

    Returns (current_results, baseline_or_none).

    Supports two formats:
    1. Combined format with benchmarks + baseline keys
    2. Criterion estimates.json format (single benchmark)
    """
    with open(json_path) as f:
        data = json.load(f)

    # Format 1: Combined file with benchmarks and baseline
    if "benchmarks" in data:
        current = {}
        for name, info in data["benchmarks"].items():
            if isinstance(info, dict) and "mean_ns" in info:
                current[name] = float(info["mean_ns"])
            elif isinstance(info, (int, float)):
                current[name] = float(info)

        baseline = None
        if "baseline" in data:
            baseline = {}
            for name, info in data["baseline"].items():
                if isinstance(info, dict) and "mean_ns" in info:
                    baseline[name] = float(info["mean_ns"])
                elif isinstance(info, (int, float)):
                    baseline[name] = float(info)

        return current, baseline

    # Format 2: Criterion estimates.json (single benchmark)
    if "mean" in data:
        mean_ns = data["mean"].get("point_estimate")
        if mean_ns is not None:
            name = Path(json_path).parent.parent.name
            return {name: float(mean_ns)}, None

    print(f"  WARN: Unrecognized JSON format in {json_path}")
    return {}, None


def load_baseline(baseline_path: Path) -> Optional[dict[str, float]]:
    """Load saved baseline from disk."""
    if not baseline_path.exists():
        return None
    try:
        with open(baseline_path) as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  WARN: Failed to load baseline {baseline_path}: {e}")
        return None


def save_baseline(baseline_path: Path, results: dict[str, float]) -> None:
    """Save current results as baseline."""
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"  Baseline saved to {baseline_path} ({len(results)} benchmarks)")


def compare_results(
    current: dict[str, float],
    baseline: dict[str, float],
    threshold: float,
) -> tuple[bool, list[str]]:
    """Compare current results against baseline.

    Returns (all_passed, list_of_result_lines).
    A positive change% means regression (slower), negative means improvement.
    """
    all_passed = True
    lines = []

    all_names = sorted(set(current.keys()) | set(baseline.keys()))

    for name in all_names:
        if name not in current:
            lines.append(f"  SKIP  {name}: not in current results")
            continue
        if name not in baseline:
            lines.append(f"  NEW   {name}: {current[name]:.1f} ns (no baseline)")
            continue

        cur_val = current[name]
        base_val = baseline[name]

        if base_val == 0:
            lines.append(f"  SKIP  {name}: baseline is zero")
            continue

        change_pct = ((cur_val - base_val) / base_val) * 100.0

        if change_pct > threshold:
            status = "FAIL"
            all_passed = False
        elif change_pct < -threshold:
            status = "IMPROVED"
        else:
            status = "PASS"

        lines.append(
            f"  {status:8s} {name}: "
            f"{base_val:.1f} -> {cur_val:.1f} ns "
            f"({change_pct:+.1f}%)"
        )

    return all_passed, lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect benchmark regressions from Criterion output"
    )
    parser.add_argument(
        "bench_path",
        help="Path to Criterion output directory or standalone JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold percentage (default: 10)",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current results as the new baseline",
    )
    args = parser.parse_args()

    bench_path = args.bench_path
    threshold = args.threshold

    print(f"Benchmark Regression Check (threshold: {threshold}%)")
    print(f"  Source: {bench_path}")
    print()

    # Parse current results
    inline_baseline = None
    if os.path.isfile(bench_path):
        current, inline_baseline = parse_json_file(bench_path)
    elif os.path.isdir(bench_path):
        current = parse_criterion_dir(bench_path)
    else:
        print(f"  ERROR: {bench_path} not found")
        return 1

    if not current:
        print("  No benchmark results found.")
        return 1

    print(f"  Found {len(current)} benchmark(s)")

    # Save baseline if requested
    if args.save_baseline:
        baseline_path = find_baseline_path(bench_path)
        save_baseline(baseline_path, current)
        return 0

    # Load baseline for comparison
    baseline = inline_baseline
    if baseline is None:
        baseline_path = find_baseline_path(bench_path)
        baseline = load_baseline(baseline_path)

    if baseline is None:
        print("  No baseline found. Saving current results as baseline.")
        baseline_path = find_baseline_path(bench_path)
        save_baseline(baseline_path, current)
        print("  PASS (first run, baseline established)")
        return 0

    # Compare
    print()
    print("Results:")
    all_passed, lines = compare_results(current, baseline, threshold)

    for line in lines:
        print(line)

    print()
    if all_passed:
        print("PASS: No regressions detected")
        return 0
    else:
        print(f"FAIL: Regression(s) detected (>{threshold}% slower)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
