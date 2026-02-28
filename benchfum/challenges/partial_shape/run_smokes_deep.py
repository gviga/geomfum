"""Run all partial-shape deep smoke benchmarks.

Usage
-----
python -m benchfum.challenges.partial_shape.run_smokes_deep
python -m benchfum.challenges.partial_shape.run_smokes_deep --seed 7
python -m benchfum.challenges.partial_shape.run_smokes_deep --stop-on-fail
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _default_datasets_root() -> Path:
    # .../geomfum/benchfum/challenges/partial_shape/run_smokes_deep.py
    # -> .../Research/datasets/partial_meshes
    return Path(__file__).resolve().parents[5] / "datasets" / "partial_meshes"


def _run_one(name: str, dataset: Path, config: Path, seed: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "benchfum.challenges.partial_shape.run",
        "--dataset",
        str(dataset),
        "--config",
        str(config),
        "--seed",
        str(seed),
    ]
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    """Run all configured partial deep smoke jobs and return process exit code."""
    parser = argparse.ArgumentParser(
        description="Run all partial-shape deep smoke benchmarks"
    )
    parser.add_argument(
        "--datasets-root",
        type=str,
        default=None,
        help="Root folder containing SHREC16, CP2P24, PARTIALSMAL",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    datasets_root = (
        Path(args.datasets_root) if args.datasets_root else _default_datasets_root()
    )

    config_root = (
        Path(__file__).resolve().parents[2] / "configs" / "benchmarks" / "partial"
    )
    jobs = [
        (
            "SHREC16 deep smoke",
            datasets_root / "SHREC16",
            config_root / "shrec16_partial_deep_smoke.json",
        ),
        (
            "CP2P24 deep smoke",
            datasets_root / "CP2P24" / "test",
            config_root / "cp2p_partial_deep_smoke.json",
        ),
        (
            "PARTIALSMAL deep smoke",
            datasets_root / "PARTIALSMAL",
            config_root / "partialsmal_partial_deep_smoke.json",
        ),
    ]

    failures = []
    for name, dataset, config in jobs:
        if not dataset.exists():
            print(f"\n=== {name} ===")
            print(f"Missing dataset path: {dataset}")
            failures.append((name, 2))
            if args.stop_on_fail:
                break
            continue

        rc = _run_one(name, dataset, config, args.seed)
        if rc != 0:
            failures.append((name, rc))
            if args.stop_on_fail:
                break

    print("\n=== Summary ===")
    if not failures:
        print("All partial deep smoke benchmarks passed.")
        return 0

    for name, rc in failures:
        print(f"{name}: FAILED (exit code {rc})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
