"""Run all full-shape smoke benchmarks.

Usage
-----
python -m benchfum.challenges.full_shape.run_smokes
python -m benchfum.challenges.full_shape.run_smokes --seed 7
python -m benchfum.challenges.full_shape.run_smokes --stop-on-fail
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _default_datasets_root() -> Path:
    # .../geomfum/benchfum/challenges/full_shape/run_smokes.py
    # -> .../Research/datasets/full_meshes
    return Path(__file__).resolve().parents[5] / "datasets" / "full_meshes"


def _run_one(name: str, dataset: Path, config: Path, seed: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "benchfum.challenges.full_shape.run",
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
    parser = argparse.ArgumentParser(description="Run all full-shape smoke benchmarks")
    parser.add_argument(
        "--datasets-root",
        type=str,
        default=None,
        help="Root folder containing full datasets (FAUST, SCAPE_r, SHREC19_r, ...)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    datasets_root = (
        Path(args.datasets_root) if args.datasets_root else _default_datasets_root()
    )

    config_root = (
        Path(__file__).resolve().parents[2] / "configs" / "benchmarks" / "full"
    )
    jobs = [
        ("FAUST smoke", datasets_root / "FAUST", config_root / "faust_smoke.json"),
        (
            "SCAPE_r smoke",
            datasets_root / "SCAPE_r",
            config_root / "scape_r_smoke.json",
        ),
        (
            "SHREC19_r smoke",
            datasets_root / "SHREC19_r",
            config_root / "shrec19_r_smoke.json",
        ),
        (
            "SHREC20 smoke",
            datasets_root / "SHREC20",
            config_root / "shrec20_smoke.json",
        ),
        ("SMAL_r smoke", datasets_root / "SMAL_r", config_root / "smal_r_smoke.json"),
        (
            "TOPKIDS smoke",
            datasets_root / "TOPKIDS",
            config_root / "topkids_smoke.json",
        ),
        ("TOSCA smoke", datasets_root / "TOSCA", config_root / "tosca_smoke.json"),
        (
            "DT4D intra smoke",
            datasets_root / "DT4D_r",
            config_root / "dt4d_intra_smoke.json",
        ),
        (
            "DT4D inter smoke",
            datasets_root / "DT4D_r",
            config_root / "dt4d_inter_smoke.json",
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
        print("All full-shape smoke benchmarks passed.")
        return 0

    for name, rc in failures:
        print(f"{name}: FAILED (exit code {rc})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
