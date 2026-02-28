"""Run all refinement smoke benchmarks.

Usage
-----
python -m benchfum.challenges.refinement.run_smokes
python -m benchfum.challenges.refinement.run_smokes --seed 7
python -m benchfum.challenges.refinement.run_smokes --stop-on-fail
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _default_datasets_root() -> Path:
    # .../geomfum/benchfum/challenges/refinement/run_smokes.py
    # -> .../Research/datasets/full_meshes
    return Path(__file__).resolve().parents[5] / "datasets" / "full_meshes"


def _run_one(name: str, dataset: Path, config: Path, seed: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "benchfum.challenges.refinement.run",
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
    """Run all configured refinement smoke jobs and return process exit code."""
    parser = argparse.ArgumentParser(description="Run all refinement smoke benchmarks")
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
        Path(__file__).resolve().parents[2] / "configs" / "benchmarks" / "smoke_tests"
    )
    jobs = [
        (
            "Refine FAUST",
            datasets_root / "FAUST",
            config_root / "smoke_refine_faust_test.json",
        ),
        (
            "Refine SCAPE_r",
            datasets_root / "SCAPE_r",
            config_root / "smoke_refine_scape_r_test.json",
        ),
        (
            "Refine SHREC19_r",
            datasets_root / "SHREC19_r",
            config_root / "smoke_refine_shrec19_r_test.json",
        ),
        (
            "Refine SHREC20",
            datasets_root / "SHREC20",
            config_root / "smoke_refine_shrec20_test.json",
        ),
        (
            "Refine SMAL_r",
            datasets_root / "SMAL_r",
            config_root / "smoke_refine_smal_r_test.json",
        ),
        (
            "Refine TOPKIDS",
            datasets_root / "TOPKIDS",
            config_root / "smoke_refine_topkids_test.json",
        ),
        (
            "Refine TOSCA",
            datasets_root / "TOSCA",
            config_root / "smoke_refine_tosca_test.json",
        ),
        (
            "Refine DT4D intra",
            datasets_root / "DT4D_r",
            config_root / "smoke_refine_dt4d_intra_test.json",
        ),
        (
            "Refine DT4D inter",
            datasets_root / "DT4D_r",
            config_root / "smoke_refine_dt4d_inter_test.json",
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
        print("All refinement smoke benchmarks passed.")
        return 0

    for name, rc in failures:
        print(f"{name}: FAILED (exit code {rc})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
