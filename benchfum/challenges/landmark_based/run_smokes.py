"""Run all landmark pairwise-FPS smoke benchmarks.

Usage
-----
python -m benchfum.challenges.landmark_based.run_smokes
python -m benchfum.challenges.landmark_based.run_smokes --seed 7
python -m benchfum.challenges.landmark_based.run_smokes --stop-on-fail
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _default_datasets_root() -> Path:
    # .../geomfum/benchfum/challenges/landmark_based/run_smokes.py
    # -> .../Research/datasets/full_meshes
    return Path(__file__).resolve().parents[5] / "datasets" / "full_meshes"


def _run_one(name: str, dataset: Path, config: Path, seed: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "benchfum.challenges.landmark_based.run",
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
    parser = argparse.ArgumentParser(
        description="Run all landmark pairwise-FPS smoke benchmarks"
    )
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
        Path(__file__).resolve().parents[2] / "configs" / "benchmarks" / "ldmk"
    )
    jobs = [
        (
            "LDMK FAUST",
            datasets_root / "FAUST",
            config_root / "smoke_ldmk_faust_test.json",
        ),
        (
            "LDMK FAUST_r",
            datasets_root / "FAUST_r",
            config_root / "smoke_ldmk_faust_r_test.json",
        ),
        (
            "LDMK SCAPE_r",
            datasets_root / "SCAPE_r",
            config_root / "smoke_ldmk_scape_r_test.json",
        ),
        (
            "LDMK SHREC19_r",
            datasets_root / "SHREC19_r",
            config_root / "smoke_ldmk_shrec19_r_test.json",
        ),
        (
            "LDMK SHREC20",
            datasets_root / "SHREC20",
            config_root / "smoke_ldmk_shrec20_test.json",
        ),
        (
            "LDMK SMAL_r",
            datasets_root / "SMAL_r",
            config_root / "smoke_ldmk_smal_r_test.json",
        ),
        (
            "LDMK TOPKIDS",
            datasets_root / "TOPKIDS",
            config_root / "smoke_ldmk_topkids_test.json",
        ),
        (
            "LDMK TOSCA",
            datasets_root / "TOSCA",
            config_root / "smoke_ldmk_tosca_test.json",
        ),
        (
            "LDMK DT4D intra",
            datasets_root / "DT4D_r",
            config_root / "smoke_ldmk_dt4d_intra_test.json",
        ),
        (
            "LDMK DT4D inter",
            datasets_root / "DT4D_r",
            config_root / "smoke_ldmk_dt4d_inter_test.json",
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
        print("All landmark pairwise-FPS smoke benchmarks passed.")
        return 0

    for name, rc in failures:
        print(f"{name}: FAILED (exit code {rc})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
