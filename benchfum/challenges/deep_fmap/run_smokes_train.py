"""Run deep-learning training smoke benchmarks.

Usage
-----
python -m benchfum.challenges.deep_fmap.run_smokes_train
python -m benchfum.challenges.deep_fmap.run_smokes_train --seed 7
python -m benchfum.challenges.deep_fmap.run_smokes_train --stop-on-fail
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _default_datasets_root() -> Path:
    # .../geomfum/benchfum/challenges/deep_fmap/run_smokes_train.py
    # -> .../Research/datasets/full_meshes
    return Path(__file__).resolve().parents[5] / "datasets" / "full_meshes"


def _run_one(
    name: str,
    eval_dataset: Path,
    train_dataset: Path,
    val_dataset: Path,
    config: Path,
    seed: int,
    device: str,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "benchfum.challenges.deep_fmap.run",
        "--dataset",
        str(eval_dataset),
        "--train_dataset",
        str(train_dataset),
        "--val_dataset",
        str(val_dataset),
        "--config",
        str(config),
        "--seed",
        str(seed),
        "--device",
        device,
        "--train",
        "--n_pairs",
        "1",
    ]
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    """Run training smoke jobs and return process exit code."""
    parser = argparse.ArgumentParser(
        description="Run deep-learning training smoke benchmarks"
    )
    parser.add_argument(
        "--datasets-root",
        type=str,
        default=None,
        help="Root folder containing full datasets (FAUST, SCAPE_r, SHREC19_r, ...)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
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
            "Learn-train FAUST",
            datasets_root / "FAUST",
            datasets_root / "FAUST",
            datasets_root / "FAUST",
            config_root / "smoke_learn_faust_train_test.json",
        ),
        (
            "Learn-train SCAPE_r",
            datasets_root / "SCAPE_r",
            datasets_root / "SCAPE_r",
            datasets_root / "SCAPE_r",
            config_root / "smoke_learn_scape_r_train_test.json",
        ),
        (
            "Learn-train SHREC19_r",
            datasets_root / "SHREC19_r",
            datasets_root / "SHREC19_r",
            datasets_root / "SHREC19_r",
            config_root / "smoke_learn_shrec19_r_train_test.json",
        ),
        (
            "Learn-train SHREC20",
            datasets_root / "SHREC20",
            datasets_root / "SHREC20",
            datasets_root / "SHREC20",
            config_root / "smoke_learn_shrec20_train_test.json",
        ),
        (
            "Learn-train SMAL_r",
            datasets_root / "SMAL_r",
            datasets_root / "SMAL_r",
            datasets_root / "SMAL_r",
            config_root / "smoke_learn_smal_r_train_test.json",
        ),
        (
            "Learn-train TOPKIDS",
            datasets_root / "TOPKIDS",
            datasets_root / "TOPKIDS",
            datasets_root / "TOPKIDS",
            config_root / "smoke_learn_topkids_train_test.json",
        ),
        (
            "Learn-train TOSCA",
            datasets_root / "TOSCA",
            datasets_root / "TOSCA",
            datasets_root / "TOSCA",
            config_root / "smoke_learn_tosca_train_test.json",
        ),
        (
            "Learn-train DT4D intra",
            datasets_root / "DT4D_r",
            datasets_root / "DT4D_r",
            datasets_root / "DT4D_r",
            config_root / "smoke_learn_dt4d_intra_train_test.json",
        ),
        (
            "Learn-train DT4D inter",
            datasets_root / "DT4D_r",
            datasets_root / "DT4D_r",
            datasets_root / "DT4D_r",
            config_root / "smoke_learn_dt4d_inter_train_test.json",
        ),
    ]

    failures = []
    for name, eval_dataset, train_dataset, val_dataset, config in jobs:
        missing = [
            p for p in (eval_dataset, train_dataset, val_dataset) if not p.exists()
        ]
        if missing:
            print(f"\n=== {name} ===")
            for path in missing:
                print(f"Missing dataset path: {path}")
            failures.append((name, 2))
            if args.stop_on_fail:
                break
            continue

        rc = _run_one(
            name,
            eval_dataset,
            train_dataset,
            val_dataset,
            config,
            args.seed,
            args.device,
        )
        if rc != 0:
            failures.append((name, rc))
            if args.stop_on_fail:
                break

    print("\n=== Summary ===")
    if not failures:
        print("All deep-learning training smoke benchmarks passed.")
        return 0

    for name, rc in failures:
        print(f"{name}: FAILED (exit code {rc})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
