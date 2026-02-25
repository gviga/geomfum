r"""Landmark-based method benchmark.

This runner compares multiple matcher methods on FAUST landmark pairs.
Methods are declared in a benchmark config JSON and instantiated through
``benchfum.build_matcher_from_json``.

Usage
-----
Run benchmark declared in config:

    python -m benchfum.challenges.landmark_based.run \
        --dataset datasets/faust/train_set

Use a custom benchmark config and save results:

    python -m benchfum.challenges.landmark_based.run \
        --dataset datasets/faust/train_set \
        --config benchfum/configs/challenges/landmark_faust_benchmark.json \
        --save results/landmark_faust
"""

import argparse
import os
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchfum import build_matcher_from_json, compare
from benchfum.challenges._common import (
    build_dataset,
    load_config,
    resolve_dataset_dir,
    resolve_path,
    seed_random,
)

# ============================================================================
# BENCHMARK CONFIG
# ============================================================================
_DEFAULT_BENCHMARK_CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "challenges"
    / "landmark_faust.json"
)


# ============================================================================
# METHODS
# ============================================================================


def load_methods(config: dict, config_path: Path) -> dict:
    """Load all methods declared in the benchmark config.

    Parameters
    ----------
    config : dict
        Benchmark config dict.
    config_path : Path
        Path to benchmark config.

    Returns
    -------
    methods : dict[str, object]
    """
    methods_cfg = config.get("methods")
    if not methods_cfg:
        raise ValueError(
            "Benchmark config must define a 'methods' list. "
            "Each entry needs 'name' and 'matcher_config'."
        )

    methods = {}
    for method_cfg in methods_cfg:
        if "name" not in method_cfg:
            raise ValueError(f"Method entry missing 'name': {method_cfg!r}")
        if "matcher_config" not in method_cfg:
            raise ValueError(
                f"Method entry for {method_cfg['name']!r} is missing 'matcher_config'."
            )

        method_name = method_cfg["name"]
        matcher_path = resolve_path(config_path, method_cfg["matcher_config"])
        methods[method_name] = build_matcher_from_json(str(matcher_path))

    return methods


# ============================================================================
# MAIN RUNNER
# ============================================================================


def run_benchmark(
    dataset_dir=None,
    config_path=None,
    n_pairs: int = None,
    save_dir: str = None,
    seed=None,
):
    """Run the landmark-based benchmark.

    Parameters
    ----------
    dataset_dir : str or None
        Path to the dataset root.
    config_path : str or None
        Path to a benchmark config JSON.
    n_pairs : int or None
        Limit evaluation to this many random pairs (None = all from config).
    save_dir : str or None
        If given, save per-method JSON results here.
    seed : int or None
        Random seed for reproducible pair selection.

    Returns
    -------
    suite : ExperimentSuite
    """
    config, resolved_config_path = load_config(config_path, _DEFAULT_BENCHMARK_CONFIG_PATH)
    dataset_dir = resolve_dataset_dir(
        dataset_dir, config, resolved_config_path, default="datasets/faust/train_set"
    )

    print(f"Benchmark : {config.get('_name', 'Landmark-Based')}")
    print(f"Dataset   : {dataset_dir}")
    ds_cfg = config.get("dataset", {})
    k = ds_cfg.get("k", 200)
    lm = ds_cfg.get("landmark_indices", [])
    print(f"Spectrum  : k={k}  |  Landmarks: {len(lm)}")
    if n_pairs:
        print(f"Pairs     : {n_pairs} (random, seed={seed})")
    print()

    pairs = build_dataset(dataset_dir, config, n_pairs, seed=seed)
    methods = load_methods(config, resolved_config_path)

    print(f"Methods   : {list(methods.keys())}")

    metrics = config.get("metrics", ["geodesic_error"])

    suite = compare(methods, dataset=pairs, metrics=metrics)

    print()
    suite.print_comparison(metrics=metrics)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        suite.save_all(save_dir)
        print(f"\nResults saved to: {save_dir}")

    return suite


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landmark-based method benchmark")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to dataset root directory (must contain shapes/). Overrides dataset.root in config.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override path to benchmark config JSON.",
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=None,
        help="Number of random pairs to evaluate (default: all).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Directory to save per-method JSON results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible pair selection.",
    )
    args = parser.parse_args()

    run_benchmark(
        dataset_dir=args.dataset,
        config_path=args.config,
        n_pairs=args.n_pairs,
        save_dir=args.save,
        seed=args.seed,
    )
