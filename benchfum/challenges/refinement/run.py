r"""Refinement method benchmark.

Measures how much each refinement strategy improves a shared initial
functional map. Methods are declared in a benchmark config JSON and
built through the benchfum JSON factory.

Usage
-----
Run benchmark declared in config:

    python -m benchfum.challenges.refinement.run \
        --dataset datasets/faust/train_set
"""

import argparse
import os
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchfum import (
    build_matcher_from_json,
    build_refiner_from_json,
    compare,
)
from benchfum.challenges._common import (
    load_config,
    resolve_dataset_dir,
    resolve_path,
    seed_random,
)
from benchfum.refinement import RefinementMatcher
from geomfum.dataset.torch import PairsDataset, ShapeDataset

# ============================================================================
# BENCHMARK CONFIG
# ============================================================================
_DEFAULT_BENCHMARK_CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "challenges"
    / "refinement_faust.json"
)


# ============================================================================
# DATASET
# ============================================================================


def build_dataset(dataset_dir: str, config: dict, n_pairs: int = None, seed=None):
    """Build a PairsDataset from a benchmark config and a dataset path.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root (must contain a ``shapes/`` subdirectory).
    config : dict
        Benchmark config dict.
    n_pairs : int or None
        Override the number of pairs.
    seed : int or None
        Random seed for reproducible pair selection.

    Returns
    -------
    pairs : PairsDataset
    """
    ds_cfg = config.get("dataset", {})
    k = ds_cfg.get("k", 200)

    if n_pairs is None:
        n_pairs = config.get("n_pairs", None)

    seed_random(seed)

    shape_data = ShapeDataset(
        dataset_dir=dataset_dir,
        shape_type="mesh",
        spectral=True,
        k=k,
        distances=True,
        correspondences=True,
    )

    if n_pairs is not None:
        pair_mode = "random"
        pairs_ratio = n_pairs / len(shape_data)
    else:
        pair_mode = "all"
        pairs_ratio = 100
    return PairsDataset(shape_data, pair_mode=pair_mode, pairs_ratio=pairs_ratio)


# ============================================================================
# METHOD ASSEMBLY
# ============================================================================


def build_methods(config: dict, config_path: Path, base_matcher) -> dict:
    """Build one RefinementMatcher per refiner declared in the config.

    Parameters
    ----------
    config : dict
        Benchmark config.
    config_path : Path
        Path to benchmark config.
    base_matcher : BaseMatcher
        Shared base matcher used by all methods.

    Returns
    -------
    methods : dict[str, RefinementMatcher]
    """
    methods_cfg = config.get("methods")
    if not methods_cfg:
        raise ValueError(
            "Benchmark config must define a 'methods' list. "
            "Each entry needs 'name' and 'refiner_config'."
        )

    methods = {}
    for method_cfg in methods_cfg:
        if "name" not in method_cfg:
            raise ValueError(f"Method entry missing 'name': {method_cfg!r}")
        if "refiner_config" not in method_cfg:
            raise ValueError(
                f"Method entry for {method_cfg['name']!r} is missing 'refiner_config'."
            )

        method_name = method_cfg["name"]
        refiner_path = resolve_path(config_path, method_cfg["refiner_config"])
        refiner = build_refiner_from_json(str(refiner_path))
        methods[method_name] = RefinementMatcher(base_matcher, refiner)

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
    """Run the refinement benchmark.

    Parameters
    ----------
    dataset_dir : str or None
        Path to the dataset root.
    config_path : str or None
        Override path to benchmark config JSON.
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

    base_matcher_cfg = config.get("base_matcher_config")
    if base_matcher_cfg is None:
        raise ValueError(
            "Benchmark config must define 'base_matcher_config' "
            "(path to a matcher JSON relative to this config file)."
        )

    base_matcher_path = resolve_path(resolved_config_path, base_matcher_cfg)

    print(f"Benchmark   : {config.get('_name', 'Refinement')}")
    print(f"Dataset     : {dataset_dir}")
    print(f"Base matcher: {base_matcher_path}")
    ds_cfg = config.get("dataset", {})
    print(f"Spectrum    : k={ds_cfg.get('k', 200)}")
    if n_pairs:
        print(f"Pairs       : {n_pairs} (random, seed={seed})")
    print()

    base_matcher = build_matcher_from_json(str(base_matcher_path))
    methods = build_methods(config, resolved_config_path, base_matcher)
    print(f"Methods     : {list(methods.keys())}")

    pairs = build_dataset(dataset_dir, config, n_pairs, seed=seed)

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
    parser = argparse.ArgumentParser(description="Refinement method benchmark")
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
