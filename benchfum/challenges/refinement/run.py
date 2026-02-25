r"""Refinement method benchmark.

Measures how much each refinement strategy improves a shared initial
functional map. Methods are declared in a benchmark config JSON and
built through the benchfum JSON factory.

Usage
-----
Run benchmark declared in config:

    python -m benchfum.challenges.refinement.run \\
        --dataset datasets/faust/train_set
"""

import argparse
import json
import os
from pathlib import Path

from benchfum import (
    build_matcher_from_json,
    build_refiner_from_json,
    compare,
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
    / "refinement_faust_benchmark.json"
)


def load_benchmark_config(path=None):
    """Load the benchmark configuration.

    Parameters
    ----------
    path : str or Path, optional
        Override path to a benchmark config JSON. Defaults to
        ``benchfum/configs/challenges/refinement_faust_benchmark.json``.

    Returns
    -------
    config : dict
    """
    config_path = Path(path) if path is not None else _DEFAULT_BENCHMARK_CONFIG_PATH
    with open(config_path) as f:
        return json.load(f)


def load_challenge_config(path=None):
    """Backward-compatible alias for ``load_benchmark_config``."""
    return load_benchmark_config(path)


# ============================================================================
# DATASET
# ============================================================================


def build_dataset(dataset_dir: str, config: dict, n_pairs: int = None):
    """Build a PairsDataset from a benchmark config and a dataset path.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root (must contain a ``shapes/`` subdirectory).
    config : dict
        Benchmark config dict.
    n_pairs : int or None
        Override the number of pairs.

    Returns
    -------
    pairs : PairsDataset
    """
    ds_cfg = config.get("dataset", {})
    k = ds_cfg.get("k", 200)

    if n_pairs is None:
        n_pairs = config.get("n_pairs", None)

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


def _resolve_path(config_path: Path, relative_or_abs: str) -> Path:
    """Resolve relative paths against config file directory."""
    candidate = Path(relative_or_abs)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def _load_methods_cfg(config: dict):
    """Return methods config, with legacy fallback.

    Preferred schema:
      methods: [{name, refiner_config}, ...]
      base_matcher_config: ../matchers/fmap.json
    Legacy fallback:
      base_method: fmap
      refiners: [identity, icp, ...]
    """
    methods_cfg = config.get("methods")
    if methods_cfg:
        return methods_cfg

    refiners = config.get("refiners", [])
    if not refiners:
        raise ValueError(
            "Benchmark config must define 'methods' (preferred) or legacy 'refiners'."
        )

    return [
        {
            "name": refiner_name,
            "refiner_config": f"../refiners/{refiner_name}.json",
        }
        for refiner_name in refiners
    ]


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
    methods_cfg = _load_methods_cfg(config)
    methods = {}

    for method_cfg in methods_cfg:
        if "name" not in method_cfg:
            raise ValueError(f"Method entry missing 'name': {method_cfg!r}")
        if "refiner_config" not in method_cfg:
            raise ValueError(
                f"Method entry for {method_cfg['name']!r} is missing 'refiner_config'."
            )

        method_name = method_cfg["name"]
        refiner_path = _resolve_path(config_path, method_cfg["refiner_config"])
        refiner = build_refiner_from_json(str(refiner_path))
        methods[method_name] = RefinementMatcher(base_matcher, refiner)

    return methods


# ============================================================================
# MAIN RUNNER
# ============================================================================


def run_benchmark(
    dataset_dir: str,
    config_path: str = None,
    n_pairs: int = None,
    save_dir: str = None,
):
    """Run the refinement benchmark.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root.
    config_path : str or None
        Override path to benchmark config JSON.
    n_pairs : int or None
        Limit evaluation to this many random pairs (None = all from config).
    save_dir : str or None
        If given, save per-method JSON results here.

    Returns
    -------
    suite : ExperimentSuite
    """
    resolved_config_path = (
        Path(config_path).resolve()
        if config_path is not None
        else _DEFAULT_BENCHMARK_CONFIG_PATH
    )
    config = load_benchmark_config(resolved_config_path)

    base_matcher_cfg = config.get("base_matcher_config")
    if base_matcher_cfg is None:
        legacy_base_method = config.get("base_method")
        if legacy_base_method is None:
            raise ValueError(
                "Benchmark config must define 'base_matcher_config' "
                "(preferred) or legacy 'base_method'."
            )
        base_matcher_cfg = f"../matchers/{legacy_base_method}.json"

    base_matcher_path = _resolve_path(resolved_config_path, base_matcher_cfg)

    print(f"Benchmark   : {config.get('_name', 'Refinement')}")
    print(f"Dataset     : {dataset_dir}")
    print(f"Base matcher: {base_matcher_path}")
    ds_cfg = config.get("dataset", {})
    print(f"Spectrum    : k={ds_cfg.get('k', 200)}")
    if n_pairs:
        print(f"Pairs       : {n_pairs} (random)")
    print()

    base_matcher = build_matcher_from_json(str(base_matcher_path))
    methods = build_methods(config, resolved_config_path, base_matcher)
    print(f"Methods     : {list(methods.keys())}")

    pairs = build_dataset(dataset_dir, config, n_pairs)

    metrics = config.get("metrics", ["geodesic_error"])
    bidirectional = config.get("bidirectional", False)

    suite = compare(
        methods,
        dataset=pairs,
        metrics=metrics,
        bidirectional=bidirectional,
    )

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
        default="datasets/faust/train_set",
        help="Path to dataset root directory (must contain shapes/).",
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
    args = parser.parse_args()

    run_benchmark(
        dataset_dir=args.dataset,
        config_path=args.config,
        n_pairs=args.n_pairs,
        save_dir=args.save,
    )
