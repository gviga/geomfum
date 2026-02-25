r"""Landmark-based method benchmark.

This runner compares multiple matcher methods on FAUST landmark pairs.
Methods are declared in a benchmark config JSON and instantiated through
``benchfum.build_matcher_from_json``.

Usage
-----
Run benchmark declared in config:

    python -m benchfum.challenges.landmark_based.run \\
        --dataset datasets/faust/train_set

Use a custom benchmark config and save results:

    python -m benchfum.challenges.landmark_based.run \\
        --dataset datasets/faust/train_set \\
        --config benchfum/configs/challenges/landmark_faust_benchmark.json \\
        --save results/landmark_faust
"""

import argparse
import json
import os
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchfum import build_matcher_from_json, compare
from geomfum.dataset.torch import PairsDataset, ShapeDataset

# ============================================================================
# BENCHMARK CONFIG
# ============================================================================
_DEFAULT_BENCHMARK_CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "challenges"
    / "landmark_faust.json"
)


def load_benchmark_config(path=None):
    """Load the benchmark configuration.

    Parameters
    ----------
    path : str or Path, optional
        Override path to a benchmark config JSON. Defaults to
        ``benchfum/configs/challenges/landmark_faust_benchmark.json``.

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
        Benchmark config dict (from ``load_benchmark_config()``).
    n_pairs : int or None
        Override the number of pairs to evaluate.

    Returns
    -------
    pairs : PairsDataset
    """
    ds_cfg = config.get("dataset", {})
    k = ds_cfg.get("k", 200)
    landmark_indices = ds_cfg.get("landmark_indices", None)

    if n_pairs is None:
        n_pairs = config.get("n_pairs", None)

    shape_data = ShapeDataset(
        dataset_dir=dataset_dir,
        shape_type="mesh",
        spectral=True,
        k=k,
        distances=True,
        correspondences=True,
        landmark_indices=landmark_indices,
    )

    if n_pairs is not None:
        pair_mode = "random"
        pairs_ratio = n_pairs / len(shape_data)
    else:
        pair_mode = "all"
        pairs_ratio = 100
    return PairsDataset(shape_data, pair_mode=pair_mode, pairs_ratio=pairs_ratio)


# ============================================================================
# METHODS
# ============================================================================


def _resolve_path(config_path: Path, relative_or_abs: str) -> Path:
    """Resolve relative paths against the config file directory."""
    candidate = Path(relative_or_abs)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def _resolve_dataset_dir(
    dataset_dir: str | None,
    config: dict,
    config_path: Path,
) -> str:
    """Resolve evaluation dataset dir from CLI or config."""
    if dataset_dir is not None:
        return dataset_dir

    ds_cfg = config.get("dataset", {})
    configured_dataset_dir = ds_cfg.get("root")
    if configured_dataset_dir is not None:
        return str(_resolve_path(config_path, configured_dataset_dir))

    return "datasets/faust/train_set"


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
        baselines = config.get("baselines", [])
        if baselines:
            methods_cfg = [
                {
                    "name": baseline_name,
                    "matcher_config": f"../matchers/{baseline_name}.json",
                }
                for baseline_name in baselines
            ]
        else:
            raise ValueError(
                "Benchmark config must define 'methods' (preferred) "
                "or legacy 'baselines'."
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
        matcher_path = _resolve_path(config_path, method_cfg["matcher_config"])
        methods[method_name] = build_matcher_from_json(str(matcher_path))

    return methods


# ============================================================================
# MAIN RUNNER
# ============================================================================


def run_benchmark(
    dataset_dir: str | None = None,
    config_path: str = None,
    n_pairs: int = None,
    save_dir: str = None,
):
    """Run the landmark-based benchmark.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root.
    config_path : str or None
        Path to a benchmark config JSON. Defaults to
        ``benchfum/configs/challenges/landmark_faust_benchmark.json``.
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
    dataset_dir = _resolve_dataset_dir(dataset_dir, config, resolved_config_path)

    print(f"Benchmark : {config.get('_name', 'Landmark-Based')}")
    print(f"Dataset   : {dataset_dir}")
    ds_cfg = config.get("dataset", {})
    k = ds_cfg.get("k", 200)
    lm = ds_cfg.get("landmark_indices", [])
    print(f"Spectrum  : k={k}  |  Landmarks: {len(lm)}")
    if n_pairs:
        print(f"Pairs     : {n_pairs} (random)")
    print()

    pairs = build_dataset(dataset_dir, config, n_pairs)
    methods = load_methods(config, resolved_config_path)

    print(f"Methods   : {list(methods.keys())}")

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
    parser = argparse.ArgumentParser(description="Landmark-based method benchmark")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to dataset root directory (must contain shapes/). Overrides dataset.root in config.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override path to benchmark config JSON (default: landmark_faust_benchmark.json).",
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
