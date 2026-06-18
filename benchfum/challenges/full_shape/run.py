r"""Full-shape matching benchmark.

Evaluates one or more matcher methods on any full-shape dataset and reports
the geodesic error (and any other configured metrics).  Both classical
(descriptor + functional map) and pre-trained deep methods are supported.

The dataset is loaded via :func:`benchfum.challenges._common.build_dataset`,
which supports:

* **Generic layout** — any dataset whose root has ``shapes/``, ``corr/``,
  ``dist/`` subdirectories (or configurable alternatives via ``shapes_subdir``
  etc.).
* **Named dataset class** — set ``"type"`` in the dataset config to use a
  class from ``benchfum.datasets``, e.g. ``"FaustDataset"``,
  ``"ScapeDataset"``, ``"Shrec19rDataset"`` …

Usage
-----
Run with a benchmark config (classical methods):

    python -m benchfum.challenges.full_shape.run \
        --dataset /path/to/FAUST \
        --config benchfum/configs/benchmarks/full/faust.json

Run with a pre-trained deep model and a custom save directory:

    python -m benchfum.challenges.full_shape.run \
        --config benchfum/configs/benchmarks/full/scape_r.json \
        --save results/scape_r \
        --n_pairs 200 --seed 42

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
# DEFAULT CONFIG
# ============================================================================
_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "benchmarks"
    / "full"
    / "faust.json"
)


# ============================================================================
# METHOD LOADING
# ============================================================================


def load_methods(config: dict, config_path: Path) -> dict:
    """Build all matcher methods declared in the benchmark config.

    Each entry in ``config["methods"]`` must have:
    * ``"name"``           — display name
    * ``"matcher_config"`` — path (relative to config file) to a matcher JSON

    Parameters
    ----------
    config : dict
    config_path : Path

    Returns
    -------
    methods : dict[str, BaseMatcher]
    """
    methods_cfg = config.get("methods")
    if not methods_cfg:
        raise ValueError(
            "Benchmark config must define a non-empty 'methods' list.\n"
            "Each entry needs 'name' and 'matcher_config'."
        )

    methods = {}
    for entry in methods_cfg:
        name = entry.get("name")
        matcher_rel = entry.get("matcher_config")
        if name is None or matcher_rel is None:
            raise ValueError(
                f"Method entry must have 'name' and 'matcher_config': {entry!r}"
            )
        matcher_path = resolve_path(config_path, matcher_rel)
        methods[name] = build_matcher_from_json(str(matcher_path))

    return methods


# ============================================================================
# MAIN RUNNER
# ============================================================================


def run_benchmark(
    dataset_dir=None,
    config_path=None,
    n_pairs=None,
    save_dir=None,
    seed=None,
):
    """Run the full-shape matching benchmark.

    Parameters
    ----------
    dataset_dir : str or None
        Path to dataset root.  Overrides ``dataset.root`` in config.
    config_path : str or None
        Path to benchmark config JSON.
    n_pairs : int or None
        Limit to this many random pairs (None = use config default or all).
    save_dir : str or None
        If given, save per-method JSON results here.
    seed : int or None
        Random seed for reproducible pair selection.

    Returns
    -------
    suite : ExperimentSuite
    """
    config, resolved_config_path = load_config(config_path, _DEFAULT_CONFIG_PATH)
    dataset_dir = resolve_dataset_dir(dataset_dir, config, resolved_config_path)
    if dataset_dir is None:
        raise ValueError(
            "Dataset path required. Pass --dataset or set dataset.root in config."
        )

    seed_random(seed)

    ds_cfg = config.get("dataset", {})
    dataset_name = config.get("_name", ds_cfg.get("type", "Dataset"))
    k = ds_cfg.get("k", 200)

    print(f"Benchmark : {dataset_name}")
    print(f"Dataset   : {dataset_dir}")
    print(f"Spectrum  : k={k}")
    if n_pairs:
        print(f"Pairs     : {n_pairs} (random, seed={seed})")
    print()

    # Classical matchers (functional-map optimisation) run on CPU via scipy, so
    # default the shapes to CPU. A config can still request another device.
    device = config.get("dataset", {}).get("device") or "cpu"
    pairs = build_dataset(dataset_dir, config, n_pairs, seed=seed, device=device)
    methods = load_methods(config, resolved_config_path)

    print(f"Methods   : {list(methods.keys())}")
    metrics = config.get("metrics", ["geodesic_error"])
    print(f"Metrics   : {metrics}")
    print()

    suite = compare(methods, dataset=pairs, metrics=metrics)

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
    parser = argparse.ArgumentParser(
        description="Full-shape matching benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to dataset root directory. Overrides dataset.root in config.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to benchmark config JSON (default: configs/benchmarks/full/faust.json).",
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
