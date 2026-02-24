"""Landmark-Based Shape Matching Benchmark
==========================================
Reference benchmark for methods that use landmark correspondences.

Driven by ``benchfum/configs/challenges/landmark_faust.json``, which declares:
- The baselines to compare (loaded from ``benchfum/configs/matchers/``)
- Dataset parameters (spectrum size, landmark indices)
- Metrics and evaluation settings

Usage
-----
Run the full FAUST landmark benchmark (all baselines):

    python -m benchfum.challenges.landmark_based.run --dataset /path/to/faust

Limit to 20 random pairs for quick evaluation:

    python -m benchfum.challenges.landmark_based.run \\
        --dataset /path/to/faust --n_pairs 20

Add your own method to the comparison:

    # Edit the MyMethod class below, then:
    python -m benchfum.challenges.landmark_based.run \\
        --dataset /path/to/faust --my_method

Save per-method results to disk:

    python -m benchfum.challenges.landmark_based.run \\
        --dataset /path/to/faust --save results/landmark_faust/

How to Add Your Method
----------------------
Option A — subclass BaseMatcher (full programmatic control):

    class MyMethod(BaseMatcher):
        def __call__(self, shape_a, shape_b, bidirectional=False):
            # shape_a.landmark_indices  →  landmark vertex indices on shape A
            # shape_b.landmark_indices  →  landmark vertex indices on shape B
            # shape_a.basis.vecs        →  Laplacian eigenvectors [n_verts, k]
            # shape_a.basis.vals        →  Laplacian eigenvalues  [k]
            p2p21 = ...  # your algorithm here
            return CorrespondenceResult(fmap12=None, p2p21=p2p21)

Option B — JSON config (easy to share and reproduce):

    # Write my_method.json and run:
    METHODS["MyMethod"] = BaseMatcher.from_json("my_method.json")
"""

import argparse
import json
import os
from pathlib import Path

from benchfum import compare
from benchfum.presets import MatcherPresets
from geomfum.dataset.torch import PairsDataset, ShapeDataset
from geomfum.matcher import BaseMatcher

# ============================================================================
# CHALLENGE CONFIG
# ============================================================================
_CHALLENGE_CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "challenges"
    / "landmark_faust.json"
)


def load_challenge_config(path=None):
    """Load the challenge configuration.

    Parameters
    ----------
    path : str or Path, optional
        Override path to a challenge config JSON. Defaults to
        ``benchfum/configs/challenges/landmark_faust.json``.

    Returns
    -------
    config : dict
    """
    config_path = Path(path) if path is not None else _CHALLENGE_CONFIG_PATH
    with open(config_path) as f:
        return json.load(f)


# ============================================================================
# YOUR METHOD  — edit this section
# ============================================================================


class MyMethod(BaseMatcher):
    """Placeholder for your landmark-based matcher.

    Replace the body of ``__call__`` with your algorithm, then run with
    ``--my_method`` to include it in the comparison.

    Both shapes have ``landmark_indices`` set by the dataset loader.
    """

    def __call__(self, shape_a, shape_b, bidirectional=False):
        # Access landmarks:   shape_a.landmark_indices, shape_b.landmark_indices
        # Access eigenbasis:  shape_a.basis.vecs  [n_vertices, k]
        # Access eigenvalues: shape_a.basis.vals  [k]
        raise NotImplementedError("Implement your method here, then remove this line.")


# ============================================================================
# DATASET
# ============================================================================


def build_dataset(dataset_dir: str, config: dict, n_pairs: int = None):
    """Build a PairsDataset from a challenge config and a dataset path.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root (must contain a ``shapes/`` subdirectory).
    config : dict
        Challenge config dict (from ``load_challenge_config()``).
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
# BASELINES
# ============================================================================


def load_baselines(config: dict) -> dict:
    """Load all baseline matchers declared in the challenge config.

    Parameters
    ----------
    config : dict
        Challenge config dict.

    Returns
    -------
    baselines : dict[str, BaseMatcher]
    """
    baseline_names = config.get("baselines", [])
    return {name: MatcherPresets.build(name) for name in baseline_names}


# ============================================================================
# MAIN RUNNER
# ============================================================================


def run_benchmark(
    dataset_dir: str,
    config_path: str = None,
    n_pairs: int = None,
    save_dir: str = None,
    include_my_method: bool = False,
):
    """Run the landmark-based benchmark.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root.
    config_path : str or None
        Path to a challenge config JSON. Defaults to
        ``benchfum/configs/challenges/landmark_faust.json``.
    n_pairs : int or None
        Limit evaluation to this many random pairs (None = all from config).
    save_dir : str or None
        If given, save per-method JSON results here.
    include_my_method : bool
        Include the MyMethod placeholder in the comparison.

    Returns
    -------
    suite : ExperimentSuite
    """
    config = load_challenge_config(config_path)

    print(f"Challenge : {config.get('_name', 'Landmark-Based')}")
    print(f"Dataset   : {dataset_dir}")
    ds_cfg = config.get("dataset", {})
    k = ds_cfg.get("k", 200)
    lm = ds_cfg.get("landmark_indices", [])
    print(f"Spectrum  : k={k}  |  Landmarks: {len(lm)}")
    print(f"Baselines : {config.get('baselines', [])}")
    if n_pairs:
        print(f"Pairs     : {n_pairs} (random)")
    print()

    pairs = build_dataset(dataset_dir, config, n_pairs)
    methods = load_baselines(config)

    if include_my_method:
        methods["MyMethod"] = MyMethod()

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
    parser = argparse.ArgumentParser(
        description="Landmark-based shape matching benchmark"
    )
    parser.add_argument(
        "--dataset",
        default="datasets/faust/train_set",
        help="Path to dataset root directory (must contain shapes/).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override path to challenge config JSON (default: landmark_faust.json).",
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
        "--my_method",
        action="store_true",
        help="Include MyMethod in the comparison (must be implemented first).",
    )
    args = parser.parse_args()

    run_benchmark(
        dataset_dir=args.dataset,
        config_path=args.config,
        n_pairs=args.n_pairs,
        save_dir=args.save,
        include_my_method=args.my_method,
    )
