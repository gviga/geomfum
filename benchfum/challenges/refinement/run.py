"""Refinement Benchmark on FAUST
==================================
Measures how much each refinement strategy improves a fixed initial
functional map.  Because all methods start from the same base correspondence,
any difference in the final score is **solely due to the refinement step**.

Driven by ``benchfum/configs/challenges/refinement_faust.json``, which declares:
- The base method that produces the initial functional map
- The set of refiners to compare (from ``benchfum/configs/refiners/``)
- Dataset parameters and metrics

Usage
-----
Run all standard refiners:

    python -m benchfum.challenges.refinement.run \\
        --dataset /path/to/faust/train_set

Quick check with 20 pairs:

    python -m benchfum.challenges.refinement.run \\
        --dataset /path/to/faust/train_set --n_pairs 20

Include your own refiner:

    python -m benchfum.challenges.refinement.run \\
        --dataset /path/to/faust/train_set --my_refiner

Save results:

    python -m benchfum.challenges.refinement.run \\
        --dataset /path/to/faust/train_set --save results/refinement_faust/

How to Add Your Refiner
------------------------
Option A — implement the refiner interface directly and edit this file:

    class MyRefiner:
        def __call__(self, fmap12, basis_a, basis_b):
            # fmap12   : array [k_b, k_a]  — initial functional map
            # basis_a  : spectral basis of shape A
            # basis_b  : spectral basis of shape B
            # return   : refined fmap12 [k_b, k_a]
            ...

    Then set --my_refiner to include it in the comparison.

Option B — define it as a JSON config and add it to configs/refiners/:

    // benchfum/configs/refiners/my_refiner.json
    {"_name": "MyRefiner", "type": "IcpRefiner", "nit": 30}

    It's then accessible via RefinementPresets.build("my_refiner").

Option C — use it directly in Python:

    from benchfum import compare, MatcherPresets
    from benchfum.refinement import RefinementMatcher, RefinementPresets

    base = MatcherPresets.build("fmap")
    results = compare(
        {
            "No refinement": RefinementMatcher(base, RefinementPresets.build("identity")),
            "ICP+ZO":        RefinementMatcher(base, RefinementPresets.build("icp_zoomout")),
            "MyRefiner":     RefinementMatcher(base, MyRefiner()),
        },
        dataset=pairs,
        metrics=["geodesic_error"],
    )
    results.print_comparison()
"""

import argparse
import json
import os
from pathlib import Path

from benchfum import compare
from benchfum.presets import MatcherPresets
from benchfum.refinement import RefinementMatcher, RefinementPresets
from geomfum.dataset.torch import PairsDataset, ShapeDataset

# ============================================================================
# CHALLENGE CONFIG
# ============================================================================
_CHALLENGE_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "challenges" / "refinement_faust.json"
)


def load_challenge_config(path=None):
    """Load the challenge configuration.

    Parameters
    ----------
    path : str or Path, optional
        Override path to a challenge config JSON. Defaults to
        ``benchfum/configs/challenges/refinement_faust.json``.

    Returns
    -------
    config : dict
    """
    config_path = Path(path) if path is not None else _CHALLENGE_CONFIG_PATH
    with open(config_path) as f:
        return json.load(f)


# ============================================================================
# YOUR REFINER  — edit this section
# ============================================================================

class MyRefiner:
    """Placeholder for your refinement method.

    Implement ``__call__`` with your algorithm and run with ``--my_refiner``
    to include it in the comparison.

    Your refiner receives a functional map and the spectral bases of both
    shapes.  It must return a refined functional map of the same shape.
    """

    def __call__(self, fmap12, basis_a, basis_b):
        # fmap12       : array [k_b, k_a]  — initial functional map from A→B
        # basis_a.vecs : array [n_a, k_a]  — Laplacian eigenvectors of shape A
        # basis_a.vals : array [k_a]       — Laplacian eigenvalues of shape A
        # basis_b.vecs : array [n_b, k_b]  — Laplacian eigenvectors of shape B
        # basis_b.vals : array [k_b]       — Laplacian eigenvalues of shape B
        #
        # return a refined fmap12 of shape [k_b, k_a]
        raise NotImplementedError("Implement your refiner here, then remove this line.")


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
        Challenge config dict.
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

def build_methods(config: dict, base_matcher, include_my_refiner: bool) -> dict:
    """Build one RefinementMatcher per refiner declared in the config.

    Parameters
    ----------
    config : dict
        Challenge config.
    base_matcher : BaseMatcher
        Shared base matcher used by all methods.
    include_my_refiner : bool
        If True, also include the MyRefiner placeholder.

    Returns
    -------
    methods : dict[str, RefinementMatcher]
    """
    refiner_names = config.get("refiners", [])
    methods = {}

    for name in refiner_names:
        refiner = RefinementPresets.build(name)
        # Use the _name from describe() as the display name if available
        desc = RefinementPresets.describe(name)
        display = desc.get("_name", name) if isinstance(desc, dict) else name
        methods[display] = RefinementMatcher(base_matcher, refiner)

    if include_my_refiner:
        methods["MyRefiner"] = RefinementMatcher(base_matcher, MyRefiner())

    return methods


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_benchmark(
    dataset_dir: str,
    config_path: str = None,
    n_pairs: int = None,
    save_dir: str = None,
    include_my_refiner: bool = False,
):
    """Run the refinement benchmark.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root.
    config_path : str or None
        Override path to challenge config JSON.
    n_pairs : int or None
        Limit evaluation to this many random pairs (None = all from config).
    save_dir : str or None
        If given, save per-method JSON results here.
    include_my_refiner : bool
        Include the MyRefiner placeholder in the comparison.

    Returns
    -------
    suite : ExperimentSuite
    """
    config = load_challenge_config(config_path)
    base_name = config.get("base_method", "fmap")

    print(f"Challenge   : {config.get('_name', 'Refinement')}")
    print(f"Dataset     : {dataset_dir}")
    print(f"Base method : {base_name}")
    print(f"Refiners    : {config.get('refiners', [])}")
    ds_cfg = config.get("dataset", {})
    print(f"Spectrum    : k={ds_cfg.get('k', 200)}")
    if n_pairs:
        print(f"Pairs       : {n_pairs} (random)")
    print()

    base_matcher = MatcherPresets.build(base_name)
    methods = build_methods(config, base_matcher, include_my_refiner)

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
    parser = argparse.ArgumentParser(
        description="Refinement benchmark on FAUST"
    )
    parser.add_argument(
        "--dataset",
        default="datasets/faust/train_set",
        help="Path to dataset root directory (must contain shapes/).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override path to challenge config JSON.",
    )
    parser.add_argument(
        "--base_method",
        default=None,
        help="Override the base method (any name from MatcherPresets).",
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
        "--my_refiner",
        action="store_true",
        help="Include MyRefiner in the comparison (must be implemented first).",
    )
    args = parser.parse_args()

    # Allow CLI override of the base method
    if args.base_method is not None:
        cfg = load_challenge_config(args.config)
        cfg["base_method"] = args.base_method
        import tempfile, json as _json
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        _json.dump(cfg, tmp)
        tmp.close()
        config_path = tmp.name
    else:
        config_path = args.config

    run_benchmark(
        dataset_dir=args.dataset,
        config_path=config_path,
        n_pairs=args.n_pairs,
        save_dir=args.save,
        include_my_refiner=args.my_refiner,
    )
