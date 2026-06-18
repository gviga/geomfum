r"""Landmark-based method benchmark.

This runner compares matcher methods on landmark-aware shape pairs.
Methods are declared in a benchmark config JSON and instantiated through
``benchfum.build_matcher_from_json``.

Landmark handling supports two modes:

1) ``dataset.landmark_indices`` provided:
    keep legacy behavior (dataset-level landmarks, typically template-driven).
2) ``dataset.landmark_indices`` absent:
    generate landmarks per pair by farthest-point sampling on the source shape,
    then transfer them to target using that pair's ground-truth correspondence.

Usage
-----
Run benchmark declared in config:

    python -m benchfum.challenges.landmark_based.run \
        --dataset datasets/full_meshes/FAUST

Use a custom benchmark config and save results:

    python -m benchfum.challenges.landmark_based.run \
        --dataset datasets/full_meshes/FAUST \
        --config benchfum/configs/benchmarks/ldmk/landmark_faust.json \
        --save results/landmark_faust
"""

import argparse
import os
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchfum import build_matcher_from_json, compare
from benchfum.challenges._common import (
    build_dataset,
    load_config,
    resolve_dataset_dir,
    resolve_path,
)
from geomfum.metric import VertexEuclideanMetric
from geomfum.metric.mesh import ScipyGraphShortestPathMetric
from geomfum.sample import FarthestPointSampler

# ============================================================================
# BENCHMARK CONFIG
# ============================================================================
_DEFAULT_BENCHMARK_CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "benchmarks"
    / "ldmk"
    / "landmark_faust.json"
)


def _to_numpy_int(arr) -> np.ndarray | None:
    if arr is None:
        return None
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.int64).reshape(-1)


def _map_landmarks_from_pair_corr(
    source_corr: np.ndarray,
    target_corr: np.ndarray,
    source_landmarks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if source_corr.size == 0 or target_corr.size == 0 or source_landmarks.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    n_corr = min(len(source_corr), len(target_corr))
    if n_corr == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    source_corr = source_corr[:n_corr]
    target_corr = target_corr[:n_corr]

    src = source_landmarks[source_landmarks >= 0]
    if src.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    src_to_pos = {}
    for pos, src_vertex in enumerate(source_corr):
        src_vertex = int(src_vertex)
        if src_vertex >= 0 and src_vertex not in src_to_pos:
            src_to_pos[src_vertex] = pos

    src_mapped = []
    tgt_mapped = []
    for src_vertex in src:
        pos = src_to_pos.get(int(src_vertex))
        if pos is None:
            continue
        tgt_vertex = int(target_corr[pos])
        if tgt_vertex < 0:
            continue
        src_mapped.append(int(src_vertex))
        tgt_mapped.append(tgt_vertex)

    if not src_mapped:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    return np.asarray(src_mapped, dtype=np.int64), np.asarray(
        tgt_mapped, dtype=np.int64
    )


def _get_pair_corr(pair: dict, key: str):
    corr = pair.get(key)
    if corr is not None:
        return corr
    side = "source" if key == "source_corr" else "target"
    return pair[side].get("corr")


class PairwiseFpsLandmarkDataset:
    """Pair dataset wrapper generating landmarks per pair via FPS + GT transfer."""

    def __init__(self, base_dataset, n_landmarks=20, metric="euclidean"):
        self.base = base_dataset
        self.n_landmarks = int(n_landmarks)
        self.metric = metric
        pairs = getattr(base_dataset, "pairs", None)
        if pairs is not None:
            self.pairs = pairs
        shape_data = getattr(base_dataset, "shape_data", None)
        if shape_data is not None:
            self.shape_data = shape_data

    def __len__(self):
        """Return number of available pairs."""
        return len(self.base)

    def _ensure_metric(self, shape):
        if getattr(shape, "metric", None) is not None:
            return
        if self.metric == "geodesic" and getattr(shape, "is_mesh", False):
            shape.equip_with_metric(ScipyGraphShortestPathMetric)
        else:
            shape.equip_with_metric(VertexEuclideanMetric)

    def __getitem__(self, idx):
        """Return pair ``idx`` with pairwise-generated source/target landmarks."""
        pair = self.base[idx]
        source_shape = pair["source"]["shape"]
        target_shape = pair["target"]["shape"]

        source_corr = _to_numpy_int(_get_pair_corr(pair, "source_corr"))
        target_corr = _to_numpy_int(_get_pair_corr(pair, "target_corr"))

        if source_corr is None or target_corr is None:
            raise ValueError(
                "Pairwise landmark mode requires source/target correspondences in each pair."
            )

        self._ensure_metric(source_shape)
        sampler = FarthestPointSampler(min_n_samples=self.n_landmarks + 1)
        source_pool = np.unique(source_corr[source_corr >= 0]).astype(np.int64)
        if source_pool.size > 0:
            source_landmarks = _to_numpy_int(
                sampler.sample(source_shape, points_pool=source_pool)
            )[1:]
        else:
            source_landmarks = _to_numpy_int(sampler.sample(source_shape))[1:]

        source_lm, target_lm = _map_landmarks_from_pair_corr(
            source_corr=source_corr,
            target_corr=target_corr,
            source_landmarks=source_landmarks,
        )

        if source_lm.size == 0:
            raise ValueError(
                "Could not map sampled source landmarks to target via ground-truth correspondences."
            )

        source_shape.landmark_indices = source_lm
        target_shape.landmark_indices = target_lm

        return pair


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
    config, resolved_config_path = load_config(
        config_path, _DEFAULT_BENCHMARK_CONFIG_PATH
    )
    dataset_dir = resolve_dataset_dir(dataset_dir, config, resolved_config_path)

    print(f"Benchmark : {config.get('_name', 'Landmark-Based')}")
    print(f"Dataset   : {dataset_dir}")
    ds_cfg = config.get("dataset", {})
    ldmk_cfg = config.get("landmarks", {})
    k = ds_cfg.get("k", 200)
    lm = ds_cfg.get("landmark_indices", [])
    pairwise_n_landmarks = ldmk_cfg.get("n_landmarks", ds_cfg.get("_n_landmarks", 20))
    pairwise_metric = ldmk_cfg.get(
        "metric", ds_cfg.get("_landmark_metric", "euclidean")
    )
    if lm:
        print(f"Spectrum  : k={k}  |  Landmarks: {len(lm)} (provided)")
    else:
        print(f"Spectrum  : k={k}  |  Landmarks: pairwise FPS ({pairwise_n_landmarks})")
    if n_pairs:
        print(f"Pairs     : {n_pairs} (random, seed={seed})")
    print()

    # Classical matchers (functional-map optimisation) run on CPU via scipy, so
    # default the shapes to CPU. A config can still request another device.
    device = config.get("dataset", {}).get("device") or "cpu"
    pairs = build_dataset(dataset_dir, config, n_pairs, seed=seed, device=device)
    if not lm:
        pairs = PairwiseFpsLandmarkDataset(
            pairs,
            n_landmarks=pairwise_n_landmarks,
            metric=pairwise_metric,
        )

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
        help="Path to dataset root directory. Overrides dataset.root in config.",
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
