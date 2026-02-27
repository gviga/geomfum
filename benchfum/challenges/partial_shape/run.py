r"""Partial shape matching benchmark.

Evaluates deep learning methods on partial shape datasets (SHREC16 or CP2P
format) and reports partial-shape-specific metrics: filtered geodesic error,
overlap IoU, and PCK AUC.

The runner supports both:
- Evaluation from pre-trained checkpoints
- Full train + evaluate loops (when ``trainer_config`` is provided)

Usage
-----
Evaluate from checkpoints (SHREC16):

    python -m benchfum.challenges.partial_shape.run \
        --dataset datasets/shrec16 \
        --config benchfum/configs/benchmarks/shrec16_partial.json

Train and evaluate (EchoMatch on CP2P):

    python -m benchfum.challenges.partial_shape.run \
        --config benchfum/configs/benchmarks/cp2p_partial.json \
        --train \
        --train_dataset datasets/cp2p/train \
        --val_dataset datasets/cp2p/test

"""

import argparse
import os
from pathlib import Path

import torch

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchfum import build_model_from_json, build_trainer_from_json
from benchfum._build import _build_component_registry
from benchfum.challenges._common import load_config, resolve_path, seed_random
from benchfum.datasets.partial import (
    CP2PPairsDataset,
    PartialSmalPairsDataset,
    Shrec16PairsDataset,
)
from geomfum.learning.losses import LossManager
from geomfum.learning.wrappers import TrainedModelWrapper
from geomfum.learning.losses import OverlapIoU, PCKMetric, PartialGeodesicError

# ============================================================================
# DEFAULT CONFIG
# ============================================================================
_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "benchmarks"
    / "shrec16_partial.json"
)


# ============================================================================
# DATASET BUILDER
# ============================================================================

def _build_partial_dataset(dataset_dir, config, device, split_override=None):
    """Build a partial shape dataset from config.

    Config must have a ``dataset.format`` key: ``"shrec16"``, ``"cp2p"``,
    or ``"partialsmal"``.  All other ``dataset.*`` keys are forwarded to the
    dataset constructor, except the housekeeping keys listed below.

    Parameters
    ----------
    split_override : str or None
        When set, overrides the ``split`` key in the dataset config.
        Used so train / val / test splits can be loaded from the same root.
    """
    ds_cfg = dict(config.get("dataset", {}))
    fmt = ds_cfg.pop("format", "shrec16")
    ds_cfg.pop("root", None)
    ds_cfg.pop("train_root", None)
    ds_cfg.pop("val_root", None)
    ds_cfg.pop("train_split", None)
    ds_cfg.pop("val_split", None)

    # Build augmentation if specified
    if isinstance(ds_cfg.get("augmentation"), dict):
        registry = _build_component_registry()
        from benchfum._build import _build_component
        ds_cfg["augmentation"] = _build_component(ds_cfg["augmentation"], registry)

    if fmt == "shrec16":
        return Shrec16PairsDataset(dataset_dir, device=device, **ds_cfg)
    elif fmt == "cp2p":
        return CP2PPairsDataset(dataset_dir, device=device, **ds_cfg)
    elif fmt == "partialsmal":
        if split_override is not None:
            ds_cfg["split"] = split_override
        return PartialSmalPairsDataset(dataset_dir, device=device, **ds_cfg)
    else:
        raise ValueError(
            f"Unknown partial shape dataset format {fmt!r}. "
            "Supported: 'shrec16', 'cp2p', 'partialsmal'."
        )


# ============================================================================
# EVALUATION
# ============================================================================

def _evaluate_model(model, dataset, device, n_pairs=None, seed=None):
    """Evaluate *model* on *dataset* and return a metrics dict.

    Parameters
    ----------
    model : nn.Module or callable
        A matcher (``__call__(shape_a, shape_b) → CorrespondenceResult``).
    dataset : Shrec16PairsDataset or CP2PPairsDataset
    device : str
    n_pairs : int or None
        If given, evaluate on a random subset of this many pairs.
    seed : int or None
        Random seed for pair subsetting.

    Returns
    -------
    dict {metric_name: float}
    """
    import random
    import numpy as np

    if seed is not None:
        seed_random(seed)

    indices = list(range(len(dataset)))
    if n_pairs is not None and n_pairs < len(indices):
        indices = random.sample(indices, n_pairs)

    # Partial-shape evaluation metrics
    geo_metric = PartialGeodesicError()
    iou_metric = OverlapIoU()
    pck_metric = PCKMetric()

    geo_vals, iou_vals, pck_vals = [], [], []

    model_is_nn = hasattr(model, "eval") and hasattr(model, "training")
    if model_is_nn:
        model.eval()

    with torch.no_grad():
        for idx in indices:
            pair = dataset[idx]
            shape_a = pair["source"]["shape"]
            shape_b = pair["target"]["shape"]
            mask_a = pair["source"].get("mask")
            mask_b = pair["target"].get("mask")
            corr_a = pair["source"].get("corr")
            corr_b = pair["target"].get("corr")

            try:
                result = model(shape_a, shape_b)
            except Exception as exc:
                print(f"  [warn] pair {idx} failed: {exc}")
                continue

            p2p21 = result.p2p21
            overlap_ab = result.overlap_ab

            if p2p21 is None:
                continue

            # Move to CPU for numpy-based metrics
            p2p21 = p2p21.cpu() if hasattr(p2p21, "cpu") else torch.tensor(p2p21)

            # --- Geodesic error (requires dist_a) ---
            # dist_a is not pre-loaded in partial datasets; skip if absent.
            # Users can equip shapes with a metric and call dist_matrix().
            dist_a = None
            if hasattr(shape_a, "_dist_matrix"):
                dist_a = shape_a._dist_matrix

            if dist_a is not None and corr_b is not None and mask_a is not None:
                if corr_a is None:
                    corr_a = torch.arange(shape_a.n_vertices, device=corr_b.device)
                geo = geo_metric(p2p21, dist_a, corr_a, corr_b, mask_a)
                geo_vals.append(geo.item())

                pck = pck_metric(p2p21, dist_a, corr_a, corr_b, mask_a)
                pck_vals.append(1.0 - pck.item())  # convert 1-AUC → AUC

            # --- Overlap IoU ---
            if overlap_ab is not None and mask_a is not None:
                iou_loss = iou_metric(overlap_ab, mask_a)
                iou_vals.append(1.0 - iou_loss.item())  # convert 1-IoU → IoU

    results = {}
    if geo_vals:
        results["PartialGeodesicError"] = float(np.mean(geo_vals))
    if iou_vals:
        results["OverlapIoU"] = float(np.mean(iou_vals))
    if pck_vals:
        results["PCK_AUC"] = float(np.mean(pck_vals))

    return results


# ============================================================================
# METHOD LOADING
# ============================================================================

def _maybe_train(method_name, method_cfg, model, config_path, train_pairs, val_pairs):
    trainer_config = method_cfg.get("trainer_config")
    if trainer_config is None:
        print(f"  [skip-train] {method_name}: no trainer_config")
        return

    trainer_path = resolve_path(config_path, trainer_config)
    checkpoint_path = resolve_path(config_path, method_cfg.get("checkpoint"))

    trainer = build_trainer_from_json(
        str(trainer_path), model=model, train_set=train_pairs, val_set=val_pairs
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.checkpoint_path = str(checkpoint_path)

    print(f"  [train] {method_name}: trainer={trainer_path}")
    trainer.train()


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_benchmark(
    dataset_dir=None,
    config_path=None,
    n_pairs=None,
    save_dir=None,
    device=None,
    train=None,
    train_dataset_dir=None,
    val_dataset_dir=None,
    seed=None,
):
    """Run the partial shape matching benchmark.

    Parameters
    ----------
    dataset_dir : str or None
        Root directory of the test dataset.
    config_path : str or None
        Override path to benchmark config JSON.
    n_pairs : int or None
        Limit evaluation to this many pairs.
    save_dir : str or None
        Directory to save per-method metric CSV files.
    device : str or None
        PyTorch device (default: auto-detect).
    train : bool or None
        Train methods with ``trainer_config`` before evaluating.
    train_dataset_dir, val_dataset_dir : str or None
        Dataset roots for training/validation.
    seed : int or None
        Random seed.

    Returns
    -------
    dict {method_name: {metric_name: float}}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config, resolved_config_path = load_config(config_path, _DEFAULT_CONFIG_PATH)

    methods_cfg = config.get("methods")
    if not methods_cfg:
        raise ValueError(
            "Benchmark config must define a 'methods' list. "
            "Each method needs at least {'name', 'model_config'}."
        )

    # Resolve dataset_dir
    if dataset_dir is None:
        ds_cfg = config.get("dataset", {})
        raw_root = ds_cfg.get("root")
        if raw_root:
            dataset_dir = str(resolved_config_path.parent / raw_root)
    if dataset_dir is None:
        raise ValueError("Provide --dataset or set dataset.root in config.")

    if train is None:
        train = bool(config.get("train", False))

    print(f"Benchmark : {config.get('_name', 'Partial Shape Matching')}")
    print(f"Dataset   : {dataset_dir}")
    print(f"Format    : {config.get('dataset', {}).get('format', 'shrec16')}")
    print(f"Device    : {device}")
    if n_pairs:
        print(f"Pairs     : {n_pairs} (random, seed={seed})")
    print()

    # Training datasets
    train_pairs = val_pairs = None
    if train:
        if train_dataset_dir is None:
            ds_cfg = config.get("dataset", {})
            raw = ds_cfg.get("train_root")
            if raw:
                train_dataset_dir = str(resolved_config_path.parent / raw)
        if val_dataset_dir is None:
            ds_cfg = config.get("dataset", {})
            raw = ds_cfg.get("val_root")
            if raw:
                val_dataset_dir = str(resolved_config_path.parent / raw)
        if train_dataset_dir is None or val_dataset_dir is None:
            raise ValueError(
                "Provide --train_dataset and --val_dataset "
                "or set dataset.train_root / dataset.val_root in config."
            )
        ds_cfg = config.get("dataset", {})
        train_split = ds_cfg.get("train_split", "train")
        val_split = ds_cfg.get("val_split", "test")
        train_pairs = _build_partial_dataset(
            train_dataset_dir, config, device, split_override=train_split
        )
        val_pairs = _build_partial_dataset(
            val_dataset_dir, config, device, split_override=val_split
        )

    # Build and (optionally) train methods
    methods = {}
    print("Building methods...")
    for method_cfg in methods_cfg:
        method_name = method_cfg.get("name")
        model_config = method_cfg.get("model_config")
        if not method_name or not model_config:
            raise ValueError(f"Invalid method entry: {method_cfg!r}")

        model_config_path = resolve_path(resolved_config_path, model_config)
        model = build_model_from_json(str(model_config_path), device=device)

        if train:
            _maybe_train(method_name, method_cfg, model, resolved_config_path,
                         train_pairs, val_pairs)

        checkpoint_path = resolve_path(resolved_config_path, method_cfg.get("checkpoint"))
        if checkpoint_path is not None and checkpoint_path.exists():
            print(f"  [load] {method_name}: checkpoint={checkpoint_path}")
            matcher = TrainedModelWrapper(model=model, device=device,
                                          checkpoint_path=str(checkpoint_path))
        else:
            if checkpoint_path is not None:
                print(f"  [warn] {method_name}: checkpoint not found — using untrained model")
            matcher = model

        methods[method_name] = matcher

    print(f"\nMethods   : {list(methods)}")
    print()

    # Build test dataset
    test_dataset = _build_partial_dataset(dataset_dir, config, device)

    # Evaluate
    all_results = {}
    for method_name, matcher in methods.items():
        print(f"Evaluating {method_name} on {len(test_dataset)} pairs...")
        results = _evaluate_model(matcher, test_dataset, device, n_pairs=n_pairs, seed=seed)
        all_results[method_name] = results
        for metric, val in results.items():
            print(f"  {metric}: {val:.4f}")
        print()

    # Summary table
    if len(all_results) > 1:
        print("=" * 60)
        print(f"{'Method':<30}  {'GeoErr':>8}  {'IoU':>8}  {'PCK-AUC':>8}")
        print("-" * 60)
        for method_name, results in all_results.items():
            geo = results.get("PartialGeodesicError", float("nan"))
            iou = results.get("OverlapIoU", float("nan"))
            pck = results.get("PCK_AUC", float("nan"))
            print(f"  {method_name:<28}  {geo:>8.4f}  {iou:>8.4f}  {pck:>8.4f}")
        print("=" * 60)

    if save_dir:
        import json
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "results.json")
        with open(out_path, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nResults saved to: {out_path}")

    return all_results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partial shape matching benchmark")
    parser.add_argument("--dataset", default=None, help="Test dataset root directory.")
    parser.add_argument("--config", default=None, help="Benchmark config JSON path.")
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument("--train", dest="train", action="store_true")
    train_group.add_argument("--no-train", dest="train", action="store_false")
    parser.set_defaults(train=None)
    parser.add_argument("--train_dataset", default=None)
    parser.add_argument("--val_dataset", default=None)
    parser.add_argument("--n_pairs", type=int, default=None)
    parser.add_argument("--save", default=None, help="Directory to save results.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_benchmark(
        dataset_dir=args.dataset,
        config_path=args.config,
        n_pairs=args.n_pairs,
        save_dir=args.save,
        device=args.device,
        train=args.train,
        train_dataset_dir=args.train_dataset,
        val_dataset_dir=args.val_dataset,
        seed=args.seed,
    )
