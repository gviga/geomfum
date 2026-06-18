r"""Deep functional-map benchmark.

This runner compares deep learning methods declared in a benchmark config.
Each method can be evaluated from a checkpoint and optionally trained first
from model/training JSON configs.

Usage
-----
Evaluate methods from checkpoints declared in config:

    python -m benchfum.challenges.deep_fmap.run \
        --dataset datasets/faust/test_set

Train methods (that include ``trainer_config``) and then evaluate:

    python -m benchfum.challenges.deep_fmap.run \
        --dataset datasets/faust/test_set \
        --train --train_dataset datasets/faust/train_set --val_dataset datasets/faust/test_set
"""

import argparse
import os
from pathlib import Path

import torch

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchfum import build_model_from_json, build_trainer_from_json, compare
from benchfum.challenges._common import (
    build_dataset,
    load_config,
    resolve_dataset_dir,
    resolve_path,
)
from geomfum.matcher import DeepFMMatcher

# ============================================================================
# BENCHMARK CONFIG
# ============================================================================
_DEFAULT_BENCHMARK_CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "benchmarks"
    / "learning"
    / "deep_fmap_faust.json"
)


# ============================================================================
# METHOD LOADING
# ============================================================================


def _maybe_train_method(
    method_name,
    method_cfg,
    model,
    config_path,
    train_pairs,
    val_pairs,
):
    """Train a method if training is enabled and trainer_config is available."""
    trainer_config = method_cfg.get("trainer_config")
    if trainer_config is None:
        print(f"  [skip-train] {method_name}: no trainer_config in benchmark config")
        return

    trainer_config_path = resolve_path(config_path, trainer_config)
    trainer = build_trainer_from_json(
        str(trainer_config_path),
        model=model,
        train_set=train_pairs,
        val_set=val_pairs,
    )

    checkpoint_path = resolve_path(config_path, method_cfg.get("checkpoint"))
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.checkpoint_path = str(checkpoint_path)

    print(f"  [train] {method_name}: trainer={trainer_config_path}")
    trainer.train()


# ============================================================================
# MAIN RUNNER
# ============================================================================


def run_benchmark(
    dataset_dir=None,
    config_path=None,
    n_pairs: int = None,
    save_dir: str = None,
    device: str = None,
    train: bool | None = None,
    train_dataset_dir: str = None,
    val_dataset_dir: str = None,
    seed=None,
):
    """Run the deep functional-map benchmark.

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
    device : str or None
        PyTorch device (default: "cuda" if available, else "cpu").
    train : bool or None
        If True, train methods that include ``trainer_config`` before evaluation.
        If None, reads from config (defaults to False).
    train_dataset_dir : str or None
        Dataset root for training pairs.
    val_dataset_dir : str or None
        Dataset root for validation pairs.
    seed : int or None
        Random seed for reproducible pair selection.

    Returns
    -------
    suite : ExperimentSuite
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config, resolved_config_path = load_config(
        config_path, _DEFAULT_BENCHMARK_CONFIG_PATH
    )

    methods_cfg = config.get("methods")
    if not methods_cfg:
        raise ValueError(
            "Benchmark config must define a 'methods' list. "
            "Each method needs at least {'name', 'model_config'}."
        )

    dataset_dir = resolve_dataset_dir(dataset_dir, config, resolved_config_path)
    if dataset_dir is None:
        raise ValueError(
            "Dataset root is missing. Provide --dataset or set dataset.root in config."
        )
    if train is None:
        train = bool(config.get("train", False))

    print(f"Benchmark : {config.get('_name', 'Deep Functional Maps')}")
    print(f"Dataset   : {dataset_dir}")
    print(f"Device    : {device}")
    ds_cfg = config.get("dataset", {})
    print(f"Spectrum  : k={ds_cfg.get('k', 200)}")
    print(f"Train     : {'yes' if train else 'no'}")
    if n_pairs:
        print(f"Pairs     : {n_pairs} (random, seed={seed})")
    print()

    train_pairs = None
    val_pairs = None
    if train:
        train_dataset_dir = resolve_dataset_dir(
            train_dataset_dir,
            config,
            resolved_config_path,
            config_key="train_root",
            default=None,
        )
        val_dataset_dir = resolve_dataset_dir(
            val_dataset_dir,
            config,
            resolved_config_path,
            config_key="val_root",
            default=None,
        )
        if train_dataset_dir is None or val_dataset_dir is None:
            raise ValueError(
                "When --train is enabled, provide --train_dataset and --val_dataset "
                "or set dataset.train_root and dataset.val_root in config."
            )
        train_pairs = build_dataset(train_dataset_dir, config, device=device)
        val_pairs = build_dataset(val_dataset_dir, config, device=device)

    methods = {}
    print("Building methods...")
    for method_cfg in methods_cfg:
        method_name = method_cfg.get("name")
        model_config = method_cfg.get("model_config")
        if method_name is None or model_config is None:
            raise ValueError(
                f"Invalid method entry: {method_cfg!r}. "
                "Each entry must have 'name' and 'model_config'."
            )

        model_config_path = resolve_path(resolved_config_path, model_config)
        model = build_model_from_json(str(model_config_path), device=device)

        if train:
            _maybe_train_method(
                method_name=method_name,
                method_cfg=method_cfg,
                model=model,
                config_path=resolved_config_path,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
            )

        checkpoint_path = resolve_path(
            resolved_config_path, method_cfg.get("checkpoint")
        )
        if checkpoint_path is not None:
            if checkpoint_path.exists():
                print(f"  [load] {method_name}: checkpoint={checkpoint_path}")
                # Inference only (n_iters=0): DeepFMMatcher loads the checkpoint
                # and runs the model in eval/no-grad, like the old wrapper.
                methods[method_name] = DeepFMMatcher(
                    model=model,
                    device=device,
                    checkpoint=str(checkpoint_path),
                    n_iters=0,
                )
                continue
            else:
                print(
                    f"  [skip] {method_name}: checkpoint not found at '{checkpoint_path}'"
                )
                if not train:
                    continue

        methods[method_name] = model

    print()

    if not methods:
        print("No methods to compare — check method config and checkpoints.")
        return None

    print(f"Methods   : {list(methods)}")
    print()

    pairs = build_dataset(dataset_dir, config, n_pairs, seed=seed, device=device)

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
    parser = argparse.ArgumentParser(description="Deep functional-map benchmark")
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
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument(
        "--train",
        dest="train",
        action="store_true",
        help="Train methods (with trainer_config) before evaluation.",
    )
    train_group.add_argument(
        "--no-train",
        dest="train",
        action="store_false",
        help="Disable training and evaluate from checkpoints only.",
    )
    parser.set_defaults(train=None)
    parser.add_argument(
        "--train_dataset",
        default=None,
        help="Training dataset root. Overrides dataset.train_root in config.",
    )
    parser.add_argument(
        "--val_dataset",
        default=None,
        help="Validation dataset root. Overrides dataset.val_root in config.",
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
        "--device",
        default=None,
        help="PyTorch device: 'cuda', 'cpu', 'cuda:1', etc. (default: auto).",
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
        device=args.device,
        train=args.train,
        train_dataset_dir=args.train_dataset,
        val_dataset_dir=args.val_dataset,
        seed=args.seed,
    )
