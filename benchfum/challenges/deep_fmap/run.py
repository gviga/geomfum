"""Deep Functional Maps Benchmark on FAUST
==========================================
Reference benchmark for deep learning-based functional map methods.

Driven by ``benchfum/configs/challenges/deep_fmap_faust.json``, which declares:
- The baseline architectures (FMNet, RobustFMNet) and their model presets
- Dataset parameters
- Metrics and evaluation settings

All methods require pretrained checkpoints. If you don't have baseline
checkpoints yet, train them first:

    python -m benchfum.challenges.deep_fmap.train_baselines \\
        --train_dataset /path/to/faust/train_set \\
        --val_dataset   /path/to/faust/test_set  \\
        --out_dir       checkpoints/deep_fmap/

Usage
-----
Compare your model against pretrained FMNet and RobustFMNet:

    python -m benchfum.challenges.deep_fmap.run \\
        --dataset           /path/to/faust \\
        --fmnet_ckpt        checkpoints/deep_fmap/FMNet.pth \\
        --robust_fmnet_ckpt checkpoints/deep_fmap/RobustFMNet.pth \\
        --my_model_ckpt     checkpoints/my_model.pth

Quick evaluation with only your model (skip baselines you don't have):

    python -m benchfum.challenges.deep_fmap.run \\
        --dataset       /path/to/faust \\
        --my_model_ckpt checkpoints/my_model.pth

Save results to disk:

    python -m benchfum.challenges.deep_fmap.run ... --save results/deep_fmap/

How to Add Your Model
---------------------
Option A — define a custom nn.Module and wrap it:

    import torch.nn as nn
    from geomfum.learning.wrappers import TrainedModelWrapper

    class MyDeepFMNet(nn.Module):
        def forward(self, shape_a, shape_b, bidirectional=False):
            # shape_a.basis.vecs  [n_vertices, k]  — Laplacian eigenvectors
            # shape_a.basis.vals  [k]               — Laplacian eigenvalues
            # return a CorrespondenceResult
            ...

    model = TrainedModelWrapper(MyDeepFMNet(), checkpoint_path="my_model.pth")

Option B — use an existing architecture with a custom configuration:

    from benchfum.learning_presets import ModelPresets
    from geomfum.learning.wrappers import TrainedModelWrapper

    model = ModelPresets.build("fmnet_diffusion_large", device="cuda")
    wrapped = TrainedModelWrapper(model, checkpoint_path="my_model.pth")

Option C — edit the MyDeepModel class below and run with --my_model_ckpt:

    # Edit MyDeepModel.__init__ and forward(), then:
    python -m benchfum.challenges.deep_fmap.run \\
        --dataset /path/to/faust --my_model_ckpt checkpoints/my_model.pth
"""

import argparse
import json
import os
from pathlib import Path

import torch

from benchfum import compare
from benchfum.learning_presets import ModelPresets
from geomfum.dataset.torch import PairsDataset, ShapeDataset
from geomfum.learning.wrappers import TrainedModelWrapper
from geomfum.matcher import CorrespondenceResult

# ============================================================================
# CHALLENGE CONFIG
# ============================================================================
_CHALLENGE_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "challenges" / "deep_fmap_faust.json"
)


def load_challenge_config(path=None):
    """Load the challenge configuration.

    Parameters
    ----------
    path : str or Path, optional
        Override path to a challenge config JSON. Defaults to
        ``benchfum/configs/challenges/deep_fmap_faust.json``.

    Returns
    -------
    config : dict
    """
    config_path = Path(path) if path is not None else _CHALLENGE_CONFIG_PATH
    with open(config_path) as f:
        return json.load(f)


# ============================================================================
# YOUR MODEL  — edit this section
# ============================================================================

class MyDeepModel(torch.nn.Module):
    """Placeholder for your deep functional map model.

    Replace ``__init__`` and ``forward`` with your architecture,
    load a checkpoint with ``--my_model_ckpt``, and it will appear in the
    comparison table alongside FMNet and RobustFMNet.

    The model must accept ``(shape_a, shape_b, bidirectional=False)`` and
    return a ``CorrespondenceResult``.
    """

    def __init__(self):
        super().__init__()
        # Define your layers here, e.g.:
        #   self.feature_net = DiffusionNet(...)
        #   self.fmap_layer  = FMapLayer(...)
        raise NotImplementedError(
            "Define your model architecture in MyDeepModel.__init__(), "
            "then remove this line."
        )

    def forward(self, shape_a, shape_b, bidirectional=False):
        # shape_a.basis.vecs  [n_vertices, k]  — Laplacian eigenvectors
        # shape_a.basis.vals  [k]               — Laplacian eigenvalues
        raise NotImplementedError(
            "Implement your forward pass, then remove this line."
        )


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
        Override the number of pairs to evaluate.

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
# BASELINE LOADING
# ============================================================================

def load_baselines(config: dict, checkpoints: dict, device: str) -> dict:
    """Load all baseline deep models declared in the challenge config.

    Each baseline is built from a model preset and optionally loaded from a
    checkpoint. Models without checkpoints are skipped with a warning.

    Parameters
    ----------
    config : dict
        Challenge config dict.
    checkpoints : dict[str, str]
        Mapping from baseline name to checkpoint path (or None).
    device : str
        Device to load models on.

    Returns
    -------
    baselines : dict[str, TrainedModelWrapper]
    """
    baseline_specs = config.get("baselines", {})
    result = {}

    for name, model_preset in baseline_specs.items():
        ckpt = checkpoints.get(name)
        if ckpt is None:
            print(f"  [skip] {name}: no checkpoint provided (--{_ckpt_flag(name)})")
            continue

        if not Path(ckpt).exists():
            print(f"  [skip] {name}: checkpoint not found at '{ckpt}'")
            continue

        print(f"  [load] {name}: preset={model_preset!r}  ckpt={ckpt}")
        model = ModelPresets.build(model_preset, device=device)
        result[name] = TrainedModelWrapper(model, device=device, checkpoint_path=ckpt)

    return result


def _ckpt_flag(name: str) -> str:
    """Convert a baseline name to the corresponding CLI flag."""
    return name.lower().replace(" ", "_").replace("-", "_") + "_ckpt"


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_benchmark(
    dataset_dir: str,
    checkpoints: dict = None,
    my_model_ckpt: str = None,
    config_path: str = None,
    n_pairs: int = None,
    save_dir: str = None,
    device: str = None,
):
    """Run the deep functional maps benchmark.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root.
    checkpoints : dict[str, str], optional
        Mapping from baseline name to checkpoint path.
        e.g. ``{"FMNet": "ckpts/fmnet.pth", "RobustFMNet": "ckpts/rob.pth"}``
    my_model_ckpt : str, optional
        Path to checkpoint for MyDeepModel. If None, MyDeepModel is excluded.
    config_path : str or None
        Override path to challenge config JSON.
    n_pairs : int or None
        Limit evaluation to this many random pairs (None = all from config).
    save_dir : str or None
        If given, save per-method JSON results here.
    device : str, optional
        PyTorch device (default: "cuda" if available, else "cpu").

    Returns
    -------
    suite : ExperimentSuite
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_challenge_config(config_path)
    if checkpoints is None:
        checkpoints = {}

    print(f"Challenge : {config.get('_name', 'Deep Functional Maps')}")
    print(f"Dataset   : {dataset_dir}")
    print(f"Device    : {device}")
    ds_cfg = config.get("dataset", {})
    print(f"Spectrum  : k={ds_cfg.get('k', 200)}")
    print()

    print("Loading baselines...")
    methods = load_baselines(config, checkpoints, device)

    if my_model_ckpt is not None:
        print(f"  [load] MyModel: ckpt={my_model_ckpt}")
        my_model = MyDeepModel()
        methods["MyModel"] = TrainedModelWrapper(
            my_model, device=device, checkpoint_path=my_model_ckpt
        )
    print()

    if not methods:
        print("No methods to compare — provide at least one checkpoint.")
        return None

    print(f"Methods   : {list(methods)}")
    print()

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
        description="Deep functional maps benchmark on FAUST"
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
        "--fmnet_ckpt",
        default=None,
        help="Path to pretrained FMNet checkpoint (.pth).",
    )
    parser.add_argument(
        "--robust_fmnet_ckpt",
        default=None,
        help="Path to pretrained RobustFMNet checkpoint (.pth).",
    )
    parser.add_argument(
        "--my_model_ckpt",
        default=None,
        help="Path to your model checkpoint (.pth). Requires MyDeepModel to be implemented.",
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
    args = parser.parse_args()

    checkpoints = {
        "FMNet":        args.fmnet_ckpt,
        "RobustFMNet":  args.robust_fmnet_ckpt,
    }
    # Remove entries with no checkpoint so load_baselines can skip them
    checkpoints = {k: v for k, v in checkpoints.items() if v is not None}

    run_benchmark(
        dataset_dir=args.dataset,
        checkpoints=checkpoints,
        my_model_ckpt=args.my_model_ckpt,
        config_path=args.config,
        n_pairs=args.n_pairs,
        save_dir=args.save,
        device=args.device,
    )
