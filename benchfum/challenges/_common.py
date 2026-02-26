"""Shared utilities for benchfum challenge runners."""

import json
from pathlib import Path


def load_config(path, default_path: Path):
    """Load a benchmark config JSON and return (config_dict, resolved_path).

    Parameters
    ----------
    path : str or Path or None
        Override path. If None, ``default_path`` is used.
    default_path : Path
        Path used when ``path`` is None.

    Returns
    -------
    config : dict
    resolved_path : Path
        Absolute path of the file that was loaded.
    """
    resolved = Path(path).resolve() if path is not None else default_path
    with open(resolved) as f:
        return json.load(f), resolved


def resolve_path(config_path: Path, relative_or_abs) -> Path | None:
    """Resolve a path relative to the directory of ``config_path``.

    Parameters
    ----------
    config_path : Path
        Absolute path of the benchmark config file.
    relative_or_abs : str or None
        The path to resolve.  Absolute paths are returned as-is.
        None is returned as None.

    Returns
    -------
    Path or None
    """
    if relative_or_abs is None:
        return None
    candidate = Path(relative_or_abs)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def resolve_dataset_dir(
    dataset_dir,
    config: dict,
    config_path: Path,
    config_key: str = "root",
    default=None,
):
    """Resolve a dataset directory with priority: CLI arg > config > default.

    Parameters
    ----------
    dataset_dir : str or None
        Value passed via CLI (highest priority).
    config : dict
        Benchmark config dict.
    config_path : Path
        Absolute path of the benchmark config file (for relative path resolution).
    config_key : str
        Key within ``config["dataset"]`` that holds the path.
    default : str or None
        Fallback when nothing else resolves.

    Returns
    -------
    str or None
    """
    if dataset_dir is not None:
        return dataset_dir
    ds_cfg = config.get("dataset", {})
    configured = ds_cfg.get(config_key)
    if configured is not None:
        return str(resolve_path(config_path, configured))
    return default


# Keys in config["dataset"] consumed by the runner infrastructure (path resolution).
# Everything else is forwarded verbatim to ShapeDataset.
_DATASET_RUNNER_KEYS = {"root", "train_root", "val_root"}

# Sensible defaults for ShapeDataset when not specified in config or CLI.
# Any of these can be overridden by adding the key to config["dataset"].
_SHAPE_DATASET_DEFAULTS = {
    "shape_type": "mesh",
    "spectral": True,
    "k": 200,
    "distances": True,
    "correspondences": True,
}


def build_dataset(dataset_dir, config, n_pairs=None, seed=None, **extra_kwargs):
    """Build a PairsDataset from a benchmark config and a dataset path.

    ``ShapeDataset`` kwargs are resolved with this priority (highest wins):

        CLI / call-site kwargs  >  config["dataset"]  >  _SHAPE_DATASET_DEFAULTS

    This means adding or changing any ``ShapeDataset`` parameter only requires
    editing the JSON config — no Python changes needed.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root (must contain a ``shapes/`` subdirectory).
    config : dict
        Benchmark config dict.
    n_pairs : int or None
        Override the number of pairs. Falls back to ``config["n_pairs"]``.
    seed : int or None
        Random seed for reproducible pair selection.
    **extra_kwargs
        Override or supplement ``ShapeDataset`` kwargs at call time
        (e.g. ``device`` from the CLI).

    Returns
    -------
    pairs : PairsDataset
    """
    from geomfum.dataset.torch import PairsDataset, ShapeDataset

    ds_cfg = config.get("dataset", {})
    shape_kwargs = {
        **_SHAPE_DATASET_DEFAULTS,
        **{k: v for k, v in ds_cfg.items() if k not in _DATASET_RUNNER_KEYS},
        **extra_kwargs,
    }

    if n_pairs is None:
        n_pairs = config.get("n_pairs", None)

    seed_random(seed)

    # If augmentation is specified as a JSON sub-object, build it via the registry.
    if isinstance(shape_kwargs.get("augmentation"), dict):
        from benchfum._build import _build_component, _build_component_registry

        shape_kwargs["augmentation"] = _build_component(
            shape_kwargs["augmentation"], _build_component_registry()
        )

    shape_data = ShapeDataset(dataset_dir=dataset_dir, **shape_kwargs)

    if n_pairs is not None:
        pair_mode = "random"
        pairs_ratio = n_pairs / len(shape_data)
    else:
        pair_mode = "all"
        pairs_ratio = 100
    return PairsDataset(
        shape_data,
        pair_mode=pair_mode,
        pairs_ratio=pairs_ratio,
        device=shape_kwargs.get("device"),
    )


def seed_random(seed) -> None:
    """Seed numpy (and torch if available) for reproducible pair selection.

    Parameters
    ----------
    seed : int or None
        Seed value.  No-op when None.
    """
    if seed is None:
        return
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
