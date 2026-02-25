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
