"""PARTIALSMAL partial-to-partial pairs dataset.

Directory layout::

    dataset_dir/
      train/
        shapes/          <- partial shapes (.off)
        maps/            <- .vts files named ``{shape_x}_{shape_y}.vts``
      test/
        shapes/
        maps/

``.vts`` text format: one integer per line.  Line j gives the 1-indexed vertex
in shape X corresponding to vertex j of shape Y; ``-1`` means no match.
Shape names may contain underscores, so pair name parsing tries all possible
split positions against the known shape name set.
"""

import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from benchfum.datasets._utils import list_shapes, load_shape, move_shape_to_device


def _load_vts_file(vts_path):
    """Parse a PARTIALSMAL ``.vts`` correspondence file.

    The file ``{X}_{Y}.vts`` has ``n_x`` lines.  Line ``j`` contains the
    1-indexed vertex in shape **Y** corresponding to vertex ``j`` of shape
    **X**, or ``-1`` for no match.

    Returns
    -------
    corr_x_to_y : np.ndarray, shape=[n_valid]
        0-indexed vertex indices in Y for valid correspondences.
    valid_x : np.ndarray, shape=[n_valid]
        Indices of vertices in X that have a valid correspondence.
    """
    raw = np.loadtxt(vts_path, dtype=np.int64)
    valid_mask = raw > 0
    valid_x = np.where(valid_mask)[0]
    corr_x_to_y = raw[valid_mask] - 1
    return corr_x_to_y, valid_x


def _parse_pair_name(stem, known_names):
    """Split ``stem`` into ``(name_x, name_y)`` by trying all ``_`` positions."""
    parts = stem.split("_")
    for i in range(1, len(parts)):
        prefix = "_".join(parts[:i])
        suffix = "_".join(parts[i:])
        if prefix in known_names and suffix in known_names:
            return prefix, suffix
    return None


class PartialSmalPairsDataset(Dataset):
    """Partial-to-partial pairs from the PARTIALSMAL dataset (``.vts`` format).

    Parameters
    ----------
    dataset_dir : str
        Root PARTIALSMAL directory containing ``train/`` and ``test/``.
    split : str, optional
        ``"train"`` or ``"test"``.  Default ``"train"``.
    spectral : bool
        Whether to compute the LBO spectral basis.
    k : int
        Number of eigenvectors.
    device : torch.device or str, optional
        Target device for tensor data.
    """

    def __init__(
        self,
        dataset_dir,
        split="train",
        spectral=True,
        k=200,
        device=None,
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        split_dir = os.path.join(dataset_dir, split)
        shapes_dir = os.path.join(split_dir, "shapes")
        maps_dir = os.path.join(split_dir, "maps")

        self._shapes = {}
        for fname in list_shapes(shapes_dir):
            base = os.path.splitext(fname)[0]
            self._shapes[base] = load_shape(
                os.path.join(shapes_dir, fname), spectral=spectral, k=k
            )

        known_names = set(self._shapes.keys())

        self._pairs = []
        self._vts_cache = {}

        for fname in sorted(os.listdir(maps_dir)):
            if not fname.endswith(".vts"):
                continue
            stem = fname[:-4]
            parsed = _parse_pair_name(stem, known_names)
            if parsed is None:
                warnings.warn(f"Cannot parse pair from .vts filename {fname}; skipping.")
                continue
            x_base, y_base = parsed
            corr_x_to_y, valid_x = _load_vts_file(os.path.join(maps_dir, fname))
            self._pairs.append((x_base, y_base))
            self._vts_cache[(x_base, y_base)] = (corr_x_to_y, valid_x)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        """Return a ``(shape_x, shape_y)`` partial pair with masks and correspondences."""
        x_base, y_base = self._pairs[idx]
        corr_x_to_y, valid_x = self._vts_cache[(x_base, y_base)]

        shape_x = self._shapes[x_base]
        shape_y = self._shapes[y_base]

        move_shape_to_device(shape_x, self.device)
        move_shape_to_device(shape_y, self.device)

        mask_x = np.zeros(shape_x.n_vertices, dtype=np.float32)
        mask_x[valid_x] = 1.0

        mask_y = np.zeros(shape_y.n_vertices, dtype=np.float32)
        mask_y[corr_x_to_y] = 1.0

        return {
            "source": {
                "shape": shape_x,
                "mask": torch.tensor(mask_x, dtype=torch.float32, device=self.device),
                "corr": torch.tensor(valid_x, dtype=torch.long, device=self.device),
            },
            "target": {
                "shape": shape_y,
                "mask": torch.tensor(mask_y, dtype=torch.float32, device=self.device),
                "corr": torch.tensor(corr_x_to_y, dtype=torch.long, device=self.device),
            },
        }
