"""CP2P partial-to-partial pairs dataset.

Directory layout::

    dataset_dir/
      shapes/            <- all shapes (.off / .obj / .ply)
      corr/              <- .map files (one per ordered pair x_y.map)

``.map`` file binary format (int32 little-endian):
  ``[size_x, size_y, corr_0, ..., corr_{size_y - 1}, mask_0, ..., mask_{size_x - 1}]``

  where ``corr[j]`` = vertex in X corresponding to vertex j in Y,
  and ``mask[i] = 1`` if vertex i of X is in the overlap with Y.
"""

import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from benchfum.datasets._utils import list_shapes, load_shape, move_shape_to_device


def _load_map_file(map_path):
    """Parse a CP2P ``.map`` file.

    Returns
    -------
    corr_y_to_x : np.ndarray, shape=[size_y]
        For each vertex in Y, the corresponding vertex index in X.
    mask_x : np.ndarray, shape=[size_x]
        Binary mask: 1 if vertex in X is in the overlap with Y.
    """
    data = np.fromfile(map_path, dtype=np.int32)
    size_x = int(data[0])
    size_y = int(data[1])
    corr = data[2 : 2 + size_y].astype(np.int64)
    mask = data[2 + size_y : 2 + size_y + size_x].astype(np.float32)
    return corr, mask


class CP2PPairsDataset(Dataset):
    """Partial-to-partial pairs in CP2P ``.map`` format.

    Parameters
    ----------
    dataset_dir : str
        Root directory with ``shapes/`` and ``corr/`` sub-directories.
    spectral : bool
        Whether to compute the LBO spectral basis.
    k : int
        Number of eigenvectors.
    device : torch.device or str, optional
    pairs_file : str, optional
        Path to a text file listing pairs (one "shape_x shape_y" per line).
        If None, all ``.map`` files in ``corr/`` define the pairs.
    """

    def __init__(
        self,
        dataset_dir,
        spectral=True,
        k=200,
        device=None,
        pairs_file=None,
    ):
        self.dataset_dir = dataset_dir
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        shapes_dir = os.path.join(dataset_dir, "shapes")
        corr_dir = os.path.join(dataset_dir, "corr")

        self._shapes = {}
        for fname in list_shapes(shapes_dir):
            base = os.path.splitext(fname)[0]
            self._shapes[base] = load_shape(
                os.path.join(shapes_dir, fname), spectral=spectral, k=k
            )

        if pairs_file is not None and os.path.exists(pairs_file):
            pairs = []
            with open(pairs_file) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        pairs.append(
                            (os.path.splitext(parts[0])[0], os.path.splitext(parts[1])[0])
                        )
            self._pairs = pairs
        else:
            self._pairs = []
            for fname in sorted(os.listdir(corr_dir)):
                if not fname.endswith(".map"):
                    continue
                stem = fname[:-4]
                parts = stem.rsplit("_", 1)
                if (
                    len(parts) == 2
                    and parts[0] in self._shapes
                    and parts[1] in self._shapes
                ):
                    self._pairs.append((parts[0], parts[1]))
                else:
                    warnings.warn(
                        f"Cannot parse pair from map filename {fname}; skipping."
                    )

        self._map_cache = {}
        for x_base, y_base in self._pairs:
            key = f"{x_base}_{y_base}"
            map_path = os.path.join(corr_dir, key + ".map")
            if not os.path.exists(map_path):
                warnings.warn(f"Map file not found: {map_path}")
                continue
            self._map_cache[key] = _load_map_file(map_path)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        """Return a ``(shape_x, shape_y)`` partial pair with partiality masks."""
        x_base, y_base = self._pairs[idx]
        key = f"{x_base}_{y_base}"
        corr_yx, mask_x = self._map_cache[key]

        shape_x = self._shapes[x_base]
        shape_y = self._shapes[y_base]

        move_shape_to_device(shape_x, self.device)
        move_shape_to_device(shape_y, self.device)

        mask_x_t = torch.tensor(mask_x, dtype=torch.float32, device=self.device)
        mask_y_t = torch.ones(
            shape_y.n_vertices, dtype=torch.float32, device=self.device
        )
        corr_t = torch.tensor(corr_yx, dtype=torch.long, device=self.device)

        return {
            "source": {"shape": shape_x, "mask": mask_x_t, "corr": None},
            "target": {"shape": shape_y, "mask": mask_y_t, "corr": corr_t},
        }
