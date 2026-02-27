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

from benchfum.datasets._utils import list_shapes, load_shape, move_shape_to_device
from geomfum.dataset.torch import BasePairsDataset


def _load_map_file(map_path, size_x=None, size_y=None):
    """Parse a CP2P ``.map`` file.

    Returns
    -------
    corr_y_to_x : np.ndarray, shape=[size_y]
        For each vertex in Y, the corresponding vertex index in X.
    mask_x : np.ndarray, shape=[size_x]
        Binary mask: 1 if vertex in X is in the overlap with Y.
    """
    # Try binary CP2P layout first
    data = np.fromfile(map_path, dtype=np.int32)
    if data.size >= 2:
        bx = int(data[0])
        by = int(data[1])
        expected = 2 + by + bx
        if bx >= 0 and by >= 0 and expected <= data.size:
            corr = data[2 : 2 + by].astype(np.int64)
            mask = data[2 + by : 2 + by + bx].astype(np.float32)
            return corr, mask

    # Fallback: text .map layout used by some CP2P24 variants
    txt = np.loadtxt(map_path, dtype=np.int64)
    txt = np.atleast_1d(txt)
    if size_y is None:
        raise ValueError(
            f"Cannot parse text .map file {map_path} without target shape size."
        )
    corr = txt[:size_y].astype(np.int64)
    if size_x is not None and txt.size >= size_y + size_x:
        mask = txt[size_y : size_y + size_x].astype(np.float32)
    elif size_x is not None:
        mask = np.zeros(size_x, dtype=np.float32)
        valid = corr[(corr >= 0) & (corr < size_x)]
        mask[valid] = 1.0
    else:
        mask = np.array([], dtype=np.float32)
    return corr, mask


class CP2PPairsDataset(BasePairsDataset):
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
        super().__init__(dataset=None, device=device)
        self.dataset_dir = dataset_dir

        shapes_dir = None
        for candidate in ("shapes", "off"):
            path = os.path.join(dataset_dir, candidate)
            if os.path.isdir(path):
                shapes_dir = path
                break
        if shapes_dir is None:
            raise FileNotFoundError(
                f"No shapes directory found under {dataset_dir}. Expected one of: shapes, off"
            )

        corr_dir = None
        for candidate in ("corr", "maps"):
            path = os.path.join(dataset_dir, candidate)
            if os.path.isdir(path):
                corr_dir = path
                break
        if corr_dir is None:
            raise FileNotFoundError(
                f"No correspondence directory found under {dataset_dir}. Expected one of: corr, maps"
            )

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
            self._map_cache[key] = map_path

        self._pairs = [(x, y) for (x, y) in self._pairs if f"{x}_{y}" in self._map_cache]

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        """Return a ``(shape_x, shape_y)`` partial pair with partiality masks."""
        x_base, y_base = self._pairs[idx]
        key = f"{x_base}_{y_base}"
        shape_x = self._shapes[x_base]
        shape_y = self._shapes[y_base]

        map_path = self._map_cache[key]
        corr_yx, mask_x = _load_map_file(
            map_path,
            size_x=shape_x.n_vertices,
            size_y=shape_y.n_vertices,
        )

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
            "source_id": x_base,
            "target_id": y_base,
            "source_corr": None,
            "target_corr": corr_t,
            "corr": corr_t,
            "meta": {
                "dataset": type(self).__name__,
                "corr_type": "partial_target_to_source",
            },
        }
