"""SHREC16-format full-to-partial pairs dataset.

Directory layout::

    dataset_dir/
      null/off/          <- full reference shapes (.off / .obj / .ply)
      cuts/off/          <- partial query shapes
      cuts/corr/         <- correspondences: one .vts or .txt per partial shape
                           each line i contains the index in the full shape
                           that vertex i of the partial shape corresponds to.

Partiality mask for the full shape is derived from the correspondences:
``mask_full[corr_partial] = 1``.  The partial shape gets ``mask = ones``.

Pairs are listed in ``cuts/pairs.txt`` (one "full_name partial_name" per line).
If the file does not exist, all (full, partial) cross-product pairs are used.
"""

import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from benchfum.datasets._utils import list_shapes, load_shape, move_shape_to_device


def _load_corr(corr_path, corr_offset=0):
    """Load a correspondence file (.vts or plain .txt, one index per line)."""
    return np.loadtxt(corr_path, dtype=np.int64) - corr_offset


def _mask_from_corr(n_full, corr):
    """Binary mask for the full shape: mask[i]=1 if vertex i appears in *corr*."""
    mask = np.zeros(n_full, dtype=np.float32)
    mask[corr] = 1.0
    return mask


class Shrec16PairsDataset(Dataset):
    """Full-to-partial pairs in SHREC16 format.

    Parameters
    ----------
    dataset_dir : str
        Root directory with ``null/off/``, ``cuts/off/``, ``cuts/corr/``
        sub-directories (or ``holes/`` instead of ``cuts/``).
    partial_split : str, optional
        Name of the partial split sub-directory.  Default is ``"cuts"``.
    spectral : bool
        Whether to compute the LBO spectral basis for each shape.
    k : int
        Number of eigenvectors for the spectral basis.
    device : torch.device or str, optional
        Target device for tensor data.
    corr_offset : int, optional
        Subtract this value from loaded correspondence indices (1 for 1-indexed).
    """

    def __init__(
        self,
        dataset_dir,
        partial_split="cuts",
        spectral=True,
        k=200,
        device=None,
        corr_offset=0,
    ):
        self.dataset_dir = dataset_dir
        self.partial_split = partial_split
        self.spectral = spectral
        self.k = k
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.corr_offset = corr_offset

        full_dir = os.path.join(dataset_dir, "null", "off")
        partial_dir = os.path.join(dataset_dir, partial_split, "off")
        corr_dir = os.path.join(dataset_dir, partial_split, "corr")

        # Load full shapes
        self._full_shapes = {}
        for fname in list_shapes(full_dir):
            shape = load_shape(os.path.join(full_dir, fname), spectral=spectral, k=k)
            self._full_shapes[os.path.splitext(fname)[0]] = shape

        # Load partial shapes + correspondences
        self._partial_shapes = {}
        self._corrs = {}
        self._masks_full = {}

        for fname in list_shapes(partial_dir):
            base = os.path.splitext(fname)[0]
            shape = load_shape(
                os.path.join(partial_dir, fname), spectral=spectral, k=k
            )
            self._partial_shapes[base] = shape

            corr_file = None
            for ext in (".vts", ".txt"):
                candidate = os.path.join(corr_dir, base + ext)
                if os.path.exists(candidate):
                    corr_file = candidate
                    break
            if corr_file is None:
                warnings.warn(f"No correspondence file found for {fname}; skipping.")
                del self._partial_shapes[base]
                continue

            corr = _load_corr(corr_file, corr_offset=corr_offset)
            self._corrs[base] = corr

            full_base = base
            if full_base not in self._full_shapes:
                full_base = "_".join(base.split("_")[:-1])
            if full_base not in self._full_shapes:
                warnings.warn(
                    f"Cannot find full shape for partial {base}; tried '{full_base}'."
                )
                del self._partial_shapes[base]
                del self._corrs[base]
                continue

            n_full = self._full_shapes[full_base].n_vertices
            self._masks_full[base] = _mask_from_corr(n_full, corr)

        pairs_file = os.path.join(dataset_dir, partial_split, "pairs.txt")
        if os.path.exists(pairs_file):
            self._pairs = []
            with open(pairs_file) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self._pairs.append(
                            (
                                os.path.splitext(parts[0])[0],
                                os.path.splitext(parts[1])[0],
                            )
                        )
        else:
            self._pairs = [
                (full_base, partial_base)
                for partial_base in self._partial_shapes
                for full_base in self._full_shapes
                if partial_base.startswith(full_base)
                or "_".join(partial_base.split("_")[:-1]) == full_base
            ]

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        """Return a ``(full_shape, partial_shape)`` pair with partiality masks."""
        full_base, partial_base = self._pairs[idx]

        full_shape = self._full_shapes[full_base]
        partial_shape = self._partial_shapes[partial_base]
        corr = self._corrs[partial_base]
        mask_full = self._masks_full[partial_base]

        move_shape_to_device(full_shape, self.device)
        move_shape_to_device(partial_shape, self.device)

        mask_full_t = torch.tensor(mask_full, dtype=torch.float32, device=self.device)
        mask_partial_t = torch.ones(
            partial_shape.n_vertices, dtype=torch.float32, device=self.device
        )
        corr_t = torch.tensor(corr, dtype=torch.long, device=self.device)

        return {
            "source": {"shape": full_shape, "mask": mask_full_t, "corr": None},
            "target": {"shape": partial_shape, "mask": mask_partial_t, "corr": corr_t},
        }
