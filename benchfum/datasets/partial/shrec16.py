"""SHREC16-format full-to-partial pairs dataset.

Directory layout::

    dataset_dir/
      null/off/          <- full reference shapes (.off / .obj / .ply)
      cuts/off/          <- partial query shapes
    cuts/corr/ or cuts/corres/ <- correspondences: one .vts or .txt per partial shape
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

from benchfum.datasets._utils import list_shapes, load_shape, move_shape_to_device
from geomfum.dataset.torch import BasePairsDataset


def _load_corr(corr_path, corr_offset=0):
    """Load a correspondence file (.vts or plain .txt, one index per line)."""
    return np.loadtxt(corr_path, dtype=np.int64) - corr_offset


def _mask_from_corr(n_full, corr):
    """Binary mask for the full shape: mask[i]=1 if vertex i appears in *corr*."""
    mask = np.zeros(n_full, dtype=np.float32)
    mask[corr] = 1.0
    return mask


class Shrec16PairsDataset(BasePairsDataset):
    """Full-to-partial pairs in SHREC16 format.

    Parameters
    ----------
    dataset_dir : str
        Root directory with ``null/off/``, ``cuts/off/``, and
        ``cuts/corr/`` or ``cuts/corres/``
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
    corr_subdir : str or None, optional
        Correspondence subdirectory name inside ``partial_split``.
        If None (default), auto-detects ``corr`` then ``corres``.
    """

    def __init__(
        self,
        dataset_dir,
        partial_split="cuts",
        spectral=True,
        k=200,
        device=None,
        corr_offset=0,
        corr_subdir=None,
    ):
        super().__init__(dataset=None, device=device)
        self.dataset_dir = dataset_dir
        self.partial_split = partial_split
        self.spectral = spectral
        self.k = k
        self.corr_offset = corr_offset

        full_dir = os.path.join(dataset_dir, "null", "off")
        partial_dir = os.path.join(dataset_dir, partial_split, "off")
        split_root = os.path.join(dataset_dir, partial_split)
        if corr_subdir is not None:
            corr_dir = os.path.join(split_root, corr_subdir)
        else:
            corr_dir = None
            for candidate in ("corr", "corres"):
                candidate_dir = os.path.join(split_root, candidate)
                if os.path.isdir(candidate_dir):
                    corr_dir = candidate_dir
                    break
        if corr_dir is None or not os.path.isdir(corr_dir):
            raise FileNotFoundError(
                f"No correspondence directory found under {split_root}. "
                "Expected one of: corr, corres (or pass corr_subdir)."
            )

        # Load full shapes
        self._full_shapes = {}
        for fname in list_shapes(full_dir):
            shape = load_shape(os.path.join(full_dir, fname), spectral=spectral, k=k)
            self._full_shapes[os.path.splitext(fname)[0]] = shape

        # Load partial shapes + correspondences
        self._partial_shapes = {}
        self._corrs = {}
        self._masks_full = {}
        self._full_for_partial = {}
        corr_index = {}
        for corr_name in sorted(os.listdir(corr_dir)):
            stem, ext = os.path.splitext(corr_name)
            if ext.lower() in (".vts", ".txt"):
                corr_index[stem] = os.path.join(corr_dir, corr_name)

        for fname in list_shapes(partial_dir):
            base = os.path.splitext(fname)[0]
            shape = load_shape(
                os.path.join(partial_dir, fname), spectral=spectral, k=k
            )
            self._partial_shapes[base] = shape

            corr_file = None
            stem_candidates = [base]
            split_prefix = f"{partial_split}_"
            if base.startswith(split_prefix):
                stem_candidates.append(base[len(split_prefix):])
            for prefix in ("cuts_", "holes_"):
                if base.startswith(prefix):
                    stem_candidates.append(base[len(prefix):])
            if "_" in base:
                stem_candidates.append(base.split("_", 1)[1])
            for stem in stem_candidates:
                if stem in corr_index:
                    corr_file = corr_index[stem]
                    break
            if corr_file is None:
                warnings.warn(f"No correspondence file found for {fname}; skipping.")
                del self._partial_shapes[base]
                continue

            corr = _load_corr(corr_file, corr_offset=corr_offset)
            self._corrs[base] = corr

            full_base = self._resolve_full_base(base)
            if full_base not in self._full_shapes:
                warnings.warn(
                    f"Cannot find full shape for partial {base}; tried '{full_base}'."
                )
                del self._partial_shapes[base]
                del self._corrs[base]
                continue

            n_full = self._full_shapes[full_base].n_vertices

            # Handle common SHREC16 .vts indexing conventions robustly.
            # If corr appears 1-indexed (max equals n_full), shift to 0-indexed.
            if corr.size > 0 and np.max(corr) == n_full and np.min(corr) >= 1:
                corr = corr - 1
                self._corrs[base] = corr

            if corr.size == 0 or np.min(corr) < 0 or np.max(corr) >= n_full:
                warnings.warn(
                    f"Invalid corr indices for {base}: range=[{int(np.min(corr)) if corr.size else 'empty'}, "
                    f"{int(np.max(corr)) if corr.size else 'empty'}], n_full={n_full}; skipping."
                )
                del self._partial_shapes[base]
                del self._corrs[base]
                continue

            self._masks_full[base] = _mask_from_corr(n_full, corr)
            self._full_for_partial[base] = full_base

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
                for partial_base, full_base in self._full_for_partial.items()
            ]

    def _resolve_full_base(self, partial_base):
        """Resolve full-shape basename from a partial-shape basename."""
        full_keys = set(self._full_shapes.keys())

        candidates = [partial_base]
        if "_" in partial_base:
            candidates.append("_".join(partial_base.split("_")[:-1]))

        # Common SHREC16 patterns: cuts_cat_shape_1 -> cat
        normalized = partial_base
        for prefix in (f"{self.partial_split}_", "cuts_", "holes_"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break

        candidates.append(normalized)
        if "_shape_" in normalized:
            candidates.append(normalized.split("_shape_", 1)[0])
        if "_" in normalized:
            candidates.append(normalized.split("_", 1)[0])

        for cand in candidates:
            if cand in full_keys:
                return cand

        return "_".join(partial_base.split("_")[:-1])

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
            "source_id": full_base,
            "target_id": partial_base,
            "source_corr": None,
            "target_corr": corr_t,
            "corr": corr_t,
            "meta": {
                "dataset": type(self).__name__,
                "corr_type": "partial_target_to_source",
            },
        }
