"""Reference-based pairs dataset base class.

Some benchmark datasets (e.g. TOPKIDS, SHREC20) are evaluated by fixing one
"reference" shape as the source and pairing it against every other shape.
Each target shape ships a ``.vts`` file that maps *target vertices* to the
reference vertices, giving the ground-truth point-to-point correspondence.

The resulting pair format is compatible with
:func:`geomfum.evaluation.EuclideanErrorMetric`:

    source["corr"][k]  = vertex in reference matching target vertex k
    target["corr"]     = identity (0 … n_target − 1)
"""

import os

import numpy as np
import torch

from benchfum.datasets._utils import load_shape, move_shape_to_device
from geomfum.dataset.torch import BasePairsDataset


class ReferencePairsDataset(BasePairsDataset):
    """Pairs dataset with a fixed reference (source) shape.

    Each item is a ``(source=ref, target=shape_i)`` pair where
    ``source["corr"]`` gives the ground-truth target→source correspondence
    loaded from the per-target ``.vts`` file.

    Parameters
    ----------
    dataset_dir : str
        Root directory.
    shapes_dir : str
        Subdirectory containing mesh files.
    corr_dir : str
        Subdirectory containing per-target correspondence files.
    ref_name : str
        Base name of the reference shape (without extension).
    corr_suffix : str
        Suffix to strip from corr file stems to recover shape names.
        E.g. ``"_ref"`` turns ``"kid01_ref.vts"`` → ``"kid01"``.
        Empty string for files named directly after the shape.
    corr_ext : str
        Extension of correspondence files.  Default ``".vts"``.
    corr_offset : int
        Value subtracted from loaded indices.  Default 1 (1-indexed files).
    spectral : bool
    k : int
    device : torch.device or str, optional
    distances : bool
        Whether to load the reference shape's geodesic distance matrix.
        Looks for ``dist/{ref_name}.mat`` (key ``"D"``).
    ref_file_ext : str or None
        Extension used to load the *reference* shape.  If ``None``, the
        loader tries ``.off``, ``.ply``, ``.obj`` in that order.
    target_file_exts : tuple of str or None
        Extensions used to load *target* shapes.  Default ``(".off",)``.
    """

    def __init__(
        self,
        dataset_dir,
        shapes_dir,
        corr_dir,
        ref_name,
        corr_suffix="_ref",
        corr_ext=".vts",
        corr_offset=1,
        spectral=False,
        k=200,
        device=None,
        distances=False,
        ref_file_ext=None,
        target_file_exts=(".off",),
    ):
        super().__init__(dataset=None, device=device)
        self.dataset_dir = dataset_dir

        off_root = os.path.join(dataset_dir, shapes_dir)
        corr_root = os.path.join(dataset_dir, corr_dir)

        # --- Load reference shape ---
        if ref_file_ext is not None:
            ref_path = os.path.join(off_root, ref_name + ref_file_ext)
        else:
            ref_path = None
            for ext in (".off", ".ply", ".obj"):
                candidate = os.path.join(off_root, ref_name + ext)
                if os.path.exists(candidate):
                    ref_path = candidate
                    break
        if ref_path is None or not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"Reference shape '{ref_name}' not found in {off_root}"
            )
        self._ref_shape = load_shape(ref_path, spectral=spectral, k=k)
        self._ref_name = ref_name

        # --- Load target shapes ---
        _tgt_exts = tuple(target_file_exts) if target_file_exts else (".off",)
        tgt_shapes = {}
        for fname in sorted(os.listdir(off_root)):
            if not fname.lower().endswith(_tgt_exts):
                continue
            base = os.path.splitext(fname)[0]
            if base == ref_name:
                continue  # skip reference shape
            tgt_shapes[base] = load_shape(
                os.path.join(off_root, fname), spectral=spectral, k=k
            )

        # --- Load corr files and build pair list ---
        self._pairs = []  # list of (tgt_name, corr_array)
        for fname in sorted(os.listdir(corr_root)):
            if not fname.endswith(corr_ext):
                continue
            stem = fname[: -len(corr_ext)]  # strip extension
            # strip trailing suffix (e.g. "_ref") to get the shape name
            if corr_suffix and stem.endswith(corr_suffix):
                tgt_name = stem[: -len(corr_suffix)]
            else:
                tgt_name = stem
            if tgt_name not in tgt_shapes:
                continue
            corr = np.loadtxt(
                os.path.join(corr_root, fname), dtype=np.int64
            ) - corr_offset
            self._pairs.append((tgt_name, tgt_shapes[tgt_name], corr))

        # --- Optionally load reference distance matrix ---
        self._dist_ref = None
        if distances:
            dist_root = os.path.join(dataset_dir, "dist")
            if os.path.isdir(dist_root):
                import scipy.io
                dist_path = os.path.join(dist_root, ref_name + ".mat")
                if os.path.exists(dist_path):
                    mat = scipy.io.loadmat(dist_path)
                    if "D" in mat:
                        self._dist_ref = mat["D"]

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        """Return a pair dict.

        ``source["corr"][k]``   = vertex in reference (source) matching
                                   vertex k of the target shape.
        ``target["corr"]``      = identity (0 … n_target − 1).
        """
        tgt_name, shape_tgt, corr = self._pairs[idx]

        move_shape_to_device(self._ref_shape, self.device)
        move_shape_to_device(shape_tgt, self.device)

        n_tgt = shape_tgt.n_vertices

        # Corr tensors stay on CPU: they are used as indices into
        # shape_a.vertices (numpy array) in normalized_euclidean_error.
        # corr_b uses len(corr) not n_tgt: handles both dense (len==n_tgt)
        # and sparse landmark correspondences (len < n_tgt).
        n_corr = len(corr)
        src_dict = {
            "shape": self._ref_shape,
            "corr": torch.tensor(corr, dtype=torch.long),
            "name": self._ref_name,
        }
        tgt_dict = {
            "shape": shape_tgt,
            "corr": torch.arange(n_corr, dtype=torch.long),
            "name": tgt_name,
        }

        if self._dist_ref is not None:
            src_dict["dist_matrix"] = torch.tensor(
                self._dist_ref, dtype=torch.float32, device=self.device
            )

        return {
            "source": src_dict,
            "target": tgt_dict,
            "source_id": self._ref_name,
            "target_id": tgt_name,
            "source_corr": src_dict["corr"],
            "target_corr": tgt_dict["corr"],
            "corr": None,
            "meta": {
                "dataset": type(self).__name__,
                "pair_type": "reference_target",
            },
        }
