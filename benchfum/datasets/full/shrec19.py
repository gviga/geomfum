"""SHREC 2019 dataset loaders (original and remeshed).

SHREC19 (original)
------------------
Layout::

    dataset_dir/
      obj/          <- mesh files (1.obj, 2.obj, ... or .ply)
      FARMgt_txt/   <- per-pair correspondence files ({i}_{j}.txt)

Each ``{i}_{j}.txt`` has ``n_i`` lines; line k gives the vertex index in
shape j corresponding to vertex k of shape i (0-indexed in the original files).
A pairs list file ``PAIRS_list_SHREC19_connectivity.txt`` is used when present.

SHREC19_r (remeshed)
--------------------
Layout::

    dataset_dir/
      off/          <- remeshed .off files (1.off, 2.off, ...)
      corres/       <- per-pair correspondence files ({i}_{j}.map, text, 1-indexed)
      dist/         <- precomputed geodesic distance .mat files (key "D")

Both classes return direct pair samples and inherit from
``geomfum.dataset.torch.BasePairsDataset`` so benchmark runners can detect
that they already yield source/target pairs.
"""

import os
import warnings

import numpy as np
import scipy
import torch

from benchfum.datasets._utils import list_shapes, load_shape, move_shape_to_device
from geomfum.dataset.torch import BasePairsDataset


class Shrec19Dataset(BasePairsDataset):
    """SHREC 2019 pair dataset with per-pair correspondence files.

    Shapes have different sizes, so correspondences are stored per pair
    (not per shape) and this dataset yields source/target pairs directly.

    Parameters
    ----------
    dataset_dir : str
        Root directory.
    shapes_dir : str
        Subdirectory containing mesh files.  Default ``"obj"``.
    corr_dir : str
        Subdirectory containing per-pair correspondence files.
        Default ``"FARMgt_txt"``.
    corr_ext : str
        Extension of correspondence files.  Default ``".txt"``.
    corr_offset : int
        Value subtracted from loaded indices.  Default 0 (0-indexed files).
    dist_dir : str or None
        Subdirectory containing precomputed ``.mat`` distance files (key "D").
        Pass ``None`` to skip distance loading.  Default ``None``.
    spectral : bool
        Whether to compute the LBO spectral basis for each shape.
    k : int
        Number of eigenvectors.
    device : torch.device or str, optional
    pairs_file : str or None
        Path to a text file listing evaluation pairs (one "i j" per line).
        Defaults to ``PAIRS_list_SHREC19_connectivity.txt`` if it exists.
    """

    def __init__(
        self,
        dataset_dir,
        shapes_dir="obj",
        corr_dir="FARMgt_txt",
        corr_ext=".txt",
        corr_offset=0,
        dist_dir=None,
        spectral=False,
        k=200,
        device=None,
        pairs_file=None,
    ):
        super().__init__(dataset=None, device=device)
        self.dataset_dir = dataset_dir
        self.dist_dir = dist_dir

        shapes_root = os.path.join(dataset_dir, shapes_dir)
        corr_root = os.path.join(dataset_dir, corr_dir)

        # Load all shapes; key by stem (e.g. "1", "2", ...)
        self._shapes = {}
        for fname in list_shapes(shapes_root, extensions={".obj", ".ply", ".off"}):
            base = os.path.splitext(fname)[0]
            self._shapes[base] = load_shape(
                os.path.join(shapes_root, fname), spectral=spectral, k=k
            )

        # Resolve pairs file
        if pairs_file is None:
            candidate = os.path.join(
                dataset_dir, "PAIRS_list_SHREC19_connectivity.txt"
            )
            pairs_file = candidate if os.path.exists(candidate) else None

        if pairs_file is not None:
            self._pairs = []
            with open(pairs_file) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        i, j = parts[0], parts[1]
                        if i in self._shapes and j in self._shapes:
                            self._pairs.append((i, j))
                        else:
                            warnings.warn(
                                f"Pair ({i}, {j}) in pairs file but shapes not found."
                            )
        else:
            # Discover pairs from correspondence filenames: {i}_{j}{corr_ext}
            self._pairs = []
            for fname in sorted(os.listdir(corr_root)):
                if not fname.endswith(corr_ext):
                    continue
                stem = fname[: -len(corr_ext)]
                parts = stem.rsplit("_", 1)
                if (
                    len(parts) == 2
                    and parts[0] in self._shapes
                    and parts[1] in self._shapes
                ):
                    self._pairs.append((parts[0], parts[1]))

        # Cache correspondence arrays
        self._corr_cache = {}
        for i, j in self._pairs:
            corr_path = os.path.join(corr_root, f"{i}_{j}{corr_ext}")
            if not os.path.exists(corr_path):
                warnings.warn(f"Correspondence file not found: {corr_path}; skipping.")
                continue
            corr = np.loadtxt(corr_path, dtype=np.int64) - corr_offset
            self._corr_cache[(i, j)] = corr

        # Cache distance matrices (optional)
        self._dist_cache = {}
        if dist_dir is not None:
            dist_root = os.path.join(dataset_dir, dist_dir)
            for base in self._shapes:
                mat_path = os.path.join(dist_root, base + ".mat")
                if os.path.exists(mat_path):
                    mat = scipy.io.loadmat(mat_path)
                    if "D" in mat:
                        self._dist_cache[base] = mat["D"]

        # Filter pairs to those with a corr file
        self._pairs = [(i, j) for (i, j) in self._pairs if (i, j) in self._corr_cache]

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        """Return a shape pair dict.

        ``source["corr"]`` is the identity map (vertices 0…n_source-1).
        ``target["corr"][k]`` is the vertex in the target shape corresponding
        to vertex k of the source shape.
        """
        i, j = self._pairs[idx]
        shape_i = self._shapes[i]
        shape_j = self._shapes[j]
        corr_ij = self._corr_cache[(i, j)]  # [n_i] → vertex in j

        move_shape_to_device(shape_i, self.device)
        move_shape_to_device(shape_j, self.device)

        # Corr tensors stay on CPU: they are used as indices into
        # shape.vertices (numpy array) in normalized_euclidean_error.
        src = {
            "shape": shape_i,
            "corr": torch.arange(shape_i.n_vertices, dtype=torch.long),
        }
        tgt = {
            "shape": shape_j,
            "corr": torch.tensor(corr_ij, dtype=torch.long),
        }

        if i in self._dist_cache:
            src["dist_matrix"] = torch.tensor(
                self._dist_cache[i], dtype=torch.float32, device=self.device
            )
        if j in self._dist_cache:
            tgt["dist_matrix"] = torch.tensor(
                self._dist_cache[j], dtype=torch.float32, device=self.device
            )

        return {
            "source": src,
            "target": tgt,
            "source_id": i,
            "target_id": j,
            "source_corr": src["corr"],
            "target_corr": tgt["corr"],
            "corr": tgt["corr"],
            "meta": {
                "dataset": type(self).__name__,
                "pair": (i, j),
                "corr_type": "pair_specific",
            },
        }


class Shrec19rDataset(Shrec19Dataset):
    """SHREC 2019 remeshed dataset.

    Same pair-wise correspondence structure as :class:`Shrec19Dataset` but with
    a different directory layout and 1-indexed correspondence files.

    Layout::

        dataset_dir/
          off/      <- remeshed .off files
          corres/   <- per-pair {i}_{j}.map text files (1-indexed)
          dist/     <- precomputed .mat distance matrices (key "D")

    Parameters
    ----------
    dataset_dir : str
        Root directory.
    **kwargs
        Forwarded to :class:`Shrec19Dataset`.  Defaults are set for the
        remeshed layout (``shapes_dir="off"``, ``corr_dir="corres"``,
        ``corr_ext=".map"``, ``corr_offset=1``, ``dist_dir="dist"``).
    """

    def __init__(self, dataset_dir, **kwargs):
        kwargs.setdefault("shapes_dir", "off")
        kwargs.setdefault("corr_dir", "corres")
        kwargs.setdefault("corr_ext", ".map")
        kwargs.setdefault("corr_offset", 1)
        kwargs.setdefault("dist_dir", "dist")
        super().__init__(dataset_dir, **kwargs)
