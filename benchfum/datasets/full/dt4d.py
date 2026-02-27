"""DT4D remeshed dataset loaders.

Layout::

    dataset_dir/
      off/
        {category}/   <- .off files per character
        ...
      corres/
        {category}/   <- .vts files (n_ref entries; corr[r] = shape vertex for ref r)
        cross_category_corres/
          {cat1}_{cat2}.vts   <- reference-to-reference correspondences
      train.txt       <- "{category}/{name}" per line (training split)
      test.txt        <- "{category}/{name}" per line (test split)

The per-shape ``.vts`` files use **reference→shape** convention: the file has
``n_ref`` entries where ``corr[r]`` is the shape vertex corresponding to
category reference vertex ``r`` (1-indexed, subtract 1 for 0-indexed).

:class:`DT4DPairsDataset` builds within-category evaluation pairs.  For a
pair (shape_i, shape_j):

    source["corr"][r]  = shape_i vertex for reference vertex r
    target["corr"][r]  = shape_j vertex for reference vertex r

Both arrays have length ``n_ref``, giving ``n_ref`` GT correspondence pairs
compatible with :func:`geomfum.eval.normalized_euclidean_error`.

Cross-category correspondences (``cross_category_corres/``) are loaded on
demand and exposed via :meth:`DT4DDataset.cross_category_corr`.
"""

import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from benchfum.datasets._utils import list_shapes, load_shape, move_shape_to_device

_DEFAULT_IGNORED = frozenset(["pumpkinhulk"])


class DT4DDataset(Dataset):
    """DT4D remeshed dataset (character motion sequences).

    Each item exposes ``corr`` mapping its vertices to the shared category
    reference. ``correspondences = True`` lets the benchmark runner detect
    that corr data is present.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing ``off/`` and ``corres/`` subdirectories.
    categories : list of str, optional
        If provided, only load shapes from these categories.
    ignored_categories : list of str, optional
        Categories to skip.  Defaults to ``["pumpkinhulk"]`` (inconsistent
        topology, as in the URRSM paper).
    spectral : bool
        Whether to compute LBO spectral basis for each shape.
    k : int
        Number of eigenvectors.
    device : torch.device or str, optional
        Target device for tensor data.
    """

    def __init__(
        self,
        dataset_dir,
        categories=None,
        ignored_categories=None,
        spectral=False,
        k=200,
        device=None,
    ):
        self.dataset_dir = dataset_dir
        self.spectral = spectral
        self.k = k
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if ignored_categories is None:
            ignored_categories = _DEFAULT_IGNORED

        off_root = os.path.join(dataset_dir, "off")
        corres_root = os.path.join(dataset_dir, "corres")

        available_cats = sorted(
            d for d in os.listdir(off_root)
            if os.path.isdir(os.path.join(off_root, d))
            and d not in ignored_categories
        )
        if categories is not None:
            available_cats = [c for c in available_cats if c in categories]

        self.categories = available_cats

        self.correspondences = True  # items expose "corr" data

        self._shapes = {}   # (cat, base) → TriangleMesh
        self._corrs = {}    # (cat, base) → np.ndarray [n_ref]  (ref→shape convention)
        self._items = []    # ordered list of (cat, base)

        for cat in available_cats:
            cat_off_dir = os.path.join(off_root, cat)
            cat_corres_dir = os.path.join(corres_root, cat)

            for fname in list_shapes(cat_off_dir):
                base = os.path.splitext(fname)[0]
                shape = load_shape(
                    os.path.join(cat_off_dir, fname), spectral=spectral, k=k
                )
                self._shapes[(cat, base)] = shape

                corr_path = None
                for ext in (".vts", ".txt"):
                    candidate = os.path.join(cat_corres_dir, base + ext)
                    if os.path.exists(candidate):
                        corr_path = candidate
                        break

                if corr_path is not None:
                    corr = np.loadtxt(corr_path, dtype=np.int64) - 1  # 0-indexed
                else:
                    warnings.warn(
                        f"No corr file for {cat}/{base}; using identity."
                    )
                    corr = np.arange(shape.n_vertices, dtype=np.int64)

                self._corrs[(cat, base)] = corr
                self._items.append((cat, base))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        """Return a shape dict with ``shape``, ``corr``, ``category``, and ``name``.

        ``corr[r]`` is the shape vertex corresponding to category reference
        vertex ``r`` (reference→shape convention, length=n_ref).
        """
        cat, base = self._items[idx]
        shape = self._shapes[(cat, base)]
        corr = self._corrs[(cat, base)]
        move_shape_to_device(shape, self.device)
        return {
            "shape": shape,
            "corr": torch.tensor(corr, dtype=torch.long, device=self.device),
            "category": cat,
            "name": base,
        }

    def cross_category_corr(self, cat1, cat2):
        """Load the reference-to-reference correspondence between two categories.

        Returns
        -------
        corr : np.ndarray, shape=[n_ref_cat1]
            ``corr[r]`` = vertex index in ``cat2`` reference corresponding
            to reference vertex ``r`` of ``cat1``.
        """
        cross_dir = os.path.join(self.dataset_dir, "corres", "cross_category_corres")
        path = os.path.join(cross_dir, f"{cat1}_{cat2}.vts")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cross-category corr file not found: {path}"
            )
        return np.loadtxt(path, dtype=np.int64) - 1  # 0-indexed

    def by_category(self, category):
        """Return a view of this dataset restricted to a single category.

        Parameters
        ----------
        category : str

        Returns
        -------
        DT4DDataset
            A new dataset object containing only shapes from *category*.
        """
        sub = object.__new__(DT4DDataset)
        sub.dataset_dir = self.dataset_dir
        sub.spectral = self.spectral
        sub.k = self.k
        sub.device = self.device
        sub.categories = [category]
        sub._shapes = {k: v for k, v in self._shapes.items() if k[0] == category}
        sub._corrs = {k: v for k, v in self._corrs.items() if k[0] == category}
        sub._items = [item for item in self._items if item[0] == category]
        return sub


def _load_phase_set(dataset_dir, phase):
    """Load the set of "{category}/{name}" strings for a given phase.

    Returns None if no phase file is found.
    """
    path = os.path.join(dataset_dir, f"{phase}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


class DT4DPairsDataset(Dataset):
    """DT4D within-category pairs dataset with composed ground-truth correspondence.

    For a pair (shape_i, shape_j) from the same category:

        source["corr"][r]  = shape_i vertex for category reference vertex r
        target["corr"][r]  = shape_j vertex for category reference vertex r

    Both arrays have length ``n_ref`` (the category reference mesh size),
    giving ``n_ref`` ground-truth correspondence pairs compatible with
    :func:`geomfum.eval.normalized_euclidean_error`.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing ``off/``, ``corres/``, and phase files.
    phase : str
        ``"test"`` (default) or ``"train"``.  If the corresponding phase file
        does not exist, all shapes are used.
    categories : list of str, optional
        Restrict to these categories.
    ignored_categories : list of str, optional
        Skip these.  Defaults to ``["pumpkinhulk"]``.
    spectral : bool
    k : int
    device : torch.device or str, optional
    """

    _is_pairs_dataset = True

    def __init__(
        self,
        dataset_dir,
        phase="test",
        categories=None,
        ignored_categories=None,
        spectral=False,
        k=200,
        device=None,
    ):
        self.dataset_dir = dataset_dir
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if ignored_categories is None:
            ignored_categories = _DEFAULT_IGNORED

        # Determine which shapes belong to the requested phase
        phase_set = _load_phase_set(dataset_dir, phase)

        off_root = os.path.join(dataset_dir, "off")
        corres_root = os.path.join(dataset_dir, "corres")

        available_cats = sorted(
            d for d in os.listdir(off_root)
            if os.path.isdir(os.path.join(off_root, d))
            and d not in ignored_categories
        )
        if categories is not None:
            available_cats = [c for c in available_cats if c in categories]

        # Load shapes grouped by category
        by_cat = {}  # cat → list of (base, shape, corr)
        for cat in available_cats:
            cat_off_dir = os.path.join(off_root, cat)
            cat_corres_dir = os.path.join(corres_root, cat)
            entries = []

            for fname in list_shapes(cat_off_dir):
                base = os.path.splitext(fname)[0]
                key = f"{cat}/{base}"
                if phase_set is not None and key not in phase_set:
                    continue

                shape = load_shape(
                    os.path.join(cat_off_dir, fname), spectral=spectral, k=k
                )

                corr_path = None
                for ext in (".vts", ".txt"):
                    candidate = os.path.join(cat_corres_dir, base + ext)
                    if os.path.exists(candidate):
                        corr_path = candidate
                        break

                if corr_path is not None:
                    corr = np.loadtxt(corr_path, dtype=np.int64) - 1
                else:
                    warnings.warn(f"No corr file for {key}; using identity.")
                    corr = np.arange(shape.n_vertices, dtype=np.int64)

                entries.append((base, shape, corr))

            if entries:
                by_cat[cat] = entries

        # Build within-category pairs (i ≠ j)
        self._pairs = []  # (name_i, shape_i, corr_i, name_j, shape_j, corr_j)
        for cat, entries in sorted(by_cat.items()):
            for ai, (ni, si, ci) in enumerate(entries):
                for aj, (nj, sj, cj) in enumerate(entries):
                    if ai != aj:
                        self._pairs.append((ni, si, ci, nj, sj, cj))

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        """Return a pair dict.

        ``source["corr"][r]`` = shape_i vertex for reference vertex r.
        ``target["corr"][r]`` = shape_j vertex for reference vertex r.
        Both have length n_ref (GT correspondence anchors).
        """
        name_i, shape_i, corr_i, name_j, shape_j, corr_j = self._pairs[idx]

        move_shape_to_device(shape_i, self.device)
        move_shape_to_device(shape_j, self.device)

        # Corr tensors stay on CPU for numpy indexing in normalized_euclidean_error.
        return {
            "source": {
                "shape": shape_i,
                "corr": torch.tensor(corr_i, dtype=torch.long),
                "name": name_i,
            },
            "target": {
                "shape": shape_j,
                "corr": torch.tensor(corr_j, dtype=torch.long),
                "name": name_j,
            },
        }
