"""TOSCA dataset loaders.

Layout::

    dataset_dir/
      cat0.off, cat1.off, ..., dog0.off, ..., wolf0.off, ...
      (flat root, mixed .off and .obj for some shapes)

Only ``.off`` files are loaded by default to avoid loading both formats
for the same shape.  Optional category filtering is supported.

All TOSCA shapes share the same mesh topology (10 008 vertices), so
ground-truth correspondence between any two shapes of the *same category*
is the identity map.  :class:`ToscaPairsDataset` exploits this to provide
a pairs dataset with euclidean-error ground truth.

Available categories: cat, dog, wolf, horse, centaur, michael, victoria,
david, gorilla (depending on TOSCA variant).
"""

import os
import re

import torch

from benchfum.datasets._utils import load_shape, move_shape_to_device
from geomfum.dataset.torch import BasePairsDataset, ShapeDataset


class ToscaDataset(ShapeDataset):
    """TOSCA dataset (81 animal and human body meshes).

    Parameters
    ----------
    dataset_dir : str
        Root directory containing the ``.off`` files (flat, no subdirectory).
    categories : list of str, optional
        If provided, only load shapes whose filename starts with one of these
        category names (e.g. ``["cat", "dog"]``).
    **kwargs
        Forwarded to :class:`~geomfum.dataset.torch.ShapeDataset`.
        ``file_extensions`` defaults to ``(".off",)`` to avoid loading
        duplicate ``.obj`` files for the same shape.
    """

    def __init__(self, dataset_dir, categories=None, **kwargs):
        kwargs.setdefault("shapes_subdir", "")
        kwargs.setdefault("file_extensions", (".off",))
        super().__init__(dataset_dir, **kwargs)
        if categories is not None:
            self.shape_files = [
                f for f in self.shape_files
                if any(f.startswith(cat) for cat in categories)
            ]


# Pattern matching standard TOSCA filenames like "cat0", "michael12", etc.
_TOSCA_NAME_RE = re.compile(r"^([a-zA-Z]+)\d+$")


class ToscaPairsDataset(BasePairsDataset):
    """TOSCA pairs dataset with identity ground-truth correspondence.

    All TOSCA shapes share the same mesh connectivity (10 008 vertices), so
    vertex k of one shape corresponds to vertex k of every other shape.
    This dataset yields within-category ``(source, target)`` pairs where both
    ``source["corr"]`` and ``target["corr"]`` are identity maps
    ``(arange(n_vertices))``, enabling evaluation with
    :func:`geomfum.evaluation.EuclideanErrorMetric`.

    Shapes named with non-standard patterns (e.g. ``cat-00.off``) are silently
    skipped.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing the ``.off`` files.
    categories : list of str, optional
        Only include shapes belonging to these categories.
    spectral : bool
        Whether to compute the LBO spectral basis.
    k : int
        Number of eigenvectors.
    device : torch.device or str, optional
    """

    def __init__(
        self,
        dataset_dir,
        categories=None,
        spectral=False,
        k=200,
        device=None,
    ):
        super().__init__(dataset=None, device=device)
        self.dataset_dir = dataset_dir

        # Load all matching shapes and group by category
        by_cat = {}  # category → list of (name, Shape)
        for fname in sorted(os.listdir(dataset_dir)):
            if not fname.lower().endswith(".off"):
                continue
            stem = os.path.splitext(fname)[0]
            m = _TOSCA_NAME_RE.match(stem)
            if m is None:
                continue  # skip non-standard names like cat-00
            cat = m.group(1)
            if categories is not None and cat not in categories:
                continue
            shape = load_shape(
                os.path.join(dataset_dir, fname), spectral=spectral, k=k
            )
            by_cat.setdefault(cat, []).append((stem, shape))

        # Build within-category pairs (i ≠ j)
        self._pairs = []  # list of (name_i, shape_i, name_j, shape_j)
        for cat, shapes in sorted(by_cat.items()):
            for ai, (ni, si) in enumerate(shapes):
                for aj, (nj, sj) in enumerate(shapes):
                    if ai != aj:
                        self._pairs.append((ni, si, nj, sj))

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        """Return a pair dict with identity ground-truth correspondences."""
        name_i, shape_i, name_j, shape_j = self._pairs[idx]

        move_shape_to_device(shape_i, self.device)
        move_shape_to_device(shape_j, self.device)

        n = shape_i.n_vertices  # same for all TOSCA shapes
        # Corr tensors stay on CPU for numpy indexing in normalized_euclidean_error.
        src_corr = torch.arange(n, dtype=torch.long)
        tgt_corr = torch.arange(n, dtype=torch.long)
        return {
            "source": {
                "shape": shape_i,
                "corr": src_corr,
                "name": name_i,
            },
            "target": {
                "shape": shape_j,
                "corr": tgt_corr,
                "name": name_j,
            },
            "source_id": name_i,
            "target_id": name_j,
            "source_corr": src_corr,
            "target_corr": tgt_corr,
            "corr": None,
            "meta": {
                "dataset": type(self).__name__,
                "corr_type": "identity",
            },
        }
