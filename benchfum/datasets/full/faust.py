"""FAUST dataset loader.

Layout::

    dataset_dir/
      tr_reg_000.off ... tr_reg_099.off   <- flat, no subdirectory

All 100 shapes share the same topology (6890 vertices), so correspondences
between any two shapes are the identity map.  No ``.vts`` files are needed.
"""

from geomfum.dataset.torch import ShapeDataset


class FaustDataset(ShapeDataset):
    """FAUST registered human body dataset (100 meshes).

    All shapes are in the dataset root (no ``shapes/`` subdirectory).
    Correspondences default to identity (vertex i ↔ vertex i).

    Parameters
    ----------
    dataset_dir : str
        Root directory containing the ``.off`` files.
    split : str or None
        ``"train"`` (shapes 0–79), ``"test"`` (shapes 80–99), or ``None``
        (all 100).  Default ``None``.
    **kwargs
        Forwarded to :class:`~geomfum.dataset.torch.ShapeDataset`.
    """

    def __init__(self, dataset_dir, split=None, **kwargs):
        kwargs.setdefault("shapes_subdir", "")
        kwargs.setdefault("correspondences", True)
        super().__init__(dataset_dir, **kwargs)
        if split == "train":
            self.shape_files = self.shape_files[:80]
        elif split == "test":
            self.shape_files = self.shape_files[80:]
        elif split is not None:
            raise ValueError(f"Unknown split {split!r}. Use 'train', 'test', or None.")
