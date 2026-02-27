"""SCAPE remeshed dataset loader.

Layout::

    dataset_dir/
      off/      <- mesh files (mesh000.off ... mesh070.off)
      corres/   <- 1-indexed .vts correspondence files

71 shapes: train split uses indices 0–50 (51 shapes),
test split uses indices 51–70 (20 shapes), matching the URRSM convention.
"""

from geomfum.dataset.torch import ShapeDataset


class ScapeDataset(ShapeDataset):
    """SCAPE remeshed human body dataset (71 meshes).

    Parameters
    ----------
    dataset_dir : str
        Root directory containing ``off/`` and ``corres/`` subdirectories.
    split : str or None
        ``"train"`` (shapes 0–50), ``"test"`` (shapes 51–70), or ``None``
        (all 71).  Default ``None``.
    **kwargs
        Forwarded to :class:`~geomfum.dataset.torch.ShapeDataset`.
        ``corr_offset`` defaults to 1 (1-indexed .vts files).
    """

    def __init__(self, dataset_dir, split=None, **kwargs):
        kwargs.setdefault("shapes_subdir", "off")
        kwargs.setdefault("corr_subdir", "corres")
        kwargs.setdefault("corr_offset", 1)
        super().__init__(dataset_dir, **kwargs)
        if split == "train":
            self.shape_files = self.shape_files[:51]
        elif split == "test":
            self.shape_files = self.shape_files[51:]
        elif split is not None:
            raise ValueError(f"Unknown split {split!r}. Use 'train', 'test', or None.")
