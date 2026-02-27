"""SMAL animal body dataset loader.

Layout::

    dataset_dir/
      all0.ply ... all300.ply   <- flat, no subdirectory

All 301 shapes share the same topology, so correspondences are identity.
"""

from geomfum.dataset.torch import ShapeDataset


class SmalDataset(ShapeDataset):
    """SMAL parameterized animal body dataset (301 meshes).

    All shapes are in the dataset root (no ``shapes/`` subdirectory).
    Correspondences default to identity.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing the ``.ply`` files.
    **kwargs
        Forwarded to :class:`~geomfum.dataset.torch.ShapeDataset`.
    """

    def __init__(self, dataset_dir, **kwargs):
        kwargs.setdefault("shapes_subdir", "")
        kwargs.setdefault("correspondences", True)
        super().__init__(dataset_dir, **kwargs)
