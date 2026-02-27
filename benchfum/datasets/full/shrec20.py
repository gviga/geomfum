"""SHREC 2020 dataset loader.

Layout::

    dataset_dir/
      off/
        bear.obj           <- reference shape (only .obj, no .vts file)
        bison.off … rhino.off   <- 10 target shapes (.off)
      corres/
        bison.vts … rhino.vts   <- target → reference correspondences (1-indexed floats)

Each ``{shape}.vts`` maps that shape's vertices to the ``bear`` reference.
Evaluation pairs are always ``(bear, shape_X)``.
"""

from ._reference_pairs import ReferencePairsDataset


class Shrec20Dataset(ReferencePairsDataset):
    """SHREC 2020 shape matching dataset (bear reference + 10 targets).

    ``bear`` is the fixed reference loaded from ``bear.obj``.  Every item is
    a ``(bear, shape_X)`` pair with per-target corr from
    ``corres/{shape_X}.vts``.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing ``off/`` and ``corres/`` subdirectories.
    **kwargs
        Forwarded to :class:`ReferencePairsDataset`.
    """

    def __init__(self, dataset_dir, **kwargs):
        kwargs.setdefault("shapes_dir", "off")
        kwargs.setdefault("corr_dir", "corres")
        kwargs.setdefault("ref_name", "bear")
        kwargs.setdefault("ref_file_ext", ".obj")   # bear only has .obj
        kwargs.setdefault("corr_suffix", "")        # "bison.vts" → "bison"
        kwargs.setdefault("corr_ext", ".vts")
        kwargs.setdefault("corr_offset", 1)         # 1-indexed (stored as floats)
        kwargs.setdefault("target_file_exts", (".off",))  # only .off targets
        super().__init__(dataset_dir, **kwargs)
