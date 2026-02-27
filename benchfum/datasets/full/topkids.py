"""TOPKIDS dataset loader.

Layout::

    dataset_dir/
      off/
        kid00.off          <- reference shape (no corr file)
        kid01.off … kid25.off
      corres/
        kid01_ref.vts … kid25_ref.vts   <- target → reference correspondences

Each ``kidXX_ref.vts`` has ``n_XX`` lines; line k gives the 1-indexed vertex
in ``kid00`` (reference) corresponding to vertex k of ``kidXX``.
Evaluation pairs are always ``(kid00, kidXX)``.
"""

from ._reference_pairs import ReferencePairsDataset


class TopKidsDataset(ReferencePairsDataset):
    """TOPKIDS dataset (26 child body meshes, reference-based evaluation).

    Shape ``kid00`` is the fixed reference / source.  Every item is a
    ``(kid00, kidXX)`` pair with per-target corr loaded from
    ``corres/kidXX_ref.vts``.

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
        kwargs.setdefault("ref_name", "kid00")
        kwargs.setdefault("corr_suffix", "_ref")
        kwargs.setdefault("corr_ext", ".vts")
        kwargs.setdefault("corr_offset", 1)
        kwargs.setdefault("target_file_exts", (".off",))
        super().__init__(dataset_dir, **kwargs)
