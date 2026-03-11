"""Matcher that loads precomputed point-to-point correspondences from disk."""

import os

import numpy as np

from geomfum.matchers.base import BaseMatcher, CorrespondenceResult

_SUPPORTED_EXTENSIONS = (".npy", ".txt", ".map", ".vts")


class PrecomputedP2pMatcher(BaseMatcher):
    """Matcher that reads precomputed p2p maps from a directory on disk.

    This is useful for comparing an external method (whose outputs have
    already been saved) against other methods in a benchfum experiment,
    without re-running that method.

    Shape pairs are identified by the ``name`` attribute that
    :class:`~geomfum.dataset.torch.ShapeDataset` attaches to every loaded
    shape (the filename stem, e.g. ``"tr_reg_000"`` for
    ``"tr_reg_000.ply"``).

    File-naming convention
    ----------------------
    The file for the pair (shape_b → shape_a) is resolved as::

        <p2p_dir>/<file_pattern>.{npy|txt|map|vts}

    where ``file_pattern`` is a Python format string with two placeholders:

    * ``{src}`` — name of *shape_b* (source, the shape whose vertices the
      map covers)
    * ``{tgt}`` — name of *shape_a* (target, where vertices map to)

    The default pattern is ``"{src}_{tgt}"``, so for
    ``shape_a.name = "tr_reg_001"`` and ``shape_b.name = "tr_reg_000"``
    the matcher looks for
    ``<p2p_dir>/tr_reg_000_tr_reg_001.{npy,txt,map,vts}``.

    File formats
    ------------
    * ``.npy``  — NumPy array of shape ``[n_vertices_b]`` (int32/int64).
    * ``.txt`` / ``.vts`` — one integer index per line (plain text).
    * ``.map``  — same as ``.txt``.

    Parameters
    ----------
    p2p_dir : str
        Directory containing the precomputed p2p files.
    file_pattern : str
        Filename pattern (without extension) with ``{src}`` and ``{tgt}``
        placeholders.  Default ``"{src}_{tgt}"``.
    corr_offset : int
        Value subtracted from loaded indices.  Use ``1`` for 1-indexed
        files (e.g. MATLAB-exported ``.vts``).  Default ``0``.

    Examples
    --------
    Suppose you have already run *SomeExternalMethod* and saved its p2p maps
    as ``results/tr_reg_000_tr_reg_001.txt``, etc.:

    >>> from geomfum.matchers import PrecomputedP2pMatcher
    >>> matcher = PrecomputedP2pMatcher(p2p_dir="results/")
    >>> result = matcher(shape_a, shape_b)   # shape_a.name="tr_reg_001", shape_b.name="tr_reg_000"

    To include it in a benchfum comparison:

    >>> from benchfum import compare
    >>> results = compare(
    ...     {"ExternalMethod": PrecomputedP2pMatcher("results/"),
    ...      "FMap": my_fmap_matcher},
    ...     dataset=pairs,
    ... )
    """

    def __init__(self, p2p_dir, file_pattern="{src}_{tgt}", corr_offset=0):
        self.p2p_dir = p2p_dir
        self.file_pattern = file_pattern
        self.corr_offset = corr_offset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_name(self, shape):
        name = getattr(shape, "name", None)
        if name is None:
            raise AttributeError(
                f"Shape {shape!r} has no 'name' attribute. "
                "Shapes must be loaded via ShapeDataset (or a subclass) "
                "which sets shape.name = <filename stem> during loading."
            )
        return name

    def _resolve_path(self, stem):
        """Return the full path of the p2p file, trying all supported extensions."""
        for ext in _SUPPORTED_EXTENSIONS:
            path = os.path.join(self.p2p_dir, stem + ext)
            if os.path.exists(path):
                return path, ext
        raise FileNotFoundError(
            f"No precomputed p2p file found for stem '{stem}' "
            f"in directory '{self.p2p_dir}'. "
            f"Tried extensions: {_SUPPORTED_EXTENSIONS}. "
            "Check that p2p_dir and file_pattern are correct."
        )

    def _load(self, path, ext):
        if ext == ".npy":
            p2p = np.load(path)
        else:
            p2p = np.loadtxt(path, dtype=np.int64)
        return p2p.astype(np.int32) - self.corr_offset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, shape_a, shape_b):
        """Load and return the precomputed p2p map for the given pair.

        Parameters
        ----------
        shape_a : Shape
            Target shape (A).  Must have a ``name`` attribute.
        shape_b : Shape
            Source shape (B).  Must have a ``name`` attribute.

        Returns
        -------
        result : CorrespondenceResult
            Contains ``p2p21`` loaded from disk.  ``fmap12`` is ``None``.
        """
        name_a = self._get_name(shape_a)
        name_b = self._get_name(shape_b)
        stem = self.file_pattern.format(src=name_b, tgt=name_a)
        path, ext = self._resolve_path(stem)
        p2p21 = self._load(path, ext)
        return CorrespondenceResult(fmap12=None, p2p21=p2p21)
