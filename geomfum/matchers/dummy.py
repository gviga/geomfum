"""Dummy matchers for testing and smoke-testing the evaluation pipeline."""

import numpy as np

from geomfum.matchers.base import BaseMatcher, CorrespondenceResult


class DummySoftPermMatcher(BaseMatcher):
    """Dummy matcher that outputs a random soft permutation matrix.

    Useful for testing that the evaluation pipeline correctly handles
    soft-permutation outputs.  The output matrices are row-stochastic
    (rows sum to 1) but carry no semantic meaning.

    Parameters
    ----------
    seed : int or None
        RNG seed for reproducibility.  Default ``None`` (random).

    Notes
    -----
    ``soft_perm_ba[j, i]`` = probability that vertex *j* in B maps to
    vertex *i* in A (row-stochastic, shape ``[n_b, n_a]``).

    ``soft_perm_ab[i, j]`` = probability that vertex *i* in A maps to
    vertex *j* in B (row-stochastic, shape ``[n_a, n_b]``).

    ``p2p21`` is derived via ``argmax`` over the rows of ``soft_perm_ba``
    so that standard p2p-based metrics remain computable.
    """

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape_a, shape_b):
        """Compute a random soft permutation between two shapes.

        Parameters
        ----------
        shape_a : Shape
            Target shape (A).
        shape_b : Shape
            Source shape (B).

        Returns
        -------
        result : CorrespondenceResult
            Contains ``soft_perm_ba``, ``soft_perm_ab``, and ``p2p21``
            (argmax of ``soft_perm_ba``).  ``fmap12`` is ``None``.
        """
        rng = np.random.default_rng(self.seed)
        n_a = shape_a.n_vertices
        n_b = shape_b.n_vertices

        # Row-stochastic [n_b, n_a]: distribution over A for each vertex of B
        raw_ba = rng.random((n_b, n_a)).astype(np.float32)
        soft_perm_ba = raw_ba / raw_ba.sum(axis=1, keepdims=True)

        # Row-stochastic [n_a, n_b]: distribution over B for each vertex of A
        raw_ab = rng.random((n_a, n_b)).astype(np.float32)
        soft_perm_ab = raw_ab / raw_ab.sum(axis=1, keepdims=True)

        # Hard p2p via argmax so standard metrics stay computable
        p2p21 = np.argmax(soft_perm_ba, axis=1).astype(np.int32)

        return CorrespondenceResult(
            fmap12=None,
            p2p21=p2p21,
            soft_perm_ab=soft_perm_ab,
            soft_perm_ba=soft_perm_ba,
        )
