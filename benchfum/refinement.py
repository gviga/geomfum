"""Refinement benchmark utilities.

``RefinementMatcher`` wraps any base matcher with a swappable refiner,
so different refinement strategies can be compared on the same initial map.

The refinement challenge measures how much each refiner improves a fixed
initial correspondence produced by a shared base method.  Using the same base
correspondence for all refiners isolates the contribution of the refinement
step itself.

Typical workflow
----------------
>>> from benchfum import build_matcher, build_refiner_from_json
>>> from benchfum.refinement import RefinementMatcher
>>>
>>> base = build_matcher_from_json("configs/matchers/fmap.json")
>>> methods = {
...     "No refinement": RefinementMatcher(base, build_refiner_from_json("configs/refiners/identity.json")),
...     "ICP":           RefinementMatcher(base, build_refiner_from_json("configs/refiners/icp.json")),
...     "ICP+ZO":        RefinementMatcher(base, build_refiner_from_json("configs/refiners/icp_zoomout.json")),
...     "MyRefiner":     RefinementMatcher(base, MyRefiner()),
... }
>>> results = compare(methods, dataset=pairs, metrics=["geodesic_error"])
>>> results.print_comparison()
"""

from geomfum.convert import P2pFromFmConverter
from geomfum.matcher import BaseMatcher, CorrespondenceResult


class RefinementMatcher(BaseMatcher):
    """Compose a base matcher with a swappable refinement step.

    Runs the base matcher to obtain an initial functional map and p2p
    correspondence, then applies the refiner to produce the final p2p.

    This lets you compare refinement strategies in isolation: all methods
    share the same initial map, so any difference in the final score is
    entirely due to the refinement step.

    Parameters
    ----------
    base_matcher : BaseMatcher
        Any matcher that produces a ``CorrespondenceResult`` with a
        non-``None`` ``fmap12`` (e.g. ``FunctionalMapMatcher``).
    refiner : callable
        A refiner with signature
        ``refiner(fmap12, basis_a, basis_b) -> refined_fmap12``.
        Compatible with ``IcpRefiner``, ``ZoomOut``, ``RefinementPipeline``,
        ``IdentityRefiner``, or any custom callable following that contract.
    p2p_converter : P2pFromFmConverter, optional
        Converts the refined functional map to a point-to-point map.
        Defaults to the standard nearest-neighbour converter.

    Examples
    --------
    >>> from benchfum import build_matcher_from_json, build_refiner_from_json
    >>> from benchfum.refinement import RefinementMatcher
    >>>
    >>> base    = build_matcher_from_json("configs/matchers/fmap.json")
    >>> refiner = build_refiner_from_json("configs/refiners/icp_zoomout.json")
    >>> matcher = RefinementMatcher(base, refiner)
    >>> result  = matcher(shape_a, shape_b)
    """

    def __init__(self, base_matcher, refiner, p2p_converter=None):
        self.base_matcher = base_matcher
        self.refiner = refiner
        self.p2p_converter = p2p_converter or P2pFromFmConverter()

    def __call__(self, shape_a, shape_b, bidirectional=False):
        """Compute correspondence: run base, then apply refiner.

        Parameters
        ----------
        shape_a : Shape
            First shape (target for p2p21).
        shape_b : Shape
            Second shape (source for p2p21).
        bidirectional : bool
            If True, compute and refine correspondences in both directions.

        Returns
        -------
        result : CorrespondenceResult
            Contains:
            - ``fmap12``: initial (unrefined) functional map from A to B
            - ``p2p21``:  correspondence after refinement
            - ``refined_fmap12``: functional map after refinement
        """
        base_result = self.base_matcher(shape_a, shape_b, bidirectional=bidirectional)

        if base_result.fmap12 is not None:
            refined_fmap12 = self.refiner(
                base_result.fmap12, shape_a.basis, shape_b.basis
            )
            p2p21 = self.p2p_converter(refined_fmap12, shape_a.basis, shape_b.basis)
        else:
            refined_fmap12 = None
            p2p21 = base_result.p2p21

        refined_fmap21 = None
        p2p12 = None
        if bidirectional and base_result.fmap21 is not None:
            refined_fmap21 = self.refiner(
                base_result.fmap21, shape_b.basis, shape_a.basis
            )
            p2p12 = self.p2p_converter(refined_fmap21, shape_b.basis, shape_a.basis)

        return CorrespondenceResult(
            fmap12=base_result.fmap12,
            p2p21=p2p21,
            fmap21=base_result.fmap21,
            p2p12=p2p12,
            descr_a=base_result.descr_a,
            descr_b=base_result.descr_b,
            refined_fmap12=refined_fmap12,
            refined_fmap21=refined_fmap21,
        )


__all__ = ["RefinementMatcher"]
