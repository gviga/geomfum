"""Refinement benchmark utilities.

This module provides:

- ``RefinementMatcher``  — wraps any base matcher with a swappable refiner,
  so different refinement strategies can be compared on the same initial map.
- ``RefinementPresets``  — loads named refiner configurations from
  ``benchfum/configs/refiners/``.

The refinement challenge measures how much each refiner improves a fixed
initial correspondence produced by a shared base method.  Using the same base
correspondence for all refiners isolates the contribution of the refinement
step itself.

Typical workflow
----------------
>>> from benchfum import compare, MatcherPresets, RefinementPresets
>>> from benchfum.refinement import RefinementMatcher
>>>
>>> base   = MatcherPresets.build("fmap")          # initial FM, no refinement
>>> methods = {
...     "No refinement": RefinementMatcher(base, RefinementPresets.build("identity")),
...     "ICP":           RefinementMatcher(base, RefinementPresets.build("icp")),
...     "ZoomOut":       RefinementMatcher(base, RefinementPresets.build("zoomout")),
...     "ICP+ZO":        RefinementMatcher(base, RefinementPresets.build("icp_zoomout")),
...     "MyRefiner":     RefinementMatcher(base, MyRefiner()),
... }
>>> results = compare(methods, dataset=pairs, metrics=["geodesic_error"])
>>> results.print_comparison()
"""

import json
from pathlib import Path

from geomfum._build import build_refiner
from geomfum.convert import P2pFromFmConverter
from geomfum.matcher import BaseMatcher, CorrespondenceResult

_REFINERS_DIR = Path(__file__).parent / "configs" / "refiners"


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
    >>> from benchfum import MatcherPresets, RefinementPresets
    >>> from benchfum.refinement import RefinementMatcher
    >>>
    >>> base    = MatcherPresets.build("fmap")
    >>> refiner = RefinementPresets.build("icp_zoomout")
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
        # Step 1: obtain initial correspondence from base matcher
        base_result = self.base_matcher(shape_a, shape_b, bidirectional=bidirectional)

        # Step 2: apply refiner to fmap12
        if base_result.fmap12 is not None:
            refined_fmap12 = self.refiner(
                base_result.fmap12, shape_a.basis, shape_b.basis
            )
            p2p21 = self.p2p_converter(refined_fmap12, shape_a.basis, shape_b.basis)
        else:
            # Base matcher produced no fmap (e.g. FeatureMatcher) — pass through
            refined_fmap12 = None
            p2p21 = base_result.p2p21

        # Step 3: handle bidirectional
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


class RefinementPresets:
    """Named preset configurations for functional map refiners.

    Presets are stored as JSON files in ``benchfum/configs/refiners/``.
    Adding a new preset is as simple as dropping a ``.json`` file there.

    Available presets
    -----------------
    - ``identity``    : No refinement (pass-through). Use as the baseline.
    - ``icp``         : Iterative closest point on the functional map.
    - ``zoomout``     : ZoomOut upsampling of the functional map.
    - ``icp_zoomout`` : ICP followed by ZoomOut (standard strong baseline).

    Examples
    --------
    >>> refiner = RefinementPresets.build("icp_zoomout")
    >>> print(RefinementPresets.list_presets())
    >>> print(RefinementPresets.describe("zoomout"))
    """

    @classmethod
    def list_presets(cls):
        """List all available named refiner presets.

        Returns
        -------
        names : list[str]
            Sorted list of preset names.
        """
        if not _REFINERS_DIR.exists():
            return []
        return sorted(f.stem for f in _REFINERS_DIR.glob("*.json"))

    @classmethod
    def build(cls, name):
        """Build a refiner from a named preset.

        Parameters
        ----------
        name : str
            Preset name (e.g. ``"icp_zoomout"``).

        Returns
        -------
        refiner : refiner instance or RefinementPipeline
        """
        config_path = _REFINERS_DIR / f"{name}.json"
        if not config_path.exists():
            available = cls.list_presets()
            raise ValueError(
                f"Unknown refiner preset {name!r}. Available: {available}"
            )
        with open(config_path) as f:
            config = json.load(f)
        # Strip metadata keys from dicts; bare lists are pipeline specs and pass as-is
        if isinstance(config, dict):
            config = {k: v for k, v in config.items() if not k.startswith("_")}
        return build_refiner(config)

    @classmethod
    def describe(cls, name):
        """Return the raw configuration dict for a named preset.

        Parameters
        ----------
        name : str
            Preset name.

        Returns
        -------
        config : dict
        """
        config_path = _REFINERS_DIR / f"{name}.json"
        if not config_path.exists():
            available = cls.list_presets()
            raise ValueError(
                f"Unknown refiner preset {name!r}. Available: {available}"
            )
        with open(config_path) as f:
            return json.load(f)


__all__ = ["RefinementMatcher", "RefinementPresets"]
