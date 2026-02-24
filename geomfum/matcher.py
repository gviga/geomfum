"""Matcher framework for computing shape correspondences.

This module provides a high-level framework for computing correspondences
between shapes using functional maps. It abstracts the complexity of the
pipeline while allowing flexible configuration of each step.

Notation Convention
-------------------
Following the functional maps convention used throughout the library:
- `fmap12` = functional map from shape_a to shape_b, shape = [spectrum_size_b, spectrum_size_a]
- `p2p21` = point-to-point map from shape_b to shape_a (derived from fmap12)
    - For each vertex i in shape_b, p2p21[i] gives the corresponding vertex in shape_a
    - shape = [n_vertices_b]

The Matcher takes (shape_a, shape_b) and returns:
- `fmap12`: functional map from A to B
- `p2p21`: point-to-point correspondence from B to A

With bidirectional=True, also returns:
- `fmap21`: functional map from B to A
- `p2p12`: point-to-point correspondence from A to B
"""

import abc
import json
from dataclasses import dataclass

import gsops.backend as gs

from geomfum.convert import (
    BaseNeighborFinder,
    NeighborFinder,
    P2pFromFmConverter,
)
from geomfum.descriptor.pipeline import (
    ArangeSubsampler,
    DescriptorPipeline,
    L2InnerNormalizer,
)
from geomfum.descriptor.spectral import (
    LandmarkWaveKernelSignature,
    WaveKernelSignature,
)
from geomfum.functional_map import (
    FunctionalMapOptimizer,
    LBFactorBuilder,
    MultFactorBuilder,
    SDPFactorBuilder,
)
from geomfum.refine import (
    CorrespondenceRefinementPipeline,
    IcpRefiner,
    IdentityRefiner,
    RefinementPipeline,
    ZoomOut,
)


@dataclass
class CorrespondenceResult:
    """Result of a matching operation (for both Matcher and Model).

    This is the unified output format for all correspondence methods,
    including classical functional map matchers and learning-based models.

    Parameters
    ----------
    fmap12 : array-like, shape=[spectrum_size_b, spectrum_size_a]
        Functional map matrix from shape_a to shape_b.
    p2p21 : array-like, shape=[n_vertices_b]
        Point-to-point correspondence from shape_b to shape_a.
        For each vertex i in shape_b, p2p21[i] gives the corresponding
        vertex index in shape_a.
    fmap21 : array-like, shape=[spectrum_size_a, spectrum_size_b], optional
        Functional map matrix from shape_b to shape_a (for bidirectional).
    p2p12 : array-like, shape=[n_vertices_a], optional
        Point-to-point correspondence from shape_a to shape_b (for bidirectional).
    descr_a : array-like, shape=[n_descr, n_vertices_a], optional
        Descriptors on shape_a.
    descr_b : array-like, shape=[n_descr, n_vertices_b], optional
        Descriptors on shape_b.
    refined_fmap12 : array-like, shape=[spectrum_size_b, spectrum_size_a], optional
        Refined functional map matrix (if refinement was applied).
    refined_fmap21 : array-like, shape=[spectrum_size_a, spectrum_size_b], optional
        Refined functional map matrix from B to A (if bidirectional refinement).
    """

    fmap12: "gs.ndarray"
    p2p21: "gs.ndarray"
    fmap21: "gs.ndarray" = None
    p2p12: "gs.ndarray" = None
    descr_a: "gs.ndarray" = None
    descr_b: "gs.ndarray" = None
    refined_fmap12: "gs.ndarray" = None
    refined_fmap21: "gs.ndarray" = None

    def to_dict(self):
        """Convert to dictionary (for backward compatibility).

        Returns
        -------
        dict
            Dictionary with all non-None fields.

        Notes
        -----
        This method avoids using `asdict()` from dataclasses because it
        performs deep copying, which fails for PyTorch tensors that are
        part of the computation graph (non-leaf tensors) during training.
        """
        return {
            k: getattr(self, k)
            for k in self.__dataclass_fields__
            if getattr(self, k) is not None
        }

    @property
    def is_bidirectional(self):
        """Check if result contains bidirectional correspondences.

        Returns
        -------
        bool
            True if fmap21 and p2p12 are available.
        """
        return self.fmap21 is not None and self.p2p12 is not None


class BaseMatcher(abc.ABC):
    """Abstract base class for shape matchers."""

    @abc.abstractmethod
    def __call__(self, shape_a, shape_b):
        """Compute correspondence between two shapes.

        Parameters
        ----------
        shape_a : Shape
            First shape (target for p2p21).
        shape_b : Shape
            Second shape (source for p2p21).

        Returns
        -------
        result : CorrespondenceResult
            Correspondence result containing:
            - p2p21: point-to-point correspondence from B to A
            - fmap12: functional map from A to B (if applicable)
        """

    @classmethod
    def from_config(cls, config):
        """Build a matcher from a configuration dictionary.

        The ``"type"`` key in the config selects the matcher class
        (``"FunctionalMapMatcher"`` or ``"FeatureMatcher"``).

        Parameters
        ----------
        config : dict
            Configuration dictionary. See ``geomfum._build``
            for the full schema.

        Returns
        -------
        matcher : BaseMatcher

        Examples
        --------
        >>> config = {
        ...     "type": "FunctionalMapMatcher",
        ...     "fmap_size": 30,
        ...     "descriptor_pipeline": [
        ...         {"type": "WaveKernelSignature", "n_domain": 200, "k": 200},
        ...         {"type": "L2InnerNormalizer"},
        ...     ],
        ... }
        >>> matcher = BaseMatcher.from_config(config)
        """
        from geomfum._build import build_matcher

        return build_matcher(config)

    @classmethod
    def from_json(cls, path):
        """Build a matcher from a JSON configuration file.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file produced by ``to_config()`` or written
            manually.

        Returns
        -------
        matcher : BaseMatcher

        Examples
        --------
        >>> matcher = BaseMatcher.from_json("my_method.json")
        >>> result = matcher(shape_a, shape_b)
        """
        with open(path) as f:
            config = json.load(f)
        return cls.from_config(config)


class FeatureMatcher(BaseMatcher):
    """Feature-based matcher using nearest neighbor in descriptor space.

    This matcher directly computes correspondences by:
    1. Computing descriptors/features on both shapes either indicating descriptor or pipeline.
    2. Finding nearest neighbors in the descriptor space
    3. Optionally refining the correspondence using CorrespondenceRefinementPipeline

    This is simpler and faster than the functional map approach,
    but may be less robust for complex deformations.

    Parameters
    ----------
    descriptor_pipeline : DescriptorPipeline, optional
        Descriptor pipeline to compute descriptors. If None, uses default WKS-based pipeline.
    neighbor_finder : NeighborFinder, optional
        Nearest neighbor finder. If None, uses default.
    refiner : CorrespondenceRefinementPipeline, optional
        Correspondence refiner to apply after matching. If None, no refinement is applied.
    """

    def __init__(
        self,
        descriptor_pipeline: DescriptorPipeline = None,
        neighbor_finder: BaseNeighborFinder = None,
        refiner: CorrespondenceRefinementPipeline = None,
    ):
        self.descriptor_pipeline = (
            descriptor_pipeline or self._build_default_descriptor_pipeline()
        )
        self.neighbor_finder = neighbor_finder or NeighborFinder(n_neighbors=1)
        self.refiner = refiner

    def _build_default_descriptor_pipeline(self):
        """Build the default descriptor pipeline.

        Returns
        -------
        pipeline : DescriptorPipeline
        """
        return DescriptorPipeline(
            [WaveKernelSignature(n_domain=200, k=200), L2InnerNormalizer()]
        )

    def __call__(self, shape_a, shape_b, bidirectional=False):
        """Compute correspondence between two shapes.

        Parameters
        ----------
        shape_a : Shape
            First shape (target for p2p21).
        shape_b : Shape
            Second shape (source for p2p21).
        bidirectional : bool
            If True, compute correspondences in both directions.

        Returns
        -------
        result : CorrespondenceResult
            Matching result containing:
            - p2p21: point-to-point correspondence from B to A
            - p2p12: (if bidirectional=True) correspondence from A to B
        """
        # Compute descriptors
        descr_a = self.descriptor_pipeline.apply(shape_a)
        descr_b = self.descriptor_pipeline.apply(shape_b)

        # Find nearest neighbors in descriptor space
        # descr shape is [n_descr, n_vertices], we need [n_vertices, n_descr]
        feat_a = descr_a.T
        feat_b = descr_b.T
        # Find for each vertex in B, the nearest vertex in A (p2p21: B -> A)
        p2p21 = self.neighbor_finder(feat_b, feat_a).flatten()

        # Compute reverse direction if bidirectional
        p2p12 = None
        if bidirectional:
            p2p12 = self.neighbor_finder(feat_a, feat_b).flatten()

        # Apply correspondence refinement if available
        if self.refiner is not None:
            p2p21 = self.refiner(p2p21, shape_a.basis, shape_b.basis)
            if bidirectional:
                p2p12 = self.refiner(p2p12, shape_b.basis, shape_a.basis)

        return CorrespondenceResult(
            fmap12=None,
            p2p21=p2p21,
            fmap21=None,
            p2p12=p2p12,
            descr_a=descr_a,
            descr_b=descr_b,
            refined_fmap12=None,
            refined_fmap21=None,
        )


class FunctionalMapMatcher(BaseMatcher):
    """Functional map based matcher with configurable pipeline.

    This matcher follows the standard functional map pipeline:
    1. Compute basis (Laplacian eigenfunctions) for both shapes
    2. Compute descriptors (WKS, landmarks if available)
    3. Optimize functional map with various constraints
    4. Convert to point-to-point correspondence
    5. Optionally refine the correspondence

    Parameters
    ----------
    fmap_size : int
        Number of eigenfunctions to use for the functional map optimization.
    descriptor_pipeline : DescriptorPipeline, optional
        Descriptor pipeline to compute descriptors. If None, builds from descriptor.
    fmap_optimizer : FunctionalMapOptimizer, optional
        Optimizer for functional map. If None, uses default.
    refiner : Refiner, optional
        Refiner to apply after optimization. If None, uses IdentityRefiner.
    p2p_converter : P2pFromFmConverter, optional
        Converter from functional map to point-to-point. If None, uses default.
    """

    def __init__(
        self,
        fmap_size: int = 30,
        descriptor_pipeline: DescriptorPipeline = None,
        fmap_optimizer: FunctionalMapOptimizer = None,
        refiner=None,
        p2p_converter: P2pFromFmConverter = None,
    ):
        self.fmap_size = fmap_size
        self.descriptor_pipeline = (
            descriptor_pipeline or self._build_default_descriptor_pipeline()
        )
        self.fmap_optimizer = fmap_optimizer or FunctionalMapOptimizer()
        self.refiner = refiner or self._build_default_refiner()
        self.p2p_converter = p2p_converter or P2pFromFmConverter()

    def _build_default_descriptor_pipeline(self):
        """Build the default descriptor pipeline.

        Returns
        -------
        pipeline : DescriptorPipeline
        """
        return DescriptorPipeline(
            [
                WaveKernelSignature(n_domain=200, k=200),
                ArangeSubsampler(subsample_step=10),
                L2InnerNormalizer(),
            ]
        )

    def _build_default_refiner(self):
        """Build the default refiner.

        Returns
        -------
        refiner : RefinementPipeline
        """
        return RefinementPipeline(
            [
                IdentityRefiner(),
            ]
        )

    def __call__(self, shape_a, shape_b, bidirectional=False):
        """Compute correspondence between two shapes.

        Parameters
        ----------
        shape_a : Shape
            First shape (target for p2p21).
        shape_b : Shape
            Second shape (source for p2p21).
        bidirectional : bool
            If True, compute correspondences in both directions.

        Returns
        -------
        result : CorrespondenceResult
            Matching result containing:
            - fmap12: functional map from A to B
            - p2p21: point-to-point correspondence from B to A
            - fmap21, p2p12: (if bidirectional=True) reverse direction
        """
        # Step 1: Compute descriptors
        descr_a = self.descriptor_pipeline.apply(shape_a)
        descr_b = self.descriptor_pipeline.apply(shape_b)

        # Step 2: Set spectrum size for functional map optimization
        shape_a.basis.use_k = self.fmap_size
        shape_b.basis.use_k = self.fmap_size

        # Step 3: Optimize functional map (fmap12: A -> B)
        fmap12 = self.fmap_optimizer(shape_a.basis, shape_b.basis, descr_a, descr_b)

        # Step 4: Apply refinement
        refined_fmap12 = self.refiner(fmap12, shape_a.basis, shape_b.basis)

        # Step 5: Convert to point-to-point correspondence (p2p21: B -> A)
        p2p21 = self.p2p_converter(refined_fmap12, shape_a.basis, shape_b.basis)

        # Initialize reverse direction as None
        fmap21 = None
        refined_fmap21 = None
        p2p12 = None

        # Step 6: Compute reverse direction if bidirectional
        if bidirectional:
            fmap21 = self.fmap_optimizer(shape_b.basis, shape_a.basis, descr_b, descr_a)
            refined_fmap21 = self.refiner(fmap21, shape_b.basis, shape_a.basis)
            p2p12 = self.p2p_converter(refined_fmap21, shape_b.basis, shape_a.basis)

        return CorrespondenceResult(
            fmap12=fmap12,
            p2p21=p2p21,
            fmap21=fmap21,
            p2p12=p2p12,
            descr_a=descr_a,
            descr_b=descr_b,
            refined_fmap12=refined_fmap12,
            refined_fmap21=refined_fmap21,
        )


class QuickFunctionalMapMatcher(FunctionalMapMatcher):
    """Fast matcher with reduced settings for quick results.

    Uses smaller spectrum and fewer refinement iterations.
    """

    def __init__(self):
        super().__init__(
            fmap_size=20,
            descriptor_pipeline=DescriptorPipeline(
                [
                    WaveKernelSignature.from_registry(n_domain=200),
                    ArangeSubsampler(subsample_step=5),
                    L2InnerNormalizer(),
                ]
            ),
            fmap_optimizer=FunctionalMapOptimizer(
                factor_builders=[
                    SDPFactorBuilder(weight=1.0),
                    LBFactorBuilder(weight=1e-2),
                    MultFactorBuilder(weight=1e-1),
                ]
            ),
            refiner=RefinementPipeline(
                [
                    IcpRefiner(nit=5),
                    ZoomOut(nit=1, step=0),
                ]
            ),
        )


class DefaultFunctionalMapMatcher(FunctionalMapMatcher):
    """High-quality matcher with larger settings for better results.

    Uses larger spectrum and more refinement iterations.

    Parameters
    ----------
    use_landmarks : bool
        Whether to use landmarks in the descriptor pipeline.
    """

    def __init__(self, use_landmarks: bool = True):
        descriptors = [WaveKernelSignature(n_domain=200, k=200)]
        if use_landmarks:
            descriptors.append(LandmarkWaveKernelSignature(n_domain=200, k=200))

        super().__init__(
            fmap_size=30,
            descriptor_pipeline=DescriptorPipeline(
                [
                    *descriptors,
                    ArangeSubsampler(subsample_step=10),
                    L2InnerNormalizer(),
                ]
            ),
            fmap_optimizer=FunctionalMapOptimizer(
                factor_builders=[
                    SDPFactorBuilder(weight=1.0),
                    LBFactorBuilder(weight=1e-2),
                    MultFactorBuilder(weight=1e-1),
                ]
            ),
            refiner=RefinementPipeline(
                [
                    IcpRefiner(nit=10),
                    ZoomOut(nit=20, step=5),
                ]
            ),
        )
