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
"""

import abc
from dataclasses import dataclass

import gsops.backend as gs

from geomfum.convert import NeighborFinder, P2pFromFmConverter
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
    FactorSum,
    LBCommutativityEnforcing,
    OperatorCommutativityEnforcing,
    SpectralDescriptorPreservation,
)
from geomfum.numerics.optimization import ScipyMinimize
from geomfum.refine import IcpRefiner, IdentityRefiner, ZoomOut


@dataclass
class MatcherResult:
    """Result of a matching operation.

    Parameters
    ----------
    p2p21 : array-like, shape=[n_vertices_b]
        Point-to-point correspondence from shape_b to shape_a.
        For each vertex i in shape_b, p2p21[i] gives the corresponding
        vertex index in shape_a.
    fmap12 : array-like, shape=[spectrum_size_b, spectrum_size_a]
        Functional map matrix from shape_a to shape_b.
    descr_a : array-like, shape=[n_descr, n_vertices_a]
        Descriptors on shape_a.
    descr_b : array-like, shape=[n_descr, n_vertices_b]
        Descriptors on shape_b.
    refined_fmap12 : array-like, shape=[spectrum_size_b, spectrum_size_a], optional
        Refined functional map matrix (if refinement was applied).
    """

    p2p21: "gs.ndarray"
    fmap12: "gs.ndarray"
    descr_a: "gs.ndarray" = None
    descr_b: "gs.ndarray" = None
    refined_fmap12: "gs.ndarray" = None


@dataclass
class MatcherConfig:
    """Configuration for the Matcher.

    Parameters
    ----------
    spectrum_size : int
        Number of eigenfunctions to compute for the basis.
    fmap_size : int
        Number of eigenfunctions to use for the functional map optimization.
    descriptors : list[Descriptor] or None
        List of descriptors to compute. If None, uses default WKS-based descriptors.
        Pass a list of Descriptor instances to customize.
    subsamplers : list[Subsampler] or None
        List of subsamplers to apply to descriptors. If None, uses default.
    normalizers : list[Normalizer] or None
        List of normalizers to apply to descriptors. If None, uses L2InnerNormalizer.
    sdp_weight : float
        Weight for spectral descriptor preservation constraint.
    lb_weight : float
        Weight for Laplace-Beltrami commutativity constraint.
    mult_weight : float
        Weight for multiplication operator commutativity constraint.
    orient_weight : float
        Weight for orientation operator commutativity constraint.
    refiners : list[Refiner] or None
        List of refiners to apply in sequence. If None, uses default
        ICP + ZoomOut refinement. Pass an empty list to disable refinement.
    optimizer_method : str
        Optimization method for scipy.optimize.minimize.
    """

    spectrum_size: int = 200
    fmap_size: int = 30
    descriptors: list = None  # None means use default WKS
    subsamplers: list = None  # None means use default
    normalizers: list = None  # None means use L2InnerNormalizer
    sdp_weight: float = 1.0
    lb_weight: float = 1e-2
    mult_weight: float = 1e-1
    orient_weight: float = 0.0
    refiners: list = None  # None means use default, [] means no refinement
    optimizer_method: str = "L-BFGS-B"


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
        result : MatcherResult
            Matching result containing:
            - p2p21: point-to-point correspondence from B to A
            - fmap12: functional map from A to B (if applicable)
        """


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
    config : MatcherConfig
        Configuration for the matcher.
    descriptor_pipeline : DescriptorPipeline, optional
        Custom descriptor pipeline. If None, uses default WKS-based pipeline.
    refiner : Refiner, optional
        Custom refiner. If None, uses ICP + ZoomOut based on config.
    p2p_converter : P2pFromFmConverter, optional
        Custom pointwise map converter. If None, uses default.
    optimizer : ScipyMinimize, optional
        Custom optimizer. If None, uses default.

    """

    def __init__(
        self,
        config: MatcherConfig = None,
        descriptor_pipeline: DescriptorPipeline = None,
        refiner=None,
        p2p_converter: P2pFromFmConverter = None,
        optimizer: ScipyMinimize = None,
    ):
        self.config = config or MatcherConfig()
        self._descriptor_pipeline = descriptor_pipeline
        self._refiner = refiner
        self._p2p_converter = p2p_converter or P2pFromFmConverter()
        self._optimizer = optimizer

    @property
    def descriptor_pipeline(self):
        """Get the descriptor pipeline.

        Returns
        -------
        pipeline : DescriptorPipeline
        """
        if self._descriptor_pipeline is not None:
            return self._descriptor_pipeline

        return self._build_default_descriptor_pipeline()

    @property
    def refiner(self):
        """Get the refiner.

        Returns
        -------
        refiner : Refiner
        """
        if self._refiner is not None:
            return self._refiner

        return self._build_default_refiner()

    @property
    def optimizer(self):
        """Get the optimizer.

        Returns
        -------
        optimizer : ScipyMinimize
        """
        if self._optimizer is not None:
            return self._optimizer

        return ScipyMinimize(method=self.config.optimizer_method)

    def _build_default_descriptor_pipeline(self):
        """Build the default descriptor pipeline based on config.

        Returns
        -------
        pipeline : DescriptorPipeline
        """
        steps = []

        # Add descriptors
        if self.config.descriptors is not None:
            steps.extend(self.config.descriptors)
        else:
            # Default: WKS only (no landmarks required)
            steps.append(WaveKernelSignature.from_registry(n_domain=400))

        # Add subsamplers
        if self.config.subsamplers is not None:
            steps.extend(self.config.subsamplers)
        else:
            # Default subsampling
            steps.append(ArangeSubsampler(subsample_step=10))

        # Add normalizers
        if self.config.normalizers is not None:
            steps.extend(self.config.normalizers)
        else:
            # Default normalization
            steps.append(L2InnerNormalizer())

        return DescriptorPipeline(steps)

    def _build_default_refiner(self):
        """Build the default refiner based on config.

        Returns
        -------
        refiner : Refiner or ChainedRefiner
        """
        # If refiners explicitly set in config, use them
        if self.config.refiners is not None:
            if len(self.config.refiners) == 0:
                return IdentityRefiner()
            return ChainedRefiner(self.config.refiners)

        # Default: ICP + ZoomOut
        return ChainedRefiner(
            refiners=[
                IcpRefiner(nit=10),
                ZoomOut(nit=10, step=5),
            ]
        )

    def _ensure_basis(self, shape):
        """Ensure shape has computed basis.

        Parameters
        ----------
        shape : Shape
            Shape to check/compute basis for.
        """
        if (
            shape.basis is None
            or shape.basis.full_spectrum_size < self.config.spectrum_size
        ):
            shape.laplacian.find_spectrum(
                spectrum_size=self.config.spectrum_size, set_as_basis=True
            )

    def _build_factors(self, shape_a, shape_b, descr_a, descr_b):
        """Build optimization factors.

        Parameters
        ----------
        shape_a : Shape
            Source shape.
        shape_b : Shape
            Target shape.
        descr_a : array-like
            Descriptors on source shape.
        descr_b : array-like
            Descriptors on target shape.

        Returns
        -------
        factors : list[WeightedFactor]
            List of optimization factors.
        """
        factors = []

        # Spectral descriptor preservation
        if self.config.sdp_weight > 0:
            factors.append(
                SpectralDescriptorPreservation(
                    shape_a.basis.project(descr_a),
                    shape_b.basis.project(descr_b),
                    weight=self.config.sdp_weight,
                )
            )

        # Laplace-Beltrami commutativity
        if self.config.lb_weight > 0:
            factors.append(
                LBCommutativityEnforcing.from_bases(
                    shape_a.basis,
                    shape_b.basis,
                    weight=self.config.lb_weight,
                )
            )

        # Multiplication operator commutativity
        if self.config.mult_weight > 0:
            factors.append(
                OperatorCommutativityEnforcing.from_multiplication(
                    shape_a.basis,
                    descr_a,
                    shape_b.basis,
                    descr_b,
                    weight=self.config.mult_weight,
                )
            )

        # Orientation operator commutativity
        if self.config.orient_weight > 0:
            factors.append(
                OperatorCommutativityEnforcing.from_orientation(
                    shape_a,
                    descr_a,
                    shape_b,
                    descr_b,
                    weight=self.config.orient_weight,
                )
            )

        return factors

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
        result : MatcherResult
            Matching result containing:
            - p2p21: point-to-point correspondence from B to A
            - fmap12: functional map from A to B
        """
        # Step 1: Ensure both shapes have basis
        self._ensure_basis(shape_a)
        self._ensure_basis(shape_b)

        # Store original use_k values
        original_use_k_a = shape_a.basis.use_k
        original_use_k_b = shape_b.basis.use_k

        # Set full spectrum for descriptor computation
        shape_a.basis.use_k = self.config.spectrum_size
        shape_b.basis.use_k = self.config.spectrum_size

        # Step 2: Compute descriptors
        descr_a = self.descriptor_pipeline.apply(shape_a)
        descr_b = self.descriptor_pipeline.apply(shape_b)

        # Step 3: Set spectrum size for functional map optimization
        shape_a.basis.use_k = self.config.fmap_size
        shape_b.basis.use_k = self.config.fmap_size

        # Step 4: Build and optimize functional map (fmap12: A -> B)
        factors = self._build_factors(shape_a, shape_b, descr_a, descr_b)
        objective = FactorSum(factors)

        x0 = gs.zeros((shape_b.basis.spectrum_size, shape_a.basis.spectrum_size))

        res = self.optimizer.minimize(
            objective,
            x0,
            fun_jac=objective.gradient,
        )

        fmap12 = res.x.reshape(x0.shape)

        # Step 5: Apply refinement
        refined_fmap12 = self.refiner(fmap12, shape_a.basis, shape_b.basis)

        # Step 6: Convert to point-to-point correspondence (p2p21: B -> A)
        p2p21 = self._p2p_converter(refined_fmap12, shape_a.basis, shape_b.basis)

        # Restore original use_k values
        shape_a.basis.use_k = original_use_k_a
        shape_b.basis.use_k = original_use_k_b

        return MatcherResult(
            p2p21=p2p21,
            fmap12=fmap12,
            descr_a=descr_a,
            descr_b=descr_b,
            refined_fmap12=refined_fmap12 if refined_fmap12 is not fmap12 else None,
        )


class ChainedRefiner:
    """Chain multiple refiners together.

    Parameters
    ----------
    refiners : list[Refiner]
        List of refiners to apply in sequence.
        None values are filtered out.
    """

    def __init__(self, refiners):
        self.refiners = [r for r in refiners if r is not None]

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Apply refiners in sequence.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis of source shape.
        basis_b : Eigenbasis.
            Basis of target shape.

        Returns
        -------
        fmap_matrix : array-like
            Refined functional map matrix.
        """
        for refiner in self.refiners:
            fmap_matrix = refiner(fmap_matrix, basis_a, basis_b)

        return fmap_matrix


class QuickMatcher(FunctionalMapMatcher):
    """Fast matcher with reduced settings for quick results.

    Uses smaller spectrum and fewer refinement iterations.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to FunctionalMapMatcher.
    """

    def __init__(self, **kwargs):
        config = MatcherConfig(
            spectrum_size=50,
            fmap_size=15,
            descriptors=[WaveKernelSignature.from_registry(n_domain=100)],
            subsamplers=[ArangeSubsampler(subsample_step=5)],
            refiners=[IcpRefiner(nit=5)],
        )
        super().__init__(config=config, **kwargs)


class PreciseMatcher(FunctionalMapMatcher):
    """High-quality matcher with larger settings for better results.

    Uses larger spectrum and more refinement iterations.

    Parameters
    ----------
    use_landmarks : bool
        Whether to use landmarks.
    **kwargs
        Additional arguments passed to FunctionalMapMatcher.
    """

    def __init__(self, use_landmarks: bool = True, **kwargs):
        descriptors = [WaveKernelSignature.from_registry(n_domain=500)]
        if use_landmarks:
            descriptors.append(LandmarkWaveKernelSignature.from_registry(n_domain=500))

        config = MatcherConfig(
            spectrum_size=300,
            fmap_size=50,
            descriptors=descriptors,
            subsamplers=[ArangeSubsampler(subsample_step=5)],
            refiners=[
                IcpRefiner(nit=15),
                ZoomOut(nit=10, step=3),
            ],
        )
        super().__init__(config=config, **kwargs)


@dataclass
class FeatureMatcherConfig:
    """Configuration for the FeatureMatcher.

    Parameters
    ----------
    spectrum_size : int
        Number of eigenfunctions to compute for the basis (used for spectral descriptors).
    descriptors : list[Descriptor] or None
        List of descriptors to compute. If None, uses default WKS descriptors.
    subsamplers : list[Subsampler] or None
        List of subsamplers to apply. If None, no subsampling.
    normalizers : list[Normalizer] or None
        List of normalizers to apply. If None, uses L2InnerNormalizer.
    """

    spectrum_size: int = 200
    descriptors: list = None
    subsamplers: list = None
    normalizers: list = None


class FeatureMatcher(BaseMatcher):
    """Feature-based matcher using nearest neighbor in descriptor space.

    This matcher directly computes correspondences by:
    1. Computing descriptors/features on both shapes
    2. Finding nearest neighbors in the descriptor space

    This is simpler and faster than the functional map approach,
    but may be less robust for complex deformations.

    Parameters
    ----------
    config : FeatureMatcherConfig
        Configuration for the matcher.
    descriptor_pipeline : DescriptorPipeline, optional
        Custom descriptor pipeline. If None, uses default WKS-based pipeline.
    neighbor_finder : NeighborFinder, optional
        Nearest neighbor finder. If None, uses default.

    """

    def __init__(
        self,
        config: FeatureMatcherConfig = None,
        descriptor_pipeline: DescriptorPipeline = None,
        neighbor_finder: NeighborFinder = None,
    ):
        self.config = config or FeatureMatcherConfig()
        self._descriptor_pipeline = descriptor_pipeline
        self._neighbor_finder = neighbor_finder or NeighborFinder(n_neighbors=1)

    @property
    def descriptor_pipeline(self):
        """Get the descriptor pipeline.

        Returns
        -------
        pipeline : DescriptorPipeline
        """
        if self._descriptor_pipeline is not None:
            return self._descriptor_pipeline

        return self._build_default_descriptor_pipeline()

    def _build_default_descriptor_pipeline(self):
        """Build the default descriptor pipeline based on config.

        Returns
        -------
        pipeline : DescriptorPipeline
        """
        steps = []

        # Add descriptors
        if self.config.descriptors is not None:
            steps.extend(self.config.descriptors)
        else:
            # Default: WKS without subsampling
            steps.append(WaveKernelSignature.from_registry(n_domain=400))

        # Add subsamplers
        if self.config.subsamplers is not None:
            steps.extend(self.config.subsamplers)
        # No default subsampling for FeatureMatcher

        # Add normalizers
        if self.config.normalizers is not None:
            steps.extend(self.config.normalizers)
        else:
            # Default normalization
            steps.append(L2InnerNormalizer())

        return DescriptorPipeline(steps)

    def _ensure_basis(self, shape):
        """Ensure shape has computed basis.

        Parameters
        ----------
        shape : Shape
            Shape to check/compute basis for.
        """
        if (
            shape.basis is None
            or shape.basis.full_spectrum_size < self.config.spectrum_size
        ):
            shape.laplacian.find_spectrum(
                spectrum_size=self.config.spectrum_size, set_as_basis=True
            )

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
        result : MatcherResult
            Matching result containing:
            - p2p21: point-to-point correspondence from B to A
        """
        # Step 1: Ensure both shapes have basis (needed for spectral descriptors)
        self._ensure_basis(shape_a)
        self._ensure_basis(shape_b)

        # Store original use_k values
        original_use_k_a = shape_a.basis.use_k
        original_use_k_b = shape_b.basis.use_k

        # Set full spectrum for descriptor computation
        shape_a.basis.use_k = self.config.spectrum_size
        shape_b.basis.use_k = self.config.spectrum_size

        # Step 2: Compute descriptors
        descr_a = self.descriptor_pipeline.apply(shape_a)
        descr_b = self.descriptor_pipeline.apply(shape_b)

        # Restore original use_k values
        shape_a.basis.use_k = original_use_k_a
        shape_b.basis.use_k = original_use_k_b

        # Step 3: Find nearest neighbors in descriptor space
        # descr shape is [n_descr, n_vertices], we need [n_vertices, n_descr]
        feat_a = descr_a.T
        feat_b = descr_b.T

        # Find for each vertex in B, the nearest vertex in A (p2p21: B -> A)
        p2p21 = self._neighbor_finder(feat_b, feat_a).flatten()

        return MatcherResult(
            p2p21=p2p21,
            fmap12=None,
            descr_a=descr_a,
            descr_b=descr_b,
            refined_fmap12=None,
        )
