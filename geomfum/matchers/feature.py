"""Feature-based matchers."""

from geomfum.convert import BaseNeighborFinder, NeighborFinder
from geomfum.descriptor.pipeline import DescriptorPipeline, L2InnerNormalizer
from geomfum.descriptor.spectral import WaveKernelSignature
from geomfum.matchers.base import BaseMatcher
from geomfum.refine import CorrespondenceRefinementPipeline


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
        from geomfum.matchers.base import CorrespondenceResult

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
