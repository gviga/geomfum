"""Models for learning features for functional maps.

References
----------
.. "Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence" by Nicolas Donati, Abhishek Sharma, Maks Ovsjanikov.
.. "Deep Functional Maps: Structured Prediction for Dense Shape Correspondence" by O. Litany, T. Remez, E. Rodola, A. Bronstein, M. Bronstein.
"""

from geomfum.convert import (
    P2pFromFmConverter,
)
from geomfum.descriptor.learned import FeatureExtractor, LearnedDescriptor
from geomfum.forward_functional_map import ForwardFunctionalMap
from geomfum.matcher.base import CorrespondenceResult

from ._base import BaseModel


class FMNet(BaseModel):
    """Functional Map Network Model.

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Feature extractor to use for the descriptors.
    fmap_module : ForwardFunctionalMap
        Functional map module to use for the forward pass.
    converter : P2pFromFmConverter
        Converter to convert functional maps to point-to-point correspondences.
    """

    def __init__(
        self,
        feature_extractor=FeatureExtractor.from_registry(which="diffusionnet"),
        fmap_module=ForwardFunctionalMap(),
        converter=P2pFromFmConverter(),
    ):
        super(FMNet, self).__init__()

        self.feature_extractor = feature_extractor
        self.descriptors_module = LearnedDescriptor(
            feature_extractor=self.feature_extractor
        )
        self.fmap_module = fmap_module
        self.converter = converter

    def forward(self, mesh_a, mesh_b, bidirectional=True, as_dict=False):
        """Compute the functional map between two shapes.

        Parameters
        ----------
        mesh_a : TriangleMesh or dict
            The first shape (target for p2p21).
        mesh_b : TriangleMesh or dict
            The second shape (source for p2p21).
        bidirectional : bool, optional
            If True, compute correspondences in both directions. Default is True.
        as_dict : bool, optional
            If True, returns a dictionary instead of CorrespondenceResult.
            Deprecated, use result.to_dict() instead.

        Returns
        -------
        result : CorrespondenceResult or dict
            Matching result containing:
            - fmap12: functional map from A to B
            - p2p21: point-to-point correspondence from B to A
            - fmap21, p2p12: (if bidirectional) reverse direction
            - descr_a, descr_b: computed descriptors
        """
        desc_a = self.descriptors_module(mesh_a)
        desc_b = self.descriptors_module(mesh_b)

        fmap12, fmap21 = self.fmap_module(mesh_a, mesh_b, desc_a, desc_b)

        p2p12 = p2p21 = None
        if not self.training:
            p2p21 = self.converter(fmap12, mesh_a.basis, mesh_b.basis)
            if bidirectional:
                p2p12 = self.converter(fmap21, mesh_b.basis, mesh_a.basis)

        result = CorrespondenceResult(
            fmap12=fmap12,
            p2p21=p2p21,
            fmap21=fmap21 if bidirectional else None,
            p2p12=p2p12,
            descr_a=desc_a,
            descr_b=desc_b,
        )

        if as_dict:
            return result.to_dict()
        return result
