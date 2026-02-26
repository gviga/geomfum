"""Models for learning features for functional maps."""

import gsops.backend as gs

from geomfum.convert import (
    FmFromP2pConverter,
    P2pFromFmConverter,
    SoftmaxNeighborFinder,
)
from geomfum.descriptor.learned import FeatureExtractor, LearnedDescriptor
from geomfum.forward_functional_map import ForwardFunctionalMap
from geomfum.matcher import CorrespondenceResult

from ._base import BaseModel


class RobustFMNet(BaseModel):
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
        converter=P2pFromFmConverter(SoftmaxNeighborFinder(n_neighbors=1, tau=0.07)),
    ):
        super(RobustFMNet, self).__init__()

        self.feature_extractor = feature_extractor
        self.descriptors_module = LearnedDescriptor(
            feature_extractor=self.feature_extractor
        )
        self.fmap_module = fmap_module
        self.converter = converter
        self.fmap_converter = FmFromP2pConverter(pseudo_inverse=True)
        self.neighbor_finder = self.converter.neighbor_finder

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
            - refined_fmap12, refined_fmap21: descriptor-based fmaps
        """
        desc_a = self.descriptors_module(mesh_a)
        desc_b = self.descriptors_module(mesh_b)

        fmap12, fmap21 = self.fmap_module(mesh_a, mesh_b, desc_a, desc_b)

        desc_a_norm = desc_a / gs.linalg.norm(desc_a, axis=0, keepdims=True)
        desc_b_norm = desc_b / gs.linalg.norm(desc_b, axis=0, keepdims=True)

        P12 = self.neighbor_finder.softmax_matrix(desc_a_norm.T, desc_b_norm.T)
        P21 = self.neighbor_finder.softmax_matrix(desc_b_norm.T, desc_a_norm.T)

        # Descriptor-based fmaps (used as "refined" fmaps)
        fmap21_desc = mesh_a.basis.pinv @ (P12 @ mesh_b.basis.vecs)
        fmap12_desc = mesh_b.basis.pinv @ (P21 @ mesh_a.basis.vecs)

        p2p12 = p2p21 = None
        if not self.training:
            p2p21 = gs.to_device(
                self.converter(fmap12, mesh_a.basis, mesh_b.basis), "cpu"
            )
            if bidirectional:
                p2p12 = gs.to_device(
                    self.converter(fmap21, mesh_b.basis, mesh_a.basis), "cpu"
                )

        result = CorrespondenceResult(
            fmap12=fmap12,
            p2p21=p2p21,
            fmap21=fmap21 if bidirectional else None,
            p2p12=p2p12,
            descr_a=desc_a,
            descr_b=desc_b,
            refined_fmap12=fmap12_desc,
            refined_fmap21=fmap21_desc if bidirectional else None,
            soft_perm_ab=P12,
            soft_perm_ba=P21 if bidirectional else None,
        )

        if as_dict:
            return result.to_dict()
        return result
