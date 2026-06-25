"""Hybrid Functional Maps (Xie et al., CVPR 2024).

Solves a functional map in *two* bases — the Laplace-Beltrami basis
(:class:`~geomfum.forward_functional_map.ForwardFunctionalMap`) and the
crease-aware *elastic* basis
(:class:`~geomfum.forward_functional_map.ElasticForwardFunctionalMap`) — and, at
inference, concatenates the two spectral embeddings for a hybrid
nearest-neighbour correspondence.

The model mirrors :class:`~geomfum.learning.models.FMNet`: a
:class:`~geomfum.descriptor.learned.LearnedDescriptor` feeds two functional-map
modules and a converter; the bases come from the shape operators
(``mesh.basis`` and ``mesh.elastic_hessian.basis``).

References
----------
.. Yizheng Xie, Dongliang Cao, Florine Hartwig, Florian Bernard, et al.
    "Hybrid Functional Maps for Crease-Aware Non-Isometric Shape Matching".
    CVPR 2024.
"""

import gsops.backend as gs
from scipy.spatial import cKDTree

from geomfum.convert import P2pFromFmConverter
from geomfum.descriptor.learned import FeatureExtractor, LearnedDescriptor
from geomfum.forward_functional_map import (
    ElasticForwardFunctionalMap,
    ForwardFunctionalMap,
)
from geomfum.matcher.base import CorrespondenceResult

from ._base import BaseModel


class HybridFMNet(BaseModel):
    """Hybrid Functional-Map network (Laplace-Beltrami + elastic bases).

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Backbone (default: DiffusionNet).
    n_fmap : int
        Laplace-Beltrami functional-map size.
    n_elas : int
        Elastic functional-map size.
    lambda_lb, lambda_elas : float
        Resolvent regularization weights for the two functional-map modules.
    resolvent_gamma : float
        Resolvent exponent.
    bending_weight : float
        Bending vs membrane weight for the elastic basis.
    converter : P2pFromFmConverter
        Functional-map to point-to-point converter (used per basis if needed).
    """

    def __init__(
        self,
        feature_extractor=None,
        n_fmap=20,
        n_elas=10,
        lambda_lb=100.0,
        lambda_elas=100.0,
        resolvent_gamma=0.5,
        bending_weight=1e-2,
        converter=None,
    ):
        super().__init__()
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else FeatureExtractor.from_registry(which="diffusionnet")
        )
        self.descriptors_module = LearnedDescriptor(
            feature_extractor=self.feature_extractor
        )
        self.lb_fmap_module = ForwardFunctionalMap(
            lmbda=lambda_lb,
            resolvent_gamma=resolvent_gamma,
            bijective=True,
            fmap_shape=(n_fmap, n_fmap),
        )
        self.elas_fmap_module = ElasticForwardFunctionalMap(
            lmbda=lambda_elas,
            resolvent_gamma=resolvent_gamma,
            bijective=True,
            n_elas=n_elas,
            bending_weight=bending_weight,
        )
        self.converter = converter if converter is not None else P2pFromFmConverter()

    def _hybrid_p2p(self, mesh_a, mesh_b, fmap_ab, elastic_fmap_ab):
        """Point-to-point from the concatenated LB + elastic spectral embeddings.

        The embedding is built with backend (``gs``) ops; this is an
        inference-only nearest-neighbour query, so only the final ``cKDTree``
        lookup (scipy) drops to cpu numpy. Returns ``p2p_ba`` (for each vertex of
        ``mesh_b``, the nearest of ``mesh_a``).
        """
        # Laplace-Beltrami embedding (align the fmap dtype to the basis).
        dtype = mesh_a.basis.vecs.dtype
        lb_a = mesh_a.basis.vecs @ gs.cast(fmap_ab, dtype).T
        lb_b = mesh_b.basis.vecs

        # Elastic embedding (reduced-mass aware), using the elastic eigenbasis.
        basis_a, basis_b = mesh_a.elastic_hessian.basis, mesh_b.elastic_hessian.basis
        mk_a, mk_b = basis_a.reduced_mass, basis_b.reduced_mass
        inv_sqrt_mk_b = gs.linalg.inv(basis_b.sqrt_reduced_mass)
        el_a = (
            basis_a.vecs
            @ gs.linalg.inv(mk_a)
            @ gs.cast(elastic_fmap_ab, mk_a.dtype).T
            @ mk_b
            @ inv_sqrt_mk_b
        )
        el_b = basis_b.vecs @ inv_sqrt_mk_b

        emb_a = gs.to_numpy(gs.to_device(gs.concatenate([el_a, lb_a], axis=1), "cpu"))
        emb_b = gs.to_numpy(gs.to_device(gs.concatenate([el_b, lb_b], axis=1), "cpu"))
        _, p2p_ba = cKDTree(emb_a).query(emb_b)
        return gs.from_numpy(p2p_ba)

    def forward(self, mesh_a, mesh_b, bidirectional=True, as_dict=False):
        """Compute LB + elastic functional maps and the hybrid correspondence."""
        desc_a = self.descriptors_module(mesh_a)
        desc_b = self.descriptors_module(mesh_b)

        fmap12, fmap21 = self.lb_fmap_module(mesh_a, mesh_b, desc_a, desc_b)
        elas12, elas21 = self.elas_fmap_module(mesh_a, mesh_b, desc_a, desc_b)

        p2p21 = p2p12 = None
        if not self.training:
            p2p21 = self._hybrid_p2p(mesh_a, mesh_b, fmap12, elas12)
            if bidirectional:
                p2p12 = self._hybrid_p2p(mesh_b, mesh_a, fmap21, elas21)

        result = CorrespondenceResult(
            fmap12=fmap12,
            fmap21=fmap21,
            p2p21=p2p21,
            p2p12=p2p12,
            descr_a=desc_a,
            descr_b=desc_b,
            elastic_fmap12=elas12,
            elastic_fmap21=elas21,
        )
        return result.to_dict() if as_dict else result
