"""DUO-FM: complex / orientation-aware deep functional maps (Donati et al.).

Predicts a real functional map (in the Laplace-Beltrami basis) *and* a complex
functional map (in the connection-Laplacian / complex basis, driven by the
spectral gradient of the features). The complex map provides orientation-aware
regularization.

The model mirrors :class:`~geomfum.learning.models.FMNet`: a
:class:`~geomfum.descriptor.learned.LearnedDescriptor` feeds two functional-map
modules — :class:`~geomfum.forward_functional_map.ForwardFunctionalMap` (real,
``mesh.basis``) and
:class:`~geomfum.forward_functional_map.ComplexForwardFunctionalMap` (complex,
``mesh.connection_laplacian.basis``) — and a converter (using the real map).

References
----------
.. Nicolas Donati, Etienne Corman, Maks Ovsjanikov. "Deep orientation-aware
    functional maps: Tackling symmetry issues in Shape Matching". (DUO-FM).
"""

from geomfum.convert import P2pFromFmConverter
from geomfum.descriptor.learned import FeatureExtractor, LearnedDescriptor
from geomfum.forward_functional_map import (
    ComplexForwardFunctionalMap,
    ForwardFunctionalMap,
)
from geomfum.matcher.base import CorrespondenceResult

from ._base import BaseModel


class DUOFMNet(BaseModel):
    """DUO functional-map network (real + complex maps).

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Backbone (default: DiffusionNet).
    n_fmap : int
        Real functional-map spectral size.
    n_cfmap : int
        Complex functional-map size (0 disables the complex branch).
    lambda_, resolvent_gamma : float
        Resolvent-regularization parameters (shared by both modules).
    converter : P2pFromFmConverter
        Functional-map to point-to-point converter (uses the real map).
    """

    def __init__(
        self,
        feature_extractor=None,
        n_fmap=50,
        n_cfmap=20,
        lambda_=1e-3,
        resolvent_gamma=0.5,
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
        self.fmap_module = ForwardFunctionalMap(
            lmbda=lambda_,
            resolvent_gamma=resolvent_gamma,
            bijective=True,
            fmap_shape=(n_fmap, n_fmap),
        )
        self.cfmap_module = (
            ComplexForwardFunctionalMap(
                lmbda=lambda_,
                resolvent_gamma=resolvent_gamma,
                bijective=True,
                n_cfmap=n_cfmap,
            )
            if n_cfmap > 0
            else None
        )
        self.n_fmap = n_fmap
        self.n_cfmap = n_cfmap
        self.converter = converter if converter is not None else P2pFromFmConverter()

    def forward(self, mesh_a, mesh_b, bidirectional=True, as_dict=False):
        """Compute real + complex functional maps between two shapes."""
        desc_a = self.descriptors_module(mesh_a)
        desc_b = self.descriptors_module(mesh_b)

        fmap12, fmap21 = self.fmap_module(mesh_a, mesh_b, desc_a, desc_b)

        cfmap12 = cfmap21 = None
        if self.cfmap_module is not None:
            cfmap12, cfmap21 = self.cfmap_module(mesh_a, mesh_b, desc_a, desc_b)

        p2p21 = p2p12 = None
        if not self.training:
            p2p21 = self.converter(fmap12, mesh_a.basis, mesh_b.basis)
            if bidirectional:
                p2p12 = self.converter(fmap21, mesh_b.basis, mesh_a.basis)

        result = CorrespondenceResult(
            fmap12=fmap12,
            fmap21=fmap21,
            p2p21=p2p21,
            p2p12=p2p12,
            descr_a=desc_a,
            descr_b=desc_b,
            cfmap12=cfmap12,
            cfmap21=cfmap21,
        )
        return result.to_dict() if as_dict else result
