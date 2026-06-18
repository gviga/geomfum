"""Models for learning features for functional maps for partial shape matching.

References
----------
.. "EchoMatch: Partial-to-Partial Shape Matching via Correspondence Reflection" by Xie et al., CVPR 2025.
"""

import gsops.backend as gs
import torch
import torch.nn as nn

from geomfum.convert import P2pFromFmConverter, SoftmaxNeighborFinder
from geomfum.descriptor.learned import FeatureExtractor, LearnedDescriptor
from geomfum.forward_functional_map import ForwardFunctionalMap
from geomfum.matcher.base import CorrespondenceResult

from ._base import BaseModel

# ---------------------------------------------------------------------------
# EchoMatch (Xie et al., CVPR 2025)
# ---------------------------------------------------------------------------


class EchoScorer(nn.Module):
    """Compute per-vertex overlap scores from the echo (reflection) matrix.

    The *echo matrix* for shape A is ``echo_a = P_ab @ P_ba`` — a square
    [n_a, n_a] matrix whose diagonal captures how well each vertex in A
    "reflects" back to itself through the correspondence field.

    For efficiency we aggregate over a fixed neighbourhood of the vertex
    rather than computing the full echo matrix, following EchoMatch.

    Parameters
    ----------
    neighbor_size : int, optional
        Number of nearest neighbours used to aggregate echo scores.
        Default 64.

    Notes
    -----
    This module has *no* learnable parameters; it is a deterministic
    computation.
    """

    def __init__(self, neighbor_size=64):
        super().__init__()
        self.neighbor_size = neighbor_size

    def forward(self, P_ab, P_ba, shape_a, shape_b):
        """Compute raw overlap scores for both shapes.

        Parameters
        ----------
        P_ab : Tensor[n_a, n_b]
        P_ba : Tensor[n_b, n_a]
        shape_a : Shape
        shape_b : Shape

        Returns
        -------
        score_a : Tensor[n_a, neighbor_size]  raw echo scores (k-NN features)
        score_b : Tensor[n_b, neighbor_size]
        """
        verts_a = gs.to_torch(shape_a.vertices).float().to(P_ab.device)  # [n_a, 3]
        verts_b = gs.to_torch(shape_b.vertices).float().to(P_ab.device)  # [n_b, 3]
        score_a = self._echo_scores(P_ab, P_ba, verts_a)  # [n_a, neighbor_size]
        score_b = self._echo_scores(P_ba, P_ab, verts_b)  # [n_b, neighbor_size]
        return score_a, score_b

    def _echo_scores(self, P_xy, P_yx, verts_x):
        """Compute [n_x, neighbor_size] echo scores via Euclidean k-NN.

        Replicates the EchoMatch overlap scoring: for each vertex i, gather the
        echo matrix values at the positions of its k geometrically nearest
        neighbours.  Pads with zeros when the shape is smaller than
        ``neighbor_size``.
        """
        n_x = verts_x.shape[0]
        k = min(self.neighbor_size, n_x)

        # Euclidean distances between all pairs of vertices
        dists_x = torch.cdist(verts_x, verts_x)  # [n_x, n_x]
        _, idx_x = torch.topk(dists_x, k, largest=False, sorted=True)  # [n_x, k]

        # Echo matrix: P_xy @ P_yx  [n_x, n_x]
        matrix_x = P_xy @ P_yx

        # Gather echo values at k-NN positions → [n_x, k]
        score_x = matrix_x.gather(1, idx_x)

        # Normalise
        max_val = score_x.max()
        if max_val > 0:
            score_x = score_x / max_val

        # Zero-pad if shape is smaller than neighbor_size
        if k < self.neighbor_size:
            pad_size = self.neighbor_size - k
            score_x = torch.cat(
                [score_x, torch.zeros(n_x, pad_size, device=score_x.device)],
                dim=1,
            )

        return score_x  # [n_x, neighbor_size]


class OverlapRefiner(nn.Module):
    """Refine raw echo overlap scores via a per-vertex feature extractor.

    The feature extractor (typically a small DiffusionNet) receives the raw
    per-vertex overlap scores as input features and outputs refined overlap
    probabilities in [0, 1] (sigmoid applied after the network).

    Parameters
    ----------
    feature_extractor : DiffusionnetFeatureExtractor or compatible, optional
        Feature extractor that exposes ``.model``, ``._get_operators``,
        ``.device``, and ``.k``.  The network must map 1 input channel to
        1 output channel.  Defaults to a lightweight DiffusionNet
        (1 → 1 channels, 16 hidden, 3 blocks, 128 eigenvectors).
    """

    def __init__(self, feature_extractor=None, neighbor_size=64):
        super().__init__()
        if feature_extractor is None:
            from geomfum.wrap.diffusionnet import DiffusionnetFeatureExtractor

            feature_extractor = DiffusionnetFeatureExtractor(
                in_channels=neighbor_size,
                out_channels=1,
                hidden_channels=16,
                n_block=3,
                dropout=False,
                k=128,
            )
        self.feature_extractor = feature_extractor

    def forward(self, scores, shape):
        """Refine overlap scores using the feature extractor.

        Parameters
        ----------
        scores : Tensor[n_vertices, k]
            Raw k-NN echo scores from ``EchoScorer`` (k = neighbor_size).
        shape : TriangleMesh
            Shape with pre-computed spectral basis.

        Returns
        -------
        Tensor[n_vertices]
            Refined overlap probabilities in [0, 1].
        """
        fe = self.feature_extractor
        frames, mass, L, evals, evecs, gradX, gradY = fe._get_operators(shape, k=fe.k)

        feat = scores.float().unsqueeze(0)  # [1, V, k]
        verts = (
            gs.to_torch(shape.vertices).float().to(fe.device).unsqueeze(0)
        )  # [1, V, 3]

        out = fe.model(
            verts,
            feats=feat,
            frames=frames.unsqueeze(0).float(),
            mass=mass.unsqueeze(0).float(),
            L=L.unsqueeze(0).float(),
            evals=evals.unsqueeze(0).float(),
            evecs=evecs.unsqueeze(0).float(),
            gradX=gradX.unsqueeze(0).float(),
            gradY=gradY.unsqueeze(0).float(),
        )  # [1, V, 1]

        return torch.sigmoid(out.squeeze())  # [V]


class EchoMatchNet(BaseModel):
    """EchoMatch: Partial-to-Partial Shape Matching via Correspondence Reflection.

    Combines a DiffusionNet feature extractor with bidirectional soft
    permutation matrices (echo matching) to jointly predict:
    - a functional map C_12 between the shapes,
    - per-vertex overlap scores (which vertices are in the shared region).

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Feature extractor for computing per-vertex descriptors.
    fmap_module : ForwardFunctionalMap
        Functional map solver.
    converter : P2pFromFmConverter
        Converts functional maps to point-to-point correspondences.
    neighbor_finder : SoftmaxNeighborFinder
        Computes soft bidirectional permutation matrices (P_ab, P_ba).
        Uses a separate temperature from the converter's neighbor_finder.
    echo_scorer : EchoScorer
        Aggregates echo matrix into per-vertex overlap scores.
    overlap_refiner : OverlapRefiner
        Refines raw overlap scores via spectral diffusion.

    References
    ----------
    Xie et al., "EchoMatch: Partial-to-Partial Shape Matching via
    Correspondence Reflection", CVPR 2025.
    """

    def __init__(
        self,
        feature_extractor=None,
        fmap_module=None,
        converter=None,
        neighbor_finder=None,
        echo_scorer=None,
        overlap_refiner=None,
    ):
        super().__init__()

        if feature_extractor is None:
            feature_extractor = FeatureExtractor.from_registry(which="diffusionnet")
        if fmap_module is None:
            fmap_module = ForwardFunctionalMap(lmbda=100.0, resolvent_gamma=0.5)
        if converter is None:
            converter = P2pFromFmConverter(
                SoftmaxNeighborFinder(n_neighbors=1, tau=0.07)
            )
        if neighbor_finder is None:
            neighbor_finder = SoftmaxNeighborFinder(n_neighbors=1, tau=0.10)
        if echo_scorer is None:
            echo_scorer = EchoScorer()
        if overlap_refiner is None:
            overlap_refiner = OverlapRefiner()

        self.feature_extractor = feature_extractor
        self.descriptors_module = LearnedDescriptor(
            feature_extractor=self.feature_extractor
        )
        self.fmap_module = fmap_module
        self.converter = converter
        self.neighbor_finder = neighbor_finder
        self.echo_scorer = echo_scorer
        self.overlap_refiner = overlap_refiner

    def forward(self, mesh_a, mesh_b, bidirectional=True, as_dict=False):
        """Compute correspondences and overlap scores between two shapes.

        Parameters
        ----------
        mesh_a : TriangleMesh
            First shape (target for p2p21).
        mesh_b : TriangleMesh
            Second shape (source for p2p21).
        bidirectional : bool, optional
            Compute both directions.  Default True.
        as_dict : bool, optional
            Return a dict instead of CorrespondenceResult.

        Returns
        -------
        CorrespondenceResult with additional fields:
            overlap_ab : Tensor[n_a] — predicted overlap in shape A
            overlap_ba : Tensor[n_b] — predicted overlap in shape B
        """
        # 1. Feature extraction
        desc_a = self.descriptors_module(mesh_a)  # [n_feat, n_a]
        desc_b = self.descriptors_module(mesh_b)  # [n_feat, n_b]

        # 2. Normalise features for permutation network
        desc_a_norm = desc_a / (gs.linalg.norm(desc_a, axis=0, keepdims=True) + 1e-8)
        desc_b_norm = desc_b / (gs.linalg.norm(desc_b, axis=0, keepdims=True) + 1e-8)

        # 3. Soft permutation matrices (same pattern as RobustFMNet)
        P_ab = self.neighbor_finder.softmax_matrix(desc_a_norm.T, desc_b_norm.T)
        P_ba = self.neighbor_finder.softmax_matrix(desc_b_norm.T, desc_a_norm.T)

        # 4. Raw echo overlap scores
        raw_overlap_a, raw_overlap_b = self.echo_scorer(P_ab, P_ba, mesh_a, mesh_b)

        # 5. Refine overlap via spectral diffusion
        overlap_ab = self.overlap_refiner(raw_overlap_a, mesh_a)  # [n_a]
        overlap_ba = self.overlap_refiner(raw_overlap_b, mesh_b)  # [n_b]

        # 6. Functional map
        fmap12, fmap21 = self.fmap_module(mesh_a, mesh_b, desc_a, desc_b)

        # 7. Point-to-point (eval only)
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
            overlap_ab=overlap_ab,
            overlap_ba=overlap_ba,  # always available; echo scores are computed both ways
        )

        if as_dict:
            return result.to_dict()
        return result
