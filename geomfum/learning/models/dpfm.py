"""Deep Partial Functional Maps (Attaiki et al., 2021).

A partial-shape correspondence model: DiffusionNet features are refined by
cross-attention between the two shapes, a per-vertex overlap probability is
predicted, and a resolvent-regularized functional map is solved from the
refined features.

Faithful self-contained port of the core of https://github.com/pvnieo/DPFM
(``dpfm/model.py``), adapted to geomfum's single-pair model interface. Uses
full cross-attention (``cross_sampling_ratio = 1``) so no precomputed FPS
samples are needed.

References
----------
.. Souhaib Attaiki, Gautam Pai, Maks Ovsjanikov. "DPFM: Deep Partial Functional
    Maps". 3DV 2021.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomfum.convert import P2pFromFmConverter
from geomfum.descriptor.learned import FeatureExtractor
from geomfum.matcher.base import CorrespondenceResult

from ._base import BaseModel


def _as_float_tensor(x):
    """Convert a numpy array or (CPU/CUDA) torch tensor to a float32 tensor."""
    if isinstance(x, torch.Tensor):
        return x.float()
    import numpy as np

    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _fmap_for_basis(fmap, basis):
    """Return ``fmap`` as the backend type/device of ``basis.full_vecs``."""
    fv = basis.full_vecs
    if isinstance(fv, torch.Tensor):
        return fmap.detach().to(device=fv.device, dtype=fv.dtype)
    return fmap.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Helpers (ported from DPFM/dpfm/model.py and utils.py)
# ---------------------------------------------------------------------------
def _mlp(channels, do_bn=True):
    layers = []
    for i in range(1, len(channels)):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], 1, bias=True))
        if i < len(channels) - 1:
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value)


def _resolvent_mask(evals1, evals2, gamma=0.5):
    """Compute the resolvent regularization mask (as in ForwardFunctionalMap)."""
    # LBO eigenvalues can be slightly negative (~1e-14) from float error; clamp
    # before the fractional power to avoid NaN.
    evals1 = torch.clamp(evals1, min=0.0)
    evals2 = torch.clamp(evals2, min=0.0)
    scaling = max(torch.max(evals1), torch.max(evals2))
    e1, e2 = evals1 / scaling, evals2 / scaling
    g1, g2 = (e1**gamma)[None, :], (e2**gamma)[:, None]
    m_re = g2 / (g2.square() + 1) - g1 / (g1.square() + 1)
    m_im = 1 / (g2.square() + 1) - 1 / (g1.square() + 1)
    return m_re.square() + m_im.square()


class _MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, 1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [
            ll(x).view(b, self.dim, self.num_heads, -1)
            for ll, x in zip(self.proj, (query, key, value))
        ]
        x = _attention(query, key, value)
        return self.merge(x.contiguous().view(b, self.dim * self.num_heads, -1))


class _AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.attn = _MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = _mlp([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class _OverlapPredictorNet(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(feat_dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat_x, feat_y):
        nx = F.normalize(feat_x, p=2, dim=-1)
        ny = F.normalize(feat_y, p=2, dim=-1)
        return self.net(nx).squeeze(2).squeeze(0), self.net(ny).squeeze(2).squeeze(0)


class _CrossAttentionRefinementNet(nn.Module):
    """Full cross-attention feature refinement + overlap prediction."""

    def __init__(self, n_in=128, num_head=4, gnn_dim=512, n_layers=2):
        super().__init__()
        self.n_in = n_in
        self.first_lin = nn.Linear(n_in, gnn_dim)
        self.layers = nn.ModuleList(
            [_AttentionalPropagation(gnn_dim, num_head) for _ in range(n_layers)]
        )
        self.last_lin = nn.Linear(gnn_dim, n_in)
        self.overlap_predictor = _OverlapPredictorNet(n_in)

    def forward(self, features_x, features_y):
        desc0 = self.first_lin(features_x).transpose(1, 2)  # (1, gnn_dim, nx)
        desc1 = self.first_lin(features_y).transpose(1, 2)
        for layer in self.layers:
            desc0 = desc0 + layer(desc0, desc1)
            desc1 = desc1 + layer(desc1, desc0)
        ref_x = self.last_lin(desc0.transpose(1, 2))  # (1, nx, n_in)
        ref_y = self.last_lin(desc1.transpose(1, 2))
        ov_x, ov_y = self.overlap_predictor(ref_x, ref_y)
        return ref_x, ref_y, ov_x, ov_y


class _RegularizedFMNet(nn.Module):
    """Resolvent-regularized functional map from features."""

    def __init__(self, lambda_=1e-3, resolvent_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvent_gamma = resolvent_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        f_hat = evecs_trans_x @ feat_x[0]  # (k, C)
        g_hat = evecs_trans_y @ feat_y[0]
        a, b = f_hat, g_hat
        d = _resolvent_mask(evals_x, evals_y, self.resolvent_gamma)  # (k_y, k_x)
        a_t = a.transpose(0, 1)
        a_a_t = a @ a_t
        b_a_t = b @ a_t
        rows = []
        for i in range(evals_y.size(0)):
            lhs = a_a_t + self.lambda_ * torch.diag(d[i, :])
            rows.append((torch.inverse(lhs) @ b_a_t[i, :].unsqueeze(1)).transpose(0, 1))
        return torch.cat(rows, dim=0)  # (k_y, k_x)


class DPFMNet(BaseModel):
    """Deep Partial Functional Maps network.

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Backbone (default: DiffusionNet) producing per-vertex features.
    n_fmap : int
        Functional-map spectral size.
    num_head, gnn_dim, n_layers : int
        Cross-attention refinement hyper-parameters.
    lambda_ : float
        Resolvent regularization weight.
    resolvent_gamma : float
        Resolvent exponent.
    robust : bool
        If True, the fmap uses the cross-attention-refined features; otherwise
        the raw backbone features.
    converter : P2pFromFmConverter
        Functional-map to point-to-point converter.
    """

    def __init__(
        self,
        feature_extractor=None,
        n_fmap=50,
        num_head=4,
        gnn_dim=512,
        n_layers=2,
        lambda_=1e-3,
        resolvent_gamma=0.5,
        robust=True,
        converter=None,
    ):
        super().__init__()
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else FeatureExtractor.from_registry(which="diffusionnet")
        )
        n_feat = self.feature_extractor.out_channels
        # geomfum sets a float64 default dtype, but the DiffusionNet backbone is
        # float32; keep the new submodules float32 to match the features.
        self.feat_refiner = _CrossAttentionRefinementNet(
            n_in=n_feat, num_head=num_head, gnn_dim=gnn_dim, n_layers=n_layers
        ).float()
        self.fmreg_net = _RegularizedFMNet(lambda_, resolvent_gamma).float()
        self.n_fmap = n_fmap
        self.robust = robust
        self.converter = converter if converter is not None else P2pFromFmConverter()

    @staticmethod
    def _spectral(mesh, k):
        """Return evecs (n,k), evals (k,), and evecs_trans (k,n)=evecs^T diag(mass)."""
        evecs = _as_float_tensor(mesh.basis.full_vecs)[:, :k]
        evals = _as_float_tensor(mesh.basis.full_vals)[:k].to(evecs.device)
        mass = _as_float_tensor(mesh.vertex_areas).reshape(-1).to(evecs.device)
        evecs_trans = evecs.transpose(0, 1) * mass[None, :]  # (k, n), avoids dense diag
        return evecs, evals, evecs_trans

    def forward(self, mesh_a, mesh_b, bidirectional=True, as_dict=False):
        """Compute the partial functional map + overlap scores between shapes."""
        k = self.n_fmap
        feat_a = self.feature_extractor(mesh_a).squeeze().float().unsqueeze(0)  # (1,na,C)
        feat_b = self.feature_extractor(mesh_b).squeeze().float().unsqueeze(0)

        ref_a, ref_b, overlap_a, overlap_b = self.feat_refiner(feat_a, feat_b)
        use_a, use_b = (ref_a, ref_b) if self.robust else (feat_a, feat_b)

        evecs_a, evals_a, et_a = self._spectral(mesh_a, k)
        evecs_b, evals_b, et_b = self._spectral(mesh_b, k)

        fmap12 = self.fmreg_net(use_a, use_b, evals_a, evals_b, et_a, et_b)
        fmap21 = None
        p2p21 = p2p12 = None
        if bidirectional:
            fmap21 = self.fmreg_net(use_b, use_a, evals_b, evals_a, et_b, et_a)
        if not self.training:
            mesh_a.basis.use_k = k
            mesh_b.basis.use_k = k
            p2p21 = self.converter(
                _fmap_for_basis(fmap12, mesh_a.basis), mesh_a.basis, mesh_b.basis
            )
            if bidirectional:
                p2p12 = self.converter(
                    _fmap_for_basis(fmap21, mesh_b.basis), mesh_b.basis, mesh_a.basis
                )

        result = CorrespondenceResult(
            fmap12=fmap12,
            p2p21=p2p21,
            fmap21=fmap21,
            p2p12=p2p12,
            descr_a=use_a.squeeze(0),
            descr_b=use_b.squeeze(0),
            overlap_ab=overlap_a,
            overlap_ba=overlap_b,
        )
        return result.to_dict() if as_dict else result
