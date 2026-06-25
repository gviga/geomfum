"""Attentive Functional Maps (Li et al., 2022).

Learns multi-resolution functional maps with spectral attention: a functional
map is solved at several spectral resolutions, a small attention network scores
each resolution from its per-vertex matching residual, every resolution is
upsampled (differentiable ZoomOut) to the largest size, and the maps are
combined by the attention weights.

Faithful self-contained port of the core of
https://github.com/craigleili/AttentiveFMaps (``models/attnfmaps.py`` and
``models/utils.py``), adapted to geomfum's single-pair model interface.

References
----------
.. Lei Li, Nicolas Donati, Maks Ovsjanikov. "Learning Multi-resolution
    Functional Maps with Spectral Attention for Robust Shape Matching".
    NeurIPS 2022.
"""

import torch
import torch.nn as nn

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
# Helpers (ported from AttentiveFMaps/models/utils.py)
# ---------------------------------------------------------------------------
def _pdists(x, y, squared=False, eps=1e-12):
    """Pairwise distances between rows of ``x`` and ``y`` (batched)."""
    x2 = torch.sum(x**2, dim=-1, keepdim=True)
    y2 = torch.sum(y**2, dim=-1, keepdim=True)
    dist2 = -2.0 * torch.matmul(x, torch.transpose(y, -2, -1))
    dist2 = dist2 + x2 + torch.transpose(y2, -2, -1)
    if squared:
        return dist2
    return torch.sqrt(torch.clamp(dist2, min=eps))


def _wlstsq(a, b):
    """Batched least squares ``min_X ||A X - B||``."""
    return torch.linalg.lstsq(a, b).solution


def _fmap_reg(evecs0, evecs1, evals0, evals1, mass0, mass1, feats0, feats1, reg):
    """Regularized (resolvent) functional map solve, batched (B, k1, k0)."""
    a = torch.transpose(evecs0, 1, 2) @ (torch.unsqueeze(mass0, 2) * feats0)
    b = torch.transpose(evecs1, 1, 2) @ (torch.unsqueeze(mass1, 2) * feats1)
    aat = a @ torch.transpose(a, 1, 2)
    delta = (torch.unsqueeze(evals1, 2) - torch.unsqueeze(evals0, 1)) ** 2

    rows = []
    for ridx in range(evals1.size(-1)):
        lhs = aat + torch.diag_embed(reg * delta[:, ridx, :])
        rhs = a @ torch.transpose(b[:, ridx : ridx + 1, :], 1, 2)
        rows.append(torch.transpose(torch.linalg.inv(lhs) @ rhs, 1, 2))
    return torch.cat(rows, dim=1)


class _DiffNNSearch(nn.Module):
    """Differentiable nearest-neighbour (soft in train, straight-through eval)."""

    def __init__(self, temp_init=1.0, temp_min=1e-4):
        super().__init__()
        self.temp_min = temp_min
        self.temp = nn.Parameter(torch.tensor(temp_init, dtype=torch.float32))

    def _t(self):
        return torch.clamp(self.temp**2, min=self.temp_min)

    def forward(self, feats0, feats1):
        dists = _pdists(feats0, feats1, squared=True)
        dists = torch.softmax(-dists / self._t(), dim=-1)
        _, indices = torch.max(dists, dim=-1, keepdim=True)
        if self.training:
            asgn = dists
        else:
            hard = torch.zeros_like(dists).scatter_(-1, indices, 1.0)
            asgn = hard - dists.detach() + dists
        return asgn, torch.squeeze(indices, dim=-1)


def _diff_zoomout(evecs0, evecs1, fmap01, fmap_sizes, nnsearcher):
    """Differentiable ZoomOut upsampling from ``fmap_sizes[0]`` to ``[-1]``."""
    cur = fmap01
    for i in range(len(fmap_sizes) - 1):
        fs = fmap_sizes[i]
        corr10_mat, _ = nnsearcher(
            evecs1[..., :fs], evecs0[..., :fs] @ torch.transpose(cur, -2, -1)
        )
        fs = fmap_sizes[i + 1]
        cur = _wlstsq(evecs1[..., :fs], corr10_mat @ evecs0[..., :fs])
    return cur


# ---------------------------------------------------------------------------
# Spectral attention network (ported from models/attnfmaps.py)
# ---------------------------------------------------------------------------
class _SEBlock(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):  # x: (B, C, N)
        w = torch.mean(x, dim=2)
        w = self.fc(w).unsqueeze(2)
        return x * w


class _PointConvBlock(nn.Module):
    def __init__(self, in_c, out_c, use_norm=True, use_act=True, use_se=False):
        super().__init__()
        layers = [nn.Conv1d(in_c, out_c, 1, bias=not use_norm)]
        if use_norm:
            layers.append(nn.BatchNorm1d(out_c))
        if use_act:
            layers.append(nn.ReLU(inplace=True))
        if use_se:
            layers.append(_SEBlock(out_c, reduction=4))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _FeatureSTN(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Sequential(
            _PointConvBlock(in_c, 64),
            _PointConvBlock(64, 128),
            _PointConvBlock(128, 256),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, in_c * in_c),
        )
        self.in_c = in_c

    def forward(self, x):  # x: (B, C, N)
        b = x.shape[0]
        f = torch.max(self.conv(x), dim=2)[0]
        mat = self.fc(f).view(b, self.in_c, self.in_c)
        eye = torch.eye(self.in_c, device=x.device, dtype=x.dtype).unsqueeze(0)
        return mat + eye


class _SpectralAttentionNet(nn.Module):
    """Score each spectral resolution from its matching residual."""

    def __init__(self, nfeatures, spectral_dims):
        super().__init__()
        self.spectral_dims = spectral_dims
        fdim = 64
        self.mlp0 = _PointConvBlock(len(spectral_dims), fdim)
        self.fstn = _FeatureSTN(fdim)
        self.mlp1 = nn.Sequential(
            _PointConvBlock(fdim, nfeatures),
            _PointConvBlock(nfeatures, nfeatures, use_act=False),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(nfeatures, nfeatures),
            nn.ReLU(True),
            nn.Linear(nfeatures, len(spectral_dims)),
        )

    def forward(self, evecs0, evecs1, mass1, fmaps01):
        # residual per resolution: how well each fmap aligns the embeddings.
        residuals = []
        for fm in fmaps01:
            k1, k0 = fm.shape[-2:]
            e0 = evecs0[..., :k0] @ torch.transpose(fm.detach(), -2, -1)
            e1 = evecs1[..., :k1]
            res, _ = torch.min(_pdists(e1, e0, squared=False), dim=-1)
            residuals.append(res / k1**0.5)
        residuals = torch.stack(residuals, dim=1)  # (B, n_res, N1)

        ft = self.mlp0(residuals)
        tsfm = self.fstn(ft)
        ft = (torch.transpose(ft, 1, 2) @ tsfm).transpose(1, 2).contiguous()
        ft = self.mlp1(ft)
        ft = torch.sum(ft * torch.unsqueeze(mass1, 1), dim=-1) / torch.sum(
            mass1, dim=-1, keepdim=True
        )
        return self.mlp2(ft)  # (B, n_res) attention logits


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AttentiveFMNet(BaseModel):
    """Attentive Functional Maps network.

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Backbone (default: DiffusionNet) producing per-vertex features.
    spectral_dims : list[int]
        Functional-map resolutions to combine (e.g. ``[20, 40, ..., 120]``).
    reg : float
        Resolvent regularization weight in the per-resolution fmap solve.
    nfeatures : int
        Width of the spectral-attention network.
    converter : P2pFromFmConverter
        Functional-map to point-to-point converter.
    """

    def __init__(
        self,
        feature_extractor=None,
        spectral_dims=(20, 40, 60, 80, 100, 120),
        reg=1e-3,
        nfeatures=128,
        converter=None,
    ):
        super().__init__()
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else FeatureExtractor.from_registry(which="diffusionnet")
        )
        self.spectral_dims = list(spectral_dims)
        self.reg = reg
        # geomfum sets the default torch dtype to float64, but the DiffusionNet
        # features (and the spectral tensors) are float32; keep the attention
        # subnet float32 so its conv/linear weights match the inputs.
        self.attention = _SpectralAttentionNet(nfeatures, self.spectral_dims).float()
        self.nnsearcher = _DiffNNSearch()
        self.converter = converter if converter is not None else P2pFromFmConverter()

    @staticmethod
    def _spectral(mesh, max_k):
        """Return (evecs, evals, lumped-mass) from a mesh basis as float tensors."""
        evecs = _as_float_tensor(mesh.basis.full_vecs)[:, :max_k]
        evals = _as_float_tensor(mesh.basis.full_vals)[:max_k].to(evecs.device)
        mass = _as_float_tensor(mesh.vertex_areas).reshape(-1).to(evecs.device)
        return evecs, evals, mass

    def forward(self, mesh_a, mesh_b, bidirectional=True, as_dict=False):
        """Compute the attentive functional map between two shapes."""
        max_k = self.spectral_dims[-1]

        feats_a = self.feature_extractor(mesh_a).squeeze().float()  # (na, C)
        feats_b = self.feature_extractor(mesh_b).squeeze().float()

        evecs_a, evals_a, mass_a = self._spectral(mesh_a, max_k)
        evecs_b, evals_b, mass_b = self._spectral(mesh_b, max_k)

        def _combined(evecs0, evals0, mass0, feats0, evecs1, evals1, mass1, feats1):
            # batch dim of 1 to reuse the batched modules.
            e0, e1 = evecs0[None], evecs1[None]
            ev0, ev1 = evals0[None], evals1[None]
            m0, m1 = mass0[None], mass1[None]
            f0, f1 = feats0[None], feats1[None]

            # Multi-resolution fmaps (solve at max, slice — "fast solve").
            fmap_full = _fmap_reg(
                e0[..., :max_k], e1[..., :max_k], ev0, ev1, m0, m1, f0, f1, self.reg
            )
            fmaps_init = [fmap_full[..., :sd, :sd] for sd in self.spectral_dims]

            logits = self.attention(e0, e1, m1, fmaps_init)
            attn = torch.softmax(logits, dim=1)  # (1, n_res)

            fmaps_dzo = [
                _diff_zoomout(e0, e1, fmaps_init[i], [sd, max_k], self.nnsearcher)
                for i, sd in enumerate(self.spectral_dims)
            ]
            stacked = torch.stack(fmaps_dzo, dim=1)  # (1, n_res, k, k)
            fmap = torch.sum(attn.view(1, -1, 1, 1) * stacked, dim=1)[0]
            return fmap  # (max_k, max_k)

        fmap12 = _combined(
            evecs_a, evals_a, mass_a, feats_a, evecs_b, evals_b, mass_b, feats_b
        )
        fmap21 = None
        p2p21 = p2p12 = None
        if not self.training:
            mesh_a.basis.use_k = max_k
            mesh_b.basis.use_k = max_k
            p2p21 = self.converter(
                _fmap_for_basis(fmap12, mesh_a.basis), mesh_a.basis, mesh_b.basis
            )
        if bidirectional:
            fmap21 = _combined(
                evecs_b, evals_b, mass_b, feats_b, evecs_a, evals_a, mass_a, feats_a
            )
            if not self.training:
                p2p12 = self.converter(
                    _fmap_for_basis(fmap21, mesh_b.basis), mesh_b.basis, mesh_a.basis
                )

        result = CorrespondenceResult(
            fmap12=fmap12,
            p2p21=p2p21,
            fmap21=fmap21,
            p2p12=p2p12,
            descr_a=feats_a,
            descr_b=feats_b,
        )
        return result.to_dict() if as_dict else result
