"""Optimization of the functional map with a forward pass."""

import abc

import gsops.backend as gs
import torch.nn as nn


def _kron(a, b):
    """Kronecker product of two matrices via broadcasting.

    Backend-agnostic and works on transposed/non-contiguous views, unlike
    ``gs.kron`` which requires contiguous operands under the pytorch backend.
    """
    m, n = a.shape
    p, q = b.shape
    return gs.reshape(a[:, None, :, None] * b[None, :, None, :], (m * p, n * q))


class ForwardFunctionalMap(abc.ABC, nn.Module):
    """Class for the forward pass of the functional map.

    Parameters
    ----------
    lmbda : float
        Weight of the mask (default: 1e3).
    resolvent_gamma: float
        Resolvant of the regularized functional map (default: 1).
    bijective: bool
        Whether we compute the map in both the directions (default: True).
    fmap_shape: tuple, optional
        Shape of fmap12, i.e (spectrum_size_b, spectrum_size_a). If None, the shape is inferred from the input shapes.
    """

    def __init__(self, lmbda=1e3, resolvent_gamma=1, bijective=True, fmap_shape=None):
        super(ForwardFunctionalMap, self).__init__()
        self.lmbda = lmbda
        self.resolvent_gamma = resolvent_gamma
        self.bijective = bijective
        self.fmap_shape = fmap_shape

    def _compute_functional_map(self, sdescr_a, sdescr_b, mask):
        """Compute the functional map between two shapes.

        Parameters
        ----------
        sdescr_a : array-like, shape=[..., spectrum_size_a]
            Spectral descriptors on first basis.
        sdescr_b : array-like, shape=[..., spectrum_size_b]
            Spectral descriptors on second basis.
        mask: array-like, shape=[..., spectrum_size_b, spectrum_size_a]
            Mask for the functional map.

        Returns
        -------
            fmap12 : array-like, shape=[..., spectrum_size_b, spectrum_size_a]
                Functional map from shape a to shape b.
        """
        At_A = sdescr_a.T @ sdescr_a
        Bt_A = sdescr_b.T @ sdescr_a

        fmap = []
        for i in range(mask.shape[0]):
            if self.lmbda == 0:
                map_row = gs.linalg.inv(At_A) @ Bt_A[i, :].reshape(-1, 1)
            else:
                MASK_i = gs.diag(mask[i, :].flatten())
                map_row = gs.linalg.inv(At_A + self.lmbda * MASK_i) @ Bt_A[
                    i, :
                ].reshape(-1, 1)
            fmap.append(map_row.T)

        fmap = gs.concatenate(fmap, 0)

        return fmap

    def __call__(self, mesh_a, mesh_b, descr_a, descr_b):
        """Compute the functional map between two shapes.

        Parameters
        ----------
        mesh_a : TriangleMesh
            Mesh object representing the first shape.
        mesh_b : TriangleMesh
            Mesh object representing the second shape.
        descr_a : array-like, shape=[D, ...]
            Spectral descriptors on the first shape.
        descr_b : array-like, shape=[D, ...]
            Spectral descriptors on the second shape.

        Returns
        -------
        fmap_12 : array-like, shape[spectrum_size_b, spectrum_size_a]
            Functional map from shape a to shape b.
        fmap_21: array-like, shape=[spectrum_size_a, spectrum_size_b] or None
            Functional map from shape b to shape a if bijective, otherwise None.
        """
        if self.fmap_shape is not None:
            mesh_a.basis.use_k = self.fmap_shape[1]
            mesh_b.basis.use_k = self.fmap_shape[0]

        evals_a = mesh_a.basis.vals
        sdescr_a = mesh_a.basis.project(descr_a)
        evals_b = mesh_b.basis.vals
        sdescr_b = mesh_b.basis.project(descr_b)

        mask = self._compute_mask(evals_a, evals_b, self.resolvent_gamma)
        fmap_12 = self._compute_functional_map(sdescr_a, sdescr_b, mask)
        fmap_21 = None
        if self.bijective:
            mask = self._compute_mask(evals_b, evals_a, self.resolvent_gamma)
            fmap_21 = self._compute_functional_map(sdescr_b, sdescr_a, mask)
        return fmap_12, fmap_21

    def _compute_mask(self, evals_a, evals_b, resolvant_gamma):
        """Compute the mask for the functional map.

        Parameters
        ----------
        evals_a : array-like, shape=[..., spectrum_size_a]
            Eigenvalues of the first shape.
        evals_b : array-like, shape=[..., spectrum_size_b]
            Eigenvalues of the second shape.
        resolvant_gamma : float
            Resolvent of the regularized functional map.

        Returns
        -------
        mask : array-like, shape=[..., spectrum_size_b, spectrum_size_a]
            Mask for the functional map.
        """
        evals_a = gs.array(evals_a)
        evals_b = gs.array(evals_b)

        # Laplacian eigenvalues are theoretically non-negative, but can be
        # slightly negative (~1e-14) due to floating-point precision.
        # Clamp before raising to a fractional power to avoid NaN.
        evals_a = gs.clip(evals_a, 0, None)
        evals_b = gs.clip(evals_b, 0, None)

        scaling_factor = max(max(evals_a), max(evals_b))
        evals_a, evals_b = evals_a / scaling_factor, evals_b / scaling_factor
        evals_gamma_a = gs.power(evals_a, resolvant_gamma)[None, :]
        evals_gamma_b = gs.power(evals_b, resolvant_gamma)[:, None]
        M_re = evals_gamma_b / (gs.square(evals_gamma_b) + 1) - evals_gamma_a / (
            gs.square(evals_gamma_a) + 1
        )
        M_im = 1 / (gs.square(evals_gamma_b) + 1) - 1 / (gs.square(evals_gamma_a) + 1)
        return gs.square(M_re) + gs.square(M_im)


class ElasticForwardFunctionalMap(ForwardFunctionalMap):
    """Forward functional map in the (non-orthonormal) elastic basis.

    Companion to :class:`ForwardFunctionalMap` for Hybrid Functional Maps
    (Xie et al.). Because the elastic basis ``shape.elastic_hessian.basis`` is
    not mass-orthonormal, the map is solved with a Hilbert-Schmidt-adapted
    resolvent system using the reduced-mass square root ``sqrt(Mk)`` exposed by
    :class:`~geomfum.basis.ElasticEigenBasis`.

    Parameters
    ----------
    lmbda : float
        Resolvent regularization weight.
    resolvent_gamma : float
        Resolvent exponent.
    bijective : bool
        Whether to also compute the reverse map.
    n_elas : int
        Elastic functional-map size.
    bending_weight : float
        Bending vs membrane weight for the elastic basis.
    """

    def __init__(
        self, lmbda=100.0, resolvent_gamma=0.5, bijective=True, n_elas=10, bending_weight=1e-2
    ):
        super().__init__(lmbda, resolvent_gamma, bijective)
        self.n_elas = n_elas
        self.bending_weight = bending_weight

    def elastic_basis(self, mesh):
        """Return the (cached) elastic eigenbasis truncated to ``n_elas``."""
        operator = mesh.elastic_hessian
        operator.bending_weight = self.bending_weight
        if operator._basis is None or operator.basis.full_spectrum_size < self.n_elas:
            operator.find_spectrum(spectrum_size=self.n_elas, recompute=True)
        basis = operator.basis
        basis.use_k = self.n_elas
        return basis

    def _compute_functional_map(self, a, b, evals_a, evals_b, sqrt_a, sqrt_b):
        """Solve the expanded resolvent system in the reduced-mass metric.

        Parameters
        ----------
        a, b : array-like, shape=[k, n_features]
            Projected descriptors ``evecs_trans @ features`` for each shape.
        evals_a, evals_b : array-like, shape=[k]
        sqrt_a, sqrt_b : array-like, shape=[k, k]
            Square roots of the reduced mass ``Mk``.
        """
        k = a.shape[0]
        b = sqrt_b @ b
        at_ik = _kron(a.T, sqrt_b)
        vec_b = gs.reshape(b.T, (-1, 1))
        first = at_ik.T @ at_ik

        inv_sqrt_a = gs.linalg.inv(sqrt_a)
        scaling = max(gs.amax(evals_a), gs.amax(evals_b))
        g1 = (evals_a / scaling) ** self.resolvent_gamma
        g2 = (evals_b / scaling) ** self.resolvent_gamma
        rn1_re, rn1_im = g1 / (gs.square(g1) + 1), 1 / (gs.square(g1) + 1)
        rn2_re, rn2_im = g2 / (gs.square(g2) + 1), 1 / (gs.square(g2) + 1)

        lx_re = _kron(gs.diag(rn1_re) @ inv_sqrt_a, sqrt_b)
        lx_im = _kron(gs.diag(rn1_im) @ inv_sqrt_a, sqrt_b)
        ly_re = _kron(inv_sqrt_a, sqrt_b @ gs.diag(rn2_re))
        ly_im = _kron(inv_sqrt_a, sqrt_b @ gs.diag(rn2_im))
        delta_re, delta_im = ly_re - lx_re, ly_im - lx_im
        second = delta_re.T @ delta_re + delta_im.T @ delta_im

        op = first + self.lmbda * second
        c = gs.linalg.solve(op, at_ik.T @ vec_b)
        return gs.reshape(c, (k, k)).T

    def __call__(self, mesh_a, mesh_b, descr_a, descr_b):
        """Compute the elastic functional map(s) between two shapes."""
        basis_a = self.elastic_basis(mesh_a)
        basis_b = self.elastic_basis(mesh_b)

        # The basis is backend-native; align its dtype to the (learned) features.
        # projected descriptors A = evecs_trans @ features  (features = descr.T)
        a = gs.cast(basis_a.pinv, descr_a.dtype) @ descr_a.T
        b = gs.cast(basis_b.pinv, descr_b.dtype) @ descr_b.T
        evals_a = gs.cast(basis_a.vals, descr_a.dtype)
        evals_b = gs.cast(basis_b.vals, descr_b.dtype)
        sqrt_a = gs.cast(basis_a.sqrt_reduced_mass, descr_a.dtype)
        sqrt_b = gs.cast(basis_b.sqrt_reduced_mass, descr_b.dtype)

        fmap_12 = self._compute_functional_map(a, b, evals_a, evals_b, sqrt_a, sqrt_b)
        fmap_21 = None
        if self.bijective:
            fmap_21 = self._compute_functional_map(
                b, a, evals_b, evals_a, sqrt_b, sqrt_a
            )
        return fmap_12, fmap_21


class ComplexForwardFunctionalMap(ForwardFunctionalMap):
    """Forward *complex* functional map in the connection-Laplacian basis.

    Companion to :class:`ForwardFunctionalMap` for orientation-aware ("DUO")
    functional maps (Donati et al.). The map ``Q`` acts on the complex tangent
    eigenbasis ``shape.connection_laplacian.basis`` and is driven by the
    spectral gradient of the features. Reuses the (real) resolvent mask of the
    parent on the (real, non-negative) connection eigenvalues.

    Parameters
    ----------
    lmbda : float
        Resolvent regularization weight.
    resolvent_gamma : float
        Resolvent exponent.
    bijective : bool
        Whether to also compute the reverse map.
    n_cfmap : int
        Complex functional-map size.
    """

    def __init__(self, lmbda=1e-3, resolvent_gamma=0.5, bijective=True, n_cfmap=20):
        super().__init__(lmbda, resolvent_gamma, bijective)
        self.n_cfmap = n_cfmap

    def connection_basis(self, mesh):
        """Return the (cached) complex connection eigenbasis truncated to ``n_cfmap``."""
        operator = mesh.connection_laplacian
        if operator._basis is None or operator.basis.full_spectrum_size < self.n_cfmap:
            operator.find_spectrum(spectrum_size=self.n_cfmap, recompute=True)
        basis = operator.basis
        basis.use_k = self.n_cfmap
        return basis

    def _compute_functional_map(self, a, b, cevals_a, cevals_b):
        """Solve the resolvent-regularized complex functional map row by row."""
        mask = gs.cast(
            self._compute_mask(cevals_a, cevals_b, self.resolvent_gamma), gs.complex128
        )
        a_t = gs.conj(a.T)
        aat, bat = a @ a_t, b @ a_t
        rows = []
        for i in range(cevals_b.shape[0]):
            lhs = aat + self.lmbda * gs.diag(mask[i, :])
            q = gs.linalg.inv(lhs) @ gs.conj(gs.reshape(bat[i, :], (-1, 1)))
            rows.append(gs.conj(q.T))
        return gs.concatenate(rows, axis=0)

    def __call__(self, mesh_a, mesh_b, descr_a, descr_b):
        """Compute the complex functional map(s) between two shapes."""
        basis_a = self.connection_basis(mesh_a)
        basis_b = self.connection_basis(mesh_b)

        # The complex connection basis is backend-native; features are cast to
        # complex to be projected by the (complex) spectral gradient.
        descr_a_c = gs.cast(descr_a.T, gs.complex128)
        descr_b_c = gs.cast(descr_b.T, gs.complex128)
        a = basis_a.spectral_gradient @ descr_a_c
        b = basis_b.spectral_gradient @ descr_b_c
        cevals_a, cevals_b = basis_a.vals, basis_b.vals

        fmap_12 = self._compute_functional_map(a, b, cevals_a, cevals_b)
        fmap_21 = None
        if self.bijective:
            fmap_21 = self._compute_functional_map(b, a, cevals_b, cevals_a)
        return fmap_12, fmap_21
