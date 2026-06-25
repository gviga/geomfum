"""Functional map refinement machinery."""

import abc
import logging

import gsops.backend as gs
import scipy

from geomfum.convert import (
    FmFromP2pBijectiveConverter,
    FmFromP2pConverter,
    NamFromP2pConverter,
    P2pFromFmConverter,
    P2pFromNamConverter,
    SinkhornP2pFromFmConverter,
)


class Refiner(abc.ABC):
    """Functional map refiner."""

    @abc.abstractmethod
    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """


class IdentityRefiner(Refiner):
    """A dummy refiner."""

    def __call__(self, fmap_matrix, basis_a=None, basis_b=None):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis. Ignored.
        basis_b: Eigenbasis.
            Basis. Ignored.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        return fmap_matrix


class OrthogonalRefiner(Refiner):
    """Refinement using singular value decomposition.

    Parameters
    ----------
    flip_neg_det : bool
        Whether to flip negative determinant for square matrices.

    References
    ----------
    .. [OCSBG2012] Maks Ovsjanikov, Mirela Ben-Chen, Justin Solomon,
        Adrian Butscher, and Leonidas Guibas.
        “Functional Maps: A Flexible Representation of Maps between
        Shapes.” ACM Transactions on Graphics 31, no. 4 (2012): 30:1-30:11.
        https://doi.org/10.1145/2185520.2185526.
    """

    def __init__(self, flip_neg_det=True):
        self.flip_neg_det = flip_neg_det

    def __call__(self, fmap_matrix, basis_a=None, basis_b=None):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis. Ignored.
        basis_b: Eigenbasis.
            Basis. Ignored.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        k2, k1 = fmap_matrix.shape
        # scipy.linalg.svd is numpy-only; bring the factors back to the active
        # backend so the matmuls below work under the pytorch backend too.
        U, _, VT = scipy.linalg.svd(gs.to_numpy(fmap_matrix))
        U = gs.asarray(U)
        VT = gs.asarray(VT)

        if k1 != k2 or not self.flip_neg_det:
            return gs.matmul(U, gs.matmul(gs.eye(k2, k1), VT))

        opt_rot = gs.matmul(U, VT)
        if gs.linalg.det(opt_rot) < 0.0:
            diag_sign = gs.diag(gs.ones(VT.shape[0]))
            diag_sign[-1, -1] = -1
            opt_rot = gs.matmul(U, gs.matmul(diag_sign, VT))

        return opt_rot


class ProperRefiner(Refiner):
    """Refinement projecting the functional map to the proper functional map space.

    Parameters
    ----------
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.
    """

    def __init__(
        self,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        super().__init__()
        if p2p_from_fm_converter is None:
            p2p_from_fm_converter = P2pFromFmConverter()

        if fm_from_p2p_converter is None:
            fm_from_p2p_converter = FmFromP2pConverter()

        self.p2p_from_fm_converter = p2p_from_fm_converter
        self.fm_from_p2p_converter = fm_from_p2p_converter

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        p2p_21 = self.p2p_from_fm_converter(fmap_matrix, basis_a, basis_b)
        return self.fm_from_p2p_converter(p2p_21, basis_a, basis_b)


class IterativeRefiner(Refiner):
    """Iterative refinement of functional map.

    At each iteration, it computes a pointwise map,
    converts it back to a functional map, and (optionally)
    furthers refines it.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    atol : float
        Convergence tolerance.
        Ignored if step different than 1.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.
    iter_refiner : Refiner
        Refinement algorithm that runs within each iteration.
    """

    def __init__(
        self,
        nit=10,
        step=0,
        atol=None,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
        iter_refiner=None,
    ):
        super().__init__()
        if p2p_from_fm_converter is None:
            p2p_from_fm_converter = P2pFromFmConverter()

        if fm_from_p2p_converter is None:
            fm_from_p2p_converter = FmFromP2pConverter()

        if iter_refiner is None:
            iter_refiner = IdentityRefiner()

        self.nit = nit
        self.step = step
        self.atol = atol
        self.p2p_from_fm_converter = p2p_from_fm_converter
        self.fm_from_p2p_converter = fm_from_p2p_converter
        self.iter_refiner = iter_refiner

        if self._step_a != self._step_b != 0 and atol is not None:
            raise ValueError("`atol` can't be used with step different than 0.")

    @property
    def step(self):
        """How much to increase each basis per iteration.

        Returns
        -------
        step : tuple[2, int]
            Step.
        """
        return self._step_a, self._step_b

    @step.setter
    def step(self, step):
        """Set step.

        Parameters
        ----------
        step : int or tuple[2, int]
            How much to increase each basis per iteration.
        """
        if isinstance(step, int):
            self._step_a = self._step_b = step
        else:
            self._step_a, self._step_b = step

    def iter(self, fmap_matrix, basis_a, basis_b):
        """Refiner iteration.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b + step_b, spectrum_size_a + step_a]
            Refined functional map matrix.
        """
        k2, k1 = fmap_matrix.shape
        new_k1, new_k2 = k1 + self._step_a, k2 + self._step_b

        p2p_21 = self.p2p_from_fm_converter(fmap_matrix, basis_a, basis_b)

        fmap_matrix = self.fm_from_p2p_converter(
            p2p_21, basis_a.truncate(new_k1), basis_b.truncate(new_k2)
        )
        return self.iter_refiner(fmap_matrix, basis_a, basis_b)

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        k2, k1 = fmap_matrix.shape

        nit = self.nit
        if nit is None:
            nit = min(
                (k1 - basis_a.full_spectrum_size) // self._step_a,
                (k2 - basis_b.full_spectrum_size) // self._step_b,
            )
        else:
            msg = []
            if k1 + nit * self._step_a > basis_a.full_spectrum_size:
                msg.append("`basis_a`")
            if k2 + nit * self._step_b > basis_b.full_spectrum_size:
                msg.append("`basis_b`")

            if msg:
                raise ValueError(f"Not enough eigenvectors on {', '.join(msg)}.")

        for _ in range(nit):
            new_fmap_matrix = self.iter(fmap_matrix, basis_a, basis_b)

            if (
                self.atol is not None
                and gs.amax(gs.abs(new_fmap_matrix - fmap_matrix)) < self.atol
            ):
                break

            fmap_matrix = new_fmap_matrix

        else:
            if self.atol is not None:
                logging.warning(f"Maximum number of iterations reached: {nit}")

        return new_fmap_matrix


class IcpRefiner(IterativeRefiner):
    """Iterative refinement of functional map using SVD.

    Parameters
    ----------
    nit : int
        Number of iterations.
    atol : float
        Convergence tolerance.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.

    References
    ----------
    .. [OCSBG2012] Maks Ovsjanikov, Mirela Ben-Chen, Justin Solomon,
        Adrian Butscher, and Leonidas Guibas.
        “Functional Maps: A Flexible Representation of Maps between
        Shapes.” ACM Transactions on Graphics 31, no. 4 (2012): 30:1-30:11.
        https://doi.org/10.1145/2185520.2185526.
    """

    def __init__(
        self,
        nit=10,
        atol=1e-4,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        super().__init__(
            nit=nit,
            step=0,
            atol=atol,
            p2p_from_fm_converter=p2p_from_fm_converter,
            fm_from_p2p_converter=fm_from_p2p_converter,
            iter_refiner=OrthogonalRefiner(),
        )


class ZoomOut(IterativeRefiner):
    """Zoomout algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.

    References
    ----------
    .. [MRRSWO2019] Simone Melzi, Jing Ren, Emanuele Rodolà, Abhishek Sharma,
        Peter Wonka, and Maks Ovsjanikov. “ZoomOut: Spectral Upsampling
        for Efficient Shape Correspondence.” arXiv, September 12, 2019.
        http://arxiv.org/abs/1904.07865
    """

    def __init__(
        self,
        nit=10,
        step=1,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=p2p_from_fm_converter,
            fm_from_p2p_converter=fm_from_p2p_converter,
            iter_refiner=None,
        )


class AdjointBijectiveZoomOut(ZoomOut):
    """Adjoint bijective zoomout algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.

    References
    ----------
    :cite:`VM2024`
    """

    def __init__(
        self,
        nit=10,
        step=1,
    ):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=P2pFromFmConverter(adjoint=True, bijective=True),
            fm_from_p2p_converter=FmFromP2pBijectiveConverter(),
        )


class FastSinkhornFilters(ZoomOut):
    """Fast Sinkhorn filters.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    neighbor_finder : SinkhornKNeighborsFinder
        Nearest neighbor finder.

    References
    ----------
    .. [PRMWO2021] Gautam Pai, Jing Ren, Simone Melzi, Peter Wonka, and Maks Ovsjanikov.
        "Fast Sinkhorn Filters: Using Matrix Scaling for Non-Rigid Shape Correspondence
        with Functional Maps." Proceedings of the IEEE/CVF Conference on Computer Vision
        and Pattern Recognition (CVPR), 2021, pp. 11956-11965.
        https://hal.science/hal-03184936/document
    """

    def __init__(
        self,
        nit=10,
        step=1,
        neighbor_finder=None,
    ):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=SinkhornP2pFromFmConverter(neighbor_finder),
            fm_from_p2p_converter=FmFromP2pConverter(),
        )


class NeuralZoomOut(ZoomOut):
    """Neural zoomout algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.

    References
    ----------
    .. [VOM2025] Giulio Viganò, Maks Ovsjanikov, Simone Melzi.
        "NAM: Neural Adjoint Maps for refining shape correspondences".
    """

    def __init__(
        self,
        nit=10,
        step=1,
        device="cpu",
    ):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=P2pFromNamConverter(),
            fm_from_p2p_converter=NamFromP2pConverter(device=device),
        )


class DiscreteOptimizationRefiner(Refiner):
    """Discrete-optimization functional-map refinement (Ren et al., SGP 2021).

    A bijective, bidirectionally-coupled generalization of ZoomOut. Like
    ZoomOut it alternates, with increasing spectral resolution, between
    functional maps and pointwise maps; unlike ZoomOut it couples *both*
    directions through a bijective energy, keeping the map in (close to) the
    proper functional-map space and improving accuracy/smoothness.

    Each iteration, at spectral size ``k``:

    1. Solve the forward/backward functional maps from the current pointwise
       maps using the weighted bijective energy
       ``||Phi2 C12 - Pi21 Phi1||^2_A2 + w * ||Pi12 Phi2 C12 - Phi1||^2_A1``,
       then orthogonalize.
    2. Recover the pointwise maps by nearest neighbour in the concatenated
       coupling + bijectivity spectral embedding.

    This is a faithful, self-contained port of the core ``bijective`` variant
    of Robin Magnet's reference implementation
    (https://github.com/RobinMagnet/SmoothFunctionalMaps).

    Parameters
    ----------
    nit : int
        Number of iterations (spectral upsampling steps).
    step : int
        Spectral-size increase per iteration (symmetric for both shapes).
    k_init : int or None
        Starting spectral size. If None, uses the input functional map size.
    bij_weight : float
        Weight of the bijective term in both the functional-map solve and the
        pointwise-map embedding. ``0`` recovers a (bidirectional) ZoomOut.
    orthogonal : bool
        Whether to orthogonalize the functional maps each iteration.
    neighbor_finder : BaseNeighborFinder, optional
        Nearest-neighbour finder. If None, uses ``NeighborFinder(n_neighbors=1)``.

    References
    ----------
    .. [RMWO21] Jing Ren, Simone Melzi, Peter Wonka, Maks Ovsjanikov.
        "Discrete Optimization for Shape Matching". SGP 2021.
    """

    def __init__(
        self,
        nit=10,
        step=1,
        k_init=None,
        bij_weight=1.0,
        orthogonal=True,
        neighbor_finder=None,
    ):
        super().__init__()
        from geomfum.convert import NeighborFinder

        self.nit = nit
        self.step = step
        self.k_init = k_init
        self.bij_weight = bij_weight
        self.orthogonal = orthogonal
        self.neighbor_finder = neighbor_finder or NeighborFinder(n_neighbors=1)

    @staticmethod
    def _orthogonalize(fmap):
        """Project a functional map to the closest orthogonal one (SVD)."""
        import numpy as np

        k2, k1 = fmap.shape
        u, _, vt = scipy.linalg.svd(fmap)
        return u @ np.eye(k2, k1) @ vt

    @staticmethod
    def _to_scipy_sparse(mat):
        """Return ``mat`` (scipy or torch sparse/dense) as a scipy CSR matrix."""
        import scipy.sparse as sp

        if sp.issparse(mat):
            return mat.tocsr()
        try:
            import torch

            if isinstance(mat, torch.Tensor) and mat.layout != torch.strided:
                coo = mat.to_sparse_coo().coalesce()
                idx = coo.indices().cpu().numpy()
                val = coo.values().cpu().numpy()
                return sp.coo_matrix(
                    (val, (idx[0], idx[1])), shape=tuple(mat.shape)
                ).tocsr()
        except ImportError:
            pass
        return sp.csr_matrix(gs.to_numpy(mat))

    def _solve_fm(self, evecs_src, evecs_tgt, mass_tgt, mass_src, p2p_st, p2p_ts, k):
        """Weighted bijective functional map src -> tgt from both pointwise maps.

        ``p2p_st`` maps tgt vertices to src vertices (the primary direction);
        ``p2p_ts`` maps src vertices to tgt vertices (the bijective coupling).
        """
        import numpy as np

        ev_s = evecs_src[:, :k]  # (n_s, k)
        ev_t = evecs_tgt[:, :k]  # (n_t, k)
        ev_s_pb = evecs_src[p2p_st, :k]  # (n_t, k) source basis pulled to tgt
        ev_t_pb = evecs_tgt[p2p_ts, :k]  # (n_s, k) tgt basis pulled to src

        w = self.bij_weight
        a_mat = np.eye(k) + w * (ev_t_pb.T @ (mass_src @ ev_t_pb))
        b_mat = ev_t.T @ (mass_tgt @ ev_s_pb) + w * (ev_t_pb.T @ (mass_src @ ev_s))
        fmap = np.linalg.solve(a_mat, b_mat)  # (k_tgt, k_src) = (k, k)
        if self.orthogonal:
            fmap = self._orthogonalize(fmap)
        return fmap

    def _embedding(self, evecs1, evecs2, fmap12, fmap21, k):
        """Concatenated coupling + bijectivity embeddings for p2p_21 recovery."""
        import numpy as np

        cw = 1.0
        bw = self.bij_weight
        # coupling: emb1 = Phi1, emb2 = Phi2 @ FM_12
        emb1 = [np.sqrt(cw) * evecs1[:, :k]]
        emb2 = [np.sqrt(cw) * (evecs2[:, :k] @ fmap12)]
        if bw > 0:
            # bijectivity: emb1 = Phi1 @ FM_21, emb2 = Phi2
            emb1.append(np.sqrt(bw) * (evecs1[:, :k] @ fmap21))
            emb2.append(np.sqrt(bw) * evecs2[:, :k])
        return np.concatenate(emb1, axis=1), np.concatenate(emb2, axis=1)

    def _nn(self, emb_ref, emb_query):
        """For each query row, index of nearest reference row."""
        return gs.to_numpy(
            self.neighbor_finder(gs.from_numpy(emb_ref), gs.from_numpy(emb_query))
        ).flatten()

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Apply the refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map from shape_a to shape_b.
        basis_a, basis_b : Eigenbasis
            Spectral bases of the two shapes.

        Returns
        -------
        fmap_matrix : array-like, shape=[k_final_b, k_final_a]
            Refined functional map.
        """
        evecs1 = gs.to_numpy(basis_a.full_vecs)  # (n1, K)
        evecs2 = gs.to_numpy(basis_b.full_vecs)  # (n2, K)
        mass1 = self._to_scipy_sparse(basis_a._shape.laplacian.mass_matrix)
        mass2 = self._to_scipy_sparse(basis_b._shape.laplacian.mass_matrix)

        fmap12 = gs.to_numpy(fmap_matrix)
        k = self.k_init or fmap12.shape[1]
        k_max = min(evecs1.shape[1], evecs2.shape[1])
        if k + self.nit * self.step > k_max:
            raise ValueError(
                f"Not enough eigenvectors for DiscreteOptimizationRefiner: "
                f"need {k + self.nit * self.step}, have {k_max}."
            )

        # Bootstrap both pointwise maps from the input forward map.
        p2p_21 = self._nn(evecs1[:, :k], evecs2[:, :k] @ fmap12)
        fmap21 = self._solve_fm(evecs2, evecs1, mass1, mass2, p2p_21, p2p_21, k)
        p2p_12 = self._nn(evecs2[:, :k], evecs1[:, :k] @ fmap21)

        for _ in range(self.nit):
            fmap12 = self._solve_fm(evecs1, evecs2, mass2, mass1, p2p_21, p2p_12, k)
            fmap21 = self._solve_fm(evecs2, evecs1, mass1, mass2, p2p_12, p2p_21, k)

            emb1, emb2 = self._embedding(evecs1, evecs2, fmap12, fmap21, k)
            p2p_21 = self._nn(emb1, emb2)
            emb2b, emb1b = self._embedding(evecs2, evecs1, fmap21, fmap12, k)
            p2p_12 = self._nn(emb2b, emb1b)

            k += self.step

        # Final functional map at the achieved resolution.
        fmap12 = self._solve_fm(evecs1, evecs2, mass2, mass1, p2p_21, p2p_12, k)
        return gs.asarray(fmap12)


class SmoothDiscreteOptimizationRefiner(DiscreteOptimizationRefiner):
    """Smooth Discrete Optimization refinement (Magnet & Ovsjanikov).

    Extends :class:`DiscreteOptimizationRefiner` with a *primal* (spatial)
    smoothness term. In addition to the spectral coupling+bijectivity terms,
    each shape's vertices are smoothly deformed onto the other shape by solving
    a Dirichlet (harmonic) coupling problem

        min_Y  ||Y||^2_W + w_couple ||Y - X_target[p2p]||^2_A
        =>  (W + w_couple A) Y = w_couple A X_target[p2p],

    (``W`` = cotangent stiffness, ``A`` = mass) and the pointwise map is then
    recovered by nearest neighbour in the *concatenated spectral + spatial*
    embedding. The spatial term's weight is ramped up geometrically over the
    iterations, progressively enforcing geometric smoothness.

    Faithful self-contained port of the ``dirichlet`` smooth variant of
    https://github.com/RobinMagnet/SmoothFunctionalMaps.

    Parameters
    ----------
    nit, step, k_init, bij_weight, orthogonal, neighbor_finder
        See :class:`DiscreteOptimizationRefiner`.
    smooth_weight : float
        Dirichlet smoothness weight in the primal solve (smaller = smoother).
    sm_couple_weight : float
        Data-term weight in the primal solve.
    primal_weight_range : tuple[float, float]
        Geometric schedule (start, end) for the spatial term's weight in the
        nearest-neighbour query across iterations.

    References
    ----------
    .. Robin Magnet, Maks Ovsjanikov. "Memory-Scalable and Simplified Functional
        Map Learning" / Smooth Discrete Optimization (SmoothFunctionalMaps).
    """

    def __init__(
        self,
        nit=10,
        step=10,
        k_init=None,
        bij_weight=1.0,
        orthogonal=False,
        smooth_weight=1e-3,
        sm_couple_weight=1.0,
        primal_weight_range=(1e-2, 1.0),
        neighbor_finder=None,
    ):
        super().__init__(
            nit=nit,
            step=step,
            k_init=k_init,
            bij_weight=bij_weight,
            orthogonal=orthogonal,
            neighbor_finder=neighbor_finder,
        )
        self.smooth_weight = smooth_weight
        self.sm_couple_weight = sm_couple_weight
        self.primal_weight_range = primal_weight_range

    def _solve_Y(self, verts_src, verts_tgt, stiffness_src, mass_src, p2p_st):
        """Dirichlet-smoothed deformed positions of src vertices toward tgt."""
        import scipy.sparse.linalg as spla

        target = verts_tgt[p2p_st]  # (n_src, 3)
        cw = self.sm_couple_weight / self.smooth_weight
        lhs = stiffness_src + cw * mass_src
        rhs = cw * (mass_src @ target)
        return spla.spsolve(lhs.tocsc(), rhs)  # (n_src, 3)

    @staticmethod
    def _normalize_spatial(verts_ref, y_def, mass_src):
        """Area-weighted normalization shared by the two spatial embeddings."""
        import numpy as np

        areas = np.asarray(mass_src.sum(axis=1)).ravel()  # lumped vertex areas
        factor = np.sqrt(float((areas[:, None] * y_def**2).sum()))
        factor = factor if factor > 0 else 1.0
        return verts_ref / factor, y_def / factor

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Apply the smooth discrete-optimization refiner."""
        import numpy as np

        evecs1 = gs.to_numpy(basis_a.full_vecs)
        evecs2 = gs.to_numpy(basis_b.full_vecs)
        mass1 = self._to_scipy_sparse(basis_a._shape.laplacian.mass_matrix)
        mass2 = self._to_scipy_sparse(basis_b._shape.laplacian.mass_matrix)
        stiff1 = self._to_scipy_sparse(basis_a._shape.laplacian.stiffness_matrix)
        stiff2 = self._to_scipy_sparse(basis_b._shape.laplacian.stiffness_matrix)
        verts1 = gs.to_numpy(basis_a._shape.vertices)
        verts2 = gs.to_numpy(basis_b._shape.vertices)

        fmap12 = gs.to_numpy(fmap_matrix)
        k = self.k_init or fmap12.shape[1]
        k_max = min(evecs1.shape[1], evecs2.shape[1])
        if k + self.nit * self.step > k_max:
            raise ValueError(
                f"Not enough eigenvectors for SmoothDiscreteOptimizationRefiner: "
                f"need {k + self.nit * self.step}, have {k_max}."
            )

        primal_weights = np.geomspace(
            self.primal_weight_range[0], self.primal_weight_range[1], self.nit
        )

        # Bootstrap both pointwise maps from the input forward map.
        p2p_21 = self._nn(evecs1[:, :k], evecs2[:, :k] @ fmap12)
        fmap21 = self._solve_fm(evecs2, evecs1, mass1, mass2, p2p_21, p2p_21, k)
        p2p_12 = self._nn(evecs2[:, :k], evecs1[:, :k] @ fmap21)

        for it in range(self.nit):
            fmap12 = self._solve_fm(evecs1, evecs2, mass2, mass1, p2p_21, p2p_12, k)
            fmap21 = self._solve_fm(evecs2, evecs1, mass1, mass2, p2p_12, p2p_21, k)

            # Primal: smoothly deform each shape onto the other.
            y_21 = self._solve_Y(verts2, verts1, stiff2, mass2, p2p_21)  # (n2, 3)
            y_12 = self._solve_Y(verts1, verts2, stiff1, mass1, p2p_12)  # (n1, 3)

            pw = np.sqrt(self.sm_couple_weight) * float(primal_weights[it])

            # p2p_21: spectral (ref=mesh1, query=mesh2) + spatial [verts1 ; y_21]
            emb1, emb2 = self._embedding(evecs1, evecs2, fmap12, fmap21, k)
            sp1, sp2 = self._normalize_spatial(verts1, y_21, mass2)
            emb1 = np.concatenate([emb1, pw * sp1], axis=1)
            emb2 = np.concatenate([emb2, pw * sp2], axis=1)
            p2p_21 = self._nn(emb1, emb2)

            # p2p_12: spectral (ref=mesh2, query=mesh1) + spatial [verts2 ; y_12]
            emb2b, emb1b = self._embedding(evecs2, evecs1, fmap21, fmap12, k)
            sq2, sq1 = self._normalize_spatial(verts2, y_12, mass1)
            emb2b = np.concatenate([emb2b, pw * sq2], axis=1)
            emb1b = np.concatenate([emb1b, pw * sq1], axis=1)
            p2p_12 = self._nn(emb2b, emb1b)

            k += self.step

        fmap12 = self._solve_fm(evecs1, evecs2, mass2, mass1, p2p_21, p2p_12, k)
        return gs.asarray(fmap12)


class RefinementPipeline:
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


class CorrespondenceRefiner:
    """Refine point-to-point correspondences by converting to functional maps.

    This class wraps a functional map refiner to work with p2p correspondences.
    It converts the input p2p to a functional map, applies the refiner,
    and converts the result back to a p2p correspondence.

    Parameters
    ----------
    refiner : Refiner
        The functional map refiner to apply.
    fm_from_p2p_converter : FmFromP2pConverter
        Converter from pointwise map to functional map.
    p2p_from_fm_converter : P2pFromFmConverter
        Converter from functional map to pointwise map.
    """

    def __init__(
        self,
        refiner,
        fmap_init_size=10,
        fm_from_p2p_converter=None,
        p2p_from_fm_converter=None,
    ):
        if fm_from_p2p_converter is None:
            fm_from_p2p_converter = FmFromP2pConverter()

        if p2p_from_fm_converter is None:
            p2p_from_fm_converter = P2pFromFmConverter()

        self.refiner = refiner
        self.fmap_init_size = fmap_init_size
        self.fm_from_p2p_converter = fm_from_p2p_converter
        self.p2p_from_fm_converter = p2p_from_fm_converter

    def __call__(self, p2p, basis_a, basis_b):
        """Refine a point-to-point correspondence.

        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_b]
            Input pointwise map.
        basis_a : Eigenbasis.
            Basis of source shape.
        basis_b : Eigenbasis.
            Basis of target shape.

        Returns
        -------
        p2p : array-like, shape=[n_vertices_b]
            Refined pointwise map.
        """
        basis_a.use_k = self.fmap_init_size
        basis_b.use_k = self.fmap_init_size
        fmap_matrix = self.fm_from_p2p_converter(p2p, basis_a, basis_b)
        refined_fmap = self.refiner(fmap_matrix, basis_a, basis_b)
        return self.p2p_from_fm_converter(refined_fmap, basis_a, basis_b)


class ZoomOutCorrespondenceRefiner(CorrespondenceRefiner):
    """Refine p2p correspondences using ZoomOut algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.

    References
    ----------
    .. [MRRSWO2019] Simone Melzi, Jing Ren, Emanuele Rodolà, Abhishek Sharma,
        Peter Wonka, and Maks Ovsjanikov. "ZoomOut: Spectral Upsampling
        for Efficient Shape Correspondence." arXiv, September 12, 2019.
        http://arxiv.org/abs/1904.07865
    """

    def __init__(
        self,
        fmap_init_size=10,
        nit=10,
        step=1,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        if p2p_from_fm_converter is None:
            p2p_from_fm_converter = P2pFromFmConverter()

        if fm_from_p2p_converter is None:
            fm_from_p2p_converter = FmFromP2pConverter()

        refiner = ZoomOut(
            nit=nit,
            step=step,
            p2p_from_fm_converter=p2p_from_fm_converter,
            fm_from_p2p_converter=fm_from_p2p_converter,
        )

        super().__init__(
            refiner=refiner,
            fmap_init_size=fmap_init_size,
            fm_from_p2p_converter=fm_from_p2p_converter,
            p2p_from_fm_converter=p2p_from_fm_converter,
        )


class IcpCorrespondenceRefiner(CorrespondenceRefiner):
    """Refine p2p correspondences using ICP algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    atol : float
        Convergence tolerance.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.

    References
    ----------
    .. [OCSBG2012] Maks Ovsjanikov, Mirela Ben-Chen, Justin Solomon,
        Adrian Butscher, and Leonidas Guibas.
        "Functional Maps: A Flexible Representation of Maps between
        Shapes." ACM Transactions on Graphics 31, no. 4 (2012): 30:1-30:11.
        https://doi.org/10.1145/2185520.2185526.
    """

    def __init__(
        self,
        nit=10,
        atol=1e-4,
        fmap_init_size=10,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        if p2p_from_fm_converter is None:
            p2p_from_fm_converter = P2pFromFmConverter()

        if fm_from_p2p_converter is None:
            fm_from_p2p_converter = FmFromP2pConverter()

        refiner = IcpRefiner(
            nit=nit,
            atol=atol,
            p2p_from_fm_converter=p2p_from_fm_converter,
            fm_from_p2p_converter=fm_from_p2p_converter,
        )

        super().__init__(
            refiner=refiner,
            fmap_init_size=fmap_init_size,
            fm_from_p2p_converter=fm_from_p2p_converter,
            p2p_from_fm_converter=p2p_from_fm_converter,
        )


class CorrespondenceRefinementPipeline:
    """Chain multiple correspondence refiners together.

    Parameters
    ----------
    refiners : list[CorrespondenceRefiner]
        List of correspondence refiners to apply in sequence.
        None values are filtered out.
    """

    def __init__(self, refiners):
        self.refiners = [r for r in refiners if r is not None]

    def __call__(self, p2p, basis_a, basis_b):
        """Apply correspondence refiners in sequence.

        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_b]
            Input pointwise map.
        basis_a : Eigenbasis.
            Basis of source shape.
        basis_b : Eigenbasis.
            Basis of target shape.

        Returns
        -------
        p2p : array-like, shape=[n_vertices_b]
            Refined pointwise map.
        """
        for refiner in self.refiners:
            p2p = refiner(p2p, basis_a, basis_b)

        return p2p
