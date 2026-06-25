"""Elastic (thin-shell) Hessian operator and its eigenbasis.

Mirrors the Laplacian operator pipeline (:mod:`geomfum.operator.laplacian`):

* :class:`ElasticShellHessianFinder` — registry-backed algorithm that builds the
  shell-energy Hessian matrices (built-in pure-Python translation of
  https://gitlab.com/numod/shell-energy; a ``pyshell`` C++ binding can be
  registered as ``which="pyshell"``).
* :class:`ElasticSpectrumFinder` — eigendecomposition of the Hessian into an
  :class:`~geomfum.basis.ElasticEigenBasis`.
* :class:`ElasticShellHessian` — the operator object attached to a shape
  (``shape.elastic_hessian``), holding the matrices and basis with ``find`` /
  ``find_spectrum`` (parallel to :class:`~geomfum.operator.base.Laplacian`).

The elastic vibration modes (projected onto vertex normals) form the
crease-aware basis of Hybrid Functional Maps (Xie et al., CVPR 2024).
"""

import abc

import gsops.backend as gs
import igl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from geomfum._registry import ElasticShellHessianRegistry, WhichRegistryMixins
from geomfum.basis import ElasticEigenBasis
from geomfum.operator import _shell_energy
from geomfum.operator.base import FunctionalOperator


class BaseElasticShellHessianFinder(abc.ABC):
    """Algorithm to find the elastic shell Hessian."""

    @abc.abstractmethod
    def __call__(self, shape):
        """Return the elastic Hessian and (block) mass matrix of a shape.

        Returns
        -------
        hessian : scipy.sparse matrix, shape=[3 n_vertices, 3 n_vertices]
            Hessian of the shell energy at the rest configuration.
        mass3 : scipy.sparse matrix, shape=[3 n_vertices, 3 n_vertices]
            Block-diagonal (x/y/z) Voronoi mass matrix.
        """


class ElasticShellHessianFinder(WhichRegistryMixins, BaseElasticShellHessianFinder):
    """Discrete-shell elastic Hessian (pure-Python default, ``pyshell`` optional).

    Parameters
    ----------
    bending_weight : float
        Relative weight of the bending vs. membrane energy (default 1e-2).
    mu, lam : float
        Lamé-type membrane parameters (defaults 1.0, matching shell-energy).
    """

    _Registry = ElasticShellHessianRegistry

    def __init__(self, bending_weight=1e-2, mu=1.0, lam=1.0):
        super().__init__()
        self.bending_weight = bending_weight
        self.mu = mu
        self.lam = lam

    def __call__(self, shape):
        """Compute the elastic Hessian + block mass via the pure-Python energy."""
        verts = gs.to_numpy(gs.to_device(shape.vertices, "cpu")).astype(np.float64)
        faces = gs.to_numpy(gs.to_device(shape.faces, "cpu")).astype(np.int32)
        ue, emap, ef, ei = igl.edge_flaps(faces)
        hessian = _shell_energy.shell_deformed_hessian(
            verts, verts, faces, ue, emap, ef, ei, self.bending_weight, self.mu, self.lam
        )
        mass = igl.massmatrix(verts, faces, igl.MASSMATRIX_TYPE_VORONOI)
        mass3 = sp.block_diag((mass, mass, mass)).tocsc()
        return gs.sparse.from_scipy_csc(hessian.tocsc()), gs.sparse.from_scipy_csc(mass3)


class ElasticSpectrumFinder:
    """Algorithm to find the elastic Hessian spectrum.

    Parameters
    ----------
    spectrum_size : int
        Number of elastic eigenfunctions (excluding the 6 rigid-body modes).
    bending_weight : float
        Bending vs membrane weight (used to build a default Hessian finder).
    hessian_finder : BaseElasticShellHessianFinder, optional
        Algorithm to build the Hessian. Ignored if the operator already cached it.
    eps : float
        Diagonal regularization added before the generalized eigensolve.
    """

    def __init__(
        self, spectrum_size=20, bending_weight=1e-2, hessian_finder=None, eps=1e-8
    ):
        self.spectrum_size = spectrum_size
        self.bending_weight = bending_weight
        self.hessian_finder = hessian_finder
        self.eps = eps

    def __call__(self, shape, as_basis=True, recompute=False):
        """Eigendecompose the elastic Hessian, dropping the rigid-body kernel."""
        hessian, mass3 = shape.elastic_hessian.find(
            self.hessian_finder, recompute=recompute
        )
        # Bridge the (backend) sparse operators to scipy for the ARPACK solve.
        hessian = gs.sparse.to_scipy_csc(hessian)
        mass3 = gs.sparse.to_scipy_csc(mass3)
        n3 = hessian.shape[0]
        n = n3 // 3
        hreg = (hessian + self.eps * sp.identity(n3)).tocsc()
        vals, vecs = sla.eigsh(hreg, self.spectrum_size + 6, mass3, sigma=0, which="LM")
        order = np.argsort(vals)
        vals, vecs = vals[order][6:], vecs[:, order][:, 6:]

        verts = gs.to_numpy(gs.to_device(shape.vertices, "cpu")).astype(np.float64)
        faces = gs.to_numpy(gs.to_device(shape.faces, "cpu")).astype(np.int32)
        normals = igl.per_vertex_normals(verts, faces)
        elastic = (
            _shell_energy.normal_projection(normals).T @ vecs
        )  # (n_vertices, spectrum_size)
        vertex_mass = np.asarray(mass3.diagonal())[:n]

        if as_basis:
            return ElasticEigenBasis(
                shape,
                gs.from_numpy(vals),
                gs.from_numpy(elastic),
                gs.from_numpy(vertex_mass),
            )
        return gs.from_numpy(vals), gs.from_numpy(elastic)


class ElasticShellHessian(FunctionalOperator):
    """Elastic shell Hessian operator on a shape (parallels :class:`Laplacian`).

    Parameters
    ----------
    shape : Shape
    bending_weight : float
        Bending vs membrane weight.
    hessian : scipy.sparse matrix, optional
    mass : scipy.sparse matrix, optional
    """

    def __init__(self, shape, bending_weight=1e-2, hessian=None, mass=None):
        super().__init__(shape)
        self.bending_weight = bending_weight
        self._hessian = hessian
        self._mass = mass
        self._basis = None

    @property
    def hessian(self):
        """Elastic Hessian (3n x 3n)."""
        if self._hessian is None:
            self.find()
        return self._hessian

    @property
    def mass_matrix(self):
        """Block (3n x 3n) Voronoi mass matrix."""
        if self._mass is None:
            self.find()
        return self._mass

    @property
    def basis(self):
        """Elastic eigenbasis."""
        if self._basis is None:
            self.find_spectrum()
        return self._basis

    def find(self, hessian_finder=None, recompute=False):
        """Build the elastic Hessian + mass matrices via a hessian finder."""
        if not recompute and self._hessian is not None and self._mass is not None:
            return self._hessian, self._mass
        if hessian_finder is None:
            hessian_finder = ElasticShellHessianFinder(bending_weight=self.bending_weight)
        self._hessian, self._mass = hessian_finder(self._shape)
        return self._hessian, self._mass

    def find_spectrum(
        self, spectrum_size=20, spectrum_finder=None, set_as_basis=False, recompute=False
    ):
        """Compute the elastic eigenbasis of the operator."""
        if not recompute and self._basis is not None:
            return self._basis.full_vals, self._basis.full_vecs
        if spectrum_finder is None:
            spectrum_finder = ElasticSpectrumFinder(
                spectrum_size=spectrum_size, bending_weight=self.bending_weight
            )
        self._basis = spectrum_finder(self._shape, as_basis=True, recompute=recompute)
        if set_as_basis:
            self._shape.set_basis(self._basis)
        return self._basis.full_vals, self._basis.full_vecs

    def __call__(self, displacement):
        """Apply the Hessian to a stacked (3n,) displacement field."""
        import geomfum.linalg as la

        return la.matvecmul(self.hessian, displacement)
