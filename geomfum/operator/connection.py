"""Connection Laplacian operator and its induced complex eigenbasis.

Mirrors the Laplacian operator pipeline (:mod:`geomfum.operator.laplacian`):

* :class:`ConnectionLaplacianFinder` — registry-backed algorithm that builds the
  complex connection Laplacian (built-in pure-Python translation of the vector
  heat method, see :mod:`geomfum.operator._connection_geometry`).
* :class:`ConnectionSpectrumFinder` — complex eigendecomposition into a
  :class:`~geomfum.basis.ConnectionEigenBasis`.
* :class:`ConnectionLaplacian` — the operator object attached to a shape
  (``shape.connection_laplacian``), holding the matrices, the gradient operator
  and the basis with ``find`` / ``find_spectrum`` (parallel to
  :class:`~geomfum.operator.base.Laplacian`).

The complex eigenfunctions (tangent vector fields) form the orientation-aware
basis of complex / "DUO" functional maps (Donati et al., DUO-FM). The connection
Laplacian requires a closed, manifold, triangle mesh.
"""

import abc

import gsops.backend as gs
import igl
import numpy as np

from geomfum._registry import (
    ConnectionLaplacianFinderRegistry,
    WhichRegistryMixins,
)
from geomfum.basis import ConnectionEigenBasis
from geomfum.operator._connection_geometry import (
    complex_eigenbasis,
    connection_laplacian,
    spectral_gradient,
    vertex_gradient_op,
)
from geomfum.operator.base import FunctionalOperator


class BaseConnectionLaplacianFinder(abc.ABC):
    """Algorithm to find the connection (vector) Laplacian."""

    @abc.abstractmethod
    def __call__(self, shape):
        """Return ``(connection_matrix, mass, aux)`` for a shape."""


class ConnectionLaplacianFinder(WhichRegistryMixins, BaseConnectionLaplacianFinder):
    """Connection Laplacian finder (built-in pure-Python via libigl)."""

    _Registry = ConnectionLaplacianFinderRegistry

    def __call__(self, shape):
        """Build the connection Laplacian, the Voronoi mass, and halfedge aux."""
        vertices = gs.to_numpy(gs.to_device(shape.vertices, "cpu")).astype(np.float64)
        faces = gs.to_numpy(gs.to_device(shape.faces, "cpu")).astype(np.int32)
        cl, aux = connection_laplacian(vertices, faces)
        mass = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_VORONOI)
        return (
            gs.sparse.from_scipy_csc(cl.tocsc()),
            gs.sparse.from_scipy_csc(mass.tocsc()),
            aux,
        )


class ConnectionSpectrumFinder:
    """Eigendecomposition of the connection Laplacian into a complex eigenbasis.

    Parameters
    ----------
    spectrum_size : int
        Number of complex eigenfunctions.
    """

    def __init__(self, spectrum_size=20):
        self.spectrum_size = spectrum_size

    def __call__(self, shape, as_basis=True, recompute=False):
        """Solve the complex eigenproblem and attach the spectral gradient."""
        cl, mass = shape.connection_laplacian.find(recompute=recompute)
        cvals, cevecs = complex_eigenbasis(
            gs.sparse.to_scipy_csc(cl),
            gs.sparse.to_scipy_csc(mass),
            self.spectrum_size,
        )
        gradv = shape.connection_laplacian.gradient_matrix
        spec_grad = spectral_gradient(gradv, cevecs)
        if as_basis:
            return ConnectionEigenBasis(
                shape,
                gs.from_numpy(cvals),
                gs.from_numpy(cevecs),
                gs.from_numpy(spec_grad),
            )
        return gs.from_numpy(cvals), gs.from_numpy(cevecs)


class ConnectionLaplacian(FunctionalOperator):
    """Connection (vector) Laplacian operator on a shape.

    Parallels :class:`~geomfum.operator.base.Laplacian`: it holds the complex
    connection matrix, the mass matrix and the complex eigenbasis, with
    ``find`` / ``find_spectrum``.
    """

    def __init__(self, shape, matrix=None, mass=None):
        super().__init__(shape)
        self._matrix = matrix
        self._mass = mass
        self._aux = None
        self._gradient = None
        self._basis = None

    @property
    def matrix(self):
        """Complex Hermitian connection Laplacian (n x n)."""
        if self._matrix is None:
            self.find()
        return self._matrix

    @property
    def mass_matrix(self):
        """Voronoi mass matrix (n x n)."""
        if self._mass is None:
            self.find()
        return self._mass

    @property
    def gradient_matrix(self):
        """Real per-vertex gradient operator (2n x n) used for the spectral gradient."""
        if self._gradient is None:
            if self._aux is None:
                self.find()
            vertices = gs.to_numpy(gs.to_device(self._shape.vertices, "cpu")).astype(
                np.float64
            )
            faces = gs.to_numpy(gs.to_device(self._shape.faces, "cpu")).astype(np.int32)
            self._gradient = vertex_gradient_op(vertices, faces, self._aux)
        return self._gradient

    @property
    def basis(self):
        """Complex eigenbasis of the connection Laplacian."""
        if self._basis is None:
            self.find_spectrum()
        return self._basis

    def find(self, finder=None, recompute=False):
        """Build the connection matrix + mass (+ halfedge aux) via a finder."""
        if not recompute and self._matrix is not None and self._mass is not None:
            return self._matrix, self._mass
        if finder is None:
            finder = ConnectionLaplacianFinder()
        self._matrix, self._mass, self._aux = finder(self._shape)
        return self._matrix, self._mass

    def find_spectrum(
        self,
        spectrum_size=20,
        spectrum_finder=None,
        set_as_basis=False,
        recompute=False,
    ):
        """Compute the complex eigenbasis of the operator."""
        if not recompute and self._basis is not None:
            return self._basis.full_vals, self._basis.full_vecs
        if spectrum_finder is None:
            spectrum_finder = ConnectionSpectrumFinder(spectrum_size=spectrum_size)
        self._basis = spectrum_finder(self._shape, as_basis=True, recompute=recompute)
        if set_as_basis:
            self._shape.set_basis(self._basis)
        return self._basis.full_vals, self._basis.full_vecs

    def __call__(self, function):
        """Apply the connection Laplacian to a complex tangent field."""
        import geomfum.linalg as la

        return la.matvecmul(self.matrix, function)
