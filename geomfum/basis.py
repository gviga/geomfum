"""Basis implementations. This module defines various function space bases used in GeomFum. A basis is a set of functionsdefined on a shape that can be used to represent other functions on that shape."""

import abc

import gsops.backend as gs
import numpy as np

import geomfum.linalg as la


class Basis(abc.ABC):
    """Abstract base class for function space bases."""


class EigenBasis(Basis):
    """Basis formed by eigenvectors with dynamic truncation support.

    Parameters
    ----------
    vals : array-like, shape=[full_spectrum_size]
        Eigenvalues.
    vecs : array-like, shape=[dim, full_spectrum_size]
        Eigenvectors.
    use_k : int
        Number of values to use on computations.
    """

    def __init__(self, vals, vecs, use_k=None):
        self.full_vals = vals
        self.full_vecs = vecs
        self.use_k = use_k

        # NB: assumes sorted
        self._n_zeros = gs.sum(gs.isclose(vals, 0.0, atol=1e-3))

    @property
    def vals(self):
        """Currently used eigenvalues (truncated to use_k).

        Returns
        -------
        vals : array-like, shape=[spectrum_size]
            Eigenvalues.
        """
        return self.full_vals[: self.use_k]

    @property
    def vecs(self):
        """Currently used eigenvectors (truncated to use_k).

        Returns
        -------
        vecs : array-like, shape=[dim, spectrum_size]
            Eigenvectors.
        """
        return self.full_vecs[:, : self.use_k]

    @property
    def nonzero_vals(self):
        """Nonzero eigenvalues.

        Returns
        -------
        vals : array-like, shape=[spectrum_size - n_zeros]
            Eigenvalues.
        """
        return self.vals[self._n_zeros :]

    @property
    def nonzero_vecs(self):
        """Eigenvectors corresponding to nonzero eigenvalues.

        Returns
        -------
        vecs : array-like, shape=[dim, spectrum_size - n_zeros]
            Eigenvectors.
        """
        return self.vecs[:, self._n_zeros :]

    @property
    def spectrum_size(self):
        """Number of eigenvalues/eigenvectors currently in use.

        Returns
        -------
        spectrum_size : int
            Spectrum size.
        """
        return len(self.vals)

    @property
    def full_spectrum_size(self):
        """Total number of stored eigenvalues/eigenvectors.

        Returns
        -------
        spectrum_size : int
            Spectrum size.
        """
        return len(self.full_vals)

    def truncate(self, spectrum_size):
        """Create new basis with reduced spectrum size.

        Parameters
        ----------
        spectrum_size : int
            Spectrum size.

        Returns
        -------
        basis : Eigenbasis
            Truncated eigenbasis.
        """
        if spectrum_size == self.spectrum_size:
            return self

        return EigenBasis(self.vals[:spectrum_size], self.vecs[:, :spectrum_size])


class LaplaceEigenBasis(EigenBasis):
    """Eigenbasis of the Laplace-Beltrami operator with mass matrix projection.

    Parameters
    ----------
    shape : Shape
        Shape.
    vals : array-like, shape=[spectrum_size]
        Eigenvalues.
    vecs : array-like, shape=[dim, spectrum_size]
        Eigenvectors.
    use_k : int
        Number of values to use on computations.
    """

    def __init__(self, shape, vals, vecs, use_k=None):
        super().__init__(vals, vecs, use_k)
        self._shape = shape

        self._pinv = None

    @property
    def use_k(self):
        """Number of basis functions actively used in computations.

        Returns
        -------
        use_k : int
            Number of values to use on computations.
        """
        return self._use_k

    @use_k.setter
    def use_k(self, value):
        """Set number of basis functions to use (invalidates cached pinv).

        Parameters
        ----------
        use_k : int
            Number of values to use on computations.
        """
        self._pinv = None
        self._use_k = value

    @property
    def pinv(self):
        """L2 pseudo-inverse for projecting functions onto the basis.

        Return
        ------
        pinv : array-like, shape=[spectrum_size, n_vertices]
            Inverse of the eigenvectors matrix.
        """
        if self._pinv is None:
            self._pinv = self.vecs.T @ self._shape.laplacian.mass_matrix
        return self._pinv

    def truncate(self, spectrum_size):
        """Create new basis with reduced spectrum size.

        Parameters
        ----------
        spectrum_size : int
            Spectrum size.

        Returns
        -------
        basis : LaplaceEigenBasis
            Truncated eigenbasis.
        """
        if spectrum_size == self.spectrum_size:
            return self

        return LaplaceEigenBasis(
            self._shape,
            self.full_vals[:spectrum_size],
            self.full_vecs[:, :spectrum_size],
        )

    def project(self, array):
        """Project function onto the eigenbasis using L2 inner product.

        Parameters
        ----------
        array : array-like, shape=[..., n_vertices]
            Function values to project.

        Returns
        -------
        projected_array : array-like, shape=[..., spectrum_size]
            Spectral coefficients.
        """
        return la.matvecmul(
            self.vecs.T,
            la.matvecmul(self._shape.laplacian.mass_matrix, array),
        )


class ConnectionEigenBasis(EigenBasis):
    """Complex eigenbasis of the connection Laplacian (vector Laplacian).

    The eigenvectors are complex tangent vector fields (per vertex); the
    eigenvalues are real and non-negative. Used by complex / orientation-aware
    ("DUO") functional maps (Donati et al.).

    Parameters
    ----------
    shape : Shape
        Shape the basis is attached to.
    vals : array-like, shape=[full_spectrum_size]
        Real eigenvalues.
    vecs : array-like (complex), shape=[n_vertices, full_spectrum_size]
        Complex eigenvectors.
    spectral_gradient : array-like (complex), shape=[full_spectrum_size, n_vertices]
        Vertex-gradient operator projected onto the complex basis
        (``pinv(vecs) @ complex_vertex_gradient``). Used to build complex maps.
    use_k : int
        Number of values to use on computations.
    """

    def __init__(self, shape, vals, vecs, spectral_gradient=None, use_k=None):
        super().__init__(vals, vecs, use_k)
        self._shape = shape
        self._spectral_gradient = spectral_gradient

    @property
    def spectral_gradient(self):
        """Complex spectral gradient (truncated to ``use_k``).

        Returns
        -------
        spectral_gradient : array-like (complex), shape=[spectrum_size, n_vertices]
        """
        if self._spectral_gradient is None:
            return None
        return self._spectral_gradient[: self.use_k]


class ElasticEigenBasis(EigenBasis):
    """Eigenbasis of the elastic (thin-shell) Hessian.

    The vector vibration modes of the shell Hessian, projected onto the
    per-vertex normals, give scalar basis functions (the crease-aware elastic
    basis of Hybrid Functional Maps, Xie et al.). Unlike the Laplace basis it is
    *not* mass-orthonormal, so the reduced mass ``Mk = vecs^T M vecs`` and its
    square root are exposed for spectral computations.

    Parameters
    ----------
    shape : Shape
    vals : ndarray, shape=[full_spectrum_size]
        Eigenvalues (numpy; the elastic spectrum is computed with scipy).
    vecs : ndarray, shape=[n_vertices, full_spectrum_size]
        (Normal-projected) eigenvectors (numpy).
    vertex_mass : ndarray, shape=[n_vertices]
        Lumped per-vertex (Voronoi) mass vector (numpy).
    use_k : int
    """

    def __init__(self, shape, vals, vecs, vertex_mass, use_k=None):
        super().__init__(vals, vecs, use_k)
        self._shape = shape
        self.vertex_mass = vertex_mass

    @property
    def reduced_mass(self):
        """Reduced mass ``Mk = vecs^T diag(mass) vecs`` (truncated to ``use_k``).

        Returns
        -------
        mk : ndarray, shape=[spectrum_size, spectrum_size]
        """
        vecs = self.vecs
        return vecs.T @ (self.vertex_mass[:, None] * vecs)

    @property
    def sqrt_reduced_mass(self):
        """Matrix square root of :attr:`reduced_mass`.

        Returns
        -------
        sqrt_mk : array-like, shape=[spectrum_size, spectrum_size]
        """
        return gs.real(gs.linalg.sqrtm(self.reduced_mass))

    @property
    def pinv(self):
        """Mass-aware projector ``pinv(sqrt(M) vecs) sqrt(M)`` onto the basis.

        Returns
        -------
        pinv : array-like, shape=[spectrum_size, n_vertices]
        """
        sqrt_mass = gs.sqrt(self.vertex_mass)
        weighted = sqrt_mass[:, None] * self.vecs
        # gsops exposes no ``linalg.pinv``; bridge through scipy on a cpu array.
        inv = gs.from_numpy(np.linalg.pinv(gs.to_numpy(gs.to_device(weighted, "cpu"))))
        return inv * sqrt_mass[None, :]

    def project(self, array):
        """Project a function onto the elastic basis.

        Parameters
        ----------
        array : array-like, shape=[..., n_vertices]

        Returns
        -------
        projected : array-like, shape=[..., spectrum_size]
        """
        return la.matvecmul(self.pinv, array)
