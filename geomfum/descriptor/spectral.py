"""Spectral descriptors."""

import abc

import gsops.backend as gs

import geomfum.linalg as la
from geomfum._registry import (
    HeatKernelSignatureRegistry,
    LandmarkHeatKernelSignatureRegistry,
    LandmarkWaveKernelSignatureRegistry,
    WaveKernelSignatureRegistry,
    WhichRegistryMixins,
)

from ._base import Descriptor


class SpectralDescriptor(Descriptor, abc.ABC):
    """Spectral descriptor computed from Laplacian eigenfunctions with spectral filters.

    Parameters
    ----------
    spectral_filter : SpectralFilter
        Spectral filter.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(basis, n_domain)``) or
        domain points.
    sigma : float
        Standard deviation for the Gaussian.
    scale : bool
        Whether to scale weights to sum to one.
    landmarks : bool
        Whether to compute landmarks based descriptors.
    k: int, optional
        Number of eigenvalues and eigenvectors to use. If None, basis.use_k is used.
    """

    def __init__(
        self,
        spectral_filter=None,
        domain=None,
        sigma=1,
        scale=True,
        landmarks=False,
        k=None,
    ):
        super().__init__()
        self.domain = domain
        self.sigma = sigma
        self.scale = scale
        self.landmarks = landmarks
        self.spectral_filter = spectral_filter
        self.k = k

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.
        """
        if self.k is not None:
            shape.basis.use_k = self.k
        vals = shape.basis.vals
        vecs = shape.basis.vecs

        domain, sigma = (
            self.domain(shape) if callable(self.domain) else (self.domain, self.sigma)
        )

        coefs = self.spectral_filter(vals, domain, sigma)

        if self.landmarks:
            if not hasattr(shape, "landmark_indices") or shape.landmark_indices is None:
                raise AttributeError(
                    f"Shape must have 'landmark_indices' set for {self.__class__.__name__}."
                )
            return self._compute_landmark_descriptor(
                coefs, vecs, shape.landmark_indices
            )
        else:
            return self._compute_descriptor(coefs, vecs)

    def _compute_descriptor(self, coefs, vecs):
        """Compute descriptors from coefficients and eigenvectors.

        Parameters
        ----------
        coefs : array-like, shape=[n_domain, n_eigen]
            Coefficients.
        vecs : array-like, shape=[n_vertices, n_eigen]
            Eigenvectors.

        Returns
        -------
        descriptors : array-like, shape=[n_domain, n_vertices]
        """
        vecs_term = gs.square(vecs)
        if self.scale:
            coefs = la.scale_to_unit_sum(coefs)
        return gs.einsum("...j,ij->...i", coefs, vecs_term)

    def _compute_landmark_descriptor(self, coefs, vecs, landmarks):
        """Compute descriptor with landmarks.

        Parameters
        ----------
        coefs : array-like, shape=[n_domain, n_eigen]
            Coefficients.
        vecs : array-like, shape=[n_vertices, n_eigen]
            Eigenvectors.
        landmarks : array-like, shape=[n_landmarks]
            Landmark indices.

        Returns
        -------
        descriptor : array-like, shape=[n_landmarks * n_domain, n_vertices]
            Descriptor values.
        """
        # weighted_evects: (n_domain, n_landmarks, n_eigen)
        weighted_evects = vecs[None, landmarks, :] * coefs[:, None, :]

        # result: (n_landmarks, n_domain, n_vertices)
        descriptor = gs.einsum("tpk,nk->ptn", weighted_evects, vecs)

        if self.scale:
            inv_scaling = coefs.sum(1)  # (n_domain,)
            descriptor = (1 / inv_scaling)[None, :, None] * descriptor

        # reshape to (n_landmarks * n_domain, n_vertices)
        return gs.reshape(
            descriptor,
            (descriptor.shape[0] * descriptor.shape[1], vecs.shape[0]),
        )


def hks_default_domain(shape, n_domain):
    """Compute HKS default domain. The domain is a set of sampled time points.

    Parameters
    ----------
    shape : Shape.
        Shape with basis.
    n_domain : int
        Number of time points.

    Returns
    -------
    domain : array-like, shape=[n_domain]
        Time points.
    """
    nonzero_vals = shape.basis.nonzero_vals
    device = getattr(nonzero_vals, "device", None)
    return gs.to_device(
        gs.geomspace(
            4 * gs.log(10) / nonzero_vals[-1],
            4 * gs.log(10) / nonzero_vals[0],
            n_domain,
        ),
        device,
    ), None


class WksDefaultDomain:
    """Default domain generator for Wave Kernel Signature using logarithmic energy sampling.

    Parameters
    ----------
    shape : Shape.
        Shape with basis.
    n_domain : int
        Number of energy points to use.
    n_overlap : int
        Controls Gaussian overlap. Ignored if ``sigma`` is not None.
    n_trans : int
        Number of standard deviations to translate energy bound by.
    """

    def __init__(self, n_domain, sigma=None, n_overlap=7, n_trans=2):
        self.n_domain = n_domain
        self.sigma = sigma
        self.n_overlap = n_overlap
        self.n_trans = n_trans

    def __call__(self, shape):
        """Compute WKS domain.

        Parameters
        ----------
        shape : Shape.
            Shape with basis.

        Returns
        -------
        domain : array-like, shape=[n_domain]
        sigma : float
            Standard deviation.
        """
        nonzero_vals = shape.basis.nonzero_vals
        device = getattr(nonzero_vals, "device", None)

        e_min, e_max = gs.log(nonzero_vals[0]), gs.log(nonzero_vals[-1])

        sigma = (
            self.n_overlap * (e_max - e_min) / self.n_domain
            if self.sigma is None
            else self.sigma
        )

        e_min += self.n_trans * sigma
        e_max -= self.n_trans * sigma

        energy = gs.to_device(gs.linspace(e_min, e_max, self.n_domain), device)

        return energy, sigma


class SpectralFilter(abc.ABC):
    """Abstract base class for computing spectral filter coefficients from eigenvalues."""

    @abc.abstractmethod
    def __call__(self, vals, domain, sigma):
        """
        Compute filter coefficients for the given eigenvalues and domain.

        Parameters
        ----------
        vals : array-like, shape=[n_eigen]
            Eigenvalues.
        domain : array-like, shape=[n_domain]
            Domain points (e.g., time for HKS, energy for WKS).
        sigma : float or None
            Optional parameter for the filter (e.g., standard deviation for WKS).

        Returns
        -------
        coefs : array-like, shape=[n_domain, n_eigen]
            Filter coefficients.
        """


class HeatKernelFilter(SpectralFilter):
    """Heat kernel filter computing exp(-t * λ) coefficients for HKS."""

    def __call__(self, vals, domain, sigma):
        """
        Compute heat kernel filter coefficients.

        Parameters
        ----------
        vals : array-like, shape=[n_eigen]
            Eigenvalues.
        domain : array-like, shape=[n_domain]
            Time points.
        sigma : float or None
            Unused for heat kernel filter.

        Returns
        -------
        coefs : array-like, shape=[n_domain, n_eigen]
            Filter coefficients.
        """
        exp_arg = -la.scalarvecmul(domain, vals)
        return gs.exp(exp_arg)


class WaveKernelFilter(SpectralFilter):
    """Wave kernel filter using Gaussian weighting in log-eigenvalue space for WKS."""

    def __call__(self, vals, domain, sigma):
        """
        Compute wave kernel filter coefficients.

        Parameters
        ----------
        vals : array-like, shape=[n_eigen]
            Eigenvalues.
        domain : array-like, shape=[n_domain]
            Energy points (log-space).
        sigma : float
            Standard deviation for the Gaussian.

        Returns
        -------
        coefs : array-like, shape=[n_domain, n_eigen]
            Filter coefficients.
        """
        nonzero_vals = vals[gs.sum(gs.isclose(vals, 0.0, atol=1e-3)) :]
        zeros = gs.to_device(
            gs.zeros((domain.shape[0], vals.shape[0] - nonzero_vals.shape[0])),
            device=getattr(nonzero_vals, "device", None),
        )
        exp_arg = -gs.square(gs.log(nonzero_vals) - domain[:, None]) / (
            2 * gs.square(sigma)
        )
        coefs = gs.exp(exp_arg)

        if zeros.shape[1] > 0:
            coefs = gs.concatenate([zeros, coefs], axis=1)

        return coefs


class HeatKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Heat Kernel Signature descriptor using heat diffusion over time.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain], optional
        Method to compute time domain points (``f(shape)``) or time domain points.
    k : int, optional
        Number of eigenfunctions to use. If None, all eigenfunctions are used.
    """

    _Registry = HeatKernelSignatureRegistry

    def __init__(self, scale=True, n_domain=3, domain=None, k=None):
        super().__init__(
            spectral_filter=HeatKernelFilter(),
            domain=domain
            or (lambda shape: hks_default_domain(shape, n_domain=n_domain)),
            scale=scale,
            sigma=1,
            landmarks=False,
            k=k,
        )


class WaveKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Wave Kernel Signature descriptor using quantum mechanical wave propagation.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    sigma : float
        Standard deviation for the Gaussian.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain], optional
        Method to compute energy domain points (``f(shape)``) or energy domain points.
    k : int, optional
        Number of eigenfunctions to use. If None, all eigenfunctions are used.
    """

    _Registry = WaveKernelSignatureRegistry

    def __init__(self, scale=True, sigma=None, n_domain=3, domain=None, k=None):
        domain = domain or WksDefaultDomain(n_domain=n_domain, sigma=sigma)
        super().__init__(
            spectral_filter=WaveKernelFilter(),
            domain=domain,
            scale=scale,
            sigma=sigma,
            landmarks=False,
            k=k,
        )


class LandmarkHeatKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Heat Kernel Signature computed from landmark points.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain], optional
        Method to compute time domain points (``f(shape)``) or time domain points.
    k : int, optional
        Number of eigenfunctions to use.
    """

    _Registry = LandmarkHeatKernelSignatureRegistry

    def __init__(self, scale=True, n_domain=3, domain=None, k=None):
        super().__init__(
            spectral_filter=HeatKernelFilter(),
            domain=domain
            or (lambda shape: hks_default_domain(shape, n_domain=n_domain)),
            scale=scale,
            sigma=1,
            landmarks=True,
            k=k,
        )


class LandmarkWaveKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Wave Kernel Signature computed from landmark points.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    sigma : float
        Standard deviation for the Gaussian.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain], optional
        Method to compute energy domain points (``f(shape)``) or energy domain points.
    k : int, optional
        Number of eigenfunctions to use.
    """

    _Registry = LandmarkWaveKernelSignatureRegistry

    def __init__(self, scale=True, sigma=None, n_domain=3, domain=None, k=None):
        super().__init__(
            spectral_filter=WaveKernelFilter(),
            domain=domain or WksDefaultDomain(n_domain=n_domain, sigma=sigma),
            scale=scale,
            sigma=sigma,
            k=k,
            landmarks=True,
        )
