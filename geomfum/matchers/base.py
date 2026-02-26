"""Base classes for shape matchers."""

import abc
from dataclasses import dataclass

import gsops.backend as gs


@dataclass
class CorrespondenceResult:
    """Result of a matching operation (for both Matcher and Model).

    This is the unified output format for all correspondence methods,
    including classical functional map matchers and learning-based models.

    Parameters
    ----------
    fmap12 : array-like, shape=[spectrum_size_b, spectrum_size_a]
        Functional map matrix from shape_a to shape_b.
    p2p21 : array-like, shape=[n_vertices_b]
        Point-to-point correspondence from shape_b to shape_a.
        For each vertex i in shape_b, p2p21[i] gives the corresponding
        vertex index in shape_a.
    fmap21 : array-like, shape=[spectrum_size_a, spectrum_size_b], optional
        Functional map matrix from shape_b to shape_a (for bidirectional).
    p2p12 : array-like, shape=[n_vertices_a], optional
        Point-to-point correspondence from shape_a to shape_b (for bidirectional).
    descr_a : array-like, shape=[n_descr, n_vertices_a], optional
        Descriptors on shape_a.
    descr_b : array-like, shape=[n_descr, n_vertices_b], optional
        Descriptors on shape_b.
    refined_fmap12 : array-like, shape=[spectrum_size_b, spectrum_size_a], optional
        Refined functional map matrix (if refinement was applied).
    refined_fmap21 : array-like, shape=[spectrum_size_a, spectrum_size_b], optional
        Refined functional map matrix from B to A (if bidirectional refinement).
    soft_perm_ab : array-like, shape=[n_vertices_a, n_vertices_b], optional
        Soft permutation matrix mapping b vertices to a domain (P12 in RobustFMNet).
        soft_perm_ab[i, j] = probability that vertex i in a corresponds to vertex j in b.
    soft_perm_ba : array-like, shape=[n_vertices_b, n_vertices_a], optional
        Soft permutation matrix mapping a vertices to b domain (P21 in RobustFMNet).
        soft_perm_ba[i, j] = probability that vertex i in b corresponds to vertex j in a.
    """

    fmap12: "gs.ndarray"
    p2p21: "gs.ndarray"
    fmap21: "gs.ndarray" = None
    p2p12: "gs.ndarray" = None
    descr_a: "gs.ndarray" = None
    descr_b: "gs.ndarray" = None
    refined_fmap12: "gs.ndarray" = None
    refined_fmap21: "gs.ndarray" = None
    soft_perm_ab: "gs.ndarray" = None
    soft_perm_ba: "gs.ndarray" = None

    def to_dict(self):
        """Convert to dictionary (for backward compatibility).

        Returns
        -------
        dict
            Dictionary with all non-None fields.

        Notes
        -----
        This method avoids using `asdict()` from dataclasses because it
        performs deep copying, which fails for PyTorch tensors that are
        part of the computation graph (non-leaf tensors) during training.
        """
        return {
            k: getattr(self, k)
            for k in self.__dataclass_fields__
            if getattr(self, k) is not None
        }

    @property
    def is_bidirectional(self):
        """Check if result contains bidirectional correspondences.

        Returns
        -------
        bool
            True if fmap21 and p2p12 are available.
        """
        return self.fmap21 is not None and self.p2p12 is not None


class BaseMatcher(abc.ABC):
    """Abstract base class for shape matchers."""

    @abc.abstractmethod
    def __call__(self, shape_a, shape_b):
        """Compute correspondence between two shapes.

        Parameters
        ----------
        shape_a : Shape
            First shape (target for p2p21).
        shape_b : Shape
            Second shape (source for p2p21).

        Returns
        -------
        result : CorrespondenceResult
            Correspondence result containing:
            - p2p21: point-to-point correspondence from B to A
            - fmap12: functional map from A to B (if applicable)
        """

