"""Base Descriptors Classes."""

import abc

import gsops.backend as gs

import geomfum.linalg as la


class Descriptor(abc.ABC):
    """Abstract base class for shape descriptors."""


class DistanceFromLandmarksDescriptor(Descriptor):
    """Descriptor computing geodesic distances from landmark points."""

    def __init__(self, diameter_normalized=False):
        """Initialize descriptor.

        Parameters
        ----------
        diameter_normalized : bool
            Whether to normalize distances by the shape diameter.
        """
        self.diameter_normalized = diameter_normalized

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.

        Returns
        -------
        descriptor : array-like, shape=[n_landmarks]
            Descriptor values.
        """
        if not hasattr(shape, "landmark_indices"):
            raise AttributeError(
                "shape object does not have 'landmark_indices' attribute"
            )

        if shape.metric is None:
            raise ValueError("shape is not equipped with metric")
        distances_list = shape.metric.dist_from_source(shape.landmark_indices)[0]
        distances = gs.stack(distances_list)

        if self.diameter_normalized:
            diameter = shape.dist_matrix().max()
            distances /= diameter

        return distances
