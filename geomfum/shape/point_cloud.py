"""Definition of point cloud."""

import gs.backend as gs

from geomfum.io import load_pointcloud

from ._base import Shape


class PointCloud(Shape):
    """Point cloud.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Vertices of the point cloud.
    """

    def __init__(self, vertices):
        super().__init__(is_mesh=False)
        self.vertices = gs.asarray(vertices)

    @classmethod
    def from_file(cls, filename):
        """Instantiate given a file.

        Returns
        -------
        mesh : PointCloud
            A point cloud.
        """
        vertices = load_pointcloud(filename)
        return cls(vertices)

    @property
    def n_vertices(self):
        """Number of points.

        Returns
        -------
        n_vertices : int
        """
        return self.vertices.shape[0]
