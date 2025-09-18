"""geopext wrapper."""

import gs.backend as gs
import geopext
import numpy as np

import geomfum.backend as xgs
from geomfum.laplacian import BaseLaplacianFinder


class GeopextMeshLaplacianFinder(BaseLaplacianFinder):
    """Algorithm to find the Laplacian of a mesh.

    Parameters
    ----------
    data_struct : str
        Which data structure to use within rust code.
        One of: "corner_table", "half_edge".
    """

    def __init__(self, data_struct="half_edge"):
        available_ds = ("corner_table", "half_edge")
        if data_struct not in available_ds:
            raise ValueError(
                f"Unknown data structure `{data_struct}`. Choose one of the following: {', '.join(available_ds)}"
            )

        self.data_struct = data_struct

    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        stiffness_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Stiffness matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        stiff_dict, mass_vec = geopext.mesh_laplacian(
            gs.to_numpy(shape.vertices),
            gs.to_numpy(shape.faces).ravel().astype(np.uintp),
            self.data_struct,
        )

        indices_i = []
        indices_j = []
        values = []
        for (index_i, index_j), value in stiff_dict.items():
            indices_i.append(index_i)
            indices_j.append(index_j)

            values.append(-value)

        stiffness_matrix = xgs.sparse.csc_matrix(
            gs.array([indices_i, indices_j]),
            values,
            shape=(shape.n_vertices, shape.n_vertices),
            coalesce=True,
        )

        indices = range(shape.n_vertices)
        mass_matrix = xgs.sparse.csc_matrix(
            gs.array([indices, indices]),
            mass_vec,
            shape=(shape.n_vertices, shape.n_vertices),
            coalesce=True,
        )

        return stiffness_matrix, mass_matrix
