"""``pyshell`` (C++ GOAST shell-energy) backend for the elastic shell Hessian.

Registered as the ``which="pyshell"`` implementation of
:class:`~geomfum.elastic.ElasticShellHessianFinder`. Requires the ``pyshell``
bindings (https://gitlab.com/numod/shell-energy) to be installed. Used to
cross-validate the built-in pure-Python translation.
"""

import igl
import numpy as np
import scipy.sparse as sp

from geomfum.operator.elastic import BaseElasticShellHessianFinder


class PyShellHessianFinder(BaseElasticShellHessianFinder):
    """Elastic shell Hessian via the ``pyshell`` C++ bindings."""

    def __init__(self, bending_weight=1e-2, mu=1.0, lam=1.0):
        self.bending_weight = bending_weight
        self.mu = mu
        self.lam = lam

    def __call__(self, shape):
        """Compute the elastic Hessian + block mass via ``pyshell``."""
        import pyshell

        verts = np.asarray(
            shape.vertices.detach().cpu().numpy()
            if hasattr(shape.vertices, "detach")
            else shape.vertices,
            dtype=np.float64,
        )
        faces = np.asarray(shape.faces).astype(np.int32)
        ue, emap, ef, ei = igl.edge_flaps(faces)
        hessian = pyshell.shell_deformed_hessian(
            verts,
            verts,
            faces,
            ue,
            emap,
            ef,
            ei,
            self.bending_weight,
            self.mu,
            self.lam,
        )
        mass = igl.massmatrix(verts, faces, igl.MASSMATRIX_TYPE_VORONOI)
        mass3 = sp.block_diag((mass, mass, mass)).tocsc()
        return sp.csr_matrix(hessian), mass3
