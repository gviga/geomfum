"""Pure-Python connection-Laplacian geometry (vector heat method).

Discrete vector Laplacian and induced complex spectral machinery, expressed as
pure functions on ``(vertices, faces)`` numpy arrays. Faithful, self-contained
port of the mesh machinery in https://github.com/nicolasdonati/DUO-FM
(``Tools/mesh.py``). Assumes a closed, manifold, triangle mesh (the standard
non-rigid-matching setting).

This is the built-in numerics backend for
:mod:`geomfum.operator.connection` (parallel to
:mod:`geomfum.operator._shell_energy` for the elastic Hessian).
"""

import igl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def halfedge_structure(faces):
    """Build the halfedge structure of a triangle mesh.

    Returns
    -------
    edges : ndarray, shape=[3*n_faces, 2]
        Directed halfedges (i -> j).
    opposite : ndarray, shape=[3*n_faces]
        Index of the opposite halfedge (j -> i).
    nxt : ndarray, shape=[3*n_faces]
        Index of the next halfedge within the face.
    """
    # Newer libigl returns (E, uE, EMAP, uEC, uEE); older returned (E, uE, EMAP, uE2E).
    out = igl.unique_edge_map(faces)
    edges, u_edges, emap = out[0], out[1], out[2]
    if len(out) == 5:  # uEC (cumulative counts), uEE (flattened directed-edge ids)
        uec, uee = out[3], out[4]
        counts = np.diff(uec)
        if not np.all(counts == 2):
            raise ValueError(
                "connection_laplacian requires a closed manifold triangle mesh "
                "(every edge shared by exactly two faces)."
            )
        ue2e = uee.reshape(u_edges.shape[0], 2)
    else:  # legacy: list of directed-edge ids per unique edge
        ue2e = np.array(out[3])
    e2ue2e = ue2e[emap]
    ee = np.tile(np.arange(edges.shape[0]), [2, 1]).T
    opposite = e2ue2e[(e2ue2e != ee)]
    nf = faces.shape[0]
    e2f = np.remainder(np.arange(edges.shape[0]), nf)
    e2f_in = np.arange(edges.shape[0]) // nf
    nxt = ((e2f_in + 1) % 3) * nf + e2f
    return edges, opposite, nxt


def _he_start(edges, n_vertices):
    """First (reference) halfedge per vertex — the local tangent-basis direction."""
    _, he_start = np.unique(edges[:, 0], return_index=True)
    # np.unique gives one start per *present* vertex id, in sorted id order.
    assert he_start.shape[0] == n_vertices, "mesh has unreferenced vertices"
    return he_start


def _normalized_he_angles(verts, faces, edges, opposite, nxt, he_start):
    """Cumulative (flattened) tangent angles per halfedge, plus curvature K."""
    angles = igl.internal_angles(verts, faces)
    he_angles = np.concatenate([angles[:, 1], angles[:, 2], angles[:, 0]])
    he_angles[he_start] = 0.0
    K = igl.gaussian_curvature(verts, faces)  # angle deficit
    vert_angle_sum = 2 * np.pi - K
    he_angles_norm = 2 * np.pi * he_angles / vert_angle_sum[edges[:, 0]]

    # Circulate the 1-ring (vectorized over all vertices) to accumulate angles.
    rotate = np.zeros(verts.shape[0], dtype=int)
    he = np.copy(he_start)
    last_he = he
    i = 0
    while np.any(rotate == 0):
        he = nxt[opposite[he]]
        i += 1
        rot_mask = (he_start == he) * (rotate == 0)
        rotate[rot_mask] = i
        he_angles_norm[he[rotate == 0]] += he_angles_norm[last_he[rotate == 0]]
        last_he = he
        if i > 10_000:
            raise RuntimeError("halfedge circulation did not terminate")
    return he_angles_norm, K, angles


def connection_laplacian(verts, faces):
    """Compute the connection (complex) Laplacian and the data the gradient needs.

    Returns
    -------
    cl : scipy.sparse.csr_matrix (complex), shape=[n_vertices, n_vertices]
        Hermitian connection Laplacian.
    aux : dict
        Halfedge structure and angle data reused by :func:`vertex_gradient_op`.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces).astype(np.int32)
    n = verts.shape[0]

    edges, opposite, nxt = halfedge_structure(faces)
    he_start = _he_start(edges, n)
    he_angles_norm, K, angles = _normalized_he_angles(
        verts, faces, edges, opposite, nxt, he_start
    )

    # Per-halfedge complex rotation (holonomy) rho.
    rho = (he_angles_norm[opposite] + np.pi) - he_angles_norm
    r = np.cos(rho) + np.sin(rho) * 1j
    r_op = r[opposite]

    nf = faces.shape[0]
    r = r.reshape(3, nf).T
    r_op = r_op.reshape(3, nf).T
    cot_ = 0.5 / np.tan(angles)
    cot = cot_ * r
    cot_op = cot_ * r_op

    s_ = np.concatenate([cot_[:, 2], cot_[:, 0], cot_[:, 1]])
    s = np.concatenate([cot[:, 2], cot[:, 0], cot[:, 1]])
    s_op = np.concatenate([cot_op[:, 2], cot_op[:, 0], cot_op[:, 1]])

    ii = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    jj = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows = np.concatenate([ii, jj, ii, jj])
    cols = np.concatenate([jj, ii, ii, jj])
    vals = np.concatenate([-s_op, -s, s_, s_])
    cl = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    aux = dict(
        edges=edges,
        opposite=opposite,
        nxt=nxt,
        he_start=he_start,
        he_angles_norm=he_angles_norm,
        K=K,
    )
    return cl, aux


def complex_eigenbasis(cl, mass, k):
    """Generalized eigendecomposition of the connection Laplacian.

    Parameters
    ----------
    cl : scipy.sparse matrix (complex)
        Connection Laplacian.
    mass : scipy.sparse matrix
        (Voronoi) mass matrix.
    k : int
        Number of complex eigenfunctions.

    Returns
    -------
    cvals : ndarray, shape=[k]
        Real eigenvalues (sorted ascending).
    cevecs : ndarray (complex), shape=[n_vertices, k]
        Complex eigenvectors (tangent vector fields).
    """
    cvals, cevecs = sla.eigsh(cl, k, mass, sigma=0, which="LM")
    order = np.argsort(cvals)
    return np.real(cvals[order]), cevecs[:, order]


def vertex_gradient_op(verts, faces, aux):
    """Per-vertex gradient operator (2 n_vertices, n_vertices), real.

    Rows ``2 i`` / ``2 i + 1`` hold the x / y tangent components of the gradient
    at vertex ``i``. ``aux`` is the dict returned by :func:`connection_laplacian`.
    """
    verts = np.asarray(verts, dtype=np.float64)
    n = verts.shape[0]
    edges, opposite, nxt = aux["edges"], aux["opposite"], aux["nxt"]
    he_start, he_angles_norm = aux["he_start"], aux["he_angles_norm"]

    # First pass: local 1-ring coordinate systems Vjs.
    vjs = []
    rotate = np.zeros(n, dtype=int)
    he = np.copy(he_start)
    i = 0
    while np.any(rotate == 0):
        lij = np.linalg.norm(verts[edges[he][:, 1]] - verts[edges[he][:, 0]], axis=1)
        aij = he_angles_norm[he]
        vj = lij[:, None] * np.cos(np.stack([aij, np.pi / 2 - aij], axis=-1))
        vj[rotate > 0] = 0
        vjs.append(vj)
        he = nxt[opposite[he]]
        i += 1
        rotate[(he_start == he) * (rotate == 0)] = i
        if i > 10_000:
            raise RuntimeError("halfedge circulation did not terminate")
    vjs = np.stack(vjs, axis=1)
    vjs_inv = np.linalg.pinv(vjs)

    # Second pass: assemble the sparse gradient.
    rows, cols, vals = [], [], []
    rotate = np.zeros(n, dtype=int)
    he = np.copy(he_start)
    i = 0
    while np.any(rotate == 0):
        jdv = edges[he][:, 1]
        idv = edges[he][:, 0]
        rows += [2 * idv, 2 * idv, 2 * idv + 1, 2 * idv + 1]
        cols += [idv, jdv, idv, jdv]
        vals += [-vjs_inv[:, 0, i], vjs_inv[:, 0, i], -vjs_inv[:, 1, i], vjs_inv[:, 1, i]]
        he = nxt[opposite[he]]
        i += 1
        rotate[(he_start == he) * (rotate == 0)] = i
        if i > 10_000:
            raise RuntimeError("halfedge circulation did not terminate")

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return sp.csr_matrix((vals, (rows, cols)), shape=(2 * n, n))


def spectral_gradient(gradv, cevecs):
    """Project the real vertex gradient onto the complex basis (the spectral gradient).

    Parameters
    ----------
    gradv : scipy.sparse matrix, shape=[2 n_vertices, n_vertices]
        Real per-vertex gradient operator.
    cevecs : ndarray (complex), shape=[n_vertices, k]
        Complex eigenvectors.

    Returns
    -------
    spec_grad : ndarray (complex), shape=[k, n_vertices]
        Maps a per-vertex function to its complex gradient coefficients.
    """
    g = gradv.toarray() if sp.issparse(gradv) else np.asarray(gradv)
    # combine the two tangent channels into a complex gradient operator (n, n)
    g_t = g.T  # (n, 2n)
    idv = np.arange(g_t.shape[1] // 2)
    grad_vc = (g_t[:, 2 * idv] + 1j * g_t[:, 2 * idv + 1]).T  # (n, n) complex
    return np.linalg.pinv(cevecs) @ grad_vc  # (k, n)
