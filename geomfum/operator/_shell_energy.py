"""Pure-Python (vectorized) discrete-shell elastic energy Hessian.

Faithful translation of the GOAST / shell-energy C++ library
(https://gitlab.com/numod/shell-energy), used to build the *elastic* spectral
basis for Hybrid Functional Maps (Xie et al., CVPR 2024). The shell energy is
the sum of a membrane (stretching) term and a bending term; its Hessian at the
rest configuration is the elastic "stiffness" operator on vertex positions
(a 3n x 3n sparse matrix).

Only the ``*_deformed_hessian`` variants are needed (the elastic basis evaluates
the Hessian at deformed == undeformed). Everything is vectorized over faces /
edges with numpy; per-element helpers take batched ``(n, 3)`` point arrays.

This module is the pure-Python default backend; a ``pyshell`` C++ binding can be
registered as an alternative for validation (see ``geomfum/wrap/pyshell.py``).
"""

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Vectorized geometric helpers (batched over a leading axis)
# ---------------------------------------------------------------------------
_EYE = np.eye(3)


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def _norm(a):
    return np.linalg.norm(a, axis=-1)


def _outer(a, b):
    return a[:, :, None] * b[:, None, :]


def _area(pi, pj, pk):
    return 0.5 * _norm(np.cross(pk - pj, pi - pk))


def _normal(pi, pj, pk):
    n = np.cross(pk - pj, pi - pk)
    return n / _norm(n)[:, None]


def _area_gradient(pi, pj, pk):  # getAreaGradient (gradient w.r.t. Pk... see C++)
    normal = np.cross(pk - pj, pi - pk)
    normal = normal / (2.0 * _norm(normal)[:, None])
    return np.cross(normal, pj - pi)


def _area_grad_k(pi, pj, pk):  # getAreaGradK
    a, d, e = pi - pk, pk - pj, pj - pi
    area = _area(pi, pj, pk)
    t1 = (-0.25 * _dot(e, a) / area)[:, None]
    t2 = (0.25 * _dot(e, d) / area)[:, None]
    return t1 * d + t2 * a


def _cross_op(a):  # getCrossOp -> skew-symmetric (n, 3, 3)
    n = a.shape[0]
    m = np.zeros((n, 3, 3))
    m[:, 0, 1], m[:, 0, 2] = -a[:, 2], a[:, 1]
    m[:, 1, 0], m[:, 1, 2] = a[:, 2], -a[:, 0]
    m[:, 2, 0], m[:, 2, 1] = -a[:, 1], a[:, 0]
    return m


def _projection(x):  # I - x x^T  (x unit)
    return _EYE[None] - _outer(x, x)


def _reflection(x):  # I - 2 x x^T  (x unit)
    return _EYE[None] - 2.0 * _outer(x, x)


def _add_diag(h, val):
    h[:, 0, 0] += val
    h[:, 1, 1] += val
    h[:, 2, 2] += val
    return h


# --- dihedral-angle gradients ---------------------------------------------
def _theta_grad_k(pi, pj, pk):
    e = pj - pi
    grad = _normal(pi, pj, pk)
    return grad * (-0.5 * _norm(e) / _area(pi, pj, pk))[:, None]


def _theta_grad_i_left(pi, pj, pk):
    e, d = pj - pi, pk - pj
    return _theta_grad_k(pi, pj, pk) * (_dot(d, e) / _dot(e, e))[:, None]


def _theta_grad_j_left(pi, pj, pk):
    e, a = pj - pi, pi - pk
    return _theta_grad_k(pi, pj, pk) * (_dot(a, e) / _dot(e, e))[:, None]


def _theta_grad_i(pi, pj, pk, pl):
    return _theta_grad_i_left(pi, pj, pk) - _theta_grad_i_left(pi, pj, pl)


def _theta_grad_j(pi, pj, pk, pl):
    return _theta_grad_j_left(pi, pj, pk) - _theta_grad_j_left(pi, pj, pl)


# --- area Hessians ---------------------------------------------------------
def _hess_area_kk(pi, pj, pk):
    e = pj - pi
    en = e / _norm(e)[:, None]
    grad = _area_grad_k(pi, pj, pk)
    h = _outer(grad, grad)
    h += (-0.25 * _dot(e, e))[:, None, None] * _projection(en)
    return h * (-1.0 / _area(pi, pj, pk))[:, None, None]


def _hess_area_ik(pi, pj, pk):
    e, d = pj - pi, pk - pj
    h = _outer(_area_grad_k(pj, pk, pi), _area_grad_k(pi, pj, pk))
    h += 0.25 * _outer(e, d)
    _add_diag(h, -0.25 * _dot(d, e))
    h = h * (-1.0 / _area(pi, pj, pk))[:, None, None]
    h += 0.5 * _cross_op(_normal(pi, pj, pk))
    return h


# --- dihedral-angle Hessians ----------------------------------------------
def _hess_theta_kk(pi, pj, pk):
    area_sqr = _area(pi, pj, pk) ** 2
    e = pj - pi
    en = _norm(e)
    grad = _area_grad_k(pi, pj, pk)
    normal = _normal(pi, pj, pk)
    mat1 = _cross_op(e)
    mat2 = _outer(grad, normal)
    return (en / (4.0 * area_sqr))[:, None, None] * mat1 + (en / area_sqr)[
        :, None, None
    ] * mat2


def _hess_theta_ik(pi, pj, pk):
    area = _area(pi, pj, pk)
    area_sqr = area * area
    e, d = pj - pi, pk - pj
    en = _norm(e)
    grad = _area_grad_k(pj, pk, pi)
    normal = _normal(pi, pj, pk)
    mat3 = (1.0 / (2.0 * area * en))[:, None, None] * _outer(e, normal) + (
        en / (4.0 * area_sqr)
    )[:, None, None] * _cross_op(d)
    return mat3 + (en / area_sqr)[:, None, None] * _outer(grad, normal)


def _hess_theta_jk(pi, pj, pk):
    area = _area(pi, pj, pk)
    area_sqr = area * area
    e, a = pi - pj, pi - pk
    en = _norm(e)
    grad = _area_grad_k(pk, pi, pj)
    normal = _normal(pi, pj, pk)
    mat3 = (1.0 / (2.0 * area * en))[:, None, None] * _outer(e, normal) + (
        en / (4.0 * area_sqr)
    )[:, None, None] * _cross_op(a)
    return mat3 + (en / area_sqr)[:, None, None] * _outer(grad, normal)


def _hess_theta_i_left_i(pi, pj, pk):
    e, d = pj - pi, pk - pj
    en = e / _norm(e)[:, None]
    grad_k = _theta_grad_k(pi, pj, pk)
    refl = _reflection(en)
    temp = np.einsum("nij,nj->ni", refl, d)
    mat1 = _outer(temp, grad_k)
    mat2 = _hess_theta_ik(pi, pj, pk)
    esq = _dot(e, e)
    return (-1.0 / esq)[:, None, None] * mat1 + (_dot(d, e) / esq)[:, None, None] * mat2


def _hess_theta_j_left_i(pi, pj, pk):
    e, d = pj - pi, pk - pj
    en = e / _norm(e)[:, None]
    grad_k = _theta_grad_k(pi, pj, pk)
    refl = _reflection(en)
    temp = np.einsum("nij,nj->ni", refl, d - e)
    mat1 = _outer(temp, grad_k)
    mat2 = _hess_theta_jk(pi, pj, pk)
    esq = _dot(e, e)
    return (1.0 / esq)[:, None, None] * mat1 + (_dot(d, e) / esq)[:, None, None] * mat2


def _hess_theta_ii(pi, pj, pk, pl):
    return _hess_theta_i_left_i(pi, pj, pk) - _hess_theta_i_left_i(pi, pj, pl)


def _hess_theta_ji(pi, pj, pk, pl):
    edge, d, c = pj - pi, pk - pj, pj - pl
    diff, summ = d - edge, c + edge
    esq = _dot(edge, edge)
    thetak = _theta_grad_k(pi, pj, pk)
    thetal = _theta_grad_k(pj, pi, pl)
    grad = _dot(edge, d)[:, None] * thetak - _dot(edge, c)[:, None] * thetal
    hjk = _hess_theta_jk(pi, pj, pk)
    hjl = _hess_theta_ik(pj, pi, pl)
    hji = (_dot(edge, d) / esq)[:, None, None] * np.transpose(hjk, (0, 2, 1)) - (
        _dot(edge, c) / esq
    )[:, None, None] * np.transpose(hjl, (0, 2, 1))
    hji += (-2.0 / (esq * esq))[:, None, None] * _outer(grad, edge)
    hji += (1.0 / esq)[:, None, None] * _outer(thetak, diff)
    hji -= (1.0 / esq)[:, None, None] * _outer(thetal, summ)
    return hji


# ---------------------------------------------------------------------------
# Triplet assembly
# ---------------------------------------------------------------------------
class _Triplets:
    """Accumulate (row, col, val) for the 3n x 3n component-major Hessian."""

    def __init__(self, num_v, factor=1.0):
        self.num_v = num_v
        self.factor = factor
        self.rows, self.cols, self.vals = [], [], []

    def add(self, vk, vl, h):
        """Add local 3x3 blocks ``h`` (n,3,3) between vertices vk,vl (+ transpose)."""
        nv, f = self.num_v, self.factor
        for i in range(3):
            for j in range(3):
                self.rows.append(i * nv + vk)
                self.cols.append(j * nv + vl)
                self.vals.append(f * h[:, i, j])
        mask = vk != vl
        if np.any(mask):
            vkm, vlm, hm = vk[mask], vl[mask], h[mask]
            for i in range(3):
                for j in range(3):
                    self.rows.append(i * nv + vlm)
                    self.cols.append(j * nv + vkm)
                    self.vals.append(f * hm[:, j, i])

    def matrix(self):
        n3 = 3 * self.num_v
        return sp.coo_matrix(
            (
                np.concatenate(self.vals),
                (np.concatenate(self.rows), np.concatenate(self.cols)),
            ),
            shape=(n3, n3),
        ).tocsr()


# ---------------------------------------------------------------------------
# Membrane + bending Hessians (deformed == rest)
# ---------------------------------------------------------------------------
def membrane_deformed_hessian(v_undef, v_def, faces, mu=1.0, lam=1.0):
    """Hessian of the membrane (stretching) energy w.r.t. deformed positions."""
    faces = np.asarray(faces)
    num_v = v_undef.shape[0]
    tri = _Triplets(num_v, factor=1.0)
    lam_q = lam / 4.0
    mu_half_lam_q = mu / 2.0 + lam_q

    fi = faces[:, 0]
    fj = faces[:, 1]
    fk = faces[:, 2]
    und = [v_undef[faces[:, j]] for j in range(3)]
    dfm = [v_def[faces[:, j]] for j in range(3)]

    def und_edge(j):
        return und[(j + 2) % 3] - und[(j + 1) % 3]

    vol_undef = _area(und[0], und[1], und[2])
    vol_def_sqr = (2.0 * _area(dfm[0], dfm[1], dfm[2])) ** 2 / 4.0
    vol_def = np.sqrt(vol_def_sqr)

    trace = [
        -0.25 * mu * _dot(und_edge((i + 2) % 3), und_edge((i + 1) % 3)) / vol_undef
        for i in range(3)
    ]
    mixed_factor = 0.5 * lam / vol_undef + 2.0 * mu_half_lam_q * vol_undef / vol_def_sqr
    area_factor = 0.5 * lam * vol_def / vol_undef - 2.0 * mu_half_lam_q * vol_undef / vol_def

    nodes = dfm  # deformed nodes
    grad_area = [
        _area_gradient(nodes[(i + 1) % 3], nodes[(i + 2) % 3], nodes[i]) for i in range(3)
    ]
    fverts = [fi, fj, fk]

    for i in range(3):  # i == j (diagonal blocks)
        aux = _hess_area_kk(nodes[(i + 1) % 3], nodes[(i + 2) % 3], nodes[i])
        h = area_factor[:, None, None] * aux + mixed_factor[:, None, None] * _outer(
            grad_area[i], grad_area[i]
        )
        _add_diag(h, trace[(i + 1) % 3] + trace[(i + 2) % 3])
        tri.add(fverts[i], fverts[i], h)

    for i in range(3):  # i != j
        aux = _hess_area_ik(nodes[i], nodes[(i + 1) % 3], nodes[(i + 2) % 3])
        h = area_factor[:, None, None] * aux + mixed_factor[:, None, None] * _outer(
            grad_area[i], grad_area[(i + 2) % 3]
        )
        _add_diag(h, -trace[(i + 1) % 3])
        tri.add(fverts[i], fverts[(i + 2) % 3], h)

    return tri.matrix()


def bending_deformed_hessian(v_undef, v_def, faces, ue, emap, ef, ei):
    """Hessian of the bending energy w.r.t. deformed positions (factor 3 per C++)."""
    faces = np.asarray(faces)
    ue = np.asarray(ue)
    ef = np.asarray(ef)
    ei = np.asarray(ei)
    num_v = v_undef.shape[0]

    interior = (ef[:, 0] != -1) & (ef[:, 1] != -1)
    ue, ef, ei = ue[interior], ef[interior], ei[interior]

    pi_idx, pj_idx = ue[:, 0], ue[:, 1]
    pk_idx = faces[ef[:, 0], ei[:, 0]]
    pl_idx = faces[ef[:, 1], ei[:, 1]]

    def dihedral(pi, pj, pk, pl):
        nk = np.cross(pk - pj, pi - pk)
        nk /= _norm(nk)[:, None]
        nl = np.cross(pl - pi, pj - pl)
        nl /= _norm(nl)[:, None]
        cp = np.cross(nk, nl)
        aux = np.clip(_dot(nk, nl), -1.0, 1.0)
        ang = np.arccos(aux)
        return np.where(_dot(cp, pj - pi) < 0.0, -ang, ang)

    # undeformed quantities
    ui, uj, uk, ul = (v_undef[pi_idx], v_undef[pj_idx], v_undef[pk_idx], v_undef[pl_idx])
    del_theta = dihedral(ui, uj, uk, ul)
    vol = _area(ui, uj, uk) * 2.0 / 2.0 + _area(ui, uj, ul) * 2.0 / 2.0  # Ak + Al
    elen_sqr = _dot(uj - ui, uj - ui)

    # deformed quantities
    pi, pj, pk, pl = (v_def[pi_idx], v_def[pj_idx], v_def[pk_idx], v_def[pl_idx])
    del_theta = del_theta - dihedral(pi, pj, pk, pl)
    del_theta = del_theta * (-2.0 * elen_sqr / vol)
    factor = 2.0 * elen_sqr / vol

    thetak = _theta_grad_k(pi, pj, pk)
    thetal = _theta_grad_k(pj, pi, pl)
    thetai = _theta_grad_i(pi, pj, pk, pl)
    thetaj = _theta_grad_j(pi, pj, pk, pl)

    tri = _Triplets(num_v, factor=3.0)  # factor 3 as in the C++ source
    dt = del_theta[:, None, None]
    fa = factor[:, None, None]

    def block(g_a, g_b, hess):
        return fa * _outer(g_a, g_b) + dt * hess

    tri.add(pk_idx, pk_idx, block(thetak, thetak, _hess_theta_kk(pi, pj, pk)))
    tri.add(pi_idx, pk_idx, block(thetai, thetak, _hess_theta_ik(pi, pj, pk)))
    tri.add(pj_idx, pk_idx, block(thetaj, thetak, _hess_theta_jk(pi, pj, pk)))
    tri.add(pl_idx, pl_idx, block(thetal, thetal, _hess_theta_kk(pj, pi, pl)))
    tri.add(pi_idx, pl_idx, block(thetai, thetal, _hess_theta_jk(pj, pi, pl)))
    tri.add(pj_idx, pl_idx, block(thetaj, thetal, _hess_theta_ik(pj, pi, pl)))
    tri.add(pk_idx, pl_idx, fa * _outer(thetak, thetal))  # kl: Hess part is 0
    tri.add(pi_idx, pi_idx, block(thetai, thetai, _hess_theta_ii(pi, pj, pk, pl)))
    tri.add(pj_idx, pj_idx, block(thetaj, thetaj, _hess_theta_ii(pj, pi, pl, pk)))
    hji = dt * _hess_theta_ji(pi, pj, pk, pl) + fa * _outer(thetai, thetaj)
    tri.add(pi_idx, pj_idx, hji)

    return tri.matrix()


def shell_deformed_hessian(
    v_undef, v_def, faces, ue, emap, ef, ei, bending_weight, mu=1.0, lam=1.0
):
    """Full shell-energy Hessian = membrane + bending_weight * bending."""
    h = membrane_deformed_hessian(v_undef, v_def, faces, mu, lam)
    h = h + bending_weight * bending_deformed_hessian(v_undef, v_def, faces, ue, emap, ef, ei)
    return h


def normal_projection(normals):
    """Sparse (3n, n) operator mapping a scalar field to a normal-aligned field.

    Used to project the (vector) elastic vibration modes onto the per-vertex
    normals, yielding the scalar crease-aware elastic basis.
    """
    m = normals.shape[0]
    rows = np.concatenate([np.arange(m), np.arange(m) + m, np.arange(m) + 2 * m])
    cols = np.tile(np.arange(m), 3)
    vals = np.concatenate([normals[:, 0], normals[:, 1], normals[:, 2]])
    return sp.csr_matrix((vals, (rows, cols)), shape=(3 * m, m))
