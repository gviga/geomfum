import gs.backend as gs
import scipy as _scipy
import torch as _torch


def to_dense(array):
    return array.to_dense()


def from_scipy_coo(array):
    indices = gs.stack([gs.from_numpy(array.row), gs.from_numpy(array.col)])
    values = gs.from_numpy(array.data)

    return _torch.sparse_coo_tensor(indices, values, array.shape)


def from_scipy_csc(array):
    values = gs.from_numpy(array.data)
    ccol_indices = gs.from_numpy(array.indptr)
    row_indices = gs.from_numpy(array.indices)

    return _torch.sparse_csc_tensor(ccol_indices, row_indices, values, size=array.shape)


def from_scipy_csr(array):
    values = gs.from_numpy(array.data)
    crow_indices = gs.from_numpy(array.indptr)
    col_indices = gs.from_numpy(array.indices)

    return _torch.sparse_csr_tensor(crow_indices, col_indices, values, size=array.shape)


def from_scipy_dia(array):
    return from_scipy_csc(array.tocsc())


def to_scipy_csc(array):
    ccol_indices = array.ccol_indices().cpu().numpy()
    row_indices = array.row_indices().cpu().numpy()
    values = array.values().cpu().numpy()

    return _scipy.sparse.csc_matrix(
        (values, row_indices, ccol_indices), shape=array.shape
    )


def to_coo(array):
    return array.to_sparse_coo()


def to_csc(array):
    return array.to_sparse_csc()


def to_csr(array):
    return array.to_sparse_csr()


def to_torch_csc(array):
    return array


def to_torch_dia(array):
    return array


def to_torch_coo(array):
    return array


def to_scipy_dia(array):
    # assumes:
    # 1. torch uses csc (consistency with from_scipy_dia)
    # 2. indices are sorted
    return _scipy.sparse.dia_matrix(
        (array.values().cpu().numpy(), 0), shape=array.shape
    )


def coo_matrix(indices, values, shape=None, dtype=None, coalesce=False):
    mat = _torch.sparse_coo_tensor(indices, values, size=shape, dtype=dtype)
    if coalesce:
        return mat.coalesce()

    return mat


def csr_matrix(indices, values, shape=None, dtype=None, coalesce=False):
    if not coalesce:
        return _torch.sparse_csr_tensor(indices, values, size=shape, dtype=dtype)

    return coo_matrix(
        indices, values, shape=shape, dtype=dtype, coalesce=True
    ).to_sparse_csr()


def csc_matrix(indices, values, shape=None, dtype=None, coalesce=False):
    if not coalesce:
        return _torch.sparse_csr_tensor(indices, values, size=shape, dtype=dtype)

    return coo_matrix(
        indices, values, shape=shape, dtype=dtype, coalesce=True
    ).to_sparse_csc()


def dia_matrix(diagonals, offsets=0, shape=None, dtype=None):
    if shape is None:
        shape = (diagonals.shape[-1], diagonals.shape[-1])

    if isinstance(offsets, int):
        offsets = _torch.tensor([offsets])

    return _torch.sparse.spdiags(diagonals, offsets, shape, layout=_torch.sparse_csc)
