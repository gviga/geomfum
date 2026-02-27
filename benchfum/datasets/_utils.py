"""Shared internal utilities for benchfum dataset loaders."""

import os

import gsops.backend as gs
import numpy as np

_MESH_EXTS = {".off", ".ply", ".obj"}


def list_shapes(directory, extensions=None):
    """Return sorted list of mesh filenames in *directory*."""
    if extensions is None:
        extensions = _MESH_EXTS
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in extensions
    )


def load_shape(filepath, spectral=False, k=200):
    """Load a TriangleMesh and optionally compute its spectral basis."""
    from geomfum.shape import TriangleMesh

    shape = TriangleMesh.from_file(filepath)
    if spectral:
        shape.laplacian.find_spectrum(spectrum_size=k, set_as_basis=True)
    return shape


def move_shape_to_device(shape, device):
    """Move all cached spectral tensors of a shape to *device* in-place."""
    shape.vertices = gs.to_device(shape.vertices, device)
    if shape._basis is not None:
        shape.basis.full_vals = gs.to_device(shape.basis.full_vals, device)
        shape.basis.full_vecs = gs.to_device(shape.basis.full_vecs, device)
    if shape.laplacian._mass_matrix is not None:
        shape.laplacian._mass_matrix = gs.to_device(
            shape.laplacian._mass_matrix, device
        )
    shape.faces = gs.to_device(shape.faces, device)


def default_device():
    """Return CUDA device if available, else CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vts(path, offset=1):
    """Load a .vts file and return 0-indexed int64 array."""
    return np.loadtxt(path, dtype=np.int64) - offset
