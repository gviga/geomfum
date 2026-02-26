"""Datasets for partial shape matching (full-to-partial and partial-to-partial).

Supports three directory formats:

SHREC16 / full-to-partial
-------------------------
::

    dataset_dir/
      null/off/          <- full reference shapes (.off / .obj / .ply)
      cuts/off/          <- partial query shapes
      cuts/corr/         <- correspondences: one .vts or .txt per partial shape
                           each line i contains the index in the full shape
                           that vertex i of the partial shape corresponds to.

Partiality mask for the full shape is derived from the correspondences:
``mask_full[corr_partial] = 1``.  The partial shape gets ``mask = ones``.

Pairs are listed in ``cuts/pairs.txt`` (one "full_name partial_name" per line).
If the file does not exist, all (full, partial) cross-product pairs are used.

CP2P / partial-to-partial
--------------------------
::

    dataset_dir/
      shapes/            <- all shapes (.off / .obj / .ply)
      corr/              <- .map files (one per ordered pair x_y.map)

``.map`` file binary format (uint32 little-endian):
  ``[size_x, size_y, corr_0, ..., corr_{size_y - 1}, mask_0, ..., mask_{size_x - 1}]``

  where ``corr[j]`` = vertex in X corresponding to vertex j in Y,
  and ``mask[i] = 1`` if vertex i of X is in the overlap with Y.

PARTIALSMAL / partial-to-partial
---------------------------------
::

    dataset_dir/
      train/
        shapes/          <- partial shapes (.off)
        maps/            <- .vts files named ``{shape_x}_{shape_y}.vts``
      test/
        shapes/
        maps/

``.vts`` text format: one integer per line.  Line j gives the 1-indexed vertex
in shape X corresponding to vertex j of shape Y; ``-1`` means no match.
Shape names may contain underscores, so pair name parsing tries all possible
split positions against the known shape name set.
"""

import os
import warnings

import gsops.backend as gs
import meshio
import numpy as np
import torch
from torch.utils.data import Dataset

from geomfum.shape import TriangleMesh

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MESH_EXTS = {".off", ".ply", ".obj"}


def _load_shape(filepath, spectral=False, k=200):
    """Load a TriangleMesh and optionally compute its spectral basis."""
    shape = TriangleMesh.from_file(filepath)
    if spectral:
        shape.laplacian.find_spectrum(spectrum_size=k, set_as_basis=True)
    return shape


def _move_shape_to_device(shape, device):
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


def _list_shapes(directory):
    """Return sorted list of mesh filenames in *directory*."""
    return sorted(
        f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in _MESH_EXTS
    )


# ---------------------------------------------------------------------------
# SHREC16-format helpers
# ---------------------------------------------------------------------------


def _load_corr_shrec16(corr_path, corr_offset=0):
    """Load a correspondence file (.vts or plain .txt, one index per line)."""
    corr = np.loadtxt(corr_path, dtype=np.int64) - corr_offset
    return corr  # [n_partial_vertices] → vertex indices in full shape


def _mask_from_corr(n_full, corr):
    """Binary mask for the full shape: mask[i]=1 if vertex i appears in *corr*."""
    mask = np.zeros(n_full, dtype=np.float32)
    mask[corr] = 1.0
    return mask


# ---------------------------------------------------------------------------
# CP2P-format helpers
# ---------------------------------------------------------------------------


def _load_map_file(map_path):
    """Parse a CP2P `.map` file.

    Returns
    -------
    corr_y_to_x : np.ndarray, shape=[size_y]
        For each vertex in shape Y, the corresponding vertex index in shape X.
    mask_x : np.ndarray, shape=[size_x]
        Binary mask: 1 if vertex in X is in the overlap with Y.
    """
    data = np.fromfile(map_path, dtype=np.int32)
    size_x = int(data[0])
    size_y = int(data[1])
    corr = data[2 : 2 + size_y].astype(np.int64)
    mask = data[2 + size_y : 2 + size_y + size_x].astype(np.float32)
    return corr, mask


# ---------------------------------------------------------------------------
# SHREC16 dataset
# ---------------------------------------------------------------------------


class Shrec16PairsDataset(Dataset):
    """Full-to-partial pairs in SHREC16 format.

    Parameters
    ----------
    dataset_dir : str
        Root directory with ``null/off/``, ``cuts/off/``, ``cuts/corr/``
        sub-directories (or ``holes/`` instead of ``cuts/``).
    partial_split : str, optional
        Name of the partial split sub-directory.  Default is ``"cuts"``.
    spectral : bool
        Whether to compute the LBO spectral basis for each shape.
    k : int
        Number of eigenvectors for the spectral basis.
    device : torch.device or str, optional
        Target device for tensor data.
    corr_offset : int, optional
        Subtract this value from loaded correspondence indices (1 for 1-indexed).
    """

    def __init__(
        self,
        dataset_dir,
        partial_split="cuts",
        spectral=True,
        k=200,
        device=None,
        corr_offset=0,
    ):
        self.dataset_dir = dataset_dir
        self.partial_split = partial_split
        self.spectral = spectral
        self.k = k
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.corr_offset = corr_offset

        full_dir = os.path.join(dataset_dir, "null", "off")
        partial_dir = os.path.join(dataset_dir, partial_split, "off")
        corr_dir = os.path.join(dataset_dir, partial_split, "corr")

        # Load full shapes
        self._full_shapes = {}
        for fname in _list_shapes(full_dir):
            ext = os.path.splitext(fname)[1].lower()[1:]
            if ext not in meshio._helpers._writer_map:
                warnings.warn(f"Skipped unsupported mesh file: {fname}")
                continue
            shape = _load_shape(os.path.join(full_dir, fname), spectral=spectral, k=k)
            self._full_shapes[os.path.splitext(fname)[0]] = shape

        # Load partial shapes + correspondences
        self._partial_shapes = {}
        self._corrs = {}
        self._masks_full = {}

        for fname in _list_shapes(partial_dir):
            ext = os.path.splitext(fname)[1].lower()[1:]
            if ext not in meshio._helpers._writer_map:
                warnings.warn(f"Skipped unsupported mesh file: {fname}")
                continue
            base = os.path.splitext(fname)[0]
            shape = _load_shape(
                os.path.join(partial_dir, fname), spectral=spectral, k=k
            )
            self._partial_shapes[base] = shape

            # Find correspondence file (.vts or .txt)
            corr_file = None
            for ext in (".vts", ".txt"):
                candidate = os.path.join(corr_dir, base + ext)
                if os.path.exists(candidate):
                    corr_file = candidate
                    break
            if corr_file is None:
                warnings.warn(f"No correspondence file found for {fname}; skipping.")
                del self._partial_shapes[base]
                continue

            corr = _load_corr_shrec16(corr_file, corr_offset=corr_offset)
            self._corrs[base] = corr

            # Infer full shape name — SHREC16 partials are named e.g. "cat0_r" for "cat0"
            # Try exact match first, then strip suffix after last underscore
            full_base = base
            if full_base not in self._full_shapes:
                # strip trailing _XXXX suffix
                full_base = "_".join(base.split("_")[:-1])
            if full_base not in self._full_shapes:
                warnings.warn(
                    f"Cannot find full shape for partial {base}; tried '{full_base}'."
                )
                del self._partial_shapes[base]
                del self._corrs[base]
                continue

            n_full = self._full_shapes[full_base].n_vertices
            self._masks_full[base] = _mask_from_corr(n_full, corr)

        # Build pair list: (full_base_name, partial_base_name)
        pairs_file = os.path.join(dataset_dir, partial_split, "pairs.txt")
        if os.path.exists(pairs_file):
            self._pairs = []
            with open(pairs_file) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self._pairs.append(
                            (
                                os.path.splitext(parts[0])[0],
                                os.path.splitext(parts[1])[0],
                            )
                        )
        else:
            # All (full, partial) pairs
            self._pairs = [
                (full_base, partial_base)
                for partial_base in self._partial_shapes
                for full_base in self._full_shapes
                if partial_base.startswith(full_base)
                or "_".join(partial_base.split("_")[:-1]) == full_base
            ]

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        """Return a (full_shape, partial_shape) pair with partiality masks.

        Returns
        -------
        dict with keys:
            "source" : dict — full shape data
                "shape"  : TriangleMesh
                "mask"   : Tensor[n_full] (binary, which vertices are in overlap)
                "corr"   : None (full shape has no correspondence file)
            "target" : dict — partial shape data
                "shape"  : TriangleMesh
                "mask"   : Tensor[n_partial] (all ones — all vertices are present)
                "corr"   : Tensor[n_partial] (index into full shape for each vertex)
        """
        full_base, partial_base = self._pairs[idx]

        full_shape = self._full_shapes[full_base]
        partial_shape = self._partial_shapes[partial_base]
        corr = self._corrs[partial_base]
        mask_full = self._masks_full[partial_base]

        _move_shape_to_device(full_shape, self.device)
        _move_shape_to_device(partial_shape, self.device)

        mask_full_t = torch.tensor(mask_full, dtype=torch.float32, device=self.device)
        mask_partial_t = torch.ones(
            partial_shape.n_vertices, dtype=torch.float32, device=self.device
        )
        corr_t = torch.tensor(corr, dtype=torch.long, device=self.device)

        return {
            "source": {
                "shape": full_shape,
                "mask": mask_full_t,
                "corr": None,
            },
            "target": {
                "shape": partial_shape,
                "mask": mask_partial_t,
                "corr": corr_t,
            },
        }


# ---------------------------------------------------------------------------
# CP2P dataset (partial-to-partial)
# ---------------------------------------------------------------------------


class CP2PPairsDataset(Dataset):
    """Partial-to-partial pairs in CP2P `.map` format.

    Parameters
    ----------
    dataset_dir : str
        Root directory with ``shapes/`` and ``corr/`` sub-directories.
    spectral : bool
        Whether to compute the LBO spectral basis.
    k : int
        Number of eigenvectors.
    device : torch.device or str, optional
    pairs_file : str, optional
        Path to a text file listing pairs (one "shape_x shape_y" per line).
        If None, all `.map` files in ``corr/`` define the pairs.
    """

    def __init__(
        self,
        dataset_dir,
        spectral=True,
        k=200,
        device=None,
        pairs_file=None,
    ):
        self.dataset_dir = dataset_dir
        self.spectral = spectral
        self.k = k
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        shapes_dir = os.path.join(dataset_dir, "shapes")
        corr_dir = os.path.join(dataset_dir, "corr")

        # Load all shapes
        self._shapes = {}
        for fname in _list_shapes(shapes_dir):
            ext = os.path.splitext(fname)[1].lower()[1:]
            if ext not in meshio._helpers._writer_map:
                warnings.warn(f"Skipped unsupported mesh file: {fname}")
                continue
            base = os.path.splitext(fname)[0]
            self._shapes[base] = _load_shape(
                os.path.join(shapes_dir, fname), spectral=spectral, k=k
            )

        # Build pair list from .map files or pairs_file
        if pairs_file is not None and os.path.exists(pairs_file):
            pairs = []
            with open(pairs_file) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        pairs.append(
                            (
                                os.path.splitext(parts[0])[0],
                                os.path.splitext(parts[1])[0],
                            )
                        )
            self._pairs = pairs
        else:
            # Discover from .map filenames: "shapeX_shapeY.map"
            self._pairs = []
            for fname in sorted(os.listdir(corr_dir)):
                if not fname.endswith(".map"):
                    continue
                stem = fname[:-4]  # strip .map
                # Try splitting on last underscore
                parts = stem.rsplit("_", 1)
                if (
                    len(parts) == 2
                    and parts[0] in self._shapes
                    and parts[1] in self._shapes
                ):
                    self._pairs.append((parts[0], parts[1]))
                else:
                    warnings.warn(
                        f"Cannot parse pair from map filename {fname}; skipping."
                    )

        # Cache .map data
        self._map_cache = {}
        for x_base, y_base in self._pairs:
            key = f"{x_base}_{y_base}"
            map_path = os.path.join(corr_dir, key + ".map")
            if not os.path.exists(map_path):
                warnings.warn(f"Map file not found: {map_path}")
                continue
            corr_yx, mask_x = _load_map_file(map_path)
            self._map_cache[key] = (corr_yx, mask_x)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        """Return a (shape_x, shape_y) partial pair with partiality masks.

        Returns
        -------
        dict with keys:
            "source" : dict — shape X data
                "shape" : TriangleMesh
                "mask"  : Tensor[n_x] binary (1 = in overlap with Y)
                "corr"  : None
            "target" : dict — shape Y data
                "shape" : TriangleMesh
                "mask"  : Tensor[n_y] binary (1 = in overlap with X)
                "corr"  : Tensor[n_y] index into X for each vertex of Y
        """
        x_base, y_base = self._pairs[idx]
        key = f"{x_base}_{y_base}"
        corr_yx, mask_x = self._map_cache[key]

        shape_x = self._shapes[x_base]
        shape_y = self._shapes[y_base]

        _move_shape_to_device(shape_x, self.device)
        _move_shape_to_device(shape_y, self.device)

        # mask_y: which vertices in Y have a valid correspondence in X
        # (all vertices in .map have a defined corr, so mask_y = ones)
        mask_x_t = torch.tensor(mask_x, dtype=torch.float32, device=self.device)
        mask_y_t = torch.ones(
            shape_y.n_vertices, dtype=torch.float32, device=self.device
        )
        corr_t = torch.tensor(corr_yx, dtype=torch.long, device=self.device)

        return {
            "source": {
                "shape": shape_x,
                "mask": mask_x_t,
                "corr": None,
            },
            "target": {
                "shape": shape_y,
                "mask": mask_y_t,
                "corr": corr_t,
            },
        }


# ---------------------------------------------------------------------------
# PARTIALSMAL dataset (partial-to-partial, .vts text format)
# ---------------------------------------------------------------------------


def _load_vts_file(vts_path):
    """Parse a PARTIALSMAL ``.vts`` correspondence file.

    The file ``{X}_{Y}.vts`` has ``n_x`` lines.  Line ``j`` contains the
    1-indexed vertex in shape **Y** corresponding to vertex ``j`` of shape
    **X**, or ``-1`` for no match.

    Returns
    -------
    corr_x_to_y : np.ndarray, shape=[n_valid]
        0-indexed vertex indices in Y for valid correspondences.
    valid_x : np.ndarray, shape=[n_valid]
        Indices of vertices in X that have a valid correspondence.
    """
    raw = np.loadtxt(vts_path, dtype=np.int64)  # [n_x]
    # Valid entries are strictly positive (1-indexed); -1 means "no match" and
    # 0 is also invalid in this format (would become -1 after conversion).
    valid_mask = raw > 0
    valid_x = np.where(valid_mask)[0]
    corr_x_to_y = raw[valid_mask] - 1  # convert to 0-indexed
    return corr_x_to_y, valid_x


def _parse_pair_name(stem, known_names):
    """Split ``stem`` into ``(name_x, name_y)`` by trying all ``_`` positions.

    Both halves must be present in *known_names*.  Returns ``None`` if no
    valid split is found.
    """
    parts = stem.split("_")
    for i in range(1, len(parts)):
        prefix = "_".join(parts[:i])
        suffix = "_".join(parts[i:])
        if prefix in known_names and suffix in known_names:
            return prefix, suffix
    return None


class PartialSmalPairsDataset(Dataset):
    """Partial-to-partial pairs from the PARTIALSMAL dataset (.vts format).

    Directory layout::

        dataset_dir/
          train/
            shapes/       <- .off files, e.g. ``cuts_0_cougar_01.off``
            maps/         <- .vts files, e.g. ``cuts_0_cow_02_cuts_22_dog_05.vts``
          test/
            shapes/
            maps/

    The ``.vts`` file named ``{x}_{y}.vts`` has ``n_x`` lines.  Line ``j``
    gives the 1-indexed vertex in Y matching vertex ``j`` of X (``-1`` = none).

    Parameters
    ----------
    dataset_dir : str
        Root PARTIALSMAL directory containing ``train/`` and ``test/``.
    split : str, optional
        ``"train"`` or ``"test"``.  Default ``"train"``.
    spectral : bool
        Whether to compute the LBO spectral basis.
    k : int
        Number of eigenvectors.
    device : torch.device or str, optional
        Target device for tensor data.
    """

    def __init__(
        self,
        dataset_dir,
        split="train",
        spectral=True,
        k=200,
        device=None,
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.spectral = spectral
        self.k = k
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        split_dir = os.path.join(dataset_dir, split)
        shapes_dir = os.path.join(split_dir, "shapes")
        maps_dir = os.path.join(split_dir, "maps")

        # Load all shapes
        self._shapes = {}
        for fname in _list_shapes(shapes_dir):
            base = os.path.splitext(fname)[0]
            self._shapes[base] = _load_shape(
                os.path.join(shapes_dir, fname), spectral=spectral, k=k
            )

        known_names = set(self._shapes.keys())

        # Build pair list and cache .vts data
        self._pairs = []
        self._vts_cache = {}

        for fname in sorted(os.listdir(maps_dir)):
            if not fname.endswith(".vts"):
                continue
            stem = fname[:-4]
            parsed = _parse_pair_name(stem, known_names)
            if parsed is None:
                warnings.warn(f"Cannot parse pair from .vts filename {fname}; skipping.")
                continue
            x_base, y_base = parsed
            corr_x_to_y, valid_x = _load_vts_file(os.path.join(maps_dir, fname))
            self._pairs.append((x_base, y_base))
            self._vts_cache[(x_base, y_base)] = (corr_x_to_y, valid_x)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        """Return a (shape_x, shape_y) partial pair with masks and correspondences.

        Returns
        -------
        dict with keys:
            "source" : dict — shape X
                "shape" : TriangleMesh
                "mask"  : Tensor[n_x] — 1 for vertices of X that have a match in Y
                "corr"  : Tensor[n_valid] — 0-indexed vertex indices in X (valid matches)
            "target" : dict — shape Y
                "shape" : TriangleMesh
                "mask"  : Tensor[n_y] — 1 for vertices of Y that appear as match targets
                "corr"  : Tensor[n_valid] — 0-indexed indices in Y paired with source corr
        """
        x_base, y_base = self._pairs[idx]
        corr_x_to_y, valid_x = self._vts_cache[(x_base, y_base)]

        shape_x = self._shapes[x_base]
        shape_y = self._shapes[y_base]

        _move_shape_to_device(shape_x, self.device)
        _move_shape_to_device(shape_y, self.device)

        # mask_x: 1 for vertices of X that have a valid match in Y
        mask_x = np.zeros(shape_x.n_vertices, dtype=np.float32)
        mask_x[valid_x] = 1.0

        # mask_y: 1 for vertices of Y that appear as match targets
        mask_y = np.zeros(shape_y.n_vertices, dtype=np.float32)
        mask_y[corr_x_to_y] = 1.0

        mask_x_t = torch.tensor(mask_x, dtype=torch.float32, device=self.device)
        mask_y_t = torch.tensor(mask_y, dtype=torch.float32, device=self.device)

        # corr: paired (corr_x, corr_y) — same length, indexed into respective shapes
        corr_x_t = torch.tensor(valid_x, dtype=torch.long, device=self.device)
        corr_y_t = torch.tensor(corr_x_to_y, dtype=torch.long, device=self.device)

        return {
            "source": {
                "shape": shape_x,
                "mask": mask_x_t,
                "corr": corr_x_t,
            },
            "target": {
                "shape": shape_y,
                "mask": mask_y_t,
                "corr": corr_y_t,
            },
        }
