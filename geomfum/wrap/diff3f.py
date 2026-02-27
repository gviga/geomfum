"""Diffusion-3D-Features (Diff3F) descriptor wrapper.

Wraps the Diff3F pipeline from [1], which extracts per-vertex features by
rendering a mesh from multiple viewpoints and processing each rendering with:

- A Stable Diffusion v1.5 + ControlNet pipeline (depth + normal conditioning)
  that yields 1280-dimensional UNet intermediate features.
- A DINOv2 ViT-B/14 model that yields 768-dimensional patch features.

Combined output: 2048 dimensions per vertex.

The upstream implementation lives in ``vision/Diffusion-3D-Features/`` (a
sibling directory of the geomfum package root).  Its location is resolved via:

1. The ``DIFF3F_PATH`` environment variable (explicit override).
2. A path relative to this file: ``<repo_root>/vision/Diffusion-3D-Features/``.

References
----------
.. [1] Shue et al., "Diffusion-3D-Features", ECCV 2024.
       https://github.com/niladridutt/Diffusion-3D-Features
"""

import os
import pathlib
import sys

import numpy as np

from geomfum.descriptor._base import Descriptor


def _get_diff3f_dir():
    """Return path to the Diffusion-3D-Features directory."""
    env_path = os.environ.get("DIFF3F_PATH")
    if env_path:
        return env_path
    # __file__ = .../geomfum/geomfum/wrap/diff3f.py
    # repo_root = .../geomfum_proj/
    repo_root = pathlib.Path(__file__).parent.parent.parent.parent
    return str(repo_root / "vision" / "Diffusion-3D-Features")


def _ensure_diff3f_on_path():
    """Add the Diffusion-3D-Features directory to ``sys.path`` if needed."""
    path = _get_diff3f_dir()
    if not pathlib.Path(path).is_dir():
        raise FileNotFoundError(
            f"Diffusion-3D-Features directory not found at '{path}'. "
            "Set the DIFF3F_PATH environment variable to the correct location."
        )
    if path not in sys.path:
        sys.path.insert(0, path)


class Diff3FDescriptor(Descriptor):
    """Per-vertex descriptor from Diffusion-3D-Features (Diff3F).

    Renders the mesh from ``num_views`` viewpoints, processes each rendering
    with Stable Diffusion v1.5 + ControlNet (SD features) and DINOv2 (DINO
    features), then back-projects pixel features to mesh vertices.

    SD and DINO models are loaded lazily on the first call to ``__call__``
    and cached for subsequent calls on the same instance.

    Parameters
    ----------
    prompt : str
        Text prompt for the diffusion model (e.g. ``"a 3D human body"``).
    num_views : int
        Number of viewpoints distributed on the viewing sphere. Default 100.
    H : int
        Render height in pixels. Default 512.
    W : int
        Render width in pixels. Default 512.
    tolerance : float
        Ball-query radius as a fraction of the shape's maximum diameter used
        for pixel-to-vertex mapping.  Default 0.004.
    use_normal_map : bool
        Whether to include surface normals as a second ControlNet conditioning
        signal.  Default ``True``.
    device : str or torch.device
        Compute device.  Default ``"cuda"``.

    Notes
    -----
    Output convention follows geomfum's ``[n_features, n_vertices]`` layout:
    the raw Diff3F output ``[n_vertices, 2048]`` (float16) is transposed and
    cast to float64 before being returned.

    Required packages: ``torch``, ``pytorch3d``, ``diffusers``,
    ``transformers``, ``safetensors``, ``opencv-python``.
    """

    FEATURE_DIM = 1280 + 768  # SD UNet + DINOv2

    def __init__(
        self,
        prompt="a 3D shape",
        num_views=100,
        H=512,
        W=512,
        tolerance=0.004,
        use_normal_map=True,
        device="cuda",
    ):
        import torch

        self.prompt = prompt
        self.num_views = num_views
        self.H = H
        self.W = W
        self.tolerance = tolerance
        self.use_normal_map = use_normal_map
        self.device = torch.device(device)

        self._pipe = None
        self._dino_model = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_models_loaded(self):
        """Load SD pipeline and DINOv2 on first call; cached afterwards."""
        if self._pipe is not None:
            return

        _ensure_diff3f_on_path()

        from diffusion import init_pipe  # noqa: PLC0415 (lazy import)
        from dino import init_dino  # noqa: PLC0415

        self._pipe = init_pipe(self.device)
        self._dino_model = init_dino(self.device)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def __call__(self, shape):
        """Compute Diff3F per-vertex features.

        Parameters
        ----------
        shape : TriangleMesh
            Input mesh.  Must have ``vertices`` [n_vertices, 3] and
            ``faces`` [n_faces, 3] attributes.

        Returns
        -------
        descriptor : numpy.ndarray, shape=[2048, n_vertices], dtype=float64
            Per-vertex feature descriptors.
        """
        import torch
        from pytorch3d.renderer import TexturesVertex
        from pytorch3d.structures import Meshes

        self._ensure_models_loaded()
        _ensure_diff3f_on_path()

        from diff3f import get_features_per_vertex  # noqa: PLC0415

        # --- Convert geomfum Shape → PyTorch3D Meshes ---
        verts = torch.tensor(
            np.asarray(shape.vertices, dtype=np.float32), device=self.device
        )
        faces = torch.tensor(
            np.asarray(shape.faces, dtype=np.int64), device=self.device
        )
        # TexturesVertex expects [batch, n_vertices, 3]
        verts_rgb = torch.ones_like(verts).unsqueeze(0) * 0.8
        textures = TexturesVertex(verts_features=verts_rgb)
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

        # --- Extract features via Diff3F pipeline ---
        # Output: [n_vertices, 2048], dtype=float16
        ft = get_features_per_vertex(
            device=self.device,
            pipe=self._pipe,
            dino_model=self._dino_model,
            mesh=mesh,
            prompt=self.prompt,
            num_views=self.num_views,
            H=self.H,
            W=self.W,
            tolerance=self.tolerance,
            use_normal_map=self.use_normal_map,
        )

        # Transpose to [n_features, n_vertices] and cast to float64
        return ft.double().T.cpu().numpy()
