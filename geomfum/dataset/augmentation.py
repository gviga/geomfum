"""Data augmentation for shape datasets.

Augmentation is applied to vertex coordinates at ``__getitem__`` time,
so each training call sees a freshly randomised view of the shape while
spectral quantities (eigenvectors, eigenvalues, mass matrix) remain
unchanged — they depend only on mesh topology, not on rigid/similarity
transformations.

The URRSM paper applies ``RandomAugmentation`` when DiffusionNet is fed
XYZ coordinates as input features.  When spectral descriptors (WKS/HKS)
are used as input features, augmentation is optional since those
descriptors are already approximately rotation-invariant.
"""

import math
import random

import torch


class RandomAugmentation:
    """Random rotation + noise + scale augmentation (URRSM-style).

    Matches ``data_augmentation`` from the original URRSM codebase
    (``utils/geometry_util.py``).

    Parameters
    ----------
    rot_x : float
        Max rotation around X-axis in degrees (default: 0).
    rot_y : float
        Max rotation around Y-axis in degrees (default: 90).
    rot_z : float
        Max rotation around Z-axis in degrees (default: 0).
    std : float
        Standard deviation of the additive Gaussian noise (default: 0.01).
    noise_clip : float
        Noise is clamped to ``[-noise_clip, noise_clip]`` (default: 0.05).
    scale_min : float
        Minimum per-axis scale factor (default: 0.9).
    scale_max : float
        Maximum per-axis scale factor (default: 1.1).

    Examples
    --------
    >>> aug = RandomAugmentation()                    # URRSM defaults
    >>> aug = RandomAugmentation(rot_y=180, std=0)   # rotation only
    >>> vertices_aug = aug(shape.vertices)            # [N, 3] -> [N, 3]
    """

    def __init__(
        self,
        rot_x=0.0,
        rot_y=90.0,
        rot_z=0.0,
        std=0.01,
        noise_clip=0.05,
        scale_min=0.9,
        scale_max=1.1,
    ):
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z
        self.std = std
        self.noise_clip = noise_clip
        self.scale_min = scale_min
        self.scale_max = scale_max

    def _random_rotation_matrix(self, device, dtype):
        """Build a random Euler rotation matrix (Z @ Y @ X)."""
        thetas = []
        for deg_range in [self.rot_x, self.rot_y, self.rot_z]:
            rand_deg = random.random() * 2 * deg_range - deg_range
            thetas.append(math.radians(rand_deg))
        tx, ty, tz = thetas

        Rx = torch.tensor(
            [
                [1, 0, 0],
                [0, math.cos(tx), -math.sin(tx)],
                [0, math.sin(tx), math.cos(tx)],
            ],
            dtype=dtype,
            device=device,
        )
        Ry = torch.tensor(
            [
                [math.cos(ty), 0, math.sin(ty)],
                [0, 1, 0],
                [-math.sin(ty), 0, math.cos(ty)],
            ],
            dtype=dtype,
            device=device,
        )
        Rz = torch.tensor(
            [
                [math.cos(tz), -math.sin(tz), 0],
                [math.sin(tz), math.cos(tz), 0],
                [0, 0, 1],
            ],
            dtype=dtype,
            device=device,
        )
        return Rz @ Ry @ Rx

    def __call__(self, vertices):
        """Apply augmentation to vertex coordinates.

        Parameters
        ----------
        vertices : torch.Tensor, shape=[n_vertices, 3]
            Vertex coordinates (on any device).

        Returns
        -------
        augmented : torch.Tensor, shape=[n_vertices, 3]
            Augmented vertex coordinates on the same device.
        """
        device = vertices.device
        dtype = vertices.dtype

        # Random rotation
        R = self._random_rotation_matrix(device, dtype)
        vertices = vertices @ R.T

        # Random additive noise
        if self.std > 0:
            noise = (self.std * torch.randn_like(vertices)).clamp(
                -self.noise_clip, self.noise_clip
            )
            vertices = vertices + noise

        # Random per-axis scale
        scale = self.scale_min + torch.rand(3, device=device, dtype=dtype) * (
            self.scale_max - self.scale_min
        )
        vertices = vertices * scale

        return vertices
