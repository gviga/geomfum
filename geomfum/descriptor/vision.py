"""Vision-model descriptor base class.

Descriptors in this family extract per-vertex features by rendering the shape
from multiple viewpoints and feeding the images through large vision models
(e.g. Stable Diffusion, DINOv2).  Concrete implementations live in
``geomfum.wrap`` and are accessed via the registry.
"""

from geomfum._registry import VisionModelDescriptorRegistry, WhichRegistryMixins
from geomfum.descriptor._base import Descriptor


class VisionModelDescriptor(WhichRegistryMixins, Descriptor):
    """Factory for vision-model descriptors using multi-view renderings.

    No internal implementation is provided; use ``.from_registry(which=...)``.

    Available implementations are registered in ``geomfum.wrap.__init__``.

    Parameters
    ----------
    which : str
        Key of the registered implementation (e.g. ``"diff3f"``).
    **kwargs
        Forwarded to the implementation constructor.

    Examples
    --------
    >>> descriptor = VisionModelDescriptor.from_registry(
    ...     which="diff3f",
    ...     prompt="a 3D human body",
    ...     num_views=100,
    ...     device="cuda",
    ... )
    >>> features = descriptor(shape)  # [2048, n_vertices]
    """

    _Registry = VisionModelDescriptorRegistry
