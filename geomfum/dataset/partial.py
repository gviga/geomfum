"""Backward-compatibility shim.

Partial shape dataset classes have moved to ``benchfum.datasets.partial``.
This module re-exports them for code that still imports from here, but will
be removed in a future release.
"""

import warnings

warnings.warn(
    "Importing partial shape datasets from geomfum.dataset.partial is deprecated. "
    "Use benchfum.datasets.partial instead.",
    DeprecationWarning,
    stacklevel=2,
)

from benchfum.datasets.partial import (  # noqa: E402, F401
    CP2PPairsDataset,
    PartialSmalPairsDataset,
    Shrec16PairsDataset,
)

__all__ = [
    "Shrec16PairsDataset",
    "CP2PPairsDataset",
    "PartialSmalPairsDataset",
]
