"""Benchmark dataset classes for standard shape matching datasets.

Full-shape datasets
-------------------
- :class:`~benchfum.datasets.full.FaustDataset`
- :class:`~benchfum.datasets.full.SmalDataset`
- :class:`~benchfum.datasets.full.ScapeDataset`
- :class:`~benchfum.datasets.full.TopKidsDataset`
- :class:`~benchfum.datasets.full.Shrec20Dataset`
- :class:`~benchfum.datasets.full.Shrec19rDataset`
- :class:`~benchfum.datasets.full.ToscaDataset`
- :class:`~benchfum.datasets.full.Shrec19Dataset`
- :class:`~benchfum.datasets.full.DT4DDataset`

Partial-shape datasets
----------------------
- :class:`~benchfum.datasets.partial.Shrec16PairsDataset`
- :class:`~benchfum.datasets.partial.CP2PPairsDataset`
- :class:`~benchfum.datasets.partial.PartialSmalPairsDataset`
"""

from .full import (
    DT4DDataset,
    FaustDataset,
    ScapeDataset,
    Shrec19Dataset,
    Shrec19rDataset,
    Shrec20Dataset,
    SmalDataset,
    TopKidsDataset,
    ToscaDataset,
)
from .partial import (
    CP2PPairsDataset,
    PartialSmalPairsDataset,
    Shrec16PairsDataset,
)

__all__ = [
    # Full
    "FaustDataset",
    "SmalDataset",
    "ScapeDataset",
    "TopKidsDataset",
    "Shrec20Dataset",
    "Shrec19rDataset",
    "ToscaDataset",
    "Shrec19Dataset",
    "DT4DDataset",
    # Partial
    "Shrec16PairsDataset",
    "CP2PPairsDataset",
    "PartialSmalPairsDataset",
]
