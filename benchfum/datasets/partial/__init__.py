"""Partial shape matching datasets."""

from .cp2p import CP2PPairsDataset
from .shrec16 import Shrec16PairsDataset
from .smal import PartialSmalPairsDataset

__all__ = [
    "Shrec16PairsDataset",
    "CP2PPairsDataset",
    "PartialSmalPairsDataset",
]
