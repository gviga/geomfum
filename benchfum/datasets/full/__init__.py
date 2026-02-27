"""Full-shape mesh datasets for standard shape matching benchmarks."""

from ._reference_pairs import ReferencePairsDataset
from .dt4d import DT4DDataset, DT4DPairsDataset
from .faust import FaustDataset
from .scape import ScapeDataset
from .shrec19 import Shrec19Dataset, Shrec19rDataset
from .shrec20 import Shrec20Dataset
from .smal import SmalDataset
from .topkids import TopKidsDataset
from .tosca import ToscaDataset, ToscaPairsDataset

__all__ = [
    "ReferencePairsDataset",
    "FaustDataset",
    "SmalDataset",
    "ScapeDataset",
    "TopKidsDataset",
    "Shrec20Dataset",
    "Shrec19rDataset",
    "ToscaDataset",
    "ToscaPairsDataset",
    "Shrec19Dataset",
    "DT4DDataset",
    "DT4DPairsDataset",
]
