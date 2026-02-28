"""Full-shape mesh datasets for standard shape matching benchmarks."""

from ._reference_pairs import ReferencePairsDataset
from .dt4d import DT4DDataset, DT4DPairsDataset
from .faust import FaustDataset, FaustrDataset
from .scape import ScapeDataset
from .shrec19 import Shrec19Dataset, Shrec19rDataset
from .shrec20 import Shrec20Dataset
from .smal import SmalDataset, SmalrDataset
from .topkids import TopKidsDataset
from .tosca import ToscaDataset, ToscaPairsDataset

__all__ = [
    "ReferencePairsDataset",
    "FaustDataset",
    "FaustrDataset",
    "SmalDataset",
    "ScapeDataset",
    "TopKidsDataset",
    "Shrec20Dataset",
    "Shrec19rDataset",
    "SmalrDataset",
    "ToscaDataset",
    "ToscaPairsDataset",
    "Shrec19Dataset",
    "DT4DDataset",
    "DT4DPairsDataset",
]
