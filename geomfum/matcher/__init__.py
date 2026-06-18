"""Module containing different matching algorithms."""

from .base import (
    BaseMatcher,
    CorrespondenceResult,
    DescriptorMatcher,
    SpatialNearestNeighborMatcher,
)
from .deep_fmap import DeepFMMatcher
from .dummy import DummySoftPermMatcher
from .fmap import FunctionalMapMatcher, ZoomOutMatcher
from .precomputed import PrecomputedP2pMatcher
