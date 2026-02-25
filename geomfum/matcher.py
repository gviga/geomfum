"""Backward-compatibility shim.

The matcher classes have moved to ``geomfum.matchers``.
All existing ``from geomfum.matcher import X`` statements continue to work.
"""

from geomfum.matchers import (  # noqa: F401
    BaseMatcher,
    CorrespondenceResult,
    FeatureMatcher,
    FunctionalMapMatcher,
)
