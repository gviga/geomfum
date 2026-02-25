"""Shape matcher classes.

Adding a new matcher
--------------------
1. Create a file in this package (e.g. ``geomfum/matchers/my_matcher.py``).
2. Subclass ``BaseMatcher`` and implement ``__call__``::

       from geomfum.matchers.base import BaseMatcher, CorrespondenceResult

       class MyMatcher(BaseMatcher):
           def __call__(self, shape_a, shape_b, bidirectional=False):
               ...
               return CorrespondenceResult(fmap12=None, p2p21=p2p21)

3. Register it in ``benchfum/_build.py`` by adding it to the registry dict
   returned by ``_build_component_registry()``.  It is then available via
   ``build_matcher({"type": "MyMatcher", ...})``.
"""

from geomfum.matchers.base import BaseMatcher, CorrespondenceResult
from geomfum.matchers.feature import FeatureMatcher
from geomfum.matchers.functional_map import FunctionalMapMatcher

__all__ = [
    "BaseMatcher",
    "CorrespondenceResult",
    "FunctionalMapMatcher",
    "FeatureMatcher",
]
