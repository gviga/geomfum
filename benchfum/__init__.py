"""benchfum — benchmark and experiment framework for geomfum.

This package lives alongside the ``geomfum`` source library and contains
everything that is useful for benchmarking and comparing shape matching methods:
experiment runners, named presets, learning presets, and ready-to-run benchmark
scripts.

Quick start
-----------
>>> from benchfum import compare
>>> from geomfum.matcher import BaseMatcher, FunctionalMapMatcher
>>>
>>> # Option A: load from a JSON file
>>> matcher = BaseMatcher.from_json("my_method.json")
>>>
>>> # Option B: build programmatically
>>> matcher = FunctionalMapMatcher(fmap_size=30)
>>>
>>> # Compare methods in one call
>>> results = compare(
...     {"mine": matcher, "standard": MatcherPresets.build("standard")},
...     dataset=pairs,
... )
>>> results.print_comparison()
>>> results.save_all("results/")

Matcher presets
---------------
>>> from benchfum import MatcherPresets
>>> print(MatcherPresets.list_presets())
>>> matcher = MatcherPresets.build("standard")
>>> matcher = MatcherPresets.build("standard", fmap_size=50)

Learning presets (deep functional maps)
----------------------------------------
>>> from benchfum import ModelPresets, TrainingPresets, quick_train
>>>
>>> model = ModelPresets.build("fmnet_diffusion_standard", device="cuda")
>>> trainer = TrainingPresets.create_trainer(
...     "unsupervised_standard", model, train_set, val_set
... )
>>> trainer.train()
"""

from benchfum.experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    ExperimentSuite,
    compare,
)
from benchfum.presets import MatcherPresets, build_matcher, build_matcher_from_json
from benchfum.refinement import RefinementMatcher, RefinementPresets

# Learning presets (optional — only available when torch is installed)
try:
    from benchfum.learning_presets import ModelPresets, TrainingPresets, quick_train

    _HAS_LEARNING = True
except ImportError:
    _HAS_LEARNING = False

__all__ = [
    # One-call benchmarking
    "compare",
    # Core experiment classes
    "Experiment",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentSuite",
    # Matcher config / presets
    "MatcherPresets",
    "build_matcher",
    "build_matcher_from_json",
    # Refinement
    "RefinementMatcher",
    "RefinementPresets",
    # Learning presets (when torch is available)
    "ModelPresets",
    "TrainingPresets",
    "quick_train",
]
