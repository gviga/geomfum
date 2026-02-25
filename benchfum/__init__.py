"""benchfum — benchmark and experiment framework for geomfum.

This package lives alongside the ``geomfum`` source library and contains
everything that is useful for benchmarking and comparing shape matching methods:
experiment runners, a generic JSON→Python factory, and ready-to-run benchmark
scripts.

Every object (matcher, model, trainer, refiner) is built from a JSON config file
using the build functions below.  No preset classes — just files and functions.

Quick start — matchers
-----------------------
>>> from benchfum import compare, build_matcher, build_matcher_from_json
>>> from geomfum.matchers import FunctionalMapMatcher
>>>
>>> # Option A: load from a JSON config file
>>> matcher = build_matcher_from_json("configs/matchers/standard.json")
>>>
>>> # Option B: build programmatically — geomfum class, no factory needed
>>> matcher = FunctionalMapMatcher(fmap_size=30)
>>>
>>> results = compare({"mine": matcher}, dataset=pairs)
>>> results.print_comparison()
>>> results.save_all("results/my_run/")
>>> # Later, load saved results without re-running:
>>> saved = ExperimentSuite.load_all("results/my_run/")

Quick start — models (requires torch)
--------------------------------------
>>> from benchfum import build_model_from_json, build_trainer_from_json
>>>
>>> model   = build_model_from_json("configs/models/fmnet_standard.json", device="cuda")
>>> trainer = build_trainer_from_json(
...     "configs/training/unsupervised_standard.json",
...     model, train_set, val_set,
... )
>>> trainer.train()
"""

from benchfum._build import (
    build_converter,
    build_converter_from_json,
    build_matcher,
    build_matcher_from_json,
    build_model,
    build_model_from_json,
    build_refiner,
    build_refiner_from_json,
    build_trainer,
    build_trainer_from_json,
)
from benchfum.experiment import (
    Experiment,
    ExperimentResult,
    ExperimentSuite,
    compare,
)
from benchfum.refinement import RefinementMatcher

__all__ = [
    # One-call benchmarking
    "compare",
    # Core experiment classes
    "Experiment",
    "ExperimentResult",
    "ExperimentSuite",
    # JSON factory — matchers
    "build_matcher",
    "build_matcher_from_json",
    # JSON factory — refiners
    "build_refiner",
    "build_refiner_from_json",
    # JSON factory — converters
    "build_converter",
    "build_converter_from_json",
    # JSON factory — models
    "build_model",
    "build_model_from_json",
    # JSON factory — trainers
    "build_trainer",
    "build_trainer_from_json",
    # Refinement utilities
    "RefinementMatcher",
]
