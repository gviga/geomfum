"""Experiment framework for systematic evaluation of shape matching methods.

This module provides tools for:
- Running experiments on datasets (Experiment, ExperimentSuite)
- Grid search over parameters (ParameterGridSuite)
- Preset configurations for both classical and learning methods
- Result analysis and comparison

The preset system provides a unified interface for configuring:
- Classical matchers (MatcherPresets)
- Learning-based models (ModelPresets, TrainingPresets)
- Quick training setup (quick_train)

Examples
--------
Basic experiment:
>>> from geomfum.experiment import Experiment, ExperimentConfig
>>> from geomfum.matcher import FunctionalMapMatcher
>>>
>>> experiment = Experiment(
...     method=FunctionalMapMatcher(),
...     dataset=pairs_dataset,
...     config=ExperimentConfig(name="my_experiment")
... )
>>> result = experiment.run()

Compare multiple methods:
>>> from geomfum.experiment import ExperimentSuite
>>>
>>> methods = {
...     "fmap": FunctionalMapMatcher(),
...     "feature": FeatureMatcher(),
... }
>>> suite = ExperimentSuite(methods, pairs_dataset)
>>> results = suite.run()
>>> suite.print_comparison()

Grid search:
>>> from geomfum.experiment import ParameterGridSuite, MatcherPresets
>>>
>>> grid = ParameterGridSuite(
...     base_matcher_factory=lambda cfg: FunctionalMapMatcher(config=cfg),
...     base_config=MatcherPresets.get("standard"),
...     dataset=pairs_dataset,
... )
>>> grid.add_param("sdp_weight", [0.5, 1.0, 2.0])
>>> grid.add_param("lb_weight", [1e-3, 1e-2, 1e-1])
>>> results = grid.run()
>>> grid.print_best(metric="geodesic_error")

Use matcher presets:
>>> from geomfum.experiment import MatcherPresets
>>>
>>> # List available presets
>>> MatcherPresets.list_presets()
['minimal', 'no_refinement', 'precise', 'quick', 'standard']
>>>
>>> # Get a preset configuration
>>> config = MatcherPresets.get("standard")
>>>
>>> # Override specific parameters
>>> config = MatcherPresets.get("standard", sdp_weight=2.0, lb_weight=1e-1)

Use learning presets:
>>> from geomfum.experiment import ModelPresets, TrainingPresets, quick_train
>>>
>>> # Quick training (one-liner)
>>> trainer = quick_train("unsupervised_standard", train_set, val_set)
>>> trainer.train()
>>>
>>> # Or step-by-step
>>> model = ModelPresets.build("fmnet_diffusion_standard")
>>> trainer = TrainingPresets.create_trainer("unsupervised_standard", model, train_set, val_set)
>>> trainer.train()
"""

# Import from submodules
from geomfum.experiment.grid_search import ParameterGridSuite
from geomfum.experiment.presets import (
    # Classical matcher presets
    MatcherPresets,
    # Learning-based presets
    ModelPresets,
    TrainingPresets,
    load_config_simple,
    quick_train,
    save_config,
)
from geomfum.experiment.runner import (
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    ExperimentSuite,
)

# Define public API
__all__ = [
    # Core experiment running
    "Experiment",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentSuite",
    # Grid search
    "ParameterGridSuite",
    # Matcher presets
    "MatcherPresets",
    "save_config",
    "load_config_simple",
    # Learning presets (RECOMMENDED for training)
    "ModelPresets",
    "TrainingPresets",
    "quick_train",
]
