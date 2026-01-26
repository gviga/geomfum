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
"""

# Import from submodules
from geomfum.experiment.presets import (
    # Classical matcher presets
    MatcherPresets,
    # Learning-based presets
    ModelPresets,
    TrainingPresets,
    quick_train,
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
    "MatcherPresets",
    "save_config",
    "load_config_simple",
    # Learning presets (RECOMMENDED for training)
    "ModelPresets",
    "TrainingPresets",
    "quick_train",
]
