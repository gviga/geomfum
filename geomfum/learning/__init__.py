"""Learning module for deep functional maps and neural shape matching.

This module provides the core machinery for training learning-based
shape matching methods.

Training:
- DeepFunctionalMapTrainer: Training loop for deep functional maps
- LossManager: Manages multiple loss functions with weights
- Various loss functions: OrthonormalityLoss, BijectivityLoss, etc.

Models:
- FMNet: Functional Map Network
- RobustFMNet: Robust Functional Map Network

Evaluation:
- TrainedModelWrapper: Make trained models work with experiment framework
- ModelEvaluator: High-level interface for model evaluation

Presets:
- ModelPresets: Pre-configured model architectures
- TrainingPresets: Pre-configured training setups
- quick_train: One-liner training setup

Note: Presets are centralized in the experiment module.
Import from geomfum.experiment:
    from geomfum.experiment import ModelPresets, TrainingPresets, quick_train

"""

from geomfum.learning.losses import LossManager
from geomfum.learning.trainer import DeepFunctionalMapTrainer
from geomfum.learning.wrappers import (
    ModelEvaluator,
    TrainedModelWrapper,
    load_trained_model,
)


def __getattr__(name):
    """Lazy import for backward compatibility with presets."""
    if name in ("ModelPresets", "TrainingPresets", "quick_train"):
        from geomfum.experiment.presets import (
            ModelPresets,
            TrainingPresets,
            quick_train,
        )

        return {"ModelPresets": ModelPresets, "TrainingPresets": TrainingPresets, "quick_train": quick_train}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Training
    "DeepFunctionalMapTrainer",
    "LossManager",
    # Presets (lazy import for backward compatibility)
    # RECOMMENDED: Import from geomfum.experiment instead
    "ModelPresets",
    "TrainingPresets",
    "quick_train",
    # Evaluation
    "TrainedModelWrapper",
    "ModelEvaluator",
    "load_trained_model",
]
