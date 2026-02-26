"""Learning module for deep functional maps and neural shape matching.

Training
--------
- DeepFunctionalMapTrainer: Training loop for deep functional maps
- LossManager: Manages multiple loss functions with weights
- Various loss functions: OrthonormalityLoss, BijectivityLoss, etc.

Models
------
- FMNet: Functional Map Network
- RobustFMNet: Robust Functional Map Network

Evaluation
----------
- TrainedModelWrapper: Make trained models work with experiment framework
- ModelEvaluator: High-level interface for model evaluation

Note
----
Learning presets (ModelPresets, TrainingPresets, quick_train) have moved to
``benchfum.learning_presets``.
"""

from geomfum.learning.losses import LossManager
from geomfum.learning.trainer import DeepFunctionalMapTrainer
from geomfum.learning.wrappers import (
    ModelEvaluator,
    TestTimeRefiner,
    TrainedModelWrapper,
    load_trained_model,
)

__all__ = [
    # Training
    "DeepFunctionalMapTrainer",
    "LossManager",
    # Evaluation
    "TrainedModelWrapper",
    "ModelEvaluator",
    "TestTimeRefiner",
    "load_trained_model",
]
