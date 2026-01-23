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

Presets (imported from experiment module):
- ModelPresets: Pre-configured model architectures
- TrainingPresets: Pre-configured training setups
- quick_train: One-liner training setup

Note: Presets are now centralized in the experiment module for consistency.
You can still import them from here for backward compatibility, but we
recommend importing from geomfum.experiment:

>>> from geomfum.experiment import ModelPresets, TrainingPresets, quick_train

Examples
--------
Quick training with presets (RECOMMENDED):
>>> from geomfum.experiment import quick_train
>>>
>>> trainer = quick_train(
...     preset="unsupervised_standard",
...     train_set=train_pairs,
...     val_set=val_pairs,
... )
>>> trainer.train()

Using presets separately:
>>> from geomfum.experiment import ModelPresets, TrainingPresets
>>>
>>> # Build model from preset
>>> model = ModelPresets.build("fmnet_diffusion_standard")
>>>
>>> # Create trainer from preset
>>> trainer = TrainingPresets.create_trainer(
...     preset="unsupervised_standard",
...     model=model,
...     train_set=train_pairs,
...     val_set=val_pairs,
... )
>>> trainer.train()

Manual training (advanced):
>>> from geomfum.learning import DeepFunctionalMapTrainer, LossManager
>>> from geomfum.learning.losses import OrthonormalityLoss, BijectivityLoss
>>>
>>> losses = [OrthonormalityLoss(1.0), BijectivityLoss(1.0)]
>>> trainer = DeepFunctionalMapTrainer(
...     model=model,
...     train_set=train_dataset,
...     val_set=val_dataset,
...     train_loss_manager=LossManager(losses),
...     epochs=100,
... )
>>> trainer.train()

Evaluate trained model:
>>> from geomfum.learning import load_trained_model
>>> from geomfum.experiment import Experiment
>>>
>>> model = load_trained_model("checkpoint.pth", model_instance)
>>> experiment = Experiment(model, test_dataset)
>>> results = experiment.run()
"""

# Import presets from experiment module for backward compatibility
# Users should prefer: from geomfum.experiment import ModelPresets, ...
from geomfum.experiment.presets import (
    ModelPresets,
    TrainingPresets,
    quick_train,
)
from geomfum.learning.losses import LossManager
from geomfum.learning.trainer import DeepFunctionalMapTrainer
from geomfum.learning.wrappers import (
    ModelEvaluator,
    TrainedModelWrapper,
    load_trained_model,
)

__all__ = [
    # Training
    "DeepFunctionalMapTrainer",
    "LossManager",
    # Presets (re-exported from experiment for backward compatibility)
    # RECOMMENDED: Import from geomfum.experiment instead
    "ModelPresets",
    "TrainingPresets",
    "quick_train",
    # Evaluation
    "TrainedModelWrapper",
    "ModelEvaluator",
    "load_trained_model",
]
