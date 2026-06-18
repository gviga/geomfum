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

Evaluation / test-time optimization
-----------------------------------
Use ``geomfum.matcher.DeepFMMatcher`` to run a (possibly pretrained) model for
inference and, optionally, optimize it per pair with a given loss. It replaces
the former ``TrainedModelWrapper`` / ``ModelEvaluator`` / ``TestTimeRefiner``
wrappers. Plain models can also be passed directly to the experiment framework.

Note
----
Learning presets (ModelPresets, TrainingPresets, quick_train) have moved to
``benchfum.learning_presets``.
"""

from geomfum.learning.losses import LossManager
from geomfum.learning.trainer import DeepFunctionalMapTrainer

__all__ = [
    # Training
    "DeepFunctionalMapTrainer",
    "LossManager",
]
