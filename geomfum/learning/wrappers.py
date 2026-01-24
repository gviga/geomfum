"""Wrappers for integrating trained models with the experiment framework.

This module provides utilities to make learning-based models work seamlessly
with the classical matcher experiment framework.
"""

import torch

from geomfum.matcher import BaseMatcher


class TrainedModelWrapper(BaseMatcher):
    """Wrapper to make a trained PyTorch model compatible with experiment framework.

    This wrapper ensures that trained models can be evaluated using the same
    experiment infrastructure as classical matchers.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model that returns CorrespondenceResult.
    device : str, optional
        Device to run inference on (default: "cuda" if available, else "cpu").
    checkpoint_path : str, optional
        Path to checkpoint to load. If None, assumes model is already loaded.

    Examples
    --------
    >>> from geomfum.learning.models import FMNet
    >>> from geomfum.learning.wrappers import TrainedModelWrapper
    >>> from geomfum.experiment import Experiment
    >>>
    >>> # Load trained model
    >>> model = FMNet(...)
    >>> wrapped_model = TrainedModelWrapper(
    ...     model,
    ...     checkpoint_path="checkpoints/best_model.pth"
    ... )
    >>>
    >>> # Use in experiments just like a classical matcher
    >>> experiment = Experiment(wrapped_model, dataset)
    >>> results = experiment.run()
    """

    def __init__(self, model, device=None, checkpoint_path=None):
        self.model = model

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Assume checkpoint is just the state dict
            self.model.load_state_dict(checkpoint)

    def __call__(self, shape_a, shape_b, bidirectional=False):
        """Compute correspondence between two shapes.

        Parameters
        ----------
        shape_a : Shape
            First shape (target for p2p21).
        shape_b : Shape
            Second shape (source for p2p21).
        bidirectional : bool
            If True, compute correspondences in both directions.

        Returns
        -------
        result : CorrespondenceResult
            Correspondence result.
        """
        # Ensure model is in eval mode and on correct device
        self.model.eval()

        with torch.no_grad():
            # Call model
            result = self.model(shape_a, shape_b, bidirectional=bidirectional)

        return result


class ModelEvaluator:
    """Utility for evaluating trained models with automatic device management.

    This class provides a high-level interface for loading and evaluating
    trained models, handling device placement and checkpoint loading automatically.

    Parameters
    ----------
    model_class : class
        Model class (not instance).
    model_kwargs : dict
        Arguments to pass to model constructor.
    checkpoint_path : str
        Path to trained checkpoint.
    device : str, optional
        Device for inference (default: auto-detect).

    Examples
    --------
    >>> from geomfum.learning.models import FMNet
    >>> from geomfum.learning.wrappers import ModelEvaluator
    >>>
    >>> evaluator = ModelEvaluator(
    ...     model_class=FMNet,
    ...     model_kwargs={'feature_extractor': ..., 'fmap_module': ...},
    ...     checkpoint_path='best_model.pth'
    ... )
    >>>
    >>> # Use directly in experiments
    >>> experiment = Experiment(evaluator, dataset)
    >>> results = experiment.run()
    """

    def __init__(self, model_class, model_kwargs, checkpoint_path, device=None):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.checkpoint_path = checkpoint_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Build and load model
        self._build_model()

    def _build_model(self):
        """Build model and load checkpoint."""
        # Create model instance
        self.model = self.model_class(**self.model_kwargs)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(self, shape_a, shape_b, bidirectional=False):
        """Evaluate model on a pair of shapes.

        Parameters
        ----------
        shape_a : Shape
            First shape.
        shape_b : Shape
            Second shape.
        bidirectional : bool
            Whether to compute bidirectional correspondence.

        Returns
        -------
        result : CorrespondenceResult
            Correspondence result.
        """
        self.model.eval()

        with torch.no_grad():
            result = self.model(shape_a, shape_b, bidirectional=bidirectional)

        return result

    def eval(self):
        """Set model to evaluation mode (for compatibility)."""
        self.model.eval()
        return self


def load_trained_model(checkpoint_path, model, device=None):
    """Load a trained model for evaluation.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file.
    model : nn.Module
        Model instance to load weights into.
    device : str, optional
        Device to load model on.

    Returns
    -------
    wrapped_model : TrainedModelWrapper
        Wrapped model ready for evaluation.
    """
    return TrainedModelWrapper(
        model=model, device=device, checkpoint_path=checkpoint_path
    )
