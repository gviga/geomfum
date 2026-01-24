"""Preset configurations for experiments.

This module centralizes all preset configurations for both classical matchers
and learning-based methods, providing a unified interface for experiment setup.

Presets provide:
- Pre-configured setups for common use cases
- Easy parameter exploration and comparison
- Consistent API across classical and learning methods
- Configuration serialization/deserialization
"""

import copy
import json
from dataclasses import asdict
from typing import Any, Dict

import torch

from geomfum.descriptor.pipeline import ArangeSubsampler
from geomfum.descriptor.spectral import (
    LandmarkWaveKernelSignature,
    WaveKernelSignature,
)

# Learning-related imports
from geomfum.learning.losses import (
    BijectivityLoss,
    DescriptorCommutativityLoss,
    GeodesicError,
    GroundTruthSupervisionLoss,
    LaplacianCommutativityLoss,
    LossManager,
    OrthonormalityLoss,
)

# Matcher-related imports
from geomfum.matcher import MatcherConfig
from geomfum.refine import IcpRefiner, ZoomOut

# ============================================================================
# MATCHER PRESETS
# ============================================================================


class MatcherPresets:
    """Registry of preset configurations for classical matchers.

    Presets provide sensible starting points for different scenarios:
    - minimal: Bare minimum for quick testing
    - quick: Fast matching with reduced accuracy
    - standard: Balanced speed/accuracy for most use cases
    - precise: High accuracy for benchmarking
    - no_refinement: Standard config without post-processing

    Examples
    --------
    Use a preset directly:
    >>> from geomfum.experiment import MatcherPresets
    >>> from geomfum.matcher import FunctionalMapMatcher
    >>>
    >>> config = MatcherPresets.get("standard")
    >>> matcher = FunctionalMapMatcher(config=config)

    Use preset with overrides:
    >>> config = MatcherPresets.get("standard", sdp_weight=2.0, lb_weight=1e-1)
    >>> matcher = FunctionalMapMatcher(config=config)

    Register custom preset:
    >>> custom_config = MatcherConfig(spectrum_size=150, fmap_size=25)
    >>> MatcherPresets.register("my_preset", custom_config)
    """

    # Preset definitions
    _PRESETS = {
        "minimal": {
            "spectrum_size": 50,
            "fmap_size": 10,
            "sdp_weight": 1.0,
            "lb_weight": 0.0,
            "mult_weight": 0.0,
            "orient_weight": 0.0,
            "refiners": [],
        },
        "quick": {
            "spectrum_size": 100,
            "fmap_size": 20,
            "descriptors": [WaveKernelSignature.from_registry(n_domain=200)],
            "subsamplers": [ArangeSubsampler(subsample_step=5)],
            "sdp_weight": 1.0,
            "lb_weight": 1e-2,
            "mult_weight": 1e-1,
            "orient_weight": 0.0,
            "refiners": [IcpRefiner(nit=5)],
        },
        "standard": {
            "spectrum_size": 200,
            "fmap_size": 30,
            "descriptors": [WaveKernelSignature.from_registry(n_domain=400)],
            "subsamplers": [ArangeSubsampler(subsample_step=10)],
            "sdp_weight": 1.0,
            "lb_weight": 1e-2,
            "mult_weight": 1e-1,
            "orient_weight": 0.0,
            "refiners": [IcpRefiner(nit=10), ZoomOut(nit=10, step=5)],
        },
        "precise": {
            "spectrum_size": 300,
            "fmap_size": 50,
            "descriptors": [
                WaveKernelSignature.from_registry(n_domain=200),
                LandmarkWaveKernelSignature.from_registry(n_domain=200),
            ],
            "subsamplers": [ArangeSubsampler(subsample_step=10)],
            "sdp_weight": 1.0,
            "lb_weight": 1e-2,
            "mult_weight": 1e-1,
            "orient_weight": 1e-1,
            "refiners": [IcpRefiner(nit=15), ZoomOut(nit=20, step=5)],
        },
        "no_refinement": {
            "spectrum_size": 200,
            "fmap_size": 30,
            "descriptors": [WaveKernelSignature.from_registry(n_domain=400)],
            "subsamplers": [ArangeSubsampler(subsample_step=10)],
            "sdp_weight": 1.0,
            "lb_weight": 1e-2,
            "mult_weight": 1e-1,
            "orient_weight": 0.0,
            "refiners": [],
        },
    }

    @classmethod
    def get(cls, name: str, **overrides) -> MatcherConfig:
        """Get a preset configuration with optional overrides.

        Parameters
        ----------
        name : str
            Name of the preset (e.g., "standard", "quick", "precise").
        **overrides
            Keyword arguments to override preset values.
            Only simple parameters can be overridden (weights, sizes).
            For descriptors/refiners, use the returned config and modify.

        Returns
        -------
        config : MatcherConfig
            Configuration instance.

        Raises
        ------
        ValueError
            If preset name is not recognized.

        Examples
        --------
        >>> config = MatcherPresets.get("standard")
        >>> config = MatcherPresets.get("standard", sdp_weight=2.0)
        """
        if name not in cls._PRESETS:
            available = ", ".join(cls.list_presets())
            raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")

        # Deep copy preset to avoid mutation
        config_dict = copy.deepcopy(cls._PRESETS[name])

        # Apply overrides (only for simple types)
        for key, value in overrides.items():
            if key in config_dict:
                config_dict[key] = value
            else:
                raise ValueError(
                    f"Invalid override '{key}' for preset '{name}'. "
                    f"Valid keys: {list(config_dict.keys())}"
                )

        return MatcherConfig(**config_dict)

    @classmethod
    def list_presets(cls):
        """List available preset names.

        Returns
        -------
        list[str]
            List of available preset names.
        """
        return sorted(cls._PRESETS.keys())

    @classmethod
    def register(cls, name: str, config: MatcherConfig):
        """Register a new preset configuration.

        Parameters
        ----------
        name : str
            Name for the preset.
        config : MatcherConfig
            Configuration to register.

        Examples
        --------
        >>> custom = MatcherConfig(spectrum_size=150, fmap_size=25)
        >>> MatcherPresets.register("my_custom", custom)
        >>> config = MatcherPresets.get("my_custom")
        """
        cls._PRESETS[name] = asdict(config)

    @classmethod
    def describe(cls, name: str) -> Dict[str, Any]:
        """Get detailed description of a preset.

        Parameters
        ----------
        name : str
            Preset name.

        Returns
        -------
        dict
            Dictionary with preset parameters.
        """
        if name not in cls._PRESETS:
            raise ValueError(f"Unknown preset '{name}'")

        return copy.deepcopy(cls._PRESETS[name])


# ============================================================================
# MODEL PRESETS (Learning)
# ============================================================================


class ModelPresets:
    """Registry of preset model configurations for learning-based methods.

    Provides pre-configured models for common architectures and feature extractors.

    Available presets:
    - fmnet_diffusion_small: FMNet with small DiffusionNet (fast training)
    - fmnet_diffusion_standard: FMNet with standard DiffusionNet (recommended)
    - fmnet_diffusion_large: FMNet with large DiffusionNet (best accuracy)

    Examples
    --------
    Build a preset model:
    >>> from geomfum.experiment import ModelPresets
    >>>
    >>> model = ModelPresets.build("fmnet_diffusion_standard", device="cuda")

    Build with overrides:
    >>> model = ModelPresets.build(
    ...     "fmnet_diffusion_standard",
    ...     k_eig=300,  # More eigenfunctions
    ...     device="cuda"
    ... )

    List available presets:
    >>> ModelPresets.list_presets()
    """

    _PRESETS = {
        "fmnet_diffusion_small": {
            "feature_extractor_type": "diffusionnet",
            "k_eig": 128,
            "in_channels": 64,
            "descriptor_n_domain": 64,
            "fmap_weight": 1e3,
            "fmap_reduce_dim": 1,
        },
        "fmnet_diffusion_standard": {
            "feature_extractor_type": "diffusionnet",
            "k_eig": 200,
            "in_channels": 128,
            "descriptor_n_domain": 128,
            "fmap_weight": 1e3,
            "fmap_reduce_dim": 1,
        },
        "fmnet_diffusion_large": {
            "feature_extractor_type": "diffusionnet",
            "k_eig": 300,
            "in_channels": 256,
            "descriptor_n_domain": 200,
            "fmap_weight": 1e3,
            "fmap_reduce_dim": 1,
        },
    }

    @classmethod
    def build(cls, name: str, device: str = None, **overrides) -> "torch.nn.Module":
        """Build a model from a preset configuration.

        Parameters
        ----------
        name : str
            Name of the preset.
        device : str, optional
            Device to place model on. If None, uses "cuda" if available.
        **overrides
            Override specific preset parameters.

        Returns
        -------
        model : torch.nn.Module
            Configured model instance.

        Raises
        ------
        ValueError
            If preset name is not recognized.

        Examples
        --------
        >>> model = ModelPresets.build("fmnet_diffusion_standard", device="cuda")
        >>> model = ModelPresets.build("fmnet_diffusion_small", k_eig=150)
        """
        if name not in cls._PRESETS:
            available = ", ".join(cls.list_presets())
            raise ValueError(f"Unknown model preset '{name}'. Available: {available}")

        # Get preset configuration
        config = copy.deepcopy(cls._PRESETS[name])
        config.update(overrides)

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build model based on type
        from geomfum.convert import P2pFromFmConverter
        from geomfum.descriptor.learned import FeatureExtractor
        from geomfum.descriptor.spectral import WaveKernelSignature
        from geomfum.forward_functional_map import ForwardFunctionalMap
        from geomfum.learning.models import FMNet

        # Build feature extractor
        feature_extractor = FeatureExtractor.from_registry(
            which=config["feature_extractor_type"],
            device=device,
            k_eig=config["k_eig"],
            in_channels=config["in_channels"],
            descriptor=WaveKernelSignature(n_domain=config["descriptor_n_domain"]),
        )

        # Build functional map module
        fmap_module = ForwardFunctionalMap(
            weight=config["fmap_weight"],
            reduce_dim=config["fmap_reduce_dim"],
            bidirectional=True,
        )

        # Build complete model
        model = FMNet(
            feature_extractor=feature_extractor,
            fmap_module=fmap_module,
            converter=P2pFromFmConverter(),
        )

        return model.to(device)

    @classmethod
    def list_presets(cls):
        """List available model preset names.

        Returns
        -------
        list[str]
            List of available preset names.
        """
        return sorted(cls._PRESETS.keys())

    @classmethod
    def describe(cls, name: str) -> Dict[str, Any]:
        """Get detailed description of a preset.

        Parameters
        ----------
        name : str
            Preset name.

        Returns
        -------
        dict
            Dictionary with preset parameters.
        """
        if name not in cls._PRESETS:
            raise ValueError(f"Unknown preset '{name}'")
        return copy.deepcopy(cls._PRESETS[name])


# ============================================================================
# TRAINING PRESETS (Learning)
# ============================================================================


class TrainingPresets:
    """Registry of preset training configurations for learning-based methods.

    Provides pre-configured training setups including losses, optimizers,
    and training parameters.

    Available presets:
    - unsupervised_quick: Fast unsupervised training (5 epochs, basic losses)
    - unsupervised_standard: Standard unsupervised training (recommended)
    - unsupervised_precise: Longer training with more constraints
    - supervised: Supervised training with ground truth
    - supervised_plus_regularization: Supervised + unsupervised regularization

    Examples
    --------
    Create a trainer from preset:
    >>> from geomfum.experiment import TrainingPresets, ModelPresets
    >>>
    >>> model = ModelPresets.build("fmnet_diffusion_standard")
    >>> trainer = TrainingPresets.create_trainer(
    ...     preset="unsupervised_standard",
    ...     model=model,
    ...     train_set=train_pairs,
    ...     val_set=val_pairs,
    ... )
    >>> trainer.train()

    With custom parameters:
    >>> trainer = TrainingPresets.create_trainer(
    ...     preset="unsupervised_standard",
    ...     model=model,
    ...     train_set=train_pairs,
    ...     val_set=val_pairs,
    ...     epochs=50,  # Override
    ...     lr=1e-4,    # Override
    ... )
    """

    _PRESETS = {
        "unsupervised_quick": {
            "epochs": 5,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "train_losses": [
                ("OrthonormalityLoss", {"weight": 1.0}),
                ("BijectivityLoss", {"weight": 1.0}),
            ],
            "val_losses": [
                ("GeodesicError", {}),
            ],
            "monitor_metric": "GeodesicError",
            "mode": "min",
        },
        "unsupervised_standard": {
            "epochs": 50,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "train_losses": [
                ("OrthonormalityLoss", {"weight": 1.0}),
                ("BijectivityLoss", {"weight": 1.0}),
                ("LaplacianCommutativityLoss", {"weight": 1e-3}),
            ],
            "val_losses": [
                ("GeodesicError", {}),
            ],
            "monitor_metric": "GeodesicError",
            "mode": "min",
        },
        "unsupervised_precise": {
            "epochs": 100,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "train_losses": [
                ("OrthonormalityLoss", {"weight": 1.0}),
                ("BijectivityLoss", {"weight": 1.0}),
                ("LaplacianCommutativityLoss", {"weight": 1e-3}),
                ("DescriptorCommutativityLoss", {"weight": 1e-2}),
            ],
            "val_losses": [
                ("GeodesicError", {}),
            ],
            "monitor_metric": "GeodesicError",
            "mode": "min",
        },
        "supervised": {
            "epochs": 50,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "train_losses": [
                ("GroundTruthSupervisionLoss", {"weight": 1.0}),
            ],
            "val_losses": [
                ("GeodesicError", {}),
            ],
            "monitor_metric": "GeodesicError",
            "mode": "min",
        },
        "supervised_plus_regularization": {
            "epochs": 50,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "train_losses": [
                ("GroundTruthSupervisionLoss", {"weight": 1.0}),
                ("OrthonormalityLoss", {"weight": 0.1}),
                ("BijectivityLoss", {"weight": 0.1}),
            ],
            "val_losses": [
                ("GeodesicError", {}),
            ],
            "monitor_metric": "GeodesicError",
            "mode": "min",
        },
    }

    # Mapping of loss names to classes
    _LOSS_CLASSES = {
        "OrthonormalityLoss": OrthonormalityLoss,
        "BijectivityLoss": BijectivityLoss,
        "LaplacianCommutativityLoss": LaplacianCommutativityLoss,
        "DescriptorCommutativityLoss": DescriptorCommutativityLoss,
        "GroundTruthSupervisionLoss": GroundTruthSupervisionLoss,
        "GeodesicError": GeodesicError,
    }

    @classmethod
    def _build_losses(cls, loss_specs):
        """Build loss instances from specifications."""
        losses = []
        for loss_name, loss_kwargs in loss_specs:
            loss_class = cls._LOSS_CLASSES[loss_name]
            losses.append(loss_class(**loss_kwargs))
        return losses

    @classmethod
    def create_trainer(
        cls,
        preset: str,
        model,
        train_set,
        val_set,
        checkpoint_path: str = None,
        device: str = None,
        **overrides,
    ):
        """Create a trainer from a preset configuration.

        Parameters
        ----------
        preset : str
            Name of the training preset.
        model : torch.nn.Module
            Model to train.
        train_set : Dataset
            Training dataset (PairsDataset).
        val_set : Dataset
            Validation dataset (PairsDataset).
        checkpoint_path : str, optional
            Path to save checkpoints. If None, no checkpointing.
        device : str, optional
            Device for training. If None, uses "cuda" if available.
        **overrides
            Override preset parameters (epochs, lr, weight_decay, etc.).

        Returns
        -------
        trainer : DeepFunctionalMapTrainer
            Configured trainer instance.

        Raises
        ------
        ValueError
            If preset name is not recognized.

        Examples
        --------
        >>> trainer = TrainingPresets.create_trainer(
        ...     preset="unsupervised_standard",
        ...     model=model,
        ...     train_set=train_pairs,
        ...     val_set=val_pairs,
        ...     checkpoint_path="checkpoints/best.pth",
        ... )
        >>> trainer.train()
        """
        if preset not in cls._PRESETS:
            available = ", ".join(cls.list_presets())
            raise ValueError(
                f"Unknown training preset '{preset}'. Available: {available}"
            )

        # Get preset configuration
        config = copy.deepcopy(cls._PRESETS[preset])

        # Apply overrides
        for key in ["epochs", "lr", "weight_decay", "monitor_metric", "mode"]:
            if key in overrides:
                config[key] = overrides[key]

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build losses
        train_losses = cls._build_losses(config["train_losses"])
        val_losses = cls._build_losses(config["val_losses"])

        train_loss_manager = LossManager(train_losses)
        val_loss_manager = LossManager(val_losses)

        # Build optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        # Build trainer
        from geomfum.learning.trainer import DeepFunctionalMapTrainer

        trainer = DeepFunctionalMapTrainer(
            model=model,
            train_set=train_set,
            val_set=val_set,
            train_loss_manager=train_loss_manager,
            val_loss_manager=val_loss_manager,
            optimizer=optimizer,
            epochs=config["epochs"],
            device=device,
            checkpoint_path=checkpoint_path,
            monitor_metric=config["monitor_metric"],
            mode=config["mode"],
        )

        return trainer

    @classmethod
    def list_presets(cls):
        """List available training preset names.

        Returns
        -------
        list[str]
            List of available preset names.
        """
        return sorted(cls._PRESETS.keys())

    @classmethod
    def describe(cls, name: str) -> Dict[str, Any]:
        """Get detailed description of a preset.

        Parameters
        ----------
        name : str
            Preset name.

        Returns
        -------
        dict
            Dictionary with preset parameters.
        """
        if name not in cls._PRESETS:
            raise ValueError(f"Unknown preset '{name}'")
        return copy.deepcopy(cls._PRESETS[name])


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def quick_train(
    preset: str,
    train_set,
    val_set,
    model_preset: str = "fmnet_diffusion_standard",
    checkpoint_path: str = "checkpoints/model.pth",
    device: str = None,
    **kwargs,
):
    """Quick training with preset configurations (one-liner setup).

    Convenience function that builds model and trainer from presets.

    Parameters
    ----------
    preset : str
        Training preset name.
    train_set : Dataset
        Training dataset.
    val_set : Dataset
        Validation dataset.
    model_preset : str, optional
        Model preset name (default: "fmnet_diffusion_standard").
    checkpoint_path : str, optional
        Path to save checkpoints.
    device : str, optional
        Device for training.
    **kwargs
        Additional overrides for model or training config.

    Returns
    -------
    trainer : DeepFunctionalMapTrainer
        Trainer that can be used to call .train()

    Examples
    --------
    >>> from geomfum.experiment import quick_train
    >>>
    >>> # One-liner training setup
    >>> trainer = quick_train(
    ...     preset="unsupervised_standard",
    ...     train_set=train_pairs,
    ...     val_set=val_pairs,
    ... )
    >>> trainer.train()

    >>> # With custom parameters
    >>> trainer = quick_train(
    ...     preset="unsupervised_standard",
    ...     model_preset="fmnet_diffusion_large",
    ...     train_set=train_pairs,
    ...     val_set=val_pairs,
    ...     epochs=100,
    ...     lr=5e-4,
    ... )
    """
    # Separate model kwargs from training kwargs
    model_kwargs = {}
    training_kwargs = {}

    known_training_params = ["epochs", "lr", "weight_decay"]
    for key, value in kwargs.items():
        if key in known_training_params:
            training_kwargs[key] = value
        else:
            model_kwargs[key] = value

    # Build model
    model = ModelPresets.build(model_preset, device=device, **model_kwargs)

    # Create trainer
    trainer = TrainingPresets.create_trainer(
        preset=preset,
        model=model,
        train_set=train_set,
        val_set=val_set,
        checkpoint_path=checkpoint_path,
        device=device,
        **training_kwargs,
    )

    return trainer


# ============================================================================
# CONFIGURATION UTILITIES (Matchers)
# ============================================================================


def save_config(config: MatcherConfig, path: str):
    """Save matcher configuration to JSON file.

    Parameters
    ----------
    config : MatcherConfig
        Configuration to save.
    path : str
        Path to save the JSON file.

    Note
    ----
    Complex objects (descriptors, refiners) are saved as type information only.
    Full reconstruction requires manual intervention.
    """
    config_dict = asdict(config)

    # Handle non-serializable objects
    for key in ["descriptors", "subsamplers", "normalizers", "refiners"]:
        if config_dict[key] is not None:
            config_dict[key] = [
                {
                    "type": type(item).__name__,
                    "module": type(item).__module__,
                }
                for item in config_dict[key]
            ]

    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)


def load_config_simple(path: str) -> Dict[str, Any]:
    """Load configuration from JSON (returns dict, not MatcherConfig).

    Parameters
    ----------
    path : str
        Path to JSON file.

    Returns
    -------
    dict
        Configuration dictionary.

    Note
    ----
    This loads only simple parameters. Descriptors/refiners need manual reconstruction.
    Use MatcherPresets for full configurations.
    """
    with open(path, "r") as f:
        return json.load(f)
