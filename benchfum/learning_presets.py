"""Preset configurations for learning-based shape matching methods.

This module provides pre-configured model architectures and training setups
for deep functional maps approaches (FMNet, RobustFMNet).

Usage
-----
>>> from benchfum.learning_presets import ModelPresets, TrainingPresets, quick_train
>>>
>>> # Build a model
>>> model = ModelPresets.build("fmnet_diffusion_standard", device="cuda")
>>>
>>> # Create a trainer
>>> trainer = TrainingPresets.create_trainer(
...     preset="unsupervised_standard",
...     model=model,
...     train_set=train_pairs,
...     val_set=val_pairs,
... )
>>> trainer.train()
>>>
>>> # Or use the one-liner
>>> trainer = quick_train("unsupervised_standard", train_pairs, val_pairs)
>>> trainer.train()
"""

import copy
from typing import Any, Dict

import torch

from geomfum.convert import P2pFromFmConverter, SoftmaxNeighborFinder
from geomfum.descriptor.spectral import WaveKernelSignature
from geomfum.forward_functional_map import ForwardFunctionalMap
from geomfum.learning.losses import (
    BijectivityLoss,
    DescriptorCommutativityLoss,
    GeodesicError,
    GroundTruthSupervisionLoss,
    LaplacianCommutativityLoss,
    LossManager,
    OrthonormalityLoss,
)
from geomfum.learning.models import FMNet, RobustFMNet
from geomfum.learning.trainer import DeepFunctionalMapTrainer

# ============================================================================
# UTILITY: deep config merge
# ============================================================================


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Deep merge overrides into base configuration.

    Supports two override styles:
    1. Nested dicts: ``{"feature_extractor": {"k_eig": 300}}``
    2. Double-underscore: ``{"feature_extractor__k_eig": 300}``

    Parameters
    ----------
    base : dict
        Base configuration (not modified in place; a deepcopy is expected
        to be passed by callers).
    overrides : dict
        Overrides to apply.

    Returns
    -------
    merged : dict
        Merged configuration.
    """
    expanded: dict = {}
    for key, value in overrides.items():
        if "__" in key:
            parts = key.split("__")
            current = expanded
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        else:
            expanded[key] = value

    _deep_merge_recursive(base, expanded)
    return base


def _deep_merge_recursive(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge_recursive(base[key], value)
        else:
            base[key] = value


# ============================================================================
# MODEL PRESETS
# ============================================================================


class ModelPresets:
    """Pre-configured model architectures for learning-based shape matching.

    Available presets
    -----------------
    - ``fmnet_diffusion_small``    : FMNet + small DiffusionNet (fast training)
    - ``fmnet_diffusion_standard`` : FMNet + standard DiffusionNet (recommended)
    - ``fmnet_diffusion_large``    : FMNet + large DiffusionNet (best accuracy)
    - ``robust_fmnet_small``       : RobustFMNet + small DiffusionNet
    - ``robust_fmnet_standard``    : RobustFMNet + standard DiffusionNet
    - ``robust_fmnet_large``       : RobustFMNet + large DiffusionNet

    Examples
    --------
    >>> model = ModelPresets.build("fmnet_diffusion_standard", device="cuda")
    >>> model = ModelPresets.build("fmnet_diffusion_standard",
    ...     feature_extractor__k_eig=300, device="cuda")
    >>> print(ModelPresets.list_presets())
    """

    _PRESETS: Dict[str, dict] = {
        "fmnet_diffusion_small": {
            "type": "fmnet",
            "feature_extractor": {
                "extractor_type": "diffusionnet",
                "k_eig": 128,
                "in_channels": 64,
                "descriptor_n_domain": 64,
            },
            "fmap_module": {"lmbda": 1e3, "resolvent_gamma": 1.0, "bijective": True},
            "p2p_converter": None,
        },
        "fmnet_diffusion_standard": {
            "type": "fmnet",
            "feature_extractor": {
                "extractor_type": "diffusionnet",
                "k_eig": 200,
                "in_channels": 128,
                "descriptor_n_domain": 128,
            },
            "fmap_module": {"lmbda": 1e3, "resolvent_gamma": 1.0, "bijective": True},
            "p2p_converter": None,
        },
        "fmnet_diffusion_large": {
            "type": "fmnet",
            "feature_extractor": {
                "extractor_type": "diffusionnet",
                "k_eig": 300,
                "in_channels": 256,
                "descriptor_n_domain": 200,
            },
            "fmap_module": {"lmbda": 1e3, "resolvent_gamma": 1.0, "bijective": True},
            "p2p_converter": None,
        },
        "robust_fmnet_small": {
            "type": "robust_fmnet",
            "feature_extractor": {
                "extractor_type": "diffusionnet",
                "k_eig": 128,
                "in_channels": 64,
                "descriptor_n_domain": 64,
            },
            "fmap_module": {"lmbda": 1e3, "resolvent_gamma": 1.0, "bijective": True},
            "p2p_converter": {"n_neighbors": 1, "tau": 0.07},
        },
        "robust_fmnet_standard": {
            "type": "robust_fmnet",
            "feature_extractor": {
                "extractor_type": "diffusionnet",
                "k_eig": 200,
                "in_channels": 128,
                "descriptor_n_domain": 128,
            },
            "fmap_module": {"lmbda": 1e3, "resolvent_gamma": 1.0, "bijective": True},
            "p2p_converter": {"n_neighbors": 1, "tau": 0.07},
        },
        "robust_fmnet_large": {
            "type": "robust_fmnet",
            "feature_extractor": {
                "extractor_type": "diffusionnet",
                "k_eig": 300,
                "in_channels": 256,
                "descriptor_n_domain": 200,
            },
            "fmap_module": {"lmbda": 1e3, "resolvent_gamma": 1.0, "bijective": True},
            "p2p_converter": {"n_neighbors": 1, "tau": 0.07},
        },
    }

    @classmethod
    def list_presets(cls):
        """List available model preset names."""
        return sorted(cls._PRESETS)

    @classmethod
    def describe(cls, name: str) -> Dict[str, Any]:
        """Return the raw configuration dict for a named preset."""
        if name not in cls._PRESETS:
            raise ValueError(
                f"Unknown preset {name!r}. Available: {cls.list_presets()}"
            )
        return copy.deepcopy(cls._PRESETS[name])

    @classmethod
    def build(cls, name: str, device: str = None, **overrides) -> "torch.nn.Module":
        """Build a model from a named preset.

        Parameters
        ----------
        name : str
            Preset name.
        device : str, optional
            Device to place model on. Defaults to CUDA if available, else CPU.
        **overrides
            Top-level or double-underscore overrides, e.g.
            ``feature_extractor__k_eig=300``.

        Returns
        -------
        model : torch.nn.Module
        """
        if name not in cls._PRESETS:
            raise ValueError(
                f"Unknown preset {name!r}. Available: {cls.list_presets()}"
            )

        config = _deep_merge(copy.deepcopy(cls._PRESETS[name]), overrides)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_type = config.get("type", "fmnet")
        if model_type == "fmnet":
            model = cls._build_fmnet(config, device)
        elif model_type == "robust_fmnet":
            model = cls._build_robust_fmnet(config, device)
        else:
            raise ValueError(
                f"Unknown model type {model_type!r}. Choose 'fmnet' or 'robust_fmnet'."
            )
        return model.to(device)

    # ---- private builders --------------------------------------------------

    @classmethod
    def _build_fmnet(cls, config, device):
        from geomfum.descriptor.learned import FeatureExtractor

        fe = FeatureExtractor.from_registry(
            which=config["feature_extractor"].get("extractor_type", "diffusionnet"),
            device=device,
            k=config["feature_extractor"].get("k_eig", 128),
            in_channels=config["feature_extractor"].get("in_channels", 64),
            descriptor=WaveKernelSignature(
                n_domain=config["feature_extractor"].get("descriptor_n_domain", 64)
            ),
        )
        fm = cls._build_fmap_module(config["fmap_module"])
        conv = cls._build_p2p_converter(config.get("p2p_converter"))
        return FMNet(feature_extractor=fe, fmap_module=fm, converter=conv)

    @classmethod
    def _build_robust_fmnet(cls, config, device):
        from geomfum.descriptor.learned import FeatureExtractor

        fe = FeatureExtractor.from_registry(
            which=config["feature_extractor"].get("extractor_type", "diffusionnet"),
            device=device,
            k=config["feature_extractor"].get("k_eig", 128),
            in_channels=config["feature_extractor"].get("in_channels", 64),
            descriptor=WaveKernelSignature(
                n_domain=config["feature_extractor"].get("descriptor_n_domain", 64)
            ),
        )
        fm = cls._build_fmap_module(config["fmap_module"])
        conv = cls._build_robust_p2p_converter(config.get("p2p_converter"))
        return RobustFMNet(feature_extractor=fe, fmap_module=fm, converter=conv)

    @staticmethod
    def _build_fmap_module(cfg):
        return ForwardFunctionalMap(
            lmbda=cfg.get("lmbda", 1e3),
            resolvent_gamma=cfg.get("resolvent_gamma", 1.0),
            bijective=cfg.get("bijective", True),
            fmap_shape=cfg.get("fmap_shape"),
        )

    @staticmethod
    def _build_p2p_converter(cfg):
        if cfg is None or isinstance(cfg, P2pFromFmConverter):
            return P2pFromFmConverter() if cfg is None else cfg
        return P2pFromFmConverter(
            neighbor_finder=cfg.get("neighbor_finder"),
            adjoint=cfg.get("adjoint", False),
            bijective=cfg.get("bijective", False),
        )

    @staticmethod
    def _build_robust_p2p_converter(cfg):
        if cfg is None:
            cfg = {}
        if isinstance(cfg, P2pFromFmConverter):
            return cfg
        nf = SoftmaxNeighborFinder(
            n_neighbors=cfg.get("n_neighbors", 1),
            tau=cfg.get("tau", 0.07),
        )
        return P2pFromFmConverter(neighbor_finder=nf)


# ============================================================================
# TRAINING PRESETS
# ============================================================================


class TrainingPresets:
    """Pre-configured training setups for deep functional maps.

    Available presets
    -----------------
    - ``unsupervised_quick``             : 5 epochs, basic losses
    - ``unsupervised_standard``          : 50 epochs (recommended)
    - ``unsupervised_precise``           : 100 epochs, more constraints
    - ``supervised``                     : Ground-truth supervision
    - ``supervised_plus_regularization`` : Supervised + unsupervised reg.

    Examples
    --------
    >>> trainer = TrainingPresets.create_trainer(
    ...     preset="unsupervised_standard",
    ...     model=model,
    ...     train_set=train_pairs,
    ...     val_set=val_pairs,
    ... )
    >>> trainer.train()
    """

    _LOSS_CLASSES = {
        "OrthonormalityLoss": OrthonormalityLoss,
        "BijectivityLoss": BijectivityLoss,
        "LaplacianCommutativityLoss": LaplacianCommutativityLoss,
        "DescriptorCommutativityLoss": DescriptorCommutativityLoss,
        "GroundTruthSupervisionLoss": GroundTruthSupervisionLoss,
        "GeodesicError": GeodesicError,
    }

    _PRESETS: Dict[str, dict] = {
        "unsupervised_quick": {
            "optimizer": {"lr": 1e-3, "weight_decay": 1e-5},
            "scheduler": None,
            "training": {"epochs": 5, "monitor_metric": "GeodesicError", "mode": "min"},
            "train_losses": [
                ("OrthonormalityLoss", {"weight": 1.0}),
                ("BijectivityLoss", {"weight": 1.0}),
            ],
            "val_losses": [("GeodesicError", {})],
        },
        "unsupervised_standard": {
            "optimizer": {"lr": 1e-3, "weight_decay": 1e-5},
            "scheduler": None,
            "training": {
                "epochs": 50,
                "monitor_metric": "GeodesicError",
                "mode": "min",
            },
            "train_losses": [
                ("OrthonormalityLoss", {"weight": 1.0}),
                ("BijectivityLoss", {"weight": 1.0}),
                ("LaplacianCommutativityLoss", {"weight": 1e-3}),
            ],
            "val_losses": [("GeodesicError", {})],
        },
        "unsupervised_precise": {
            "optimizer": {"lr": 1e-3, "weight_decay": 1e-5},
            "scheduler": None,
            "training": {
                "epochs": 100,
                "monitor_metric": "GeodesicError",
                "mode": "min",
            },
            "train_losses": [
                ("OrthonormalityLoss", {"weight": 1.0}),
                ("BijectivityLoss", {"weight": 1.0}),
                ("LaplacianCommutativityLoss", {"weight": 1e-3}),
                ("DescriptorCommutativityLoss", {"weight": 1e-2}),
            ],
            "val_losses": [("GeodesicError", {})],
        },
        "supervised": {
            "optimizer": {"lr": 1e-3, "weight_decay": 1e-5},
            "scheduler": None,
            "training": {
                "epochs": 50,
                "monitor_metric": "GeodesicError",
                "mode": "min",
            },
            "train_losses": [("GroundTruthSupervisionLoss", {"weight": 1.0})],
            "val_losses": [("GeodesicError", {})],
        },
        "supervised_plus_regularization": {
            "optimizer": {"lr": 1e-3, "weight_decay": 1e-5},
            "scheduler": None,
            "training": {
                "epochs": 50,
                "monitor_metric": "GeodesicError",
                "mode": "min",
            },
            "train_losses": [
                ("GroundTruthSupervisionLoss", {"weight": 1.0}),
                ("OrthonormalityLoss", {"weight": 0.1}),
                ("BijectivityLoss", {"weight": 0.1}),
            ],
            "val_losses": [("GeodesicError", {})],
        },
    }

    @classmethod
    def list_presets(cls):
        """List available training preset names."""
        return sorted(cls._PRESETS)

    @classmethod
    def describe(cls, name: str) -> Dict[str, Any]:
        """Return the raw configuration dict for a named preset."""
        if name not in cls._PRESETS:
            raise ValueError(
                f"Unknown preset {name!r}. Available: {cls.list_presets()}"
            )
        return copy.deepcopy(cls._PRESETS[name])

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
    ) -> "DeepFunctionalMapTrainer":
        """Create a trainer from a named preset.

        Parameters
        ----------
        preset : str
            Training preset name.
        model : torch.nn.Module
            Model to train.
        train_set : Dataset
            Training dataset (PairsDataset).
        val_set : Dataset
            Validation dataset (PairsDataset).
        checkpoint_path : str, optional
            Where to save the best checkpoint.
        device : str, optional
            Device for training. Defaults to CUDA if available, else CPU.
        **overrides
            Top-level or double-underscore overrides, e.g.
            ``optimizer__lr=5e-4``, ``training__epochs=100``.

        Returns
        -------
        trainer : DeepFunctionalMapTrainer
        """
        if preset not in cls._PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. Available: {cls.list_presets()}"
            )

        config = _deep_merge(copy.deepcopy(cls._PRESETS[preset]), overrides)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        train_losses = LossManager(cls._build_losses(config["train_losses"]))
        val_losses = LossManager(cls._build_losses(config["val_losses"]))

        optimizer = cls._build_optimizer(config["optimizer"], model)
        scheduler = cls._build_scheduler(config.get("scheduler"), optimizer)

        tc = config["training"]
        return DeepFunctionalMapTrainer(
            model=model,
            train_set=train_set,
            val_set=val_set,
            train_loss_manager=train_losses,
            val_loss_manager=val_losses,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=tc["epochs"],
            device=device,
            checkpoint_path=checkpoint_path,
            monitor_metric=tc["monitor_metric"],
            mode=tc["mode"],
        )

    # ---- private builders --------------------------------------------------

    @classmethod
    def _build_losses(cls, specs):
        return [cls._LOSS_CLASSES[name](**kwargs) for name, kwargs in specs]

    @staticmethod
    def _build_optimizer(cfg, model):
        opt_type = cfg.get("type", "Adam")
        params = model.parameters()
        lr = cfg.get("lr", 1e-3)
        wd = cfg.get("weight_decay", 1e-5)
        if opt_type == "Adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_type == "AdamW":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_type == "SGD":
            return torch.optim.SGD(
                params, lr=lr, weight_decay=wd, momentum=cfg.get("momentum", 0.9)
            )
        raise ValueError(f"Unknown optimizer type {opt_type!r}.")

    @staticmethod
    def _build_scheduler(cfg, optimizer):
        if cfg is None:
            return None
        stype = cfg.get("type", "StepLR")
        if stype == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.get("step_size", 10),
                gamma=cfg.get("gamma", 0.1),
            )
        elif stype == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.get("T_max", 50), eta_min=cfg.get("eta_min", 0)
            )
        elif stype == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=cfg.get("mode", "min"),
                factor=cfg.get("factor", 0.1),
                patience=cfg.get("patience", 10),
            )
        raise ValueError(f"Unknown scheduler type {stype!r}.")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def quick_train(
    preset: str,
    train_set,
    val_set,
    model_preset: str = "fmnet_diffusion_standard",
    checkpoint_path: str = "checkpoints/model.pth",
    device: str = None,
    **kwargs,
) -> "DeepFunctionalMapTrainer":
    """One-liner: build a model and trainer from named presets.

    Parameters
    ----------
    preset : str
        Training preset name (e.g. ``"unsupervised_standard"``).
    train_set : Dataset
        Training dataset (PairsDataset).
    val_set : Dataset
        Validation dataset (PairsDataset).
    model_preset : str
        Model preset name (default ``"fmnet_diffusion_standard"``).
    checkpoint_path : str
        Where to save the best checkpoint.
    device : str, optional
        Device for training.
    **kwargs
        Mixed overrides routed to model or training by prefix:
        model keys: ``feature_extractor``, ``fmap_module``, ``p2p_converter``
        training keys: ``optimizer``, ``scheduler``, ``training``,
        ``train_losses``, ``val_losses``

    Returns
    -------
    trainer : DeepFunctionalMapTrainer

    Examples
    --------
    >>> trainer = quick_train("unsupervised_standard", train_pairs, val_pairs)
    >>> trainer.train()
    >>>
    >>> trainer = quick_train(
    ...     "unsupervised_standard",
    ...     train_pairs, val_pairs,
    ...     model_preset="fmnet_diffusion_large",
    ...     optimizer__lr=5e-4,
    ...     training__epochs=100,
    ... )
    """
    _MODEL_KEYS = {"feature_extractor", "fmap_module", "p2p_converter"}

    model_kwargs, training_kwargs = {}, {}
    for key, value in kwargs.items():
        base = key.split("__")[0] if "__" in key else key
        if base in _MODEL_KEYS:
            model_kwargs[key] = value
        else:
            training_kwargs[key] = value

    model = ModelPresets.build(model_preset, device=device, **model_kwargs)
    return TrainingPresets.create_trainer(
        preset=preset,
        model=model,
        train_set=train_set,
        val_set=val_set,
        checkpoint_path=checkpoint_path,
        device=device,
        **training_kwargs,
    )
