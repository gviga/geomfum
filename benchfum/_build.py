"""JSON-to-Python factory for geomfum matchers, models, and trainers.

This module belongs in ``benchfum`` because JSON config management is a
benchmarking concern, not a core algorithm concern.  ``geomfum`` contains
only classes; ``benchfum`` owns the mapping from config dicts/files to
instantiated objects.

The builder is fully generic: every JSON key must match the Python constructor
parameter name exactly, and every nested Python object must have a ``"type"``
key naming its class.  There are no special-case builders per type.

Keys prefixed with ``_`` (e.g. ``_description``, ``_name``) are treated as
metadata and silently ignored during construction.

Learning classes (``FMNet``, ``RobustFMNet``, losses, trainer) are added to
the registry automatically when ``torch`` is available.

Usage — matchers
----------------
>>> from benchfum._build import build_matcher, build_matcher_from_json
>>>
>>> matcher = build_matcher({
...     "type": "FunctionalMapMatcher",
...     "fmap_size": 30,
...     "descriptor_pipeline": {
...         "type": "DescriptorPipeline",
...         "steps": [
...             {"type": "WaveKernelSignature", "n_domain": 200, "k": 200},
...             {"type": "L2InnerNormalizer"},
...         ],
...     },
... })
>>> matcher = build_matcher_from_json("my_method.json")

Usage — models
--------------
>>> from benchfum._build import build_model, build_model_from_json
>>>
>>> model = build_model({
...     "type": "FMNet",
...     "feature_extractor": {
...         "type": "FeatureExtractor",
...         "which": "diffusionnet",
...         "k": 200,
...         "in_channels": 128,
...         "descriptor": {"type": "WaveKernelSignature", "n_domain": 128},
...     },
...     "fmap_module": {
...         "type": "ForwardFunctionalMap",
...         "lmbda": 1000.0,
...         "resolvent_gamma": 1.0,
...         "bijective": True,
...     },
... }, device="cuda")
>>> model = build_model_from_json("my_model.json", device="cuda")

Usage — trainers
----------------
>>> from benchfum._build import build_trainer, build_trainer_from_json
>>>
>>> trainer = build_trainer({
...     "optimizer": {"type": "Adam", "lr": 1e-3},
...     "epochs": 50,
...     "train_loss_manager": {
...         "type": "LossManager",
...         "losses": [
...             {"type": "OrthonormalityLoss", "weight": 1.0},
...             {"type": "BijectivityLoss", "weight": 1.0},
...         ],
...     },
... }, model=model, train_set=train_pairs, val_set=val_pairs)
>>> trainer = build_trainer_from_json("my_training.json", model, train_set, val_set)
"""

import functools
import json


def _sync_module_device_attributes(module, device):
    """Best-effort sync for custom `.device` attrs after `module.to(device)`.

    Some geomfum wrappers store device in custom attributes and rely on that
    field in `forward()`. `nn.Module.to(...)` moves parameters/buffers but does
    not update arbitrary attributes.
    """
    import torch

    resolved_device = torch.device(device)
    for submodule in module.modules():
        if hasattr(submodule, "device"):
            try:
                submodule.device = resolved_device
            except Exception:
                pass


@functools.lru_cache(maxsize=None)
def _build_component_registry():
    """Build the mapping from type-name strings to geomfum classes.

    Imported lazily to avoid circular imports at module load time.
    To support a new geomfum class in JSON configs, add it here.

    The result is cached so imports only happen once per process.
    Learning classes are added only when ``torch`` is available.
    """
    from geomfum.convert import (
        NeighborFinder,
        P2pFromFmConverter,
        SoftmaxNeighborFinder,
    )
    from geomfum.descriptor.pipeline import (
        ArangeSubsampler,
        DescriptorPipeline,
        L2InnerNormalizer,
    )
    from geomfum.descriptor.spectral import (
        HeatKernelSignature,
        LandmarkHeatKernelSignature,
        LandmarkWaveKernelSignature,
        WaveKernelSignature,
    )
    from geomfum.functional_map import (
        FunctionalMapOptimizer,
        LBFactorBuilder,
        MultFactorBuilder,
        OrientFactorBuilder,
        SDPFactorBuilder,
    )
    from geomfum.matchers import FeatureMatcher, FunctionalMapMatcher
    from geomfum.refine import (
        AdjointBijectiveZoomOut,
        CorrespondenceRefinementPipeline,
        IcpRefiner,
        IdentityRefiner,
        OrthogonalRefiner,
        RefinementPipeline,
        ZoomOut,
    )

    registry = {
        # --- Matchers ---
        "FunctionalMapMatcher": FunctionalMapMatcher,
        "FeatureMatcher": FeatureMatcher,
        # --- Descriptors ---
        "WaveKernelSignature": WaveKernelSignature,
        "HeatKernelSignature": HeatKernelSignature,
        "LandmarkWaveKernelSignature": LandmarkWaveKernelSignature,
        "LandmarkHeatKernelSignature": LandmarkHeatKernelSignature,
        # --- Pipeline steps ---
        "ArangeSubsampler": ArangeSubsampler,
        "L2InnerNormalizer": L2InnerNormalizer,
        "DescriptorPipeline": DescriptorPipeline,
        # --- Functional map factors ---
        "SDPFactorBuilder": SDPFactorBuilder,
        "LBFactorBuilder": LBFactorBuilder,
        "MultFactorBuilder": MultFactorBuilder,
        "OrientFactorBuilder": OrientFactorBuilder,
        "FunctionalMapOptimizer": FunctionalMapOptimizer,
        # --- Refiners ---
        "IdentityRefiner": IdentityRefiner,
        "OrthogonalRefiner": OrthogonalRefiner,
        "IcpRefiner": IcpRefiner,
        "ZoomOut": ZoomOut,
        "AdjointBijectiveZoomOut": AdjointBijectiveZoomOut,
        "RefinementPipeline": RefinementPipeline,
        "CorrespondenceRefinementPipeline": CorrespondenceRefinementPipeline,
        # --- Converters ---
        "P2pFromFmConverter": P2pFromFmConverter,
        "NeighborFinder": NeighborFinder,
        "SoftmaxNeighborFinder": SoftmaxNeighborFinder,
    }

    # Learning classes — only when torch is available
    try:
        from geomfum.descriptor.learned import FeatureExtractor
        from geomfum.forward_functional_map import ForwardFunctionalMap
        from geomfum.learning.losses import (
            BijectivityLoss,
            DescriptorCommutativityLoss,
            FmapDescriptorsSupervisionLoss,
            GeodesicError,
            GroundTruthSupervisionLoss,
            LaplacianCommutativityLoss,
            LossManager,
            OrthonormalityLoss,
        )
        from geomfum.learning.models import FMNet, RobustFMNet

        def _feature_extractor_builder(**kwargs):
            return FeatureExtractor.from_registry(**kwargs)

        registry.update(
            {
                # FeatureExtractor uses from_registry(which=...) not __init__
                "FeatureExtractor": _feature_extractor_builder,
                "ForwardFunctionalMap": ForwardFunctionalMap,
                "FMNet": FMNet,
                "RobustFMNet": RobustFMNet,
                "OrthonormalityLoss": OrthonormalityLoss,
                "BijectivityLoss": BijectivityLoss,
                "LaplacianCommutativityLoss": LaplacianCommutativityLoss,
                "DescriptorCommutativityLoss": DescriptorCommutativityLoss,
                "GroundTruthSupervisionLoss": GroundTruthSupervisionLoss,
                "FmapDescriptorsSupervisionLoss": FmapDescriptorsSupervisionLoss,
                "GeodesicError": GeodesicError,
                "LossManager": LossManager,
            }
        )
    except ImportError:
        pass

    return registry


def _build_value(v, registry):
    """Recursively build a value: instantiate objects, recurse into lists, pass scalars."""
    if isinstance(v, dict) and "type" in v:
        return _build_component(v, registry)
    if isinstance(v, list):
        return [_build_value(item, registry) for item in v]
    return v


def _build_component(config, registry):
    """Build a single component from a config dict with a ``"type"`` key.

    Keys starting with ``_`` are metadata and are silently ignored.
    All other keys (except ``"type"``) are passed as keyword arguments
    to the class constructor, with values recursively resolved.
    """
    type_name = config.get("type")
    if type_name is None:
        raise ValueError(
            f"Component config is missing a 'type' key: {config!r}. "
            "Each component dict must have a 'type' key naming its class."
        )

    cls = registry.get(type_name)
    if cls is None:
        raise ValueError(
            f"Unknown component type: {type_name!r}. "
            f"Available types: {sorted(registry)}"
        )

    kwargs = {
        k: _build_value(v, registry)
        for k, v in config.items()
        if k != "type" and not k.startswith("_")
    }
    return cls(**kwargs)


def _build_optimizer(config, model):
    """Build a torch optimizer, binding ``model.parameters()``."""
    import torch.optim

    opt_type = config.get("type", "Adam")
    lr = config.get("lr", 1e-3)
    wd = config.get("weight_decay", 1e-5)
    params = model.parameters()
    if opt_type == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if opt_type == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt_type == "SGD":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=wd, momentum=config.get("momentum", 0.9)
        )
    raise ValueError(
        f"Unknown optimizer type {opt_type!r}. Supported: 'Adam', 'AdamW', 'SGD'."
    )


def _build_scheduler(config, optimizer):
    """Build a torch lr_scheduler, binding to ``optimizer``."""
    import torch.optim.lr_scheduler

    if config is None:
        return None
    stype = config.get("type", "StepLR")
    if stype == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.1),
        )
    if stype == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("T_max", 50),
            eta_min=config.get("eta_min", 0),
        )
    if stype == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get("mode", "min"),
            factor=config.get("factor", 0.1),
            patience=config.get("patience", 10),
        )
    raise ValueError(
        f"Unknown scheduler type {stype!r}. "
        "Supported: 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'."
    )


# ---------------------------------------------------------------------------
# Public API — matchers
# ---------------------------------------------------------------------------


def build_matcher(config):
    """Build a matcher from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Must include a ``"type"`` key naming the matcher class.

    Returns
    -------
    matcher : BaseMatcher
    """
    registry = _build_component_registry()
    return _build_component(config, registry)


def build_matcher_from_json(path):
    """Build a matcher from a JSON file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    matcher : BaseMatcher
    """
    with open(path) as f:
        config = json.load(f)
    return build_matcher(config)


# ---------------------------------------------------------------------------
# Public API — refiners
# ---------------------------------------------------------------------------


def build_refiner(config):
    """Build a refiner from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Must include a ``"type"`` key (e.g. ``"IcpRefiner"``,
        ``"RefinementPipeline"``).

    Returns
    -------
    refiner : refiner instance
    """
    registry = _build_component_registry()
    return _build_component(config, registry)


def build_refiner_from_json(path):
    """Build a refiner from a JSON file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    refiner : refiner instance
    """
    with open(path) as f:
        config = json.load(f)
    return build_refiner(config)


# ---------------------------------------------------------------------------
# Public API — learning models
# ---------------------------------------------------------------------------


def build_model(config, device=None):
    """Build a learning model from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Must include a ``"type"`` key (e.g. ``"FMNet"``, ``"RobustFMNet"``).
    device : str, optional
        If given, the model is moved to this device after construction.

    Returns
    -------
    model : nn.Module
    """
    registry = _build_component_registry()
    model = _build_component(config, registry)
    if device is not None:
        model = model.to(device)
        _sync_module_device_attributes(model, device)
    return model


def build_model_from_json(path, device=None):
    """Build a learning model from a JSON file.

    Parameters
    ----------
    path : str or Path
    device : str, optional

    Returns
    -------
    model : nn.Module
    """
    with open(path) as f:
        config = json.load(f)
    return build_model(config, device=device)


# ---------------------------------------------------------------------------
# Public API — trainers
# ---------------------------------------------------------------------------


def build_trainer(config, model, train_set, val_set):
    """Build a ``DeepFunctionalMapTrainer`` from a configuration dictionary.

    ``optimizer`` and ``scheduler`` are built with ``model.parameters()``
    and the optimizer respectively (they cannot be purely serialized).
    All other keys are resolved recursively by the generic builder.

    Parameters
    ----------
    config : dict
        Trainer configuration.  Special keys:

        - ``optimizer`` : ``{"type": "Adam", "lr": 1e-3, ...}``
        - ``scheduler`` : ``{"type": "StepLR", ...}`` or ``null``
        - ``train_loss_manager`` / ``val_loss_manager`` : ``LossManager``
          config dicts built recursively.
        - Scalar kwargs forwarded directly: ``epochs``, ``device``,
          ``checkpoint_path``, ``monitor_metric``, ``mode``,
          ``scheduler_step_on``.

    model : nn.Module
    train_set, val_set : Dataset

    Returns
    -------
    trainer : DeepFunctionalMapTrainer
    """
    from geomfum.learning.trainer import DeepFunctionalMapTrainer

    registry = _build_component_registry()

    optimizer = (
        _build_optimizer(config["optimizer"], model) if "optimizer" in config else None
    )
    scheduler = _build_scheduler(config.get("scheduler"), optimizer)

    kwargs = {
        k: _build_value(v, registry)
        for k, v in config.items()
        if k != "type" and not k.startswith("_") and k not in ("optimizer", "scheduler")
    }
    return DeepFunctionalMapTrainer(
        model=model,
        train_set=train_set,
        val_set=val_set,
        optimizer=optimizer,
        scheduler=scheduler,
        **kwargs,
    )


def build_trainer_from_json(path, model, train_set, val_set):
    """Build a trainer from a JSON file.

    Parameters
    ----------
    path : str or Path
    model : nn.Module
    train_set, val_set : Dataset

    Returns
    -------
    trainer : DeepFunctionalMapTrainer
    """
    with open(path) as f:
        config = json.load(f)
    return build_trainer(config, model, train_set, val_set)
