"""Component registry and matcher factory for JSON-based configuration.

This module provides the machinery for building matchers from configuration
dictionaries or JSON files, enabling serializable, shareable method definitions.

The component registry maps string type names to classes, allowing full
round-trip serialization of any matcher configuration.

Usage
-----
>>> # Build from a config dict
>>> config = {
...     "type": "FunctionalMapMatcher",
...     "fmap_size": 30,
...     "descriptor_pipeline": [
...         {"type": "WaveKernelSignature", "n_domain": 200, "k": 200},
...         {"type": "ArangeSubsampler", "subsample_step": 10},
...         {"type": "L2InnerNormalizer"},
...     ],
...     "fmap_optimizer": {
...         "factors": [
...             {"type": "SDPFactorBuilder", "weight": 1.0},
...             {"type": "LBFactorBuilder", "weight": 0.01},
...         ]
...     },
...     "refiner": [
...         {"type": "IcpRefiner", "nit": 10},
...         {"type": "ZoomOut", "nit": 20, "step": 5},
...     ],
... }
>>> matcher = build_matcher(config)

>>> # Load from a JSON file
>>> matcher = build_matcher_from_json("my_method.json")
"""

import json


def _build_component_registry():
    """Build the mapping from type-name strings to classes.

    Imported lazily to avoid circular imports at module load time.
    """
    from geomfum.convert import NeighborFinder, P2pFromFmConverter, SoftmaxNeighborFinder
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
    from geomfum.matcher import FeatureMatcher, FunctionalMapMatcher
    from geomfum.refine import (
        AdjointBijectiveZoomOut,
        CorrespondenceRefinementPipeline,
        IcpRefiner,
        IdentityRefiner,
        OrthogonalRefiner,
        RefinementPipeline,
        ZoomOut,
    )

    return {
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


def _build_component(config, registry):
    """Recursively build a single component from a config dict.

    Parameters
    ----------
    config : dict
        Configuration dict with a "type" key.
    registry : dict
        Component registry mapping type names to classes.

    Returns
    -------
    component : object
        Instantiated component.
    """
    if not isinstance(config, dict):
        return config

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

    kwargs = {k: v for k, v in config.items() if k != "type"}
    return cls(**kwargs)


def _build_descriptor_pipeline(steps_config, registry):
    from geomfum.descriptor.pipeline import DescriptorPipeline

    steps = [_build_component(s, registry) for s in steps_config]
    return DescriptorPipeline(steps)


def _build_fm_optimizer(opt_config, registry):
    from geomfum.functional_map import FunctionalMapOptimizer

    factors = [_build_component(f, registry) for f in opt_config.get("factors", [])]
    extra = {k: v for k, v in opt_config.items() if k not in ("factors", "type")}
    return FunctionalMapOptimizer(factor_builders=factors, **extra)


def _build_refiner(refiner_config, registry):
    """Build a RefinementPipeline (for FunctionalMapMatcher) from config.

    Accepts either a list of refiner configs (shorthand) or a single dict
    with type="RefinementPipeline".
    """
    from geomfum.refine import RefinementPipeline

    if isinstance(refiner_config, list):
        refiners = [_build_component(r, registry) for r in refiner_config]
        return RefinementPipeline(refiners)
    return _build_component(refiner_config, registry)


def _build_corr_refiner(refiner_config, registry):
    """Build a CorrespondenceRefinementPipeline (for FeatureMatcher) from config."""
    from geomfum.refine import CorrespondenceRefinementPipeline

    if isinstance(refiner_config, list):
        refiners = [_build_component(r, registry) for r in refiner_config]
        return CorrespondenceRefinementPipeline(refiners)
    return _build_component(refiner_config, registry)


def build_matcher(config):
    """Build a matcher from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dict. Must include a ``"type"`` key set to either
        ``"FunctionalMapMatcher"`` or ``"FeatureMatcher"``.

    Returns
    -------
    matcher : BaseMatcher
        Configured matcher instance.

    Examples
    --------
    Functional map matcher:

    >>> config = {
    ...     "type": "FunctionalMapMatcher",
    ...     "fmap_size": 30,
    ...     "descriptor_pipeline": [
    ...         {"type": "WaveKernelSignature", "n_domain": 200, "k": 200},
    ...         {"type": "ArangeSubsampler", "subsample_step": 10},
    ...         {"type": "L2InnerNormalizer"},
    ...     ],
    ...     "fmap_optimizer": {
    ...         "factors": [
    ...             {"type": "SDPFactorBuilder", "weight": 1.0},
    ...             {"type": "LBFactorBuilder", "weight": 0.01},
    ...         ]
    ...     },
    ...     "refiner": [
    ...         {"type": "IcpRefiner", "nit": 10},
    ...         {"type": "ZoomOut", "nit": 20, "step": 5},
    ...     ],
    ... }
    >>> matcher = build_matcher(config)

    Feature matcher:

    >>> config = {
    ...     "type": "FeatureMatcher",
    ...     "descriptor_pipeline": [
    ...         {"type": "WaveKernelSignature", "n_domain": 200, "k": 200},
    ...         {"type": "L2InnerNormalizer"},
    ...     ],
    ... }
    >>> matcher = build_matcher(config)
    """
    registry = _build_component_registry()
    matcher_type = config.get("type")

    if matcher_type == "FunctionalMapMatcher":
        return _build_fm_matcher(config, registry)
    elif matcher_type == "FeatureMatcher":
        return _build_feature_matcher(config, registry)
    else:
        raise ValueError(
            f"Unknown matcher type: {matcher_type!r}. "
            "Supported types: 'FunctionalMapMatcher', 'FeatureMatcher'."
        )


def _build_fm_matcher(config, registry):
    from geomfum.matcher import FunctionalMapMatcher

    kwargs = {}
    if "fmap_size" in config:
        kwargs["fmap_size"] = config["fmap_size"]
    if "descriptor_pipeline" in config:
        kwargs["descriptor_pipeline"] = _build_descriptor_pipeline(
            config["descriptor_pipeline"], registry
        )
    if "fmap_optimizer" in config:
        kwargs["fmap_optimizer"] = _build_fm_optimizer(
            config["fmap_optimizer"], registry
        )
    if "refiner" in config:
        kwargs["refiner"] = _build_refiner(config["refiner"], registry)
    if "p2p_converter" in config:
        kwargs["p2p_converter"] = _build_component(config["p2p_converter"], registry)
    return FunctionalMapMatcher(**kwargs)


def _build_feature_matcher(config, registry):
    from geomfum.matcher import FeatureMatcher

    kwargs = {}
    if "descriptor_pipeline" in config:
        kwargs["descriptor_pipeline"] = _build_descriptor_pipeline(
            config["descriptor_pipeline"], registry
        )
    if "neighbor_finder" in config:
        kwargs["neighbor_finder"] = _build_component(
            config["neighbor_finder"], registry
        )
    if "refiner" in config:
        kwargs["refiner"] = _build_corr_refiner(config["refiner"], registry)
    return FeatureMatcher(**kwargs)


def build_matcher_from_json(path):
    """Build a matcher from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to a JSON configuration file.

    Returns
    -------
    matcher : BaseMatcher
        Configured matcher instance.

    Examples
    --------
    >>> matcher = build_matcher_from_json("my_method.json")
    >>> result = matcher(shape_a, shape_b)
    """
    with open(path) as f:
        config = json.load(f)
    return build_matcher(config)


def build_refiner(config):
    """Build a refiner from a configuration dictionary or list.

    Parameters
    ----------
    config : dict or list
        A single refiner dict (e.g. ``{"type": "IcpRefiner", "nit": 10}``)
        or a list of refiner dicts (builds a ``RefinementPipeline``).

    Returns
    -------
    refiner : refiner instance or RefinementPipeline

    Examples
    --------
    Single refiner:

    >>> refiner = build_refiner({"type": "IcpRefiner", "nit": 10})

    Pipeline:

    >>> refiner = build_refiner([
    ...     {"type": "IcpRefiner", "nit": 10},
    ...     {"type": "ZoomOut",    "nit": 20, "step": 5},
    ... ])
    """
    registry = _build_component_registry()
    return _build_refiner(config, registry)
