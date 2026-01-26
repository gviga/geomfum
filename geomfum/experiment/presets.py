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
from typing import Any, Dict

import torch

# Converter imports
from geomfum.convert import NeighborFinder, P2pFromFmConverter

# Descriptor imports
from geomfum.descriptor.learned import FeatureExtractor
from geomfum.descriptor.pipeline import (
    ArangeSubsampler,
    DescriptorPipeline,
    L2InnerNormalizer,
)
from geomfum.descriptor.spectral import (
    LandmarkWaveKernelSignature,
    WaveKernelSignature,
)

# Functional map imports
from geomfum.forward_functional_map import ForwardFunctionalMap
from geomfum.functional_map import (
    FunctionalMapOptimizer,
    LBFactorBuilder,
    MultFactorBuilder,
    OrientFactorBuilder,
    SDPFactorBuilder,
)

# Learning imports
from geomfum.learning.losses import (
    BijectivityLoss,
    DescriptorCommutativityLoss,
    GeodesicError,
    GroundTruthSupervisionLoss,
    LaplacianCommutativityLoss,
    LossManager,
    OrthonormalityLoss,
)
from geomfum.learning.models import FMNet
from geomfum.learning.trainer import DeepFunctionalMapTrainer

# Matcher imports
from geomfum.matcher import BaseMatcher, FeatureMatcher, FunctionalMapMatcher

# Refinement imports
from geomfum.refine import (
    IcpRefiner,
    IdentityRefiner,
    RefinementPipeline,
    ZoomOut,
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Deep merge overrides into base configuration.

    Supports two override styles:
    1. Nested dicts: {"fmap_optimizer": {"fmap_size": 40}}
    2. Double-underscore: {"fmap_optimizer__fmap_size": 40}

    Parameters
    ----------
    base : dict
        Base configuration (will be modified in place).
    overrides : dict
        Overrides to apply.

    Returns
    -------
    merged : dict
        Merged configuration.
    """
    # First, expand double-underscore keys into nested dicts
    expanded_overrides = {}

    for key, value in overrides.items():
        if "__" in key:
            # Expand "fmap_optimizer__fmap_size" to nested dict
            parts = key.split("__")
            current = expanded_overrides
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        else:
            expanded_overrides[key] = value

    # Now recursively merge
    _deep_merge_recursive(base, expanded_overrides)
    return base


def _deep_merge_recursive(base: dict, overrides: dict) -> None:
    """Recursively merge overrides into base dict."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge_recursive(base[key], value)
        else:
            base[key] = value


# ============================================================================
# MATCHER PRESETS
# ============================================================================


class MatcherPresets:
    """Registry of preset matchers for classical shape matching.

    Provides pre-configured matchers for common scenarios:
    - minimal: Bare minimum FunctionalMapMatcher for quick testing
    - quick: Fast FunctionalMapMatcher with reduced accuracy
    - standard: Balanced FunctionalMapMatcher for most use cases
    - precise: High accuracy FunctionalMapMatcher for benchmarking
    - no_refinement: FunctionalMapMatcher without post-processing
    - feature_quick: Fast FeatureMatcher using descriptor nearest neighbors
    - feature_standard: Standard FeatureMatcher configuration

    Configuration Structure
    -----------------------
    Each preset is a dict with hierarchical structure. The "type" field
    determines which matcher class is built.

    For FunctionalMapMatcher (type="functional_map")::

        {
            "type": "functional_map",
            "descriptor_pipeline": {
                "descriptors": [...],     # List of Descriptor instances
                "subsamplers": [...],     # List of Subsampler instances
                "normalizer": ...,        # Normalizer instance or None
            },
            "fmap_optimizer": {
                "fmap_size": int,         # Size of functional map
                "sdp_weight": float,      # Spectral descriptor preservation weight
                "lb_weight": float,       # Laplacian commutativity weight
                "mult_weight": float,     # Multiplication operator weight
                "orient_weight": float,   # Orientation operator weight
            },
            "refinement_pipeline": {
                "refiners": [...],        # List of Refiner instances
            },
            "p2p_converter": None,        # P2pFromFmConverter config or None
        }

    For FeatureMatcher (type="feature")::

        {
            "type": "feature",
            "descriptor_pipeline": {
                "descriptors": [...],     # List of Descriptor instances
                "subsamplers": [...],     # List of Subsampler instances
                "normalizer": ...,        # Normalizer instance or None
            },
            "neighbor_finder": {          # Optional, defaults to n_neighbors=1
                "n_neighbors": int,
            },
        }

    Examples
    --------
    Build a preset matcher:
    >>> from geomfum.experiment import MatcherPresets
    >>>
    >>> matcher = MatcherPresets.build("standard")

    Build a feature matcher:
    >>> matcher = MatcherPresets.build("feature_standard")

    Build with hierarchical overrides:
    >>> matcher = MatcherPresets.build("standard",
    ...     fmap_optimizer={"fmap_size": 40, "lb_weight": 1e-3}
    ... )

    Build with double-underscore overrides:
    >>> matcher = MatcherPresets.build("standard",
    ...     fmap_optimizer__fmap_size=40
    ... )

    List available presets:
    >>> MatcherPresets.list_presets()
    """

    # Preset definitions with hierarchical structure
    _PRESETS = {
        # =====================================================================
        # FunctionalMapMatcher presets
        # =====================================================================
        "minimal": {
            "type": "functional_map",
            "descriptor_pipeline": {
                "descriptors": [WaveKernelSignature.from_registry(n_domain=200)],
                "subsamplers": [ArangeSubsampler(subsample_step=10)],
                "normalizer": None,
            },
            "fmap_optimizer": {
                "fmap_size": 10,
                "sdp_weight": 1.0,
                "lb_weight": 0.0,
                "mult_weight": 0.0,
                "orient_weight": 0.0,
            },
            "refinement_pipeline": {
                "refiners": [],
            },
            "p2p_converter": None,
        },
        "quick": {
            "type": "functional_map",
            "descriptor_pipeline": {
                "descriptors": [WaveKernelSignature.from_registry(n_domain=200)],
                "subsamplers": [ArangeSubsampler(subsample_step=5)],
                "normalizer": None,
            },
            "fmap_optimizer": {
                "fmap_size": 20,
                "sdp_weight": 1.0,
                "lb_weight": 1e-2,
                "mult_weight": 1e-1,
                "orient_weight": 0.0,
            },
            "refinement_pipeline": {
                "refiners": [IcpRefiner(nit=5)],
            },
            "p2p_converter": None,
        },
        "standard": {
            "type": "functional_map",
            "descriptor_pipeline": {
                "descriptors": [WaveKernelSignature.from_registry(n_domain=400)],
                "subsamplers": [ArangeSubsampler(subsample_step=10)],
                "normalizer": None,
            },
            "fmap_optimizer": {
                "fmap_size": 30,
                "sdp_weight": 1.0,
                "lb_weight": 1e-2,
                "mult_weight": 1e-1,
                "orient_weight": 0.0,
            },
            "refinement_pipeline": {
                "refiners": [IcpRefiner(nit=10), ZoomOut(nit=10, step=5)],
            },
            "p2p_converter": None,
        },
        "precise": {
            "type": "functional_map",
            "descriptor_pipeline": {
                "descriptors": [
                    WaveKernelSignature.from_registry(n_domain=200),
                    LandmarkWaveKernelSignature.from_registry(n_domain=200),
                ],
                "subsamplers": [ArangeSubsampler(subsample_step=10)],
                "normalizer": None,
            },
            "fmap_optimizer": {
                "fmap_size": 50,
                "sdp_weight": 1.0,
                "lb_weight": 1e-2,
                "mult_weight": 1e-1,
                "orient_weight": 1e-1,
            },
            "refinement_pipeline": {
                "refiners": [IcpRefiner(nit=15), ZoomOut(nit=20, step=5)],
            },
            "p2p_converter": None,
        },
        "no_refinement": {
            "type": "functional_map",
            "descriptor_pipeline": {
                "descriptors": [WaveKernelSignature.from_registry(n_domain=400)],
                "subsamplers": [ArangeSubsampler(subsample_step=10)],
                "normalizer": None,
            },
            "fmap_optimizer": {
                "fmap_size": 30,
                "sdp_weight": 1.0,
                "lb_weight": 1e-2,
                "mult_weight": 1e-1,
                "orient_weight": 0.0,
            },
            "refinement_pipeline": {
                "refiners": [],
            },
            "p2p_converter": None,
        },
        # =====================================================================
        # FeatureMatcher presets
        # =====================================================================
        "feature_quick": {
            "type": "feature",
            "descriptor_pipeline": {
                "descriptors": [WaveKernelSignature.from_registry(n_domain=200)],
                "subsamplers": [],
                "normalizer": None,
            },
            "neighbor_finder": {
                "n_neighbors": 1,
            },
        },
        "feature_standard": {
            "type": "feature",
            "descriptor_pipeline": {
                "descriptors": [WaveKernelSignature.from_registry(n_domain=400)],
                "subsamplers": [],
                "normalizer": None,
            },
            "neighbor_finder": {
                "n_neighbors": 1,
            },
        },
        "feature_precise": {
            "type": "feature",
            "descriptor_pipeline": {
                "descriptors": [
                    WaveKernelSignature.from_registry(n_domain=200),
                    LandmarkWaveKernelSignature.from_registry(n_domain=200),
                ],
                "subsamplers": [],
                "normalizer": None,
            },
            "neighbor_finder": {
                "n_neighbors": 1,
            },
        },
    }

    @classmethod
    def build(cls, name: str, **overrides) -> "BaseMatcher":
        """Build a matcher from a preset configuration.

        Parameters
        ----------
        name : str
            Name of the preset.
        **overrides
            Override specific preset parameters. Supports hierarchical overrides:
            - Nested dict: fmap_optimizer={"fmap_size": 40}
            - Double-underscore: fmap_optimizer__fmap_size=40

        Returns
        -------
        matcher : BaseMatcher
            Configured matcher instance.

        Raises
        ------
        ValueError
            If preset name is not recognized.

        Examples
        --------
        >>> matcher = MatcherPresets.build("standard")
        >>> matcher = MatcherPresets.build("feature_standard")
        >>> matcher = MatcherPresets.build("standard",
        ...     fmap_optimizer={"fmap_size": 40, "lb_weight": 1e-3}
        ... )
        >>> matcher = MatcherPresets.build("standard",
        ...     fmap_optimizer__fmap_size=40
        ... )
        """
        if name not in cls._PRESETS:
            available = ", ".join(cls.list_presets())
            raise ValueError(f"Unknown matcher preset '{name}'. Available: {available}")

        preset = cls._PRESETS[name]

        # If preset is already a matcher instance, return it directly
        if isinstance(preset, BaseMatcher):
            if overrides:
                import warnings

                warnings.warn(
                    f"Overrides ignored for preset '{name}' which was registered "
                    "as a matcher instance. Register a config dict to use overrides."
                )
            return preset

        # Deep merge preset with overrides
        config = _deep_merge(copy.deepcopy(preset), overrides)

        # Dispatch based on matcher type
        matcher_type = config.get("type", "functional_map")

        if matcher_type == "functional_map":
            return cls._build_functional_map_matcher(config)
        elif matcher_type == "feature":
            return cls._build_feature_matcher(config)
        else:
            raise ValueError(
                f"Unknown matcher type '{matcher_type}'. "
                "Available: 'functional_map', 'feature'"
            )

    @classmethod
    def _build_functional_map_matcher(cls, config: dict) -> FunctionalMapMatcher:
        """Build a FunctionalMapMatcher from config."""
        descriptor_pipeline = cls._build_descriptor_pipeline(
            config["descriptor_pipeline"]
        )
        fmap_optimizer = cls._build_fmap_optimizer(config["fmap_optimizer"])
        refiner = cls._build_refinement_pipeline(config["refinement_pipeline"])
        p2p_converter = cls._build_p2p_converter(config.get("p2p_converter"))

        return FunctionalMapMatcher(
            fmap_size=config["fmap_optimizer"]["fmap_size"],
            descriptor_pipeline=descriptor_pipeline,
            fmap_optimizer=fmap_optimizer,
            refiner=refiner,
            p2p_converter=p2p_converter,
        )

    @classmethod
    def _build_feature_matcher(cls, config: dict) -> FeatureMatcher:
        """Build a FeatureMatcher from config."""
        descriptor_pipeline = cls._build_descriptor_pipeline(
            config["descriptor_pipeline"]
        )
        neighbor_finder = cls._build_neighbor_finder(config.get("neighbor_finder"))

        return FeatureMatcher(
            descriptor_pipeline=descriptor_pipeline,
            neighbor_finder=neighbor_finder,
        )

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
    def register(cls, name: str, matcher_or_config):
        """Register a new preset.

        Parameters
        ----------
        name : str
            Name for the preset.
        matcher_or_config : BaseMatcher or dict
            Either a pre-configured matcher instance, or a hierarchical
            configuration dictionary.

        Examples
        --------
        Register a matcher instance directly:
        >>> custom_matcher = FunctionalMapMatcher(fmap_size=25, ...)
        >>> MatcherPresets.register("my_matcher", custom_matcher)
        >>> matcher = MatcherPresets.build("my_matcher")

        Register a config dict (supports overrides in build):
        >>> custom_config = {
        ...     "descriptor_pipeline": {
        ...         "descriptors": [WaveKernelSignature.from_registry(n_domain=300)],
        ...         "subsamplers": [ArangeSubsampler(subsample_step=5)],
        ...         "normalizer": None,
        ...     },
        ...     "fmap_optimizer": {
        ...         "fmap_size": 40,
        ...         "sdp_weight": 1.0,
        ...         "lb_weight": 1e-2,
        ...         "mult_weight": 1e-1,
        ...         "orient_weight": 0.0,
        ...     },
        ...     "refinement_pipeline": {
        ...         "refiners": [IcpRefiner(nit=10)],
        ...     },
        ...     "p2p_converter": None,
        ... }
        >>> MatcherPresets.register("my_config", custom_config)
        >>> matcher = MatcherPresets.build("my_config",
        ...     fmap_optimizer__fmap_size=50
        ... )
        """
        cls._PRESETS[name] = matcher_or_config

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
            Dictionary with preset parameters, or matcher info if registered
            as an instance.
        """
        if name not in cls._PRESETS:
            raise ValueError(f"Unknown preset '{name}'")

        preset = cls._PRESETS[name]

        # If preset is a matcher instance, return info about it
        if isinstance(preset, BaseMatcher):
            return {
                "type": type(preset).__name__,
                "fmap_size": getattr(preset, "fmap_size", None),
                "_registered_as": "matcher_instance",
            }

        return copy.deepcopy(preset)

    @classmethod
    def _build_descriptor_pipeline(cls, config: dict):
        """Build DescriptorPipeline from config dict.

        Parameters
        ----------
        config : dict
            Configuration with keys: descriptors, subsamplers, normalizer

        Returns
        -------
        pipeline : DescriptorPipeline
        """
        components = []
        components.extend(config.get("descriptors", []))
        components.extend(config.get("subsamplers", []))

        normalizer = config.get("normalizer")
        if normalizer is None:
            normalizer = L2InnerNormalizer()
        components.append(normalizer)

        return DescriptorPipeline(components)

    @classmethod
    def _build_fmap_optimizer(cls, config: dict):
        """Build FunctionalMapOptimizer from config dict.

        Parameters
        ----------
        config : dict
            Configuration with keys: fmap_size, sdp_weight, lb_weight,
            mult_weight, orient_weight, or factor_builders directly.

        Returns
        -------
        optimizer : FunctionalMapOptimizer
        """
        # Check if factor_builders are provided directly
        if "factor_builders" in config:
            return FunctionalMapOptimizer(
                fmap_size=config.get("fmap_size"),
                factor_builders=config["factor_builders"],
            )

        # Build factor builders from weights
        factor_builders = []
        if config.get("sdp_weight", 0) > 0:
            factor_builders.append(SDPFactorBuilder(weight=config["sdp_weight"]))
        if config.get("lb_weight", 0) > 0:
            factor_builders.append(LBFactorBuilder(weight=config["lb_weight"]))
        if config.get("mult_weight", 0) > 0:
            factor_builders.append(MultFactorBuilder(weight=config["mult_weight"]))
        if config.get("orient_weight", 0) > 0:
            factor_builders.append(OrientFactorBuilder(weight=config["orient_weight"]))

        return FunctionalMapOptimizer(
            fmap_size=config.get("fmap_size"),
            factor_builders=factor_builders,
        )

    @classmethod
    def _build_refinement_pipeline(cls, config: dict):
        """Build RefinementPipeline from config dict.

        Parameters
        ----------
        config : dict
            Configuration with key: refiners (list of Refiner instances)

        Returns
        -------
        pipeline : RefinementPipeline
        """
        refiners = config.get("refiners", [])
        if not refiners:
            refiners = [IdentityRefiner()]

        return RefinementPipeline(refiners)

    @classmethod
    def _build_p2p_converter(cls, config):
        """Build P2pFromFmConverter from config.

        Parameters
        ----------
        config : dict or None
            Configuration for converter, or None for default.

        Returns
        -------
        converter : P2pFromFmConverter
        """
        if config is None:
            return P2pFromFmConverter()

        if isinstance(config, P2pFromFmConverter):
            return config

        # Build from dict config
        return P2pFromFmConverter(
            neighbor_finder=config.get("neighbor_finder"),
            adjoint=config.get("adjoint", False),
            bijective=config.get("bijective", False),
        )

    @classmethod
    def _build_neighbor_finder(cls, config):
        """Build NeighborFinder from config.

        Parameters
        ----------
        config : dict or None
            Configuration for neighbor finder, or None for default.

        Returns
        -------
        neighbor_finder : NeighborFinder
        """
        if config is None:
            return NeighborFinder(n_neighbors=1)

        if isinstance(config, NeighborFinder):
            return config

        # Build from dict config
        return NeighborFinder(
            n_neighbors=config.get("n_neighbors", 1),
        )


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

    Configuration Structure
    -----------------------
    Each preset is a dict with hierarchical structure::

        {
            "feature_extractor": {
                "type": str,              # "diffusionnet" or "pointnet"
                "k_eig": int,             # Number of eigenvalues
                "in_channels": int,       # Input feature dimension
                "descriptor_n_domain": int,  # Descriptor domain size
            },
            "fmap_module": {
                "lmbda": float,           # Regularization weight
                "resolvent_gamma": float, # Resolvent parameter
                "bijective": bool,        # Compute bidirectional maps
            },
            "p2p_converter": None,        # P2pFromFmConverter config or None
        }

    Examples
    --------
    Build a preset model:
    >>> from geomfum.experiment import ModelPresets
    >>>
    >>> model = ModelPresets.build("fmnet_diffusion_standard", device="cuda")

    Build with hierarchical overrides:
    >>> model = ModelPresets.build("fmnet_diffusion_standard",
    ...     feature_extractor={"k_eig": 300},
    ...     device="cuda"
    ... )

    Build with double-underscore overrides:
    >>> model = ModelPresets.build("fmnet_diffusion_standard",
    ...     feature_extractor__k_eig=300,
    ...     device="cuda"
    ... )

    List available presets:
    >>> ModelPresets.list_presets()
    """

    # Preset definitions with hierarchical structure
    _PRESETS = {
        "fmnet_diffusion_small": {
            "feature_extractor": {
                "type": "diffusionnet",
                "k_eig": 128,
                "in_channels": 64,
                "descriptor_n_domain": 64,
            },
            "fmap_module": {
                "lmbda": 1e3,
                "resolvent_gamma": 1.0,
                "bijective": True,
            },
            "p2p_converter": None,
        },
        "fmnet_diffusion_standard": {
            "feature_extractor": {
                "type": "diffusionnet",
                "k_eig": 200,
                "in_channels": 128,
                "descriptor_n_domain": 128,
            },
            "fmap_module": {
                "lmbda": 1e3,
                "resolvent_gamma": 1.0,
                "bijective": True,
            },
            "p2p_converter": None,
        },
        "fmnet_diffusion_large": {
            "feature_extractor": {
                "type": "diffusionnet",
                "k_eig": 300,
                "in_channels": 256,
                "descriptor_n_domain": 200,
            },
            "fmap_module": {
                "lmbda": 1e3,
                "resolvent_gamma": 1.0,
                "bijective": True,
            },
            "p2p_converter": None,
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
            Override specific preset parameters. Supports hierarchical overrides:
            - Nested dict: feature_extractor={"k_eig": 300}
            - Double-underscore: feature_extractor__k_eig=300

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
        >>> model = ModelPresets.build("fmnet_diffusion_small",
        ...     feature_extractor={"k_eig": 150}
        ... )
        >>> model = ModelPresets.build("fmnet_diffusion_small",
        ...     feature_extractor__k_eig=150
        ... )
        """
        if name not in cls._PRESETS:
            available = ", ".join(cls.list_presets())
            raise ValueError(f"Unknown model preset '{name}'. Available: {available}")

        # Deep merge preset with overrides
        config = _deep_merge(copy.deepcopy(cls._PRESETS[name]), overrides)

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build components using helper methods
        feature_extractor = cls._build_feature_extractor(
            config["feature_extractor"], device
        )
        fmap_module = cls._build_fmap_module(config["fmap_module"])
        converter = cls._build_p2p_converter(config.get("p2p_converter"))

        # Build complete model

        model = FMNet(
            feature_extractor=feature_extractor,
            fmap_module=fmap_module,
            converter=converter,
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

    @classmethod
    def _build_feature_extractor(cls, config: dict, device: str):
        """Build FeatureExtractor from config dict.

        Parameters
        ----------
        config : dict
            Configuration with keys: type, k_eig, in_channels, descriptor_n_domain
        device : str
            Device to place the extractor on.

        Returns
        -------
        extractor : FeatureExtractor
        """
        return FeatureExtractor.from_registry(
            which=config.get("type", "diffusionnet"),
            device=device,
            k=config.get("k_eig", 128),
            in_channels=config.get("in_channels", 64),
            descriptor=WaveKernelSignature(
                n_domain=config.get("descriptor_n_domain", 64)
            ),
        )

    @classmethod
    def _build_fmap_module(cls, config: dict):
        """Build ForwardFunctionalMap from config dict.

        Parameters
        ----------
        config : dict
            Configuration with keys: lmbda, resolvent_gamma, bijective, fmap_shape

        Returns
        -------
        fmap_module : ForwardFunctionalMap
        """
        return ForwardFunctionalMap(
            lmbda=config.get("lmbda", 1e3),
            resolvent_gamma=config.get("resolvent_gamma", 1.0),
            bijective=config.get("bijective", True),
            fmap_shape=config.get("fmap_shape"),
        )

    @classmethod
    def _build_p2p_converter(cls, config):
        """Build P2pFromFmConverter from config.

        Parameters
        ----------
        config : dict or None
            Configuration for converter, or None for default.

        Returns
        -------
        converter : P2pFromFmConverter
        """
        from geomfum.convert import P2pFromFmConverter

        if config is None:
            return P2pFromFmConverter()

        if isinstance(config, P2pFromFmConverter):
            return config

        return P2pFromFmConverter(
            neighbor_finder=config.get("neighbor_finder"),
            adjoint=config.get("adjoint", False),
            bijective=config.get("bijective", False),
        )


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

    Configuration Structure
    -----------------------
    Each preset is a dict with hierarchical structure::

        {
            "optimizer": {
                "type": str,              # "Adam", "SGD", or "AdamW"
                "lr": float,              # Learning rate
                "weight_decay": float,    # Weight decay
            },
            "scheduler": None or {        # Optional LR scheduler config
                "type": str,              # "StepLR", "CosineAnnealingLR", etc.
                ...                       # Scheduler-specific params
            },
            "training": {
                "epochs": int,            # Number of training epochs
                "monitor_metric": str,    # Metric for checkpointing
                "mode": str,              # "min" or "max"
            },
            "train_losses": [             # List of (name, kwargs) tuples
                ("LossName", {"weight": ...}),
            ],
            "val_losses": [               # List of (name, kwargs) tuples
                ("MetricName", {}),
            ],
        }

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

    With hierarchical overrides:
    >>> trainer = TrainingPresets.create_trainer(
    ...     preset="unsupervised_standard",
    ...     model=model,
    ...     train_set=train_pairs,
    ...     val_set=val_pairs,
    ...     optimizer={"lr": 5e-4},
    ...     training={"epochs": 100},
    ... )

    With double-underscore overrides:
    >>> trainer = TrainingPresets.create_trainer(
    ...     preset="unsupervised_standard",
    ...     model=model,
    ...     train_set=train_pairs,
    ...     val_set=val_pairs,
    ...     optimizer__lr=5e-4,
    ...     training__epochs=100,
    ... )
    """

    # Preset definitions with hierarchical structure
    _PRESETS = {
        "unsupervised_quick": {
            "optimizer": {
                "lr": 1e-3,
                "weight_decay": 1e-5,
            },
            "scheduler": None,
            "training": {
                "epochs": 5,
                "monitor_metric": "GeodesicError",
                "mode": "min",
            },
            "train_losses": [
                ("OrthonormalityLoss", {"weight": 1.0}),
                ("BijectivityLoss", {"weight": 1.0}),
            ],
            "val_losses": [
                ("GeodesicError", {}),
            ],
        },
        "unsupervised_standard": {
            "optimizer": {
                "lr": 1e-3,
                "weight_decay": 1e-5,
            },
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
            "val_losses": [
                ("GeodesicError", {}),
            ],
        },
        "unsupervised_precise": {
            "optimizer": {
                "lr": 1e-3,
                "weight_decay": 1e-5,
            },
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
            "val_losses": [
                ("GeodesicError", {}),
            ],
        },
        "supervised": {
            "optimizer": {
                "lr": 1e-3,
                "weight_decay": 1e-5,
            },
            "scheduler": None,
            "training": {
                "epochs": 50,
                "monitor_metric": "GeodesicError",
                "mode": "min",
            },
            "train_losses": [
                ("GroundTruthSupervisionLoss", {"weight": 1.0}),
            ],
            "val_losses": [
                ("GeodesicError", {}),
            ],
        },
        "supervised_plus_regularization": {
            "optimizer": {
                "lr": 1e-3,
                "weight_decay": 1e-5,
            },
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
            "val_losses": [
                ("GeodesicError", {}),
            ],
        },
    }

    @classmethod
    def _get_loss_classes(cls):
        """Get mapping of loss names to classes (lazy import)."""
        return {
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
        loss_classes = cls._get_loss_classes()
        losses = []
        for loss_name, loss_kwargs in loss_specs:
            loss_class = loss_classes[loss_name]
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
            Override preset parameters. Supports hierarchical overrides:
            - Nested dict: optimizer={"lr": 5e-4}
            - Double-underscore: optimizer__lr=5e-4

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

        >>> # With hierarchical overrides
        >>> trainer = TrainingPresets.create_trainer(
        ...     preset="unsupervised_standard",
        ...     model=model,
        ...     train_set=train_pairs,
        ...     val_set=val_pairs,
        ...     optimizer={"lr": 5e-4},
        ...     training={"epochs": 100},
        ... )
        """
        if preset not in cls._PRESETS:
            available = ", ".join(cls.list_presets())
            raise ValueError(
                f"Unknown training preset '{preset}'. Available: {available}"
            )

        # Deep merge preset with overrides
        config = _deep_merge(copy.deepcopy(cls._PRESETS[preset]), overrides)

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build losses

        train_losses = cls._build_losses(config["train_losses"])
        val_losses = cls._build_losses(config["val_losses"])

        train_loss_manager = LossManager(train_losses)
        val_loss_manager = LossManager(val_losses)

        # Build optimizer
        optimizer = cls._build_optimizer(config["optimizer"], model)

        # Build scheduler (optional)
        scheduler = cls._build_scheduler(config.get("scheduler"), optimizer)

        # Build trainer

        training_config = config["training"]
        trainer = DeepFunctionalMapTrainer(
            model=model,
            train_set=train_set,
            val_set=val_set,
            train_loss_manager=train_loss_manager,
            val_loss_manager=val_loss_manager,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=training_config["epochs"],
            device=device,
            checkpoint_path=checkpoint_path,
            monitor_metric=training_config["monitor_metric"],
            mode=training_config["mode"],
        )

        return trainer

    @classmethod
    def _build_optimizer(cls, config: dict, model):
        """Build optimizer from config dict.

        Parameters
        ----------
        config : dict
            Configuration with keys: lr, weight_decay, and optionally type
        model : torch.nn.Module
            Model whose parameters to optimize.

        Returns
        -------
        optimizer : torch.optim.Optimizer
        """
        optimizer_type = config.get("type", "Adam")

        if optimizer_type == "Adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=config.get("lr", 1e-3),
                weight_decay=config.get("weight_decay", 1e-5),
            )
        elif optimizer_type == "SGD":
            return torch.optim.SGD(
                model.parameters(),
                lr=config.get("lr", 1e-3),
                weight_decay=config.get("weight_decay", 1e-5),
                momentum=config.get("momentum", 0.9),
            )
        elif optimizer_type == "AdamW":
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.get("lr", 1e-3),
                weight_decay=config.get("weight_decay", 1e-5),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    @classmethod
    def _build_scheduler(cls, config, optimizer):
        """Build learning rate scheduler from config.

        Parameters
        ----------
        config : dict or None
            Configuration for scheduler, or None for no scheduler.
        optimizer : torch.optim.Optimizer
            Optimizer to schedule.

        Returns
        -------
        scheduler : LRScheduler or None
        """
        if config is None:
            return None

        scheduler_type = config.get("type", "StepLR")

        if scheduler_type == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("step_size", 10),
                gamma=config.get("gamma", 0.1),
            )
        elif scheduler_type == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get("T_max", 50),
                eta_min=config.get("eta_min", 0),
            )
        elif scheduler_type == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get("mode", "min"),
                factor=config.get("factor", 0.1),
                patience=config.get("patience", 10),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

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
        Model overrides: feature_extractor, fmap_module, p2p_converter
        Training overrides: optimizer, scheduler, training, train_losses, val_losses

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

    >>> # With custom parameters (hierarchical overrides)
    >>> trainer = quick_train(
    ...     preset="unsupervised_standard",
    ...     model_preset="fmnet_diffusion_large",
    ...     train_set=train_pairs,
    ...     val_set=val_pairs,
    ...     feature_extractor={"k_eig": 256},
    ...     optimizer={"lr": 5e-4},
    ...     training={"epochs": 100},
    ... )
    """
    # Separate model kwargs from training kwargs based on hierarchical keys
    model_kwargs = {}
    training_kwargs = {}

    # Known top-level keys for each preset type
    model_keys = {"feature_extractor", "fmap_module", "p2p_converter"}
    training_keys = {"optimizer", "scheduler", "training", "train_losses", "val_losses"}

    for key, value in kwargs.items():
        # Check for double-underscore prefix
        base_key = key.split("__")[0] if "__" in key else key

        if base_key in model_keys:
            model_kwargs[key] = value
        elif base_key in training_keys:
            training_kwargs[key] = value
        else:
            # Unknown key - pass to training by default
            training_kwargs[key] = value

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
