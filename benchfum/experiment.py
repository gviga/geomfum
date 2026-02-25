"""Experiment framework for running and evaluating shape matching methods.

This module provides a unified way to run experiments with both classical
matchers and learning-based models on shape datasets.

The Experiment class mirrors the Trainer pattern but focuses on evaluation
rather than training, making it easy to benchmark different methods.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

import gsops.backend as gs
import numpy as np
from tqdm import tqdm

from geomfum.eval import evaluate_correspondence
from geomfum.matcher import CorrespondenceResult

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_dataset_attr(dataset, attr):
    """Recursively get attribute from Subset or base dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset (possibly wrapped in Subset).
    attr : str
        The attribute name to retrieve.

    Returns
    -------
    value
        The attribute value.
    """
    # Handle torch Subset
    try:
        import torch

        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    except ImportError:
        pass

    return getattr(dataset, attr, None)


@dataclass
class ExperimentResult:
    """Result of an experiment run.

    Parameters
    ----------
    metrics : dict
        Aggregated metrics (mean values).
    per_pair_metrics : list
        List of metrics for each pair.
    pair_indices : list
        List of (source_idx, target_idx) tuples.
    method_name : str
        Name of the method used.
    dataset_name : str
        Name of the dataset.
    """

    metrics: Dict[str, float]
    per_pair_metrics: List[Dict[str, float]]
    pair_indices: List[tuple]
    method_name: str = ""
    dataset_name: str = ""

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save results to JSON file.

        Parameters
        ----------
        path : str
            Path to save the results.
        """

        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        data = self.to_dict()

        # Recursively convert numpy types
        def deep_convert(d):
            if isinstance(d, dict):
                return {k: deep_convert(v) for k, v in d.items()}
            if isinstance(d, list):
                return [deep_convert(i) for i in d]
            return convert(d)

        with open(path, "w") as f:
            json.dump(deep_convert(data), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load results from JSON file.

        Parameters
        ----------
        path : str
            Path to load the results from.

        Returns
        -------
        ExperimentResult
            Loaded experiment result.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment.

    Parameters
    ----------
    name : str
        Name of the experiment.
    bidirectional : bool
        Whether to compute bidirectional correspondences.
    metrics : list[str] or None
        List of metrics to compute. If None, computes all available.
    save_correspondences : bool
        Whether to save the correspondence results.
    progress_bar : bool
        Whether to show progress bar.
    """

    name: str = "experiment"
    bidirectional: bool = False
    metrics: List[str] = None
    save_correspondences: bool = False
    progress_bar: bool = True


class Experiment:
    """Run experiments with matchers or models on shape datasets.

    This class provides a unified interface for benchmarking different
    shape matching methods (both classical and learning-based) on datasets.

    Parameters
    ----------
    method : BaseMatcher or nn.Module
        The matching method to evaluate. Can be a Matcher or a Model.
    dataset : PairsDataset
        Dataset of shape pairs to evaluate on.
    config : ExperimentConfig, optional
        Configuration for the experiment.
    """

    def __init__(
        self,
        method,
        dataset,
        config: ExperimentConfig = None,
    ):
        self.method = method
        self.dataset = dataset
        self.config = config or ExperimentConfig()

        # Determine if method is a model (has .eval()) or matcher
        self._is_model = hasattr(method, "eval") and callable(method.eval)

    def _get_correspondence(self, shape_a, shape_b) -> CorrespondenceResult:
        """Get correspondence between two shapes.

        Parameters
        ----------
        shape_a : Shape
            Source shape.
        shape_b : Shape
            Target shape.

        Returns
        -------
        result : CorrespondenceResult
            The correspondence result.
        """
        if self._is_model:
            # For models, ensure eval mode
            self.method.eval()
            import torch

            with torch.no_grad():
                result = self.method(
                    shape_a, shape_b, bidirectional=self.config.bidirectional
                )
        else:
            # For matchers
            result = self.method(
                shape_a, shape_b, bidirectional=self.config.bidirectional
            )

        return result

    def _evaluate_pair(
        self,
        result: CorrespondenceResult,
        shape_a,
        shape_b,
        corr_a=None,
        corr_b=None,
        dist_a=None,
        dist_b=None,
    ) -> Dict[str, float]:
        """Evaluate a single pair.

        Parameters
        ----------
        result : CorrespondenceResult
            The correspondence result.
        shape_a : Shape
            Source shape (target for p2p21).
        shape_b : Shape
            Target shape (source for p2p21).
        corr_a : array-like, optional
            Ground truth on shape A.
        corr_b : array-like, optional
            Ground truth on shape B.
        dist_a : array-like, optional
            Geodesic distance matrix on shape A.
        dist_b : array-like, optional
            Geodesic distance matrix on shape B.

        Returns
        -------
        metrics : dict
            Dictionary of metric values.
        """
        metrics = evaluate_correspondence(
            shape_a=shape_a,
            shape_b=shape_b,
            p2p21=result.p2p21,
            corr_a=corr_a,
            corr_b=corr_b,
            dist_a=dist_a,
        )

        # If bidirectional, also evaluate reverse direction
        if self.config.bidirectional and result.p2p12 is not None:
            metrics_rev = evaluate_correspondence(
                shape_a=shape_b,
                shape_b=shape_a,
                p2p21=result.p2p12,
                corr_a=corr_b,
                corr_b=corr_a,
                dist_a=dist_b,
            )
            # Add reverse metrics with suffix
            for k, v in metrics_rev.items():
                metrics[f"{k}_rev"] = v

        # Filter metrics if specified
        if self.config.metrics is not None:
            metrics = {k: v for k, v in metrics.items() if k in self.config.metrics}

        return metrics

    def run(self) -> ExperimentResult:
        """Run the experiment on all pairs.

        Returns
        -------
        result : ExperimentResult
            Aggregated experiment results.
        """
        per_pair_metrics = []
        pair_indices = []
        correspondences = [] if self.config.save_correspondences else None

        # Check if dataset has correspondences and distances
        has_correspondences = get_dataset_attr(
            self.dataset.shape_data, "correspondences"
        )
        has_distances = get_dataset_attr(self.dataset.shape_data, "distances")

        # Setup iterator
        iterator = self.dataset
        if self.config.progress_bar:
            iterator = tqdm(iterator, desc=f"Running {self.config.name}", unit="pair")

        for idx, pair in enumerate(iterator):
            shape_a = pair["source"]["shape"]
            shape_b = pair["target"]["shape"]

            # Get ground truth correspondences if available
            corr_a = pair["source"].get("corr") if has_correspondences else None
            corr_b = pair["target"].get("corr") if has_correspondences else None

            # Get distance matrices if available
            dist_a = pair["source"].get("dist_matrix") if has_distances else None
            dist_b = pair["target"].get("dist_matrix") if has_distances else None

            # Compute correspondence
            result = self._get_correspondence(shape_a, shape_b)

            # Evaluate
            metrics = self._evaluate_pair(
                result, shape_a, shape_b, corr_a, corr_b, dist_a, dist_b
            )
            per_pair_metrics.append(metrics)
            pair_indices.append(self.dataset.pairs[idx])

            if self.config.save_correspondences:
                correspondences.append(
                    {
                        "p2p21": gs.to_numpy(result.p2p21).tolist(),
                        "p2p12": (
                            gs.to_numpy(result.p2p12).tolist()
                            if result.p2p12 is not None
                            else None
                        ),
                    }
                )

            # Update progress bar
            if self.config.progress_bar:
                # Show running average of geodesic error if available
                if "geodesic_error" in metrics:
                    avg_error = np.mean(
                        [m.get("geodesic_error", 0) for m in per_pair_metrics]
                    )
                    iterator.set_postfix({"geo_err": f"{avg_error:.4f}"})

        # Aggregate metrics
        aggregated = self._aggregate_metrics(per_pair_metrics)

        # Get method and dataset names
        method_name = getattr(self.method, "__class__", type(self.method)).__name__
        dataset_name = getattr(self.dataset, "dataset_dir", "unknown")
        if hasattr(self.dataset, "shape_data"):
            dataset_name = getattr(self.dataset.shape_data, "dataset_dir", dataset_name)

        result = ExperimentResult(
            metrics=aggregated,
            per_pair_metrics=per_pair_metrics,
            pair_indices=pair_indices,
            method_name=method_name,
            dataset_name=str(dataset_name),
        )

        logging.info(f"Experiment '{self.config.name}' completed:")
        for k, v in aggregated.items():
            logging.info(f"  {k}: {v:.4f}")

        return result

    def _aggregate_metrics(
        self, per_pair_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate metrics across all pairs.

        Parameters
        ----------
        per_pair_metrics : list[dict]
            List of per-pair metrics.

        Returns
        -------
        aggregated : dict
            Aggregated metrics (mean, std).
        """
        if not per_pair_metrics:
            return {}

        # Get all metric keys
        all_keys = set()
        for m in per_pair_metrics:
            all_keys.update(m.keys())

        aggregated = {}
        for key in all_keys:
            values = [m[key] for m in per_pair_metrics if key in m]
            if values:
                aggregated[key] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))

        return aggregated


class ExperimentSuite:
    """Run multiple experiments and compare results.

    Parameters
    ----------
    methods : dict[str, BaseMatcher or nn.Module]
        Dictionary mapping method names to methods.
    dataset : PairsDataset
        Dataset to evaluate on.
    config : ExperimentConfig, optional
        Base configuration (name will be overridden per method).

    Examples
    --------
    >>> from geomfum.matcher import FunctionalMapMatcher, FeatureMatcher
    >>>
    >>> methods = {
    ...     "FMap": FunctionalMapMatcher(),
    ...     "Feature": FeatureMatcher(),
    ... }
    >>>
    >>> suite = ExperimentSuite(methods, pairs)
    >>> suite.run()
    >>> suite.print_comparison()
    """

    def __init__(
        self,
        methods: Dict,
        dataset,
        config: ExperimentConfig = None,
    ):
        self.methods = methods
        self.dataset = dataset
        self.base_config = config or ExperimentConfig()
        self.results: Dict[str, ExperimentResult] = {}

    def run(self) -> Dict[str, ExperimentResult]:
        """Run all experiments.

        Returns
        -------
        results : dict[str, ExperimentResult]
            Dictionary mapping method names to results.
        """
        for name, method in self.methods.items():
            logging.info(f"Running experiment: {name}")
            config = ExperimentConfig(
                name=name,
                bidirectional=self.base_config.bidirectional,
                metrics=self.base_config.metrics,
                save_correspondences=self.base_config.save_correspondences,
                progress_bar=self.base_config.progress_bar,
            )
            experiment = Experiment(method, self.dataset, config)
            self.results[name] = experiment.run()

        return self.results

    def print_comparison(self, metrics: List[str] = None):
        """Print comparison table of results.

        Parameters
        ----------
        metrics : list[str], optional
            Metrics to include in comparison. If None, uses geodesic_error.
        """
        if not self.results:
            logging.warning("No results to compare. Run experiments first.")
            return

        if metrics is None:
            metrics = ["geodesic_error"]

        # Print header
        header = f"{'Method':<20}"
        for metric in metrics:
            header += f" | {metric:<15}"
        print(header)
        print("-" * len(header))

        # Print rows
        for name, result in self.results.items():
            row = f"{name:<20}"
            for metric in metrics:
                value = result.metrics.get(metric, float("nan"))
                std = result.metrics.get(f"{metric}_std", 0)
                row += f" | {value:.4f}±{std:.4f}"
            print(row)

    def save_all(self, directory: str):
        """Save all results to a directory.

        Parameters
        ----------
        directory : str
            Directory to save results.
        """
        os.makedirs(directory, exist_ok=True)
        for name, result in self.results.items():
            path = os.path.join(directory, f"{name}.json")
            result.save(path)
            logging.info(f"Saved {name} results to {path}")


def compare(methods, dataset, metrics=None, bidirectional=False, progress_bar=True):
    """Compare multiple matching methods on a dataset.

    This is the main entry point for benchmarking. It runs all methods on
    every pair in the dataset and returns an ``ExperimentSuite`` with results.

    Parameters
    ----------
    methods : dict[str, BaseMatcher or nn.Module]
        Dictionary mapping method names to matcher or model instances.
    dataset : PairsDataset
        Dataset of shape pairs to evaluate on.
    metrics : list[str], optional
        Metrics to compute (e.g. ``["geodesic_error"]``).
        If ``None``, all available metrics are computed.
    bidirectional : bool
        Whether to compute correspondences in both directions.
    progress_bar : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    suite : ExperimentSuite
        Populated suite with ``results``, ``print_comparison()`` and
        ``save_all()`` methods.

    Examples
    --------
    >>> from benchfum import compare, build_matcher_from_json
    >>>
    >>> results = compare(
    ...     {
    ...         "FunctionalMap": build_matcher_from_json("configs/matchers/fmap.json"),
    ...         "MyMethod":      MyMatcher(),
    ...     },
    ...     dataset=pairs,
    ... )
    >>> results.print_comparison()
    >>> results.save_all("results/my_experiment/")
    """
    config = ExperimentConfig(
        bidirectional=bidirectional,
        metrics=metrics,
        progress_bar=progress_bar,
    )
    suite = ExperimentSuite(methods, dataset, config)
    suite.run()
    return suite
