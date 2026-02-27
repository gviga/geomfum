"""Experiment framework for running and evaluating shape matching methods.

This module provides a unified way to run experiments with both classical
matchers and learning-based models on shape datasets.

The Experiment class focuses on evaluation rather than training, making it
easy to benchmark different methods on the same dataset.
"""

import json
import logging
import numbers
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from geomfum.eval import evaluate_correspondence
from geomfum.matcher import CorrespondenceResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
        Aggregated metrics (mean and std values).
    per_pair_metrics : list
        List of metric dicts for each pair.
    pair_indices : list
        List of (source_idx, target_idx) tuples.
    method_name : str
        Name of the method used.
    dataset_name : str
        Name of the dataset.
    per_pair_time_s : list[float]
        Wall-clock time in seconds for each pair (correspondence only).
    total_time_s : float
        Total wall-clock time for the full run.
    """

    metrics: Dict[str, float]
    per_pair_metrics: List[Dict[str, float]]
    pair_indices: List[tuple]
    method_name: str = ""
    dataset_name: str = ""
    per_pair_time_s: List[float] = field(default_factory=list)
    total_time_s: float = 0.0

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

        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def deep_convert(d):
            if isinstance(d, dict):
                return {k: deep_convert(v) for k, v in d.items()}
            if isinstance(d, list):
                return [deep_convert(i) for i in d]
            return convert(d)

        with open(path, "w") as f:
            json.dump(deep_convert(self.to_dict()), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentResult":
        """Load results from a JSON file.

        Parameters
        ----------
        path : str
            Path to load the results from.

        Returns
        -------
        ExperimentResult
        """
        with open(path, "r") as f:
            data = json.load(f)
        # Tolerate older saves that pre-date timing fields
        data.setdefault("per_pair_time_s", [])
        data.setdefault("total_time_s", 0.0)
        return cls(**data)


class Experiment:
    """Run an experiment with a single matcher or model on a shape dataset.

    Parameters
    ----------
    method : BaseMatcher or nn.Module
        The matching method to evaluate.
    dataset : PairsDataset
        Dataset of shape pairs to evaluate on.
    name : str
        Name shown in the progress bar and log output.
    metrics : list[str] or None
        Metrics to compute. If None, all available metrics are computed.
    save_correspondences : bool
        Whether to store the raw p2p maps in the result.
    progress_bar : bool
        Whether to show a tqdm progress bar.
    progress_accuracy_threshold : float
        Geodesic-error threshold used when reporting running accuracy in tqdm.
    """

    def __init__(
        self,
        method,
        dataset,
        name: str = "experiment",
        metrics: Optional[List[str]] = None,
        save_correspondences: bool = False,
        progress_bar: bool = True,
        progress_accuracy_threshold: float = 0.05,
    ):
        self.method = method
        self.dataset = dataset
        self.name = name
        self.metrics = metrics
        self.save_correspondences = save_correspondences
        self.progress_bar = progress_bar
        self.progress_accuracy_threshold = progress_accuracy_threshold

        # Detect models (have .eval()) vs plain matchers
        self._is_model = hasattr(method, "eval") and callable(method.eval)

    def _get_correspondence(self, shape_a, shape_b) -> CorrespondenceResult:
        if self._is_model:
            self.method.eval()
            import torch

            with torch.no_grad():
                result = self.method(shape_a, shape_b)
        else:
            result = self.method(shape_a, shape_b)
        return result

    def _evaluate_pair(
        self,
        result: CorrespondenceResult,
        shape_a,
        shape_b,
        corr_a=None,
        corr_b=None,
        dist_a=None,
    ) -> Dict[str, float]:
        metrics = evaluate_correspondence(
            shape_a=shape_a,
            shape_b=shape_b,
            p2p21=result.p2p21,
            corr_a=corr_a,
            corr_b=corr_b,
            dist_a=dist_a,
            metrics=self.metrics,
        )
        if self.metrics is not None:
            metrics = {k: v for k, v in metrics.items() if k in self.metrics}
        return metrics

    def run(self) -> ExperimentResult:
        """Run the experiment on all pairs.

        Returns
        -------
        result : ExperimentResult
            Aggregated experiment results with per-pair timing.
        """
        per_pair_metrics = []
        per_pair_times = []
        pair_indices = []
        correspondences = [] if self.save_correspondences else None

        if hasattr(self.dataset, "shape_data"):
            has_correspondences = get_dataset_attr(
                self.dataset.shape_data, "correspondences"
            )
            has_distances = get_dataset_attr(self.dataset.shape_data, "distances")
        else:
            # Pairs-style dataset: items already contain corr/dist_matrix; always
            # attempt to read them from the pair dict (None if absent).
            has_correspondences = True
            has_distances = True

        iterator = self.dataset
        if self.progress_bar:
            iterator = tqdm(
                iterator,
                desc=f"Running {self.name}",
                unit="pair",
                leave=False,
            )

        running_sums = {}
        running_counts = {}
        geodesic_hits = 0
        total_start = time.perf_counter()

        for idx, pair in enumerate(iterator):
            shape_a = pair["source"]["shape"]
            shape_b = pair["target"]["shape"]

            corr_a = None
            corr_b = None
            if has_correspondences:
                corr_a = pair.get("source_corr")
                corr_b = pair.get("target_corr")
                if corr_a is None and "source" in pair:
                    corr_a = pair["source"].get("corr")
                if corr_b is None and "target" in pair:
                    corr_b = pair["target"].get("corr")
            dist_a = pair["source"].get("dist_matrix") if has_distances else None

            pair_start = time.perf_counter()
            result = self._get_correspondence(shape_a, shape_b)
            per_pair_times.append(time.perf_counter() - pair_start)

            metrics = self._evaluate_pair(
                result, shape_a, shape_b, corr_a, corr_b, dist_a
            )
            per_pair_metrics.append(metrics)
            pair_indices.append(
                self.dataset.pairs[idx] if hasattr(self.dataset, "pairs") else idx
            )

            for key, value in metrics.items():
                if isinstance(value, numbers.Number):
                    running_sums[key] = running_sums.get(key, 0.0) + float(value)
                    running_counts[key] = running_counts.get(key, 0) + 1

            if "geodesic_error" in metrics:
                if metrics["geodesic_error"] <= self.progress_accuracy_threshold:
                    geodesic_hits += 1

            if self.save_correspondences:
                p2p = result.p2p21
                if hasattr(p2p, "cpu"):
                    p2p = p2p.detach().cpu().numpy()
                correspondences.append({"p2p21": np.asarray(p2p).tolist()})

            if self.progress_bar:
                postfix = {}
                if "geodesic_error" in metrics:
                    geo_avg = (
                        running_sums["geodesic_error"]
                        / running_counts["geodesic_error"]
                    )
                    postfix["geo(cur/avg)"] = (
                        f"{metrics['geodesic_error']:.4f}/{geo_avg:.4f}"
                    )
                    postfix[f"acc@{self.progress_accuracy_threshold:g}"] = (
                        f"{100.0 * geodesic_hits / (idx + 1):.1f}%"
                    )
                if "euclidean_error" in metrics:
                    euc_avg = (
                        running_sums["euclidean_error"]
                        / running_counts["euclidean_error"]
                    )
                    postfix["euc(avg)"] = f"{euc_avg:.4f}"
                if "coverage" in metrics:
                    cov_avg = running_sums["coverage"] / running_counts["coverage"]
                    postfix["cov(avg)"] = f"{cov_avg:.3f}"
                if postfix:
                    iterator.set_postfix(postfix)

        total_time_s = time.perf_counter() - total_start
        aggregated = self._aggregate_metrics(per_pair_metrics)

        method_name = type(self.method).__name__
        dataset_name = getattr(self.dataset, "dataset_dir", "unknown")
        if hasattr(self.dataset, "shape_data"):
            dataset_name = getattr(self.dataset.shape_data, "dataset_dir", dataset_name)

        result = ExperimentResult(
            metrics=aggregated,
            per_pair_metrics=per_pair_metrics,
            pair_indices=pair_indices,
            method_name=method_name,
            dataset_name=str(dataset_name),
            per_pair_time_s=per_pair_times,
            total_time_s=total_time_s,
        )

        avg_ms = 1000.0 * float(np.mean(per_pair_times)) if per_pair_times else 0.0
        logger.info(
            "Experiment '%s' completed in %.1fs (avg %.1fms/pair):",
            self.name,
            total_time_s,
            avg_ms,
        )
        for k, v in aggregated.items():
            if not k.endswith("_std"):
                logger.info("  %s: %.4f ± %.4f", k, v, aggregated.get(f"{k}_std", 0.0))

        return result

    def _aggregate_metrics(
        self, per_pair_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        if not per_pair_metrics:
            return {}

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
    metrics : list[str] or None
        Metrics to compute for each experiment. If None, computes all.
    save_correspondences : bool
        Whether to store raw p2p maps in results.
    progress_bar : bool
        Whether to show tqdm progress bars.
    progress_accuracy_threshold : float
        Geodesic-error threshold for reporting running accuracy.

    Examples
    --------
    >>> suite = ExperimentSuite({"FMap": matcher_a, "Mine": matcher_b}, pairs)
    >>> suite.run()
    >>> suite.print_comparison()
    """

    def __init__(
        self,
        methods: Dict,
        dataset,
        metrics: Optional[List[str]] = None,
        save_correspondences: bool = False,
        progress_bar: bool = True,
        progress_accuracy_threshold: float = 0.05,
    ):
        self.methods = methods
        self.dataset = dataset
        self.metrics = metrics
        self.save_correspondences = save_correspondences
        self.progress_bar = progress_bar
        self.progress_accuracy_threshold = progress_accuracy_threshold
        self.results: Dict[str, ExperimentResult] = {}

    def run(self) -> Dict[str, ExperimentResult]:
        """Run all experiments.

        Returns
        -------
        results : dict[str, ExperimentResult]
        """
        methods_items = list(self.methods.items())
        methods_iterator = methods_items
        if self.progress_bar:
            methods_iterator = tqdm(methods_items, desc="Methods", unit="method")

        best_geo = float("inf")
        best_method = None

        for name, method in methods_iterator:
            logger.info("Running experiment: %s", name)
            experiment = Experiment(
                method=method,
                dataset=self.dataset,
                name=name,
                metrics=self.metrics,
                save_correspondences=self.save_correspondences,
                progress_bar=self.progress_bar,
                progress_accuracy_threshold=self.progress_accuracy_threshold,
            )
            result = experiment.run()
            self.results[name] = result

            if self.progress_bar:
                geo = result.metrics.get("geodesic_error")
                if geo is not None and geo < best_geo:
                    best_geo = geo
                    best_method = name

                postfix = {}
                if geo is not None:
                    postfix["last_geo"] = f"{geo:.4f}"
                if best_method is not None:
                    postfix["best"] = f"{best_method}:{best_geo:.4f}"
                if postfix:
                    methods_iterator.set_postfix(postfix)

        return self.results

    def print_comparison(self, metrics: Optional[List[str]] = None):
        """Print comparison table of results with timing.

        Parameters
        ----------
        metrics : list[str], optional
            Metrics to include. If None, uses geodesic_error.
        """
        if not self.results:
            logger.warning("No results to compare. Run experiments first.")
            return

        if metrics is None:
            metrics = ["geodesic_error"]

        header = f"{'Method':<22} | {'ms/pair':>8}"
        for metric in metrics:
            header += f" | {metric:<20}"
        sep = "-" * len(header)
        print(header)
        print(sep)

        for name, result in self.results.items():
            avg_ms = (
                1000.0 * float(np.mean(result.per_pair_time_s))
                if result.per_pair_time_s
                else 0.0
            )
            row = f"{name:<22} | {avg_ms:>7.1f}"
            for metric in metrics:
                value = result.metrics.get(metric, float("nan"))
                std = result.metrics.get(f"{metric}_std", 0.0)
                row += f" | {value:.4f}±{std:.4f}          "
            print(row.rstrip())

    def save_all(self, directory: str):
        """Save all results to a directory.

        Parameters
        ----------
        directory : str
            Directory to save results (created if needed).
        """
        os.makedirs(directory, exist_ok=True)
        for name, result in self.results.items():
            path = os.path.join(directory, f"{name}.json")
            result.save(path)
            logger.info("Saved %s results to %s", name, path)

    @staticmethod
    def load_all(directory: str) -> Dict[str, ExperimentResult]:
        """Load results previously saved with save_all.

        Parameters
        ----------
        directory : str
            Directory containing ``<method_name>.json`` files.

        Returns
        -------
        results : dict[str, ExperimentResult]

        Examples
        --------
        >>> suite.save_all("results/my_experiment/")
        >>> results = ExperimentSuite.load_all("results/my_experiment/")
        """
        results = {}
        for path in sorted(Path(directory).glob("*.json")):
            results[path.stem] = ExperimentResult.load(str(path))
        return results


def compare(
    methods,
    dataset,
    metrics=None,
    progress_bar=True,
    save_correspondences=False,
    progress_accuracy_threshold=0.05,
):
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
    progress_bar : bool
        Whether to display tqdm progress bars.
    save_correspondences : bool
        Whether to store raw p2p maps in results.
    progress_accuracy_threshold : float
        Geodesic-error threshold for reporting running accuracy in tqdm.

    Returns
    -------
    suite : ExperimentSuite
        Populated suite with ``results``, ``print_comparison()``,
        ``save_all()``, and ``load_all()`` methods.

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
    >>> # Later:
    >>> saved = ExperimentSuite.load_all("results/my_experiment/")
    """
    suite = ExperimentSuite(
        methods,
        dataset,
        metrics=metrics,
        progress_bar=progress_bar,
        save_correspondences=save_correspondences,
        progress_accuracy_threshold=progress_accuracy_threshold,
    )
    suite.run()
    return suite
