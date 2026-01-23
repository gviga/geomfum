"""Grid search utilities for systematic parameter exploration."""

import logging
from itertools import product
from typing import Callable, Dict, List, Any
from dataclasses import asdict

from geomfum.experiment.runner import Experiment, ExperimentConfig, ExperimentResult


class ParameterGridSuite:
    """Run experiments over a grid of parameters.

    This enables systematic testing of different configurations by varying
    individual components while keeping others fixed.

    Parameters
    ----------
    base_matcher_factory : callable
        Factory function that creates a matcher from a config.
        Signature: config -> matcher
    base_config : MatcherConfig
        Base configuration to start from.
    dataset : PairsDataset
        Dataset to evaluate on.
    experiment_config : ExperimentConfig, optional
        Base experiment configuration.

    Examples
    --------
    Test different weight combinations:
    >>> from geomfum.experiment import ParameterGridSuite, MatcherPresets
    >>> from geomfum.matcher import FunctionalMapMatcher
    >>>
    >>> grid = ParameterGridSuite(
    ...     base_matcher_factory=lambda cfg: FunctionalMapMatcher(config=cfg),
    ...     base_config=MatcherPresets.get("standard"),
    ...     dataset=pairs_dataset,
    ... )
    >>> grid.add_param("sdp_weight", [0.5, 1.0, 2.0])
    >>> grid.add_param("lb_weight", [1e-3, 1e-2, 1e-1])
    >>> results = grid.run()
    >>> grid.print_best(metric="geodesic_error")

    Save results for later analysis:
    >>> grid.save_summary("results/weight_grid.csv")
    """

    def __init__(
        self,
        base_matcher_factory: Callable,
        base_config,
        dataset,
        experiment_config: ExperimentConfig = None,
    ):
        self.base_matcher_factory = base_matcher_factory
        self.base_config = base_config
        self.dataset = dataset
        self.experiment_config = experiment_config or ExperimentConfig()
        self.param_grid = {}
        self.results: Dict[str, ExperimentResult] = {}

    def add_param(self, name: str, values: List[Any]):
        """Add a parameter to vary in the grid search.

        Parameters
        ----------
        name : str
            Name of the config parameter (e.g., "sdp_weight", "fmap_size").
            Must be a valid MatcherConfig field.
        values : list
            List of values to try for this parameter.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        self.param_grid[name] = values
        return self

    def _generate_configs(self):
        """Generate all combinations of parameters.

        Yields
        ------
        param_dict : dict
            Dictionary of parameter values for this combination.
        config : MatcherConfig
            Configuration instance with these parameters.
        """
        if not self.param_grid:
            # No grid specified, just use base config
            yield {}, self.base_config
            return

        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        for combo in product(*param_values):
            # Create config dict with overrides
            config_dict = asdict(self.base_config)
            param_dict = dict(zip(param_names, combo))
            config_dict.update(param_dict)

            # Import MatcherConfig to create instance
            from geomfum.matcher import MatcherConfig
            config = MatcherConfig(**config_dict)

            yield param_dict, config

    def run(self) -> Dict[str, ExperimentResult]:
        """Run experiments for all parameter combinations.

        Returns
        -------
        results : dict[str, ExperimentResult]
            Dictionary mapping parameter combination strings to results.
        """
        total_combinations = 1
        for values in self.param_grid.values():
            total_combinations *= len(values)

        logging.info(
            f"Starting grid search with {total_combinations} combinations"
        )

        for param_dict, config in self._generate_configs():
            # Create name from parameters
            if param_dict:
                name = "_".join(f"{k}={v}" for k, v in param_dict.items())
            else:
                name = "base"

            logging.info(f"Running configuration: {name}")

            # Create matcher with this config
            matcher = self.base_matcher_factory(config)

            # Run experiment
            exp_config = ExperimentConfig(
                name=name,
                bidirectional=self.experiment_config.bidirectional,
                metrics=self.experiment_config.metrics,
                save_correspondences=self.experiment_config.save_correspondences,
                progress_bar=self.experiment_config.progress_bar,
            )
            experiment = Experiment(matcher, self.dataset, exp_config)
            self.results[name] = experiment.run()

        logging.info("Grid search completed")
        return self.results

    def print_best(self, metric: str = "geodesic_error", minimize: bool = True, top_k: int = 5):
        """Print best configurations for a given metric.

        Parameters
        ----------
        metric : str
            Metric to optimize for (default: "geodesic_error").
        minimize : bool
            If True, lower is better. If False, higher is better.
        top_k : int
            Number of top configurations to display.
        """
        if not self.results:
            logging.warning("No results available. Run experiments first.")
            return

        # Sort results by metric
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].metrics.get(
                metric, float('inf' if minimize else '-inf')
            ),
            reverse=not minimize,
        )

        print(f"\n{'='*70}")
        print(f"Top {top_k} configurations for {metric}")
        print(f"({'minimize' if minimize else 'maximize'})")
        print(f"{'='*70}\n")

        for rank, (name, result) in enumerate(sorted_results[:top_k], 1):
            value = result.metrics.get(metric, float('nan'))
            std = result.metrics.get(f"{metric}_std", 0)
            print(f"{rank}. {name}")
            print(f"   {metric}: {value:.6f} ± {std:.6f}\n")

    def save_summary(self, path: str, metrics: List[str] = None):
        """Save summary of all results to CSV.

        Parameters
        ----------
        path : str
            Path to save CSV file.
        metrics : list[str], optional
            List of metrics to include. If None, includes all metrics.
        """
        import csv

        if not self.results:
            logging.warning("No results to save")
            return

        # Get all unique metric keys
        all_metrics = set()
        for result in self.results.values():
            all_metrics.update(result.metrics.keys())

        # Filter out std metrics, we'll handle them separately
        base_metrics = sorted([m for m in all_metrics if not m.endswith('_std')])

        if metrics is not None:
            base_metrics = [m for m in base_metrics if m in metrics]

        # Create header with mean ± std columns
        fieldnames = ['config']
        for metric in base_metrics:
            fieldnames.append(f"{metric}_mean")
            fieldnames.append(f"{metric}_std")

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for name, result in sorted(self.results.items()):
                row = {'config': name}
                for metric in base_metrics:
                    row[f"{metric}_mean"] = result.metrics.get(metric, '')
                    row[f"{metric}_std"] = result.metrics.get(f"{metric}_std", '')
                writer.writerow(row)

        logging.info(f"Saved grid search summary to {path}")
