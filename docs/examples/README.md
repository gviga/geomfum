# GeomFUM Examples

This directory contains minimal, runnable examples demonstrating key features of GeomFUM.

## Available Examples

### [quick_start_presets.py](quick_start_presets.py)

Demonstrates the **preset pattern** used throughout GeomFUM:

**Covered Topics:**
1. Classical matcher presets (`MatcherPresets`)
2. Learning-based model presets (`ModelPresets`, `TrainingPresets`)
3. Quick training with `quick_train()`
4. Comparing trained models with classical matchers
5. Exploring and inspecting presets

**Key Patterns:**
```python
from geomfum.experiment import MatcherPresets, quick_train, ExperimentSuite
from geomfum.matcher import FunctionalMapMatcher
from geomfum.learning import load_trained_model

# Classical matcher
matcher = FunctionalMapMatcher(config=MatcherPresets.get("standard"))

# Learning-based training (one-liner)
trainer = quick_train("unsupervised_standard", train_set, val_set)
trainer.train()

# Comparison
methods = {
    "Trained": load_trained_model("checkpoint.pth", model),
    "Classical": FunctionalMapMatcher(config=MatcherPresets.get("standard")),
}
ExperimentSuite(methods, dataset).run()
```

**Important:** All presets (MatcherPresets, ModelPresets, TrainingPresets, quick_train) are now centralized in the `geomfum.experiment` module for consistency.

## For Full Tutorials

See the [how-to notebooks](../notebooks/how_to/) for detailed, step-by-step tutorials:

- **[21_experiment.ipynb](../notebooks/how_to/21_experiment.ipynb)** - Basic experiment framework
- **[22_configuration_presets.ipynb](../notebooks/how_to/22_configuration_presets.ipynb)** - Matcher presets in detail
- **[23_systematic_experiments.ipynb](../notebooks/how_to/23_systematic_experiments.ipynb)** - Grid search and parameter exploration
- **[24_train_and_evaluate_models.ipynb](../notebooks/how_to/24_train_and_evaluate_models.ipynb)** - Training and evaluating learning-based models

## Running Examples

Make sure you have GeomFUM installed:

```bash
pip install -e .
```

Then run any example:

```bash
python docs/examples/quick_start_presets.py
```

Note: Update the dataset paths in the examples to point to your actual data.

## Philosophy

These examples emphasize:
- **Preset-first approach**: Start with presets, customize as needed
- **Unified interface**: Same pattern for classical and learning methods
- **Minimal boilerplate**: Get results with few lines of code
- **Easy comparison**: Compare methods using the same evaluation framework
