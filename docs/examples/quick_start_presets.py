"""A Quick Start Examples on Using Presets in GeomFUM.

This file demonstrates the preset pattern used throughout GeomFUM for
both classical matchers and learning-based methods.

The preset system provides:
- Pre-configured setups for common use cases
- Easy parameter exploration and comparison
- Consistent API across classical and learning methods
"""

# ============================================================================
# 1. CLASSICAL MATCHER PRESETS
# ============================================================================

from torch.utils.data import random_split

from geomfum.dataset.torch import MeshDataset, PairsDataset
from geomfum.experiment import (
    Experiment,
    ExperimentSuite,
    MatcherPresets,
    ModelPresets,
    TrainingPresets,
    quick_train,
)
from geomfum.learning import load_trained_model
from geomfum.matcher import FunctionalMapMatcher

# Load dataset
dataset_dir = "path/to/dataset"
shapes = MeshDataset(
    dataset_dir=dataset_dir,
    spectral=True,
    distances=True,
    correspondences=True,
)
pairs = PairsDataset(shapes, pairs_ratio=0.1)

# --- Example 1A: Single preset matcher ---
matcher = FunctionalMapMatcher(config=MatcherPresets.get("standard"))
result = Experiment(matcher, pairs).run()

# --- Example 1B: Compare multiple matcher presets ---
methods = {
    "Quick": FunctionalMapMatcher(config=MatcherPresets.get("quick")),
    "Standard": FunctionalMapMatcher(config=MatcherPresets.get("standard")),
    "Precise": FunctionalMapMatcher(config=MatcherPresets.get("precise")),
}
suite = ExperimentSuite(methods, pairs)
results = suite.run()
suite.print_comparison()

# --- Example 1C: Override preset parameters ---
custom_config = MatcherPresets.get("standard", fmap_size=40, spectrum_size=250)
matcher_custom = FunctionalMapMatcher(config=custom_config)


# ============================================================================
# 2. LEARNING-BASED MODEL PRESETS
# ============================================================================

# Prepare training data
train_shapes = MeshDataset(
    dataset_dir="path/to/train",
    spectral=True,
    distances=False,  # Not needed for unsupervised training
    correspondences=False,
)
train_pairs = PairsDataset(train_shapes, pair_mode="all")

# Split into train/val

train_size = int(0.8 * len(train_pairs))
val_size = len(train_pairs) - train_size
train_set, val_set = random_split(train_pairs, [train_size, val_size])

# --- Example 2A: One-liner training (QUICKEST) ---
trainer = quick_train(
    preset="unsupervised_standard",
    train_set=train_set,
    val_set=val_set,
    checkpoint_path="checkpoints/model.pth",
)
trainer.train()

# --- Example 2B: Step-by-step with presets ---
# Build model from preset
model = ModelPresets.build("fmnet_diffusion_standard")

# Create trainer from preset
trainer = TrainingPresets.create_trainer(
    preset="unsupervised_standard",
    model=model,
    train_set=train_set,
    val_set=val_set,
    checkpoint_path="checkpoints/model.pth",
)
trainer.train()

# --- Example 2C: Override training parameters ---
trainer_custom = quick_train(
    preset="unsupervised_quick",
    train_set=train_set,
    val_set=val_set,
    model_preset="fmnet_diffusion_large",  # Different model
    epochs=100,  # Override epochs
    lr=5e-4,  # Override learning rate
    checkpoint_path="checkpoints/custom.pth",
)


# ============================================================================
# 3. COMPARING LEARNING AND CLASSICAL METHODS
# ============================================================================

# Load trained model
trained_model = load_trained_model(
    checkpoint_path="checkpoints/model.pth",
    model=ModelPresets.build("fmnet_diffusion_standard"),
)

# Compare trained model with classical matchers
# Both use the same preset pattern!
comparison_methods = {
    # Learning-based
    "FMNet (trained)": trained_model,
    # Classical
    "FunctionalMap (standard)": FunctionalMapMatcher(
        config=MatcherPresets.get("standard")
    ),
    "FunctionalMap (precise)": FunctionalMapMatcher(
        config=MatcherPresets.get("precise")
    ),
}

# Evaluate on test set
test_shapes = MeshDataset(
    dataset_dir="path/to/test",
    spectral=True,
    distances=True,
    correspondences=True,
)
test_pairs = PairsDataset(test_shapes, pairs_ratio=0.1)

suite = ExperimentSuite(comparison_methods, test_pairs)
results = suite.run()
suite.print_comparison(metrics=["geodesic_error", "coverage"])


# ============================================================================
# 4. EXPLORING PRESETS
# ============================================================================

# --- List available presets ---
print("Matcher Presets:")
print(MatcherPresets.list_presets())

print("\nModel Presets:")
print(ModelPresets.list_presets())

print("\nTraining Presets:")
print(TrainingPresets.list_presets())

# --- Inspect preset details ---
print("\nStandard Matcher Config:")
print(MatcherPresets.describe("standard"))

print("\nStandard Model Config:")
print(ModelPresets.describe("fmnet_diffusion_standard"))

print("\nUnsupervised Training Config:")
print(TrainingPresets.describe("unsupervised_standard"))


# ============================================================================
# 5. TRAINING MULTIPLE MODELS WITH DIFFERENT PRESETS
# ============================================================================

# Train models with different configurations
training_configs = {
    "quick": "unsupervised_quick",
    "standard": "unsupervised_standard",
    "supervised": "supervised",
}

trained_models = {}
for name, preset in training_configs.items():
    trainer = quick_train(
        preset=preset,
        train_set=train_set,
        val_set=val_set,
        checkpoint_path=f"checkpoints/model_{name}.pth",
    )
    trainer.train()

    # Load for evaluation
    model = ModelPresets.build("fmnet_diffusion_standard")
    trained_models[f"FMNet ({name})"] = load_trained_model(
        f"checkpoints/model_{name}.pth", model
    )

# Compare all trained models
suite = ExperimentSuite(trained_models, test_pairs)
results = suite.run()
suite.print_comparison()


# ============================================================================
# SUMMARY
# ============================================================================

"""
The preset pattern in GeomFUM provides consistency across all components:

ALL PRESETS ARE IN geomfum.experiment:
- from geomfum.experiment import MatcherPresets, ModelPresets, TrainingPresets, quick_train

CLASSICAL MATCHERS:
- MatcherPresets.get(name, **overrides) → MatcherConfig
- Use with: FunctionalMapMatcher(config=...)

LEARNING MODELS:
- ModelPresets.build(name, **overrides) → Model
- TrainingPresets.create_trainer(name, model, ..., **overrides) → Trainer
- quick_train(preset, train_set, val_set, **overrides) → Trainer (one-liner)

EVALUATION:
- Experiment(method, dataset) → Single method evaluation
- ExperimentSuite(methods_dict, dataset) → Compare multiple methods

All presets support:
- .list_presets() - Show available options
- .describe(name) - Inspect configuration
- **overrides - Customize specific parameters

This unified interface makes it easy to:
- Switch between methods
- Compare configurations
- Share reproducible setups
- Quickly prototype ideas

NOTE: Presets can still be imported from geomfum.learning for backward compatibility,
but the recommended location is geomfum.experiment.
"""
