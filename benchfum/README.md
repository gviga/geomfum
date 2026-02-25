# benchfum

**benchfum** is the benchmarking and experiment framework for [geomfum](../geomfum/), a library for shape matching with functional maps.

The core idea is simple: **you define methods, benchfum runs and compares them**. Methods can be defined either programmatically (subclassing a base class) or declaratively (writing a JSON config file). No other integration work is required.

---

## How it works

```
Your method (Python class or JSON config)
        ↓
  build_matcher_from_json() / build_model_from_json()
        ↓
      compare({"name": method, ...}, dataset=pairs)
        ↓
  ExperimentSuite  →  print_comparison()  →  save_all()
```

The `compare()` function runs every method on every pair in the dataset, collects metrics, measures wall-clock time per pair, and returns an `ExperimentSuite` you can print, save, or query. That's the whole loop.

---

## File structure

```
benchfum/
├── __init__.py           # Public API
├── experiment.py         # Experiment, ExperimentSuite, compare()
├── refinement.py         # RefinementMatcher
├── _build.py             # JSON → Python factory (recursive builder)
│
├── configs/
│   ├── matchers/         # One JSON per classical matcher
│   ├── refiners/         # One JSON per refiner
│   ├── models/           # One JSON per deep model architecture
│   ├── training/         # One JSON per training configuration
│   └── challenges/       # One JSON per benchmark challenge
│
└── challenges/
    ├── _common.py                 # Shared utilities for runners
    ├── landmark_based/run.py      # Classical benchmark runner
    ├── refinement/run.py          # Refinement benchmark runner
    └── deep_fmap/run.py           # Deep learning benchmark runner
```

**Design principle:** `geomfum` contains only algorithm classes. `benchfum` owns everything related to comparison, configuration, and experiment management — it only imports from `geomfum`, never the other way around.

---

## Quick start

### One-liner comparison

```python
from benchfum import compare, build_matcher_from_json
from geomfum.dataset.torch import ShapeDataset, PairsDataset

# Load a dataset
shape_data = ShapeDataset("path/to/faust/train_set", spectral=True, k=200,
                          distances=True, correspondences=True)
pairs = PairsDataset(shape_data, pair_mode="all")

# Compare any mix of methods
results = compare(
    {
        "FMap":    build_matcher_from_json("configs/matchers/fmap.json"),
        "FMap+ZO": build_matcher_from_json("configs/matchers/fmap_zo.json"),
        "Mine":    MyMatcher(),   # any BaseMatcher subclass
    },
    dataset=pairs,
    metrics=["geodesic_error"],
)

results.print_comparison()
results.save_all("results/my_experiment/")
```

Output:
```
Method                 |  ms/pair | geodesic_error
--------------------------------------------------
FMap                   |   312.4 | 0.0934±0.0201
FMap+ZO                |   891.2 | 0.0612±0.0143
Mine                   |   145.1 | 0.0721±0.0188
```

### Load saved results later

```python
from benchfum import ExperimentSuite

results = ExperimentSuite.load_all("results/my_experiment/")
# → dict[str, ExperimentResult]
```

---

## Adding a classical matcher

### Option A — Python class

Subclass `BaseMatcher` and implement `__call__`:

```python
from geomfum.matcher import BaseMatcher, CorrespondenceResult

class MyMatcher(BaseMatcher):
    def __call__(self, shape_a, shape_b):
        # shape_a.basis.vecs       — Laplacian eigenvectors [n_vertices, k]
        # shape_a.basis.vals       — Laplacian eigenvalues  [k]
        # shape_a.landmark_indices — landmark vertex indices (if loaded)
        p2p21 = ...  # array of length n_b: for each vertex in B, its match in A
        return CorrespondenceResult(p2p21=p2p21)
```

Pass it directly to `compare()` — no registration needed.

### Option B — JSON config

Write a JSON file describing the pipeline. Every nested Python object needs a `"type"` key; all other keys are constructor parameters.

```json
{
  "_name": "MyFMap",
  "_description": "Landmark FM with ZoomOut refinement.",
  "type": "FunctionalMapMatcher",
  "fmap_size": 30,
  "descriptor_pipeline": {
    "type": "DescriptorPipeline",
    "steps": [
      {"type": "WaveKernelSignature",         "n_domain": 200, "k": 200},
      {"type": "LandmarkWaveKernelSignature",  "n_domain": 200, "k": 200},
      {"type": "ArangeSubsampler", "subsample_step": 10},
      {"type": "L2InnerNormalizer"}
    ]
  },
  "fmap_optimizer": {
    "type": "FunctionalMapOptimizer",
    "factor_builders": [
      {"type": "SDPFactorBuilder",  "weight": 1.0},
      {"type": "LBFactorBuilder",   "weight": 0.05},
      {"type": "MultFactorBuilder", "weight": 0.2}
    ]
  },
  "refiner": {
    "type": "RefinementPipeline",
    "refiners": [
      {"type": "IcpRefiner",  "nit": 10},
      {"type": "ZoomOut",     "nit": 20, "step": 5}
    ]
  }
}
```

Load and use:

```python
from benchfum import build_matcher_from_json

matcher = build_matcher_from_json("my_matcher.json")
```

To make it a permanent named baseline, drop the file in `configs/matchers/` and reference it from a challenge config.

---

## Adding a deep learning model

### Option A — Python class

Subclass `nn.Module`. The `forward` signature must match:

```python
import torch.nn as nn
from geomfum.matcher import CorrespondenceResult

class MyDeepModel(nn.Module):
    def forward(self, shape_a, shape_b):
        # shape_a.basis.vecs   [n_vertices, k]
        # shape_a.basis.vals   [k]
        # ...
        return CorrespondenceResult(fmap12=fmap12, p2p21=p2p21)
```

Pass it to `compare()` directly. `Experiment` detects `nn.Module` automatically and calls `model.eval()` + `torch.no_grad()` during evaluation.

### Option B — JSON config

Write a model config and load it:

```json
{
  "_name": "MyFMNet",
  "type": "FMNet",
  "feature_extractor": {
    "type": "FeatureExtractor",
    "which": "diffusionnet",
    "k": 200,
    "in_channels": 128,
    "descriptor": {"type": "WaveKernelSignature", "n_domain": 128}
  },
  "fmap_module": {
    "type": "ForwardFunctionalMap",
    "lmbda": 1000.0,
    "bijective": true
  }
}
```

```python
from benchfum import build_model_from_json, build_trainer_from_json

model   = build_model_from_json("my_model.json", device="cuda")
trainer = build_trainer_from_json("configs/training/unsupervised_standard.json",
                                  model, train_pairs, val_pairs)
trainer.train()
```

> **Important:** deep learning runners require `GEOMSTATS_BACKEND=pytorch` to be set in the shell **before** the process starts, because the geomfum core library reads the backend at first import:
> ```bash
> GEOMSTATS_BACKEND=pytorch python my_script.py
> # or
> GEOMSTATS_BACKEND=pytorch python -m benchfum.challenges.deep_fmap.run ...
> ```

---

## Adding a refinement method

A refiner is any callable with signature `(fmap12, basis_a, basis_b) → refined_fmap12`.

### Option A — Python class

```python
from benchfum import compare, build_matcher_from_json, build_refiner_from_json, RefinementMatcher

class MyRefiner:
    def __call__(self, fmap12, basis_a, basis_b):
        # fmap12         [k_b, k_a]   initial functional map A→B
        # basis_a.vecs   [n_a, k_a]   Laplacian eigenvectors of shape A
        # basis_b.vecs   [n_b, k_b]   Laplacian eigenvectors of shape B
        return refined_fmap12

base = build_matcher_from_json("configs/matchers/fmap.json")

results = compare(
    {
        "No refinement": RefinementMatcher(base, build_refiner_from_json("configs/refiners/identity.json")),
        "ICP+ZO":        RefinementMatcher(base, build_refiner_from_json("configs/refiners/icp_zoomout.json")),
        "Mine":          RefinementMatcher(base, MyRefiner()),
    },
    dataset=pairs,
)
results.print_comparison()
```

`RefinementMatcher` shares a single base matcher instance across all methods, so all comparisons start from the **same** initial map — any score difference is entirely due to the refinement step.

### Option B — JSON config

```json
{
  "_name": "MyRefiner",
  "type": "IcpRefiner",
  "nit": 30
}
```

```python
from benchfum import build_refiner_from_json
refiner = build_refiner_from_json("my_refiner.json")
```

---

## Challenge runners

Challenges are structured benchmarks defined by a JSON config + a runner script. Three are included:

| Runner | Config | What it compares |
|---|---|---|
| `challenges/landmark_based/run.py` | `configs/challenges/landmark_faust.json` | Classical matchers side by side |
| `challenges/refinement/run.py` | `configs/challenges/refinement_faust.json` | Refiners on a shared base map |
| `challenges/deep_fmap/run.py` | `configs/challenges/deep_fmap_faust.json` | Deep models (optional train + eval) |

### Running a challenge

All runners are invoked from the `geomfum/` directory:

```bash
# Classical
python -m benchfum.challenges.landmark_based.run \
    --dataset /path/to/faust/train_set

# Refinement
python -m benchfum.challenges.refinement.run \
    --dataset /path/to/faust/train_set

# Deep (pytorch backend required)
GEOMSTATS_BACKEND=pytorch python -m benchfum.challenges.deep_fmap.run \
    --dataset /path/to/faust/test_set
```

Common flags (all runners):

| Flag | Description |
|---|---|
| `--dataset PATH` | Override the dataset root from the config |
| `--config PATH` | Use a different challenge config JSON |
| `--n_pairs N` | Evaluate on N random pairs instead of all |
| `--seed N` | Fix random seed for reproducible pair selection |
| `--save DIR` | Save per-method JSON results to DIR |

Deep runner only:

| Flag | Description |
|---|---|
| `--train` / `--no-train` | Force enable / disable training step |
| `--train_dataset PATH` | Training dataset root |
| `--val_dataset PATH` | Validation dataset root |
| `--device cuda\|cpu` | PyTorch device (default: auto) |

### Extending a challenge — no code changes needed

The challenge configs are the only place you need to edit. Add your method to the `methods` list:

**Classical** (`landmark_faust.json`):
```json
{
  "name": "MyMethod",
  "matcher_config": "../matchers/my_method.json"
}
```

**Refinement** (`refinement_faust.json`):
```json
{
  "name": "MyRefiner",
  "refiner_config": "../refiners/my_refiner.json"
}
```

**Deep** (`deep_fmap_faust.json`):
```json
{
  "name": "MyModel",
  "model_config": "../models/my_model.json",
  "trainer_config": "../training/unsupervised_standard.json",
  "checkpoint": "../checkpoints/my_model.pth"
}
```

---

## JSON config schema reference

The builder is **generic and recursive**: every nested Python object needs a `"type"` key naming its class; all other keys are constructor parameters matched by name. Keys starting with `_` (e.g. `_name`, `_description`) are metadata and silently ignored.

**Available types:**

| Type | Category |
|---|---|
| `FunctionalMapMatcher`, `FeatureMatcher` | Matchers |
| `WaveKernelSignature`, `HeatKernelSignature` | Descriptors |
| `LandmarkWaveKernelSignature`, `LandmarkHeatKernelSignature` | Descriptors |
| `DescriptorPipeline`, `L2InnerNormalizer`, `ArangeSubsampler` | Pipeline |
| `FunctionalMapOptimizer` | FM optimizer |
| `SDPFactorBuilder`, `LBFactorBuilder`, `MultFactorBuilder`, `OrientFactorBuilder` | FM factors |
| `IdentityRefiner`, `OrthogonalRefiner`, `IcpRefiner`, `ZoomOut`, `AdjointBijectiveZoomOut` | Refiners |
| `RefinementPipeline`, `CorrespondenceRefinementPipeline` | Refiner pipelines |
| `P2pFromFmConverter`, `NeighborFinder`, `SoftmaxNeighborFinder` | Converters |
| `FMNet`, `RobustFMNet` | Deep models (torch required) |
| `FeatureExtractor`, `ForwardFunctionalMap` | Deep components (torch required) |
| `LossManager`, `OrthonormalityLoss`, `BijectivityLoss`, `LaplacianCommutativityLoss`, `DescriptorCommutativityLoss`, `GroundTruthSupervisionLoss`, `FmapDescriptorsSupervisionLoss`, `GeodesicError` | Losses (torch required) |

To register a new geomfum class, add it to `_build.py` → `_build_component_registry()`.

---

## Smoke tests

Use these to verify the full pipeline on a tiny dummy dataset before running on real data:

```bash
cd geomfum

# Classical (no GPU needed)
python -m benchfum.challenges.landmark_based.run \
    --config benchfum/configs/challenges/landmark_faust_smoke.json

# Refinement (no GPU needed)
python -m benchfum.challenges.refinement.run \
    --config benchfum/configs/challenges/refinement_faust_smoke.json

# Deep (GPU recommended, pytorch backend required)
GEOMSTATS_BACKEND=pytorch python -m benchfum.challenges.deep_fmap.run \
    --config benchfum/configs/challenges/deep_fmap_faust_smoke.json
```

Smoke configs use `datasets/dummy_faust/`, cap evaluation to 1 pair, and cap deep training to 1 epoch.

---

## Available configs at a glance

### Matchers (`configs/matchers/`)

| File | Type | Description |
|---|---|---|
| `wks_nn.json` | `FeatureMatcher` | WKS + nearest-neighbour, no FM |
| `fmap.json` | `FunctionalMapMatcher` | Classic FM, WKS, no refinement |
| `fmap_zo.json` | `FunctionalMapMatcher` | FM + ICP + ZoomOut |
| `lfmap.json` | `FunctionalMapMatcher` | Landmark-constrained FM |
| `lfmap_zo.json` | `FunctionalMapMatcher` | Landmark FM + ICP + ZoomOut |
| `feature_wks.json` | `FeatureMatcher` | WKS feature matching |
| `quick.json` | `FunctionalMapMatcher` | Small k, fast |
| `standard.json` | `FunctionalMapMatcher` | Balanced default |
| `precise.json` | `FunctionalMapMatcher` | High-quality, slow |

### Refiners (`configs/refiners/`)

| File | Description |
|---|---|
| `identity.json` | No-op (pass-through baseline) |
| `icp.json` | ICP only |
| `zoomout.json` | ZoomOut only |
| `icp_zoomout.json` | ICP then ZoomOut |

### Models (`configs/models/`)

| File | Architecture | Notes |
|---|---|---|
| `fmnet_small.json` | FMNet | k=128, fast |
| `fmnet_standard.json` | FMNet | k=200, recommended |
| `fmnet_large.json` | FMNet | k=300, best accuracy |
| `robust_fmnet_small.json` | RobustFMNet | k=128, fast |
| `robust_fmnet_standard.json` | RobustFMNet | k=200, recommended |
| `robust_fmnet_large.json` | RobustFMNet | k=300, best accuracy |

### Training (`configs/training/`)

| File | Epochs | Description |
|---|---|---|
| `unsupervised_quick.json` | 5 | Fast sanity check |
| `unsupervised_standard.json` | 50 | Recommended default |
| `unsupervised_precise.json` | 100 | Best unsupervised quality |
| `supervised.json` | 50 | Ground-truth supervision |
| `supervised_plus_regularization.json` | 50 | GT + regularization |
| `unsupervised_smoke_1epoch.json` | 1 | Smoke test only |
