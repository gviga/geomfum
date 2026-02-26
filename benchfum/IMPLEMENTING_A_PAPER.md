# Implementing a Shape Matching Paper in geomfum + benchfum

This guide walks through every step needed to add a new paper to this codebase,
using URRSM (SIGGRAPH 2023) as the worked example throughout.

---

## 0. Architecture at a glance

```
geomfum/          ← pure Python algorithm library (no JSON, no config logic)
│   matchers/     ← classical matchers (FunctionalMapMatcher, …)
│   descriptor/   ← WKS, HKS, learned, spectral utilities
│   learning/     ← nn.Module models, losses, trainer, wrappers
│   dataset/      ← ShapeDataset, PairsDataset, augmentation
│   shape.py      ← TriangleMesh / PointCloud
│   forward_functional_map.py
│   …

benchfum/         ← benchmark runner and JSON→Python factory
│   _build.py     ← THE factory (recursive JSON→Python, single registry)
│   __init__.py   ← public API  (build_* functions, compare, Experiment…)
│   experiment.py ← Experiment / ExperimentSuite / compare()
│   refinement.py ← RefinementMatcher
│   challenges/   ← CLI benchmark runners
│   configs/
│       matchers/     ← *.json for classical matchers
│       refiners/     ← *.json for refinement pipelines
│       models/       ← *.json for learned models
│       training/     ← *.json for trainer configs
│       benchmarks/   ← *.json for full benchmark runs
│       checkpoints/  ← pretrained .pth files
```

**Rule of thumb:** all algorithm code lives in `geomfum/`; everything about
*running*, *configuring*, and *comparing* methods lives in `benchfum/`.

---

## 1. Identify what your paper contributes

Most shape matching papers touch one or more of these pieces:

| Contribution | Where it goes |
|---|---|
| New network architecture | `geomfum/learning/models.py` |
| New loss function | `geomfum/learning/losses.py` |
| New descriptor | `geomfum/descriptor/spectral.py` or a new file |
| New descriptor normalizer / pipeline step | `geomfum/descriptor/pipeline.py` |
| New functional map solver | `geomfum/forward_functional_map.py` |
| New p2p converter | `geomfum/matchers/base.py` or `functional_map.py` |
| New classical matcher | `geomfum/matchers/` |
| New data augmentation | `geomfum/dataset/augmentation.py` |
| Training schedule / test-time refinement | config JSON only (no new Python needed) |

---

## 2. Add the Python class

Write your class in the appropriate geomfum module.  Follow the existing
conventions:

### 2a. New model (`nn.Module`)

Subclass `torch.nn.Module`.  The `forward` method must return a
`CorrespondenceResult` (from `geomfum.matchers.base`):

```python
# geomfum/learning/models.py
from geomfum.matchers.base import CorrespondenceResult

class MyModel(nn.Module):
    def __init__(self, feature_extractor, fmap_module, converter):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fmap_module       = fmap_module
        self.converter         = converter

    def forward(self, shape_a, shape_b, bidirectional=False):
        # ... compute features, fmap, p2p ...
        return CorrespondenceResult(
            p2p21     = p2p21,        # [n_b]  — for each vertex in B, match in A
            fmap12    = fmap12,       # [K_b, K_a]
            fmap21    = fmap21,       # [K_a, K_b]  (only if bidirectional)
            # optional extras consumed by losses:
            refined_fmap12 = ...,
            refined_fmap21 = ...,
            soft_perm_ab   = P12,    # [n_a, n_b]  (needed by DirichletLoss)
            soft_perm_ba   = P21,    # [n_b, n_a]
        )
```

`CorrespondenceResult` is a dataclass; add any new optional field you need for
a new loss at the definition site in `geomfum/matchers/base.py`.

### 2b. New loss

Subclass `torch.nn.Module` and declare `required_inputs` — a list of keys the
trainer will extract from `result.to_dict()` (plus `"shape_a"` / `"shape_b"`):

```python
# geomfum/learning/losses.py
class MyLoss(nn.Module):
    required_inputs = ["fmap12", "shape_a"]   # keys from CorrespondenceResult.to_dict()

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, fmap12, shape_a):
        return self.weight * ...
```

> **Key names:** verify that the key names in `required_inputs` exactly match
> the field names in `CorrespondenceResult.to_dict()`.  A mismatch causes a
> `KeyError` at training time that is hard to debug.

### 2c. New descriptor / domain

Add to `geomfum/descriptor/spectral.py` (for spectral descriptors) or create
a new file.  Spectral descriptors follow this pattern:

```python
class MyDomain:
    """Energy domain for some descriptor."""
    def __init__(self, n_domain=128, **kwargs):
        self.n_domain = n_domain

    def get_domain(self, evals):   # evals: [K]
        # return [n_domain] energy levels
        ...
```

### 2d. New descriptor normalizer / pipeline step

Add to `geomfum/descriptor/pipeline.py`:

```python
class MyNormalizer:
    def __call__(self, features):
        # features: [n_vertices, n_channels]
        return normalized_features
```

### 2e. New data augmentation

Add to `geomfum/dataset/augmentation.py`.  Augmentation is a callable that
takes `vertices: Tensor[N,3]` and returns `Tensor[N,3]`:

```python
class MyAugmentation:
    def __call__(self, vertices):
        # vertices are already on the correct device
        return augmented_vertices
```

It is applied at `__getitem__` time via a shallow copy of the shape, so cached
spectral quantities (eigenvectors, mass matrix) are never mutated.

---

## 3. Register the class in `benchfum/_build.py`

`_build_component_registry()` returns a flat `{name: class}` dict.  Add your
class there so JSON configs can reference it by name:

```python
# benchfum/_build.py  — inside _build_component_registry()

# At the top of the try block, import your class:
from geomfum.learning.models import MyModel
from geomfum.learning.losses import MyLoss
from geomfum.descriptor.spectral import MyDomain
from geomfum.dataset.augmentation import MyAugmentation

# Then add entries to registry.update({...}):
registry.update({
    ...
    "MyModel":         MyModel,
    "MyLoss":          MyLoss,
    "MyDomain":        MyDomain,
    "MyAugmentation":  MyAugmentation,
})
```

> **Important:** `_build_component_registry` is decorated with
> `@functools.lru_cache(maxsize=None)`.  If you add new classes mid-session in
> a REPL, restart the Python process or call
> `_build_component_registry.cache_clear()` to pick up the change.

---

## 4. Write the JSON config files

Every nested Python object in the config needs a `"type"` key that matches an
entry in the registry.  All other keys are forwarded verbatim as constructor
keyword arguments.

Keys starting with `_` (e.g. `"_name"`, `"_description"`) are metadata and are
silently ignored by the builder.

### 4a. Model config — `configs/models/my_paper.json`

```json
{
  "_name": "MyNet (My Paper, Venue Year)",
  "_description": "Short description of what makes this model different.",
  "type": "MyModel",
  "feature_extractor": {
    "type": "FeatureExtractor",
    "which": "diffusionnet",
    "k": 128,
    "in_channels": 128,
    "out_channels": 256,
    "descriptor": {
      "type": "NormalizedDescriptor",
      "descriptor": {
        "type": "WaveKernelSignature",
        "k": 128,
        "domain": {"type": "UrrsmWksDomain", "n_domain": 128},
        "scale": true
      },
      "normalizer": {"type": "L2InnerNormalizer"}
    }
  },
  "fmap_module": {
    "type": "ForwardFunctionalMap",
    "lmbda": 100.0,
    "resolvent_gamma": 0.5,
    "bijective": true
  },
  "converter": {
    "type": "P2pFromFmConverter",
    "neighbor_finder": {
      "type": "SoftmaxNeighborFinder",
      "n_neighbors": 1,
      "tau": 0.07
    }
  }
}
```

**`FeatureExtractor` special case:** it is built via
`FeatureExtractor.from_registry(which=..., k=..., in_channels=..., descriptor=..., ...)`.
The `descriptor` sub-object is built recursively and passed as the
`descriptor` kwarg.

### 4b. Training config — `configs/training/my_paper.json`

```json
{
  "_name": "My Paper Training",
  "optimizer": {"type": "Adam", "lr": 0.001},
  "scheduler": {"type": "CosineAnnealingLR", "T_max": 15, "eta_min": 0.0001},
  "scheduler_step_on": "epoch",
  "epochs": 15,
  "grad_clip_norm": 1.0,
  "monitor_metric": "GeodesicError",
  "mode": "min",
  "train_loss_manager": {
    "type": "LossManager",
    "losses": [
      {"type": "OrthonormalityLoss",          "weight": 1.0},
      {"type": "BijectivityLoss",             "weight": 1.0},
      {"type": "FmapDescriptorsSupervisionLoss", "weight": 1.0}
    ]
  },
  "val_loss_manager": {
    "type": "LossManager",
    "losses": [{"type": "GeodesicError"}]
  }
}
```

Available optimizer types: any `torch.optim.*` class name (e.g. `"AdamW"`,
`"SGD"`).  Available scheduler types: any `torch.optim.lr_scheduler.*` class
name.

### 4c. Test-time refinement config — `configs/training/my_paper_ttr.json`

```json
{
  "_name": "My Paper Test-Time Refinement",
  "n_steps": 5,
  "optimizer_config": {"type": "Adam", "lr": 0.001},
  "grad_clip_norm": 1.0,
  "restore_weights": true,
  "loss_manager": {
    "type": "LossManager",
    "losses": [
      {"type": "OrthonormalityLoss",             "weight": 1.0},
      {"type": "BijectivityLoss",                "weight": 1.0},
      {"type": "FmapDescriptorsSupervisionLoss", "weight": 1.0},
      {"type": "DirichletLoss",                  "weight": 5.0}
    ]
  }
}
```

> **`optimizer_config` is NOT recursively built** — it is passed as a raw dict
> to `TestTimeRefiner`, which constructs the optimizer per pair using
> `torch.optim`.  Do not nest it further.

### 4d. Dataset augmentation in a benchmark config

Any key in `config["dataset"]` that is not a path key (`root`, `train_root`,
`val_root`) is forwarded to `ShapeDataset`.  To enable augmentation:

```json
{
  "dataset": {
    "root": "../data/my_dataset/",
    "k": 200,
    "spectral": true,
    "augmentation": {
      "type": "RandomAugmentation",
      "rot_y": 90.0,
      "std": 0.01,
      "noise_clip": 0.05,
      "scale_min": 0.9,
      "scale_max": 1.1
    }
  }
}
```

The `build_dataset` helper in `challenges/_common.py` detects when
`augmentation` is a dict and builds it via the component registry before
passing it to `ShapeDataset`.

---

## 5. Wire it up — the public API

```python
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"   # must be set BEFORE any geomfum import

from benchfum import (
    build_model_from_json,
    build_trainer_from_json,
    build_test_time_refiner_from_json,
)
from geomfum.dataset.torch import ShapeDataset, PairsDataset
from geomfum.dataset.augmentation import RandomAugmentation
from geomfum.learning.wrappers import TrainedModelWrapper

# ── Build model ──────────────────────────────────────────────────────────────
model = build_model_from_json("configs/models/my_paper.json", device="cuda")

# ── Load a pretrained checkpoint ─────────────────────────────────────────────
# If the checkpoint is a raw DiffusionNet state_dict (keys: "first_linear.*",
# "blocks.*") and FeatureExtractor wraps the network as `.model`, remap keys:
import torch
raw_sd = torch.load("checkpoints/my_paper_faust.pth", map_location="cuda",
                    weights_only=False)
remapped = {"model." + k: v for k, v in raw_sd.items()}
model.feature_extractor.load_state_dict(remapped)

# ── Inference ────────────────────────────────────────────────────────────────
matcher = TrainedModelWrapper(model, device="cuda")
result  = matcher(shape_a, shape_b)
# result.p2p21 : [n_b]  — for each vertex in B, its match in A
# result.fmap12: [K_b, K_a]

# ── Test-time refinement ─────────────────────────────────────────────────────
ttr = build_test_time_refiner_from_json(
    "configs/training/my_paper_ttr.json", model=model
)
result_refined = ttr(shape_a, shape_b)

# ── Training from scratch ────────────────────────────────────────────────────
train_shapes = ShapeDataset(
    "data/faust/train/",
    spectral=True, k=200, device="cuda",
    augmentation=RandomAugmentation(),   # URRSM-style: rot ±90° Y, noise, scale
)
train_pairs = PairsDataset(train_shapes, pair_mode="all")

test_shapes = ShapeDataset("data/faust/test/", spectral=True, distances=True,
                            k=200, device="cuda")
test_pairs  = PairsDataset(test_shapes, pair_mode="all")

trainer = build_trainer_from_json(
    "configs/training/my_paper.json",
    model=model,
    train_set=train_pairs,
    val_set=test_pairs,
)
trainer.train()
```

---

## 6. Worked example — URRSM checklist

URRSM required the following additions to the codebase.  Use this as a template
for estimating what a new paper needs.

| # | What | File(s) |
|---|------|---------|
| 1 | `UrrsmWksDomain` / `UrrsmHksDomain` (energy domain) | `geomfum/descriptor/spectral.py` |
| 2 | `NormalizedDescriptor` + `L2InnerNormalizer` (pipeline) | `geomfum/descriptor/pipeline.py` |
| 3 | `RobustFMNet` (`nn.Module`, `ForwardFunctionalMap` with resolvent) | `geomfum/learning/models.py` |
| 4 | `FmapDescriptorsSupervisionLoss` | `geomfum/learning/losses.py` |
| 5 | `DirichletLoss` (spectral approximation) | `geomfum/learning/losses.py` |
| 6 | `soft_perm_ab` / `soft_perm_ba` fields on `CorrespondenceResult` | `geomfum/matchers/base.py` |
| 7 | `TestTimeRefiner` (transductive per-pair refinement) | `geomfum/learning/wrappers.py` |
| 8 | `RandomAugmentation` | `geomfum/dataset/augmentation.py` |
| 9 | Augmentation support in `ShapeDataset` | `geomfum/dataset/torch.py` |
| 10 | Model JSON | `benchfum/configs/models/robust_fmnet.json` |
| 11 | Training JSON | `benchfum/configs/training/urrsm.json` |
| 12 | Test-time refinement JSON | `benchfum/configs/training/urrsm_test_time.json` |
| 13 | Register all new classes | `benchfum/_build.py` |
| 14 | Demo notebook | `geomfum/notebooks/demos/URRSM-Demo.ipynb` |

---

## 7. Common gotchas

### `GEOMSTATS_BACKEND` must be set first

`geomfum` imports `gsops.backend` at module load time.  Once locked to numpy,
PyTorch ops will fail on CUDA tensors.  Always set the env var **before** any
`import geomfum`:

```python
import os; os.environ["GEOMSTATS_BACKEND"] = "pytorch"
```

In challenge runners, set it in the shell before calling `python -m benchfum.challenges.*`.

### Eigenvalue clamping

Laplacian eigenvalues can be slightly negative (~1e-14) due to floating-point
precision.  Any `gs.power(negative, 0.5)` silently returns `NaN`, which then
propagates to make the functional map solver singular.  Always clamp:

```python
evals = gs.clip(evals, 0, None)   # NOT gs.clamp — that doesn't exist in gsops
```

### `loss.required_inputs` key names must match `result.to_dict()`

`CorrespondenceResult.to_dict()` returns field names exactly as they appear in
the dataclass definition.  If you store a quantity as `refined_fmap12`, your
loss must declare `required_inputs = [..., "refined_fmap12"]`, not `"fmap12_desc"`.

### Checkpoint key remapping

Original URRSM checkpoints are raw DiffusionNet `state_dict`s (keys:
`first_linear.weight`, `blocks.0.*`, …).  `FeatureExtractor` wraps DiffusionNet
as `.model`, so all keys need a `"model."` prefix before `load_state_dict`:

```python
remapped = {"model." + k: v for k, v in raw_sd.items()}
model.feature_extractor.load_state_dict(remapped)
```

This pattern applies whenever a checkpoint was saved from a network that is
later wrapped.

### `optimizer_config` must not be recursively built

`TestTimeRefiner` receives `optimizer_config` as a plain dict and calls
`getattr(torch.optim, type)` internally.  The component registry does not know
about `torch.optim` classes, so passing `optimizer_config` through
`_build_component` raises `ValueError: Unknown component type: 'Adam'`.
The builder special-cases this key — do not remove that special case.

### Shallow copy for augmentation

`ShapeDataset` caches shapes in memory.  If you mutate `shape.vertices`
in-place, the cache is corrupted and every subsequent call to `__getitem__`
will return augmented-twice (or more) vertices.  Always use `copy.copy(shape)`
before reassigning `.vertices`:

```python
shape = copy.copy(shape)          # shallow copy — all other attrs shared
shape.vertices = augment(shape.vertices)  # only rebinds the reference
```

### Run challenge scripts from the `geomfum/` subdirectory

```bash
cd geomfum/
python -m benchfum.challenges.deep_fmap.run --config my_config.json
```

### `gs.array(tensor)` fails on CUDA tensors

`gs.array` calls `.numpy()` internally; this fails if the tensor is on GPU.
Use `gs.to_device(tensor, device)` for device moves, and avoid `gs.array` on
anything that might be a CUDA tensor.

---

## 8. Quick reference — all available registry types

### Models
| JSON `"type"` | Python class |
|---|---|
| `"FMNet"` | `FMNet` |
| `"RobustFMNet"` | `RobustFMNet` |
| `"FeatureExtractor"` | `FeatureExtractor.from_registry(...)` |
| `"ForwardFunctionalMap"` | `ForwardFunctionalMap` |
| `"P2pFromFmConverter"` | `P2pFromFmConverter` |
| `"SoftmaxNeighborFinder"` | `SoftmaxNeighborFinder` |
| `"KNNNeighborFinder"` | `KNNNeighborFinder` |

### Losses
| JSON `"type"` | Notes |
|---|---|
| `"OrthonormalityLoss"` | `‖C^T C - I‖_F` |
| `"BijectivityLoss"` | `‖C12 C21 - I‖_F` |
| `"LaplacianCommutativityLoss"` | `‖C Λ_a - Λ_b C‖_F` |
| `"DescriptorCommutativityLoss"` | `‖C F_a - F_b‖_F` |
| `"GroundTruthSupervisionLoss"` | supervised (needs GT fmap) |
| `"FmapDescriptorsSupervisionLoss"` | URRSM alignment loss |
| `"DirichletLoss"` | spectral Dirichlet energy of transported verts |
| `"GeodesicError"` | evaluation metric (val only) |

### Descriptors / pipeline
| JSON `"type"` | Notes |
|---|---|
| `"WaveKernelSignature"` | WKS with configurable domain |
| `"HeatKernelSignature"` | HKS with configurable domain |
| `"UrrsmWksDomain"` | URRSM-exact WKS energy domain |
| `"UrrsmHksDomain"` | URRSM-exact HKS time domain |
| `"NormalizedDescriptor"` | wraps any descriptor + normalizer |
| `"L2InnerNormalizer"` | mass-weighted L2 normalization |

### Augmentation
| JSON `"type"` | Notes |
|---|---|
| `"RandomAugmentation"` | rotation + noise + scale (URRSM-style) |

### Matchers (classical)
| JSON `"type"` | Notes |
|---|---|
| `"FunctionalMapMatcher"` | full classical pipeline |
| `"DescriptorPipeline"` | chain of descriptor steps |
| `"FunctionalMapOptimizer"` | with optional factor builders |
| `"RefinementPipeline"` | fmap-level refinement chain |
| `"CorrespondenceRefinementPipeline"` | p2p-level refinement chain |
| `"ZoomOut"` | spectral upsampling refiner |
| `"ICPRefinement"` | ICP-based p2p refiner |

---

*Last updated during URRSM replication (February 2026).*
