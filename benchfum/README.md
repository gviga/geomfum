# benchfum

**benchfum** is the benchmarking and experiment framework for [geomfum](../geomfum/), a library for shape matching with functional maps. It provides:

- **Named baseline matchers** — all classical methods available by name, backed by JSON configs
- **Benchmark challenges** — ready-to-run scripts comparing your method against the state of the art
- **Experiment routines** — the `compare()` function and `ExperimentSuite` for any custom comparison
- **Learning presets** — quick setup for deep functional map models (FMNet, RobustFMNet)

---

## Repository Structure

```
benchfum/
├── __init__.py                # Public API: compare, MatcherPresets, ...
├── experiment.py              # Experiment, ExperimentSuite, compare()
├── presets.py                 # MatcherPresets — load any matcher by name
├── learning_presets.py        # ModelPresets, TrainingPresets, quick_train
│
├── configs/
│   ├── matchers/              # One JSON file per named matcher
│   │   ├── wks_nn.json        # WKS nearest-neighbour (simplest baseline)
│   │   ├── fmap.json          # Classic FM, no refinement
│   │   ├── fmap_zo.json       # FM + ICP + ZoomOut
│   │   ├── lfmap.json         # Landmark-constrained FM
│   │   ├── lfmap_zo.json      # Landmark FM + ICP + ZoomOut (strongest classical)
│   │   ├── quick.json         # Fast preset (k=20, ICP only)
│   │   ├── standard.json      # Balanced preset (k=30, ICP+ZO)
│   │   ├── precise.json       # High-quality preset (k=50, more iterations)
│   │   └── feature_wks.json   # Feature-based WKS matching
│   │
│   └── challenges/            # One JSON file per benchmark challenge
│       ├── landmark_faust.json      # Classical: baselines declared by matcher name
│       ├── deep_fmap_faust.json     # Deep: baselines declared by model preset
│       └── refinement_faust.json   # Refinement: base method + list of refiners
│
├── refinement.py              # RefinementMatcher, RefinementPresets
│
└── challenges/                # One sub-package per benchmark category
    ├── landmark_based/
    │   └── run.py             # Classical landmark-based runner
    ├── deep_fmap/
    │   └── run.py             # Deep FM runner (requires checkpoints)
    └── refinement/
        └── run.py             # Refinement runner (fixed base map, swap refiners)
```

**Design principle:** `geomfum` is the pure algorithm library. `benchfum` never modifies
`geomfum` — it only imports from it. All experiment, comparison, and preset logic lives here.

---

## Benchmark Challenges

Each challenge is defined by a config file in `configs/challenges/` and a runner in `challenges/`.

| Challenge | Config | Runner | Type | Description |
|-----------|--------|--------|------|-------------|
| Landmark-Based FAUST | `landmark_faust.json` | `challenges/landmark_based/` | Classical | Landmark-guided FM on FAUST |
| Deep FM FAUST | `deep_fmap_faust.json` | `challenges/deep_fmap/` | Deep | FMNet & RobustFMNet on FAUST |
| Refinement FAUST | `refinement_faust.json` | `challenges/refinement/` | Refinement | Compare refiners on a fixed initial FM |

---

## Classical Methods — Landmark-Based Challenge

### Quick Start

```bash
pip install -e .

# Run all classical baselines
python -m benchfum.challenges.landmark_based.run \
    --dataset /path/to/faust/train_set

# Limit to 20 random pairs for a quick check
python -m benchfum.challenges.landmark_based.run \
    --dataset /path/to/faust/train_set --n_pairs 20

# Save per-method JSON results
python -m benchfum.challenges.landmark_based.run \
    --dataset /path/to/faust/train_set --save results/landmark_faust/
```

Expected output:
```
Challenge : Landmark-Based FAUST
Dataset   : /path/to/faust/train_set
Spectrum  : k=200  |  Landmarks: 7
Baselines : ['wks_nn', 'fmap', 'fmap_zo', 'lfmap', 'lfmap_zo']

Method               | geodesic_error
---------------------|----------------
wks_nn               | 0.1823±0.0412
fmap                 | 0.0934±0.0201
fmap_zo              | 0.0612±0.0143
lfmap                | 0.0541±0.0129
lfmap_zo             | 0.0387±0.0097
```

### Add Your Classical Method

**Option A — Subclass BaseMatcher** (full programmatic control):

```python
from geomfum.matcher import BaseMatcher, CorrespondenceResult

class MyMethod(BaseMatcher):
    def __call__(self, shape_a, shape_b, bidirectional=False):
        # shape_a.landmark_indices  — landmark vertex indices on shape A
        # shape_b.landmark_indices  — landmark vertex indices on shape B
        # shape_a.basis.vecs        — Laplacian eigenvectors [n_vertices, k]
        # shape_a.basis.vals        — Laplacian eigenvalues  [k]
        p2p21 = ...  # your algorithm
        return CorrespondenceResult(fmap12=None, p2p21=p2p21)
```

Then run the comparison in Python:

```python
from benchfum import compare, MatcherPresets

results = compare(
    {
        "MyMethod":  MyMethod(),
        "LFMap+ZO":  MatcherPresets.build("lfmap_zo"),
        "FMap+ZO":   MatcherPresets.build("fmap_zo"),
    },
    dataset=pairs,
    metrics=["geodesic_error"],
)
results.print_comparison()
results.save_all("results/my_method/")
```

**Option B — JSON Config** (no code, fully reproducible):

Write `my_method.json` describing your FM pipeline:

```json
{
  "_name": "MyMethod",
  "type": "FunctionalMapMatcher",
  "fmap_size": 40,
  "descriptor_pipeline": [
    {"type": "WaveKernelSignature",         "n_domain": 200, "k": 200},
    {"type": "LandmarkWaveKernelSignature", "n_domain": 200, "k": 200},
    {"type": "ArangeSubsampler", "subsample_step": 10},
    {"type": "L2InnerNormalizer"}
  ],
  "fmap_optimizer": {
    "factors": [
      {"type": "SDPFactorBuilder",  "weight": 1.0},
      {"type": "LBFactorBuilder",   "weight": 0.05},
      {"type": "MultFactorBuilder", "weight": 0.2}
    ]
  },
  "refiner": [
    {"type": "IcpRefiner", "nit": 10},
    {"type": "ZoomOut",    "nit": 20, "step": 5}
  ]
}
```

Load and use:

```python
from geomfum.matcher import BaseMatcher
my_method = BaseMatcher.from_json("my_method.json")
```

**Option C — Edit `run.py` directly**:

Open `challenges/landmark_based/run.py`, implement the `MyMethod` class, and run:

```bash
python -m benchfum.challenges.landmark_based.run \
    --dataset /path/to/faust --my_method
```

---

## Deep Learning Methods — Deep FM Challenge

### Workflow Overview

```
1. Train baselines (FMNet, RobustFMNet) on your training split
2. Train your model on the same split
3. Run the benchmark (evaluation only — loads pretrained checkpoints)
```

### Step 1 — Train Baselines

```python
from benchfum import ModelPresets, quick_train
from geomfum.dataset.torch import ShapeDataset, PairsDataset

# Load dataset
shapes = ShapeDataset(dataset_dir="faust/train", spectral=True, k=200,
                      distances=True, correspondences=True)
train_pairs = PairsDataset(shapes, pair_mode="random", pairs_ratio=80)
val_pairs   = PairsDataset(shapes, pair_mode="random", pairs_ratio=20)

# Train FMNet (standard preset)
trainer = quick_train(
    "unsupervised_standard",
    train_pairs, val_pairs,
    model_preset="fmnet_diffusion_standard",
    checkpoint_path="checkpoints/FMNet.pth",
)
trainer.train()

# Train RobustFMNet
trainer = quick_train(
    "unsupervised_standard",
    train_pairs, val_pairs,
    model_preset="robust_fmnet_standard",
    checkpoint_path="checkpoints/RobustFMNet.pth",
)
trainer.train()
```

### Step 2 — Implement and Train Your Model

```python
import torch.nn as nn
from geomfum.matcher import CorrespondenceResult

class MyDeepFMNet(nn.Module):
    """Your deep functional map architecture."""

    def __init__(self):
        super().__init__()
        # self.feature_net = ...
        # self.fmap_layer  = ...

    def forward(self, shape_a, shape_b, bidirectional=False):
        # shape_a.basis.vecs  [n_vertices, k]  — Laplacian eigenvectors
        # shape_a.basis.vals  [k]               — Laplacian eigenvalues
        # ...compute p2p21...
        return CorrespondenceResult(fmap12=fmap12, p2p21=p2p21)
```

Train using `TrainingPresets` or a custom loop:

```python
from benchfum import TrainingPresets

model = MyDeepFMNet()
trainer = TrainingPresets.create_trainer(
    "unsupervised_standard",
    model=model,
    train_set=train_pairs,
    val_set=val_pairs,
    checkpoint_path="checkpoints/my_model.pth",
)
trainer.train()
```

### Step 3 — Run the Benchmark

```bash
python -m benchfum.challenges.deep_fmap.run \
    --dataset           /path/to/faust            \
    --fmnet_ckpt        checkpoints/FMNet.pth      \
    --robust_fmnet_ckpt checkpoints/RobustFMNet.pth \
    --my_model_ckpt     checkpoints/my_model.pth
```

Or in Python using `TrainedModelWrapper`:

```python
from benchfum import compare, ModelPresets
from geomfum.learning.wrappers import TrainedModelWrapper

methods = {
    "MyDeepFMNet": TrainedModelWrapper(
        MyDeepFMNet(), checkpoint_path="checkpoints/my_model.pth"
    ),
    "FMNet": TrainedModelWrapper(
        ModelPresets.build("fmnet_diffusion_standard"),
        checkpoint_path="checkpoints/FMNet.pth",
    ),
    "RobustFMNet": TrainedModelWrapper(
        ModelPresets.build("robust_fmnet_standard"),
        checkpoint_path="checkpoints/RobustFMNet.pth",
    ),
}

results = compare(methods, dataset=test_pairs, metrics=["geodesic_error"])
results.print_comparison()
results.save_all("results/deep_fmap/")
```

---

## Refinement Methods — Refinement Challenge

The refinement challenge isolates the contribution of a **refinement step** from the rest of
the pipeline. All methods start from the **same** initial functional map (produced by a fixed
base matcher); any difference in the final score is therefore **solely due to refinement**.

### Quick Start

```bash
# Run all standard refiners (identity / ICP / ZoomOut / ICP+ZO)
python -m benchfum.challenges.refinement.run \
    --dataset /path/to/faust/train_set

# Override the base method (any name from MatcherPresets)
python -m benchfum.challenges.refinement.run \
    --dataset /path/to/faust/train_set --base_method fmap_zo

# Quick check with 20 pairs
python -m benchfum.challenges.refinement.run \
    --dataset /path/to/faust/train_set --n_pairs 20

# Include your own refiner in the comparison
python -m benchfum.challenges.refinement.run \
    --dataset /path/to/faust/train_set --my_refiner
```

Expected output:
```
Challenge   : Refinement FAUST
Dataset     : /path/to/faust/train_set
Base method : fmap
Refiners    : ['identity', 'icp', 'zoomout', 'icp_zoomout']
Spectrum    : k=200

Method          | geodesic_error
----------------|----------------
No Refinement   | 0.0934±0.0201
ICP             | 0.0712±0.0165
ZoomOut         | 0.0681±0.0153
ICP+ZoomOut     | 0.0612±0.0143
```

### Refiner Interface

A refiner is any callable with this signature:

```python
def __call__(self, fmap12, basis_a, basis_b):
    # fmap12         : np.ndarray [k_b, k_a]  — initial functional map from A→B
    # basis_a.vecs   : np.ndarray [n_a, k_a]  — Laplacian eigenvectors of shape A
    # basis_a.vals   : np.ndarray [k_a]        — Laplacian eigenvalues of shape A
    # basis_b.vecs   : np.ndarray [n_b, k_b]  — Laplacian eigenvectors of shape B
    # basis_b.vals   : np.ndarray [k_b]        — Laplacian eigenvalues of shape B
    #
    # return: refined fmap12 of shape [k_b, k_a]
    ...
```

### Add Your Refinement Method

**Option A — Implement and compare directly in Python:**

```python
from benchfum import compare, MatcherPresets, RefinementMatcher, RefinementPresets

class MyRefiner:
    def __call__(self, fmap12, basis_a, basis_b):
        # your refinement algorithm
        return refined_fmap12

base = MatcherPresets.build("fmap")
results = compare(
    {
        "No refinement": RefinementMatcher(base, RefinementPresets.build("identity")),
        "ICP+ZO":        RefinementMatcher(base, RefinementPresets.build("icp_zoomout")),
        "MyRefiner":     RefinementMatcher(base, MyRefiner()),
    },
    dataset=pairs,
    metrics=["geodesic_error"],
)
results.print_comparison()
results.save_all("results/refinement/")
```

**Option B — JSON config** (drop a file in `configs/refiners/`):

```json
{
  "_name": "MyRefiner",
  "_description": "My custom refinement strategy.",
  "type": "IcpRefiner",
  "nit": 30
}
```

Then access it by name: `RefinementPresets.build("my_refiner")`.

**Option C — Edit `challenges/refinement/run.py` directly:**

Implement the `MyRefiner` placeholder in `run.py` and run with `--my_refiner`:

```bash
python -m benchfum.challenges.refinement.run \
    --dataset /path/to/faust --my_refiner
```

### Available Refiners

```python
from benchfum import RefinementPresets

print(RefinementPresets.list_presets())     # list all named refiners
r   = RefinementPresets.build("icp_zoomout")  # build a refiner
cfg = RefinementPresets.describe("zoomout")   # inspect raw JSON config
```

| Name | Description |
|------|-------------|
| `identity` | No refinement — pass-through baseline |
| `icp` | Iterative Closest Point on the functional map |
| `zoomout` | ZoomOut upsampling (gradually increases FM size) |
| `icp_zoomout` | ICP followed by ZoomOut — strong standard baseline |

---

## Challenge Config Format

A challenge config is a JSON file that fully specifies a benchmark run.

### Classical challenge config

```jsonc
{
  "_name": "Landmark-Based FAUST",
  "_description": "...",
  "dataset": {
    "k": 200,                           // Laplacian spectrum size
    "landmark_indices": [2959, 2948]    // canonical vertex indices (optional)
  },
  // Baselines: names must match files in configs/matchers/
  "baselines": ["wks_nn", "fmap", "fmap_zo", "lfmap", "lfmap_zo"],
  "metrics": ["geodesic_error"],
  "bidirectional": false,
  "n_pairs": null                       // null = all pairs; int = random sample
}
```

### Deep learning challenge config

```jsonc
{
  "_name": "Deep Functional Maps FAUST",
  "_description": "...",
  "dataset": {"k": 200},
  // Baselines: name → ModelPresets key (checkpoint paths given at runtime)
  "baselines": {
    "FMNet":       "fmnet_diffusion_standard",
    "RobustFMNet": "robust_fmnet_standard"
  },
  "metrics": ["geodesic_error"],
  "bidirectional": false
}
```

### Refinement challenge config

```jsonc
{
  "_name": "Refinement FAUST",
  "_description": "...",
  "dataset": {"k": 200},
  "base_method": "fmap",              // any name from configs/matchers/
  // Refiners: names must match files in configs/refiners/
  "refiners": ["identity", "icp", "zoomout", "icp_zoomout"],
  "metrics": ["geodesic_error"],
  "bidirectional": false
}
```

---

## Extending benchfum

### Add a new named matcher

1. Write a JSON config in `benchfum/configs/matchers/my_matcher.json`
2. It's immediately available via `MatcherPresets.build("my_matcher")`

No code changes required.

### Add a new benchmark challenge

1. Create `benchfum/configs/challenges/my_challenge.json`
2. Copy the closest `challenges/*/run.py` to `challenges/my_challenge/run.py` and adapt:
   - Update `_CHALLENGE_CONFIG_PATH` to point at your new config
   - Adjust `build_dataset()` if your dataset has a different structure
3. Run: `python -m benchfum.challenges.my_challenge.run --dataset /path/to/data`

---

## Matcher Reference

All matchers in `configs/matchers/` are accessible via `MatcherPresets.build(name)`.

| Name | Type | Description |
|------|------|-------------|
| `wks_nn` | FeatureMatcher | WKS nearest-neighbour — simplest baseline, no FM |
| `fmap` | FunctionalMapMatcher | Classic FM with WKS, no refinement |
| `fmap_zo` | FunctionalMapMatcher | FM + ICP + ZoomOut — strong non-landmark baseline |
| `lfmap` | FunctionalMapMatcher | Landmark-constrained FM, no refinement |
| `lfmap_zo` | FunctionalMapMatcher | Landmark FM + ICP + ZoomOut — strongest classical |
| `quick` | FunctionalMapMatcher | k=20, ICP only — fast, approximate |
| `standard` | FunctionalMapMatcher | k=30, ICP + ZoomOut — good default |
| `precise` | FunctionalMapMatcher | k=50, more iterations — best quality |
| `feature_wks` | FeatureMatcher | WKS feature matching, no functional map |

```python
from benchfum import MatcherPresets

print(MatcherPresets.list_presets())         # all available names
m   = MatcherPresets.build("lfmap_zo")       # build a matcher
cfg = MatcherPresets.describe("lfmap_zo")    # inspect the raw JSON config
```

---

## Model Preset Reference

| Name | Architecture | Size | Description |
|------|-------------|------|-------------|
| `fmnet_diffusion_small` | FMNet | k=128, ch=64 | Fast training |
| `fmnet_diffusion_standard` | FMNet | k=200, ch=128 | Recommended |
| `fmnet_diffusion_large` | FMNet | k=300, ch=256 | Best accuracy |
| `robust_fmnet_small` | RobustFMNet | k=128, ch=64 | Fast training |
| `robust_fmnet_standard` | RobustFMNet | k=200, ch=128 | Recommended |
| `robust_fmnet_large` | RobustFMNet | k=300, ch=256 | Best accuracy |

| Training Preset | Epochs | Losses | Description |
|----------------|--------|--------|-------------|
| `unsupervised_quick` | 5 | Orth + Bij | Fast sanity check |
| `unsupervised_standard` | 50 | Orth + Bij + LapComm | Recommended |
| `unsupervised_precise` | 100 | + DescComm | Best quality |
| `supervised` | 50 | GroundTruth | When GT is available |
| `supervised_plus_regularization` | 50 | GT + Orth + Bij | GT + regularization |
