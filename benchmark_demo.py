"""Small benchmark demo: standard matchers vs a pretrained deep model.

Runs on a tiny FAUST mini-set (registered meshes -> identity ground truth)
and compares classical functional-map / feature matchers against a deep
FMNet whose DiffusionNet feature extractor is loaded from a pretrained
checkpoint (``saved_model_test.pth``).

Run:
    python benchmark_demo.py
"""

import torch

from benchfum import build_matcher_from_json, compare
from geomfum.convert import P2pFromFmConverter
from geomfum.dataset import PairsDataset, ShapeDataset
from geomfum.descriptor.learned import FeatureExtractor
from geomfum.forward_functional_map import ForwardFunctionalMap
from geomfum.learning.models import FMNet
from geomfum.matcher import DeepFMMatcher, DescriptorMatcher, FunctionalMapMatcher
from geomfum.refine import RefinementPipeline, ZoomOut

DATA_DIR = r"C:/Users/giuli/benchdemo"
CKPT = "saved_model_test.pth"
K = 200
FMAP_SIZE = 30


def build_pretrained_fmnet():
    """DeepFMMatcher running an FMNet with pretrained DiffusionNet weights.

    ``n_iters=0`` => pure inference; ``DeepFMMatcher`` loads the checkpoint
    into the feature extractor and runs the model in eval/no-grad.
    """
    feat = FeatureExtractor.from_registry(
        which="diffusionnet", in_channels=3, out_channels=128, k=K
    )
    fmap_module = ForwardFunctionalMap(lmbda=1e3, resolvent_gamma=1.0, bijective=True)
    model = FMNet(
        feature_extractor=feat,
        fmap_module=fmap_module,
        converter=P2pFromFmConverter(),
    )
    return DeepFMMatcher(
        model=model, checkpoint=CKPT, fmap_size=FMAP_SIZE, n_iters=0
    )


def main():
    # 1. Load the mini dataset (spectral basis computed; identity GT corr).
    shapes = ShapeDataset(
        DATA_DIR,
        spectral=True,
        k=K,
        correspondences=True,   # no .vts -> identity fallback (correct for FAUST)
        distances=False,        # euclidean_error needs no geodesic matrix
        device=torch.device("cpu"),
    )
    pairs = PairsDataset(shapes, pair_mode="all", device=torch.device("cpu"))
    print(f"Loaded {len(shapes.shape_files)} shapes -> {len(pairs)} pairs")

    # 2. Methods: standard matchers + pretrained deep model.
    methods = {
        "WKS-NN": DescriptorMatcher(),  # WKS descriptors + nearest neighbour
        "FMap-WKS": FunctionalMapMatcher(fmap_size=FMAP_SIZE),
        "FMap+ZoomOut": FunctionalMapMatcher(
            fmap_size=FMAP_SIZE,
            refiner=RefinementPipeline([ZoomOut(nit=10, step=10)]),
        ),
        "FMNet (pretrained)": build_pretrained_fmnet(),
    }

    # 3. Run the comparison.
    suite = compare(
        methods,
        pairs,
        metrics=["euclidean_error"],
        progress_bar=False,
    )
    suite.print_comparison()


if __name__ == "__main__":
    main()
