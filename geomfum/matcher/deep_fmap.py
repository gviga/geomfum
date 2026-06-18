"""Deep functional map matcher: inference + optional per-pair optimization."""

import copy

import torch

from geomfum.matcher.base import BaseMatcher


class DeepFMMatcher(BaseMatcher):
    """Run a deep functional-map model, optionally optimizing it per pair.

    A single entry point for evaluating learning-based models (``FMNet``,
    ``RobustFMNet``, …) inside the matcher / experiment framework. It covers
    the full spectrum of test-time behaviours:

    * **Inference** (``n_iters == 0``, the default): a plain forward pass of a
      (possibly pretrained) model — equivalent to passing the raw model.
    * **Test-time optimization / refinement** (``n_iters > 0``): runs
      ``n_iters`` gradient steps on ``loss_manager`` for each pair before
      extracting the correspondence. With ``restore_weights=True`` (default)
      the base model is restored after every pair, giving purely transductive
      refinement; with ``restore_weights=False`` the adaptation accumulates
      across pairs.

    The optimization loop mirrors the training stage
    (``model.forward() -> LossManager.compute_loss() -> loss.backward()``), so
    any ``BaseModel`` and any combination of ``geomfum.learning.losses`` can be
    plugged in. The loss is supplied separately via ``loss_manager``, so a
    single model can be refined with different objectives just by swapping it.

    Parameters
    ----------
    model : BaseModel, optional
        Deep FM model. Defaults to ``FMNet()`` (random DiffusionNet weights).
    loss_manager : LossManager, optional
        Loss used for per-pair optimization. Required when ``n_iters > 0``.
    fmap_size : int or tuple of int
        Number of LBO eigenfunctions for the functional map. A tuple
        ``(k_b, k_a)`` allows different sizes per shape.
    n_iters : int
        Gradient steps per pair. ``0`` (default) means inference only.
    lr : float
        Adam learning rate (used when ``optimizer_config`` is not given).
    optimizer_config : dict, optional
        Optimizer config with a ``"type"`` key and kwargs, e.g.
        ``{"type": "Adam", "lr": 1e-3}``. Overrides ``lr``.
    grad_clip_norm : float, optional
        If set, clips the gradient L2 norm after each backward pass.
    restore_weights : bool
        If True (default), restore the model weights after each pair so the
        base model is never mutated (transductive refinement).
    checkpoint : str, optional
        Path to a checkpoint to load into ``model.feature_extractor.model``
        at construction time. Accepts a bare state dict or a dict with a
        ``"model_state_dict"`` key.
    device : str or torch.device, optional
        Device for the model. Defaults to the basis device of each pair.
    verbose : bool
        If True, print loss components every 100 iterations.
    """

    def __init__(
        self,
        model=None,
        loss_manager=None,
        fmap_size=30,
        n_iters=0,
        lr=1e-3,
        optimizer_config=None,
        grad_clip_norm=None,
        restore_weights=True,
        checkpoint=None,
        device=None,
        verbose=False,
    ):
        if model is None:
            # Lazy import avoids a circular import: geomfum.learning.models
            # imports geomfum.matcher.base, which loads this package.
            from geomfum.learning.models import FMNet

            model = FMNet()
        self.model = model
        self.loss_manager = loss_manager
        self.fmap_size = (
            fmap_size if isinstance(fmap_size, tuple) else (fmap_size, fmap_size)
        )
        self.n_iters = n_iters
        self.lr = lr
        self.optimizer_config = optimizer_config
        self.grad_clip_norm = grad_clip_norm
        self.restore_weights = restore_weights
        self.device = device
        self.verbose = verbose

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint_path):
        """Load weights into ``model.feature_extractor.model``.

        Parameters
        ----------
        checkpoint_path : str
            Path to a checkpoint. Accepts a bare state dict or a dict with a
            ``"model_state_dict"`` key.
        """
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.feature_extractor.model.load_state_dict(state)

    def _build_optimizer(self):
        config = self.optimizer_config or {"type": "Adam", "lr": self.lr}
        cls = getattr(torch.optim, config.get("type", "Adam"))
        kwargs = {k: v for k, v in config.items() if k != "type"}
        return cls(self.model.parameters(), **kwargs)

    def _optimize(self, shape_a, shape_b):
        """Run ``n_iters`` gradient steps of ``loss_manager`` on the pair."""
        if self.loss_manager is None:
            raise ValueError(
                "loss_manager is required when n_iters > 0 "
                "(per-pair optimization needs a loss to minimize)."
            )
        optimizer = self._build_optimizer()
        self.model.train()
        for i in range(self.n_iters):
            optimizer.zero_grad()
            # Run bidirectionally so all fmap/soft-perm tensors the loss may
            # need (e.g. bijectivity) are available.
            outputs = self.model(shape_a, shape_b, bidirectional=True).to_dict()
            outputs["shape_a"] = shape_a
            outputs["shape_b"] = shape_b
            total_loss, loss_dict = self.loss_manager.compute_loss(outputs)
            total_loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            optimizer.step()
            if self.verbose and (i % 100 == 0 or i == self.n_iters - 1):
                parts = "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                print(f"  iter {i:4d} | {parts}")

    def __call__(self, shape_a, shape_b, bidirectional=False):
        """Match a pair, optionally optimizing the model first.

        Both shapes must have a precomputed spectral basis (``shape.basis``)
        with torch-tensor eigenvectors and mass matrix (same requirement as
        FMNet inference).

        Parameters
        ----------
        shape_a : Shape
            First shape (target for p2p21).
        shape_b : Shape
            Second shape (source for p2p21).
        bidirectional : bool
            If True, also return the reverse-direction correspondence.

        Returns
        -------
        result : CorrespondenceResult
            Model output (``fmap12``, ``p2p21``, descriptors, and the reverse
            direction when ``bidirectional``).
        """
        k_b, k_a = self.fmap_size
        shape_a.basis.use_k = k_a
        shape_b.basis.use_k = k_b

        device = self.device or shape_a.basis.vals.device
        self.model = self.model.to(device)

        if self.n_iters > 0:
            saved_state = (
                copy.deepcopy(self.model.state_dict())
                if self.restore_weights
                else None
            )
            self._optimize(shape_a, shape_b)

        self.model.eval()
        with torch.no_grad():
            result = self.model(shape_a, shape_b, bidirectional=bidirectional)

        if self.n_iters > 0 and self.restore_weights:
            self.model.load_state_dict(saved_state)

        return result
