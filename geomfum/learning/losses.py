"""Losses for Deep Functional Maps training."""

import torch
import torch.nn as nn

import geomfum.linalg as la


class LossManager:
    """
    Manages a list of loss functions and their weights for model training.

    Parameters
    ----------
    losses : list of (nn.Module, float) or list of nn.Module
        List of (loss_module, weight) tuples, or just loss modules (weight=1.0).
    """

    def __init__(self, losses):
        self.losses = losses

    def compute_loss(self, outputs):
        """Compute the total loss and a dictionary of individual losses.

        Parameters
        ----------
        outputs : dict
            Dictionary containing the outputs of the model, which should include all required inputs for the loss functions

        Returns
        -------
        total_loss : torch.Tensor
            Scalar tensor representing the total loss computed from all loss functions.
        loss_dict : dict
            Dictionary mapping loss function names to their computed values.
        """
        total_loss = 0
        loss_dict = {}
        for loss_fn in self.losses:
            # Get required input keys for this loss
            required_keys = getattr(loss_fn, "required_inputs", None)
            if required_keys is not None:
                if not all(k in outputs for k in required_keys):
                    # Skip this loss if any required input is missing
                    continue
                args = [outputs[k] for k in required_keys]
                loss_value = loss_fn(*args)
            else:
                # fallback: pass the whole dict
                loss_value = loss_fn(outputs)
            name = loss_fn.__class__.__name__
            loss_dict[name] = loss_value.item()
            total_loss += loss_value
        return total_loss, loss_dict


######################LOSS IMPLEMENTATIONS ############################


class SquaredFrobeniusLoss(nn.Module):
    """
    Computes the mean squared Frobenius norm between two input tensors.

    Parameters
    ----------
    None
    """

    def forward(self, a, b):
        """
        Forward pass.

        Parameters
        ----------
        a : torch.Tensor
            First input tensor matrix.
        b : torch.Tensor
            Second input tansor matrix, must be broadcastable to the shape of `a`.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the mean squared Frobenius norm between `a` and `b`.
        """
        return torch.mean(torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1)))


class OrthonormalityLoss(nn.Module):
    """
    Computes the orthonormality error of a functional map by measuring the mean squared Frobenius norm between C^T C and the identity matrix.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap21"]

    def forward(self, fmap12, fmap21):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor of shape ( spectrum_size_b, spectrum_size_a).
        fmap21 : torch.Tensor
            Functional map tensor of shape ( spectrum_size_a, spectrum_size_b).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted mean squared Frobenius norm between C^T C and the identity matrix.
        """
        eye_b = torch.eye(fmap12.shape[1], device=fmap12.device)
        eye_a = torch.eye(fmap21.shape[0], device=fmap21.device)
        return self.weight * (
            self.metric(torch.mm(fmap12.T, fmap12), eye_b)
            + self.metric(torch.mm(fmap21.T, fmap21), eye_a)
        )


class BijectivityLoss(nn.Module):
    """
    Computes the bijectivity error of two functional maps by measuring the mean squared Frobenius norm between fmap12 fmap21 and the identity matrix, and between fmap21 fmap12 and the identity matrix.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap21"]

    def forward(self, fmap12, fmap21):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from shape 1 to shape 2 of shape (spectrum_size_b, spectrum_size_a).
        fmap21 : torch.Tensor
            Functional map tensor from shape 2 to shape 1 of shape (spectrum_size_a, spectrum_size_b).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted mean squared Frobenius norm between fmap12 fmap21 and the identity matrix, and between fmap21 fmap12 and the identity matrix.
        """
        eye_b = torch.eye(fmap12.shape[0], device=fmap12.device)
        eye_a = torch.eye(fmap21.shape[0], device=fmap21.device)
        return self.weight * self.metric(
            torch.mm(fmap12, fmap21), eye_b
        ) + self.weight * self.metric(torch.mm(fmap21, fmap12), eye_a)


class LaplacianCommutativityLoss(nn.Module):
    """
    Computes the Laplacian commutativity error of a functional map by measuring the discrepancy between the action of the Laplacian eigenvalues and the functional map.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap21", "shape_a", "shape_b"]

    def forward(self, fmap12, fmap21, shape_a, shape_b):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from source to target shape, of shape ( spectrum_size_b, spectrum_size_a ).
        shape_a : Shape
            Shape object containing source shape information.
        shape_b : Shape
            Shape object containing target shape information.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted squared Frobenius norm of the Laplacian commutativity error.
        """
        return self.weight * self.metric(
            torch.einsum("bc,c->bc", fmap12, shape_b.basis.vals),
            torch.einsum("b,bc->bc", shape_a.basis.vals, fmap12),
        ) + self.weight * self.metric(
            torch.einsum("bc,c->bc", fmap21, shape_a.basis.vals),
            torch.einsum("b,bc->bc", shape_b.basis.vals, fmap21),
        )


class Fmap_Supervision(nn.Module):
    """
    Computes the supervision loss between predicted and ground truth functional maps.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap12_sup"]

    def forward(self, fmap12, fmap12_sup):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from source to target shape, of shape (batch_size, dim_out, dim_in).
        fmap12_sup : torch.Tensor
            Supervised functional map tensor from source to target shape, of shape (batch_size, dim_out, dim_in).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted squared Frobenius norm of the difference between predicted and supervised functional maps.
        """
        return self.weight * self.metric(fmap12, fmap12_sup)


class DescriptorCommutativityLoss(nn.Module):
    """
    Computes the descriptor commutativity loss for learning scenarios.

    This loss enforces that functional maps commute with multiplication operators
    derived from descriptors. It's equivalent to OperatorCommutativityEnforcing.from_multiplication
    but designed for PyTorch training.

    Parameters
    ----------
    weight: float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap21", "desc_a", "desc_b", "shape_a", "shape_b"]

    def _compute_multiplication_operators(self, basis, desc):
        """
        Compute multiplication operators for descriptors.

        Parameters
        ----------
        basis : Basis
            Basis object containing eigenvectors and pseudo-inverse.
        desc : torch.Tensor
            Descriptors of shape (num_vertices, num_descriptors).

        Returns
        -------
        operators : torch.Tensor
            Multiplication operators of shape (num_descriptors, spectrum_size, spectrum_size).
        """
        # desc: (num_vertices, num_descriptors)
        # basis.vecs: (num_vertices, spectrum_size)
        # basis.pinv: (spectrum_size, num_vertices)

        operators = []
        for desc_i in desc:
            operator = basis.pinv @ la.rowwise_scaling(desc_i, basis.vecs)
            operators.append(operator)

        return torch.stack(operators)  # (num_descriptors, spectrum_size, spectrum_size)

    def forward(self, fmap12, fmap21, desc_a, desc_b, shape_a, shape_b):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from shape 1 to shape 2 of shape (spectrum_size_b, spectrum_size_a).
        fmap21 : torch.Tensor
            Functional map tensor from shape 2 to shape 1 of shape (spectrum_size_a, spectrum_size_b).
        desc_a : torch.Tensor
            Descriptors for shape A of shape (num_vertices_a, num_descriptors).
        desc_b : torch.Tensor
            Descriptors for shape B of shape (num_vertices_b, num_descriptors).
        shape_a : TriangleMesh or PointCloud
            TriangleMesh object containing source shape information.
        shape_b : TriangleMesh or PointCloud
            TriangleMesh object containing target shape information.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted descriptor commutativity loss.
        """
        # Compute multiplication operators for each descriptor
        oper_a = self._compute_multiplication_operators(shape_a.basis, desc_a)
        oper_b = self._compute_multiplication_operators(shape_b.basis, desc_b)

        total_loss = 0
        # Compute commutativity loss for each descriptor
        for oper_a_i, oper_b_i in zip(oper_a, oper_b):
            left_side = torch.mm(fmap12, oper_a_i)  # (spectrum_size_b, spectrum_size_a)
            right_side = torch.mm(
                oper_b_i, fmap12
            )  # (spectrum_size_b, spectrum_size_a)
            loss_12 = self.metric(left_side, right_side)

            # For fmap21: C21 @ M_b = M_a @ C21
            left_side_21 = torch.mm(
                fmap21, oper_b_i
            )  # (spectrum_size_a, spectrum_size_b)
            right_side_21 = torch.mm(
                oper_a_i, fmap21
            )  # (spectrum_size_a, spectrum_size_b)
            loss_21 = self.metric(left_side_21, right_side_21)

            total_loss += loss_12 + loss_21

        total_loss = total_loss / oper_a.shape[0]

        return self.weight * total_loss


class GroundTruthSupervisionLoss(nn.Module):
    """
    Computes the loss of a functional map by measuring the discrepancy between the functional map and a ground truth functional map.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap21", "shape_a", "shape_b", "corr_a", "corr_b"]

    def _compute_ground_truth_map(self, shape_a, shape_b, corr_a, corr_b):
        """Compute the ground truth functional maps.

        Parameters
        ----------
        shape_a : TriangleMesh
            TriangleMesh object containing source shape information.
        shape_b : TriangleMesh
            TriangleMesh object containing target shape information.
        corr_a : torch.Tensor
            Indices of source correspondences.
        corr_b : torch.Tensor
            Indices of target correspondences.

        Returns
        -------
        fmap12_gt ,fmap21_gt : torch.Tensor
            Ground truth functional maps from shape 1 to shape 2 and from shape 2 to shape 1.
        """
        fmap12_gt = shape_b.basis.pinv[:, corr_b] @ shape_a.basis.vecs[corr_a, :]

        fmap21_gt = shape_a.basis.pinv[:, corr_a] @ shape_b.basis.vecs[corr_b, :]

        return fmap12_gt, fmap21_gt

    def forward(self, fmap12, fmap21, shape_a, shape_b, corr_a, corr_b):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from shape 1 to shape 2 of shape (spectrum_size_b, spectrum_size_a).
        fmap21 : torch.Tensor
            Functional map tensor from shape 2 to shape 1 of shape (spectrum_size_a, spectrum_size_b).
        shape_a : TriangleMesh
            TriangleMesh object containing source shape information.
        shape_b : TriangleMesh
            TriangleMesh object containing target shape information.
        corr_a : torch.Tensor
            Indices of source correspondences.
        corr_b : torch.Tensor
            Indices of target correspondences.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted mean squared Frobenius norm between fmap12 and the ground truth functional map, and between fmap21 and the ground truth functional map.
        """
        fmap12_gt, fmap21_gt = self._compute_ground_truth_map(
            shape_a, shape_b, corr_a, corr_b
        )
        return self.weight * self.metric(fmap12, fmap12_gt) + self.weight * self.metric(
            fmap21, fmap21_gt
        )


class FmapDescriptorsSupervisionLoss(nn.Module):
    """
    Computes the loss of a functional map by measuring the discrepancy between the functional map and a functional map computed by the similarity of the descriptors.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap21", "refined_fmap12", "refined_fmap21"]

    def forward(self, fmap12, fmap21, refined_fmap12, refined_fmap21):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from shape 1 to shape 2 of shape (spectrum_size_b, spectrum_size_a).
        fmap21 : torch.Tensor
            Functional map tensor from shape 2 to shape 1 of shape (spectrum_size_a, spectrum_size_b).
        refined_fmap12 : torch.Tensor
            Descriptor-based functional map from shape 1 to shape 2 of shape (spectrum_size_b, spectrum_size_a).
        refined_fmap21 : torch.Tensor
            Descriptor-based functional map from shape 2 to shape 1 of shape (spectrum_size_a, spectrum_size_b).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted mean squared Frobenius norm between fmap12 and refined_fmap12, and between fmap21 and refined_fmap21.
        """
        return self.weight * self.metric(
            fmap12, refined_fmap12
        ) + self.weight * self.metric(fmap21, refined_fmap21)


class DirichletLoss(nn.Module):
    """Dirichlet energy of transported vertex positions (URRSM test-time loss).

    Encourages correspondences to map smooth functions to smooth functions.
    Replicates the ``DirichletLoss`` used in URRSM at test-time refinement.

    For each pair, the loss computes:

    .. math::

        w \\left(
            E_{\\text{Dir}}(P_{ab} \\, v_b,\\, \\mathcal{S}_a)
            +
            E_{\\text{Dir}}(P_{ba} \\, v_a,\\, \\mathcal{S}_b)
        \\right)

    where :math:`E_{\\text{Dir}}(f, \\mathcal{S})` is approximated spectrally as

    .. math::

        \\sum_k \\lambda_k \\| \\Phi_k^\\top M f \\|^2

    using the pre-computed Laplace–Beltrami eigenvalues :math:`\\lambda_k` and
    mass-weighted pseudo-inverse :math:`\\Phi^\\top M` (``basis.pinv``).

    Parameters
    ----------
    weight : float, optional
        Loss weight (default: 1.0).
    """

    required_inputs = ["soft_perm_ab", "soft_perm_ba", "shape_a", "shape_b"]

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    @staticmethod
    def _spectral_dirichlet(perm, verts, basis):
        """Spectral Dirichlet energy of ``perm @ verts`` on the shape with ``basis``.

        Parameters
        ----------
        perm : torch.Tensor, shape=[n_target, n_source]
            Soft permutation (rows sum to 1).
        verts : torch.Tensor, shape=[n_source, 3]
            Vertex positions of the source shape.
        basis : LaplaceEigenBasis
            Basis of the target shape (provides ``vals`` and ``pinv``).

        Returns
        -------
        energy : torch.Tensor
            Scalar Dirichlet energy.
        """
        # Transport source vertices to target domain: [n_target, 3]
        transported = perm @ verts
        # Spectral coefficients: [K, 3]  (pinv = Phi^T M, shape [K, n_target])
        coeffs = basis.pinv @ transported
        # Dirichlet energy: sum_k lambda_k * ||coeffs_k||^2
        # basis.vals: [K], coeffs: [K, 3]
        energy = (basis.vals[:, None] * coeffs**2).mean()
        return energy

    def forward(self, soft_perm_ab, soft_perm_ba, shape_a, shape_b):
        """Compute Dirichlet loss.

        Parameters
        ----------
        soft_perm_ab : torch.Tensor, shape=[n_a, n_b]
            Soft permutation mapping b vertices to a domain.
        soft_perm_ba : torch.Tensor, shape=[n_b, n_a]
            Soft permutation mapping a vertices to b domain.
        shape_a : TriangleMesh
        shape_b : TriangleMesh

        Returns
        -------
        loss : torch.Tensor
        """
        e_a = self._spectral_dirichlet(soft_perm_ab, shape_b.vertices, shape_a.basis)
        e_b = self._spectral_dirichlet(soft_perm_ba, shape_a.vertices, shape_b.basis)
        return self.weight * (e_a + e_b)


class WeightedBCELoss(nn.Module):
    """Overlap supervision loss for partial shape matching.

    Binary cross-entropy between predicted overlap scores and ground-truth
    partiality masks, with class-reweighting to handle the imbalance between
    present and absent vertices (EchoMatch-style BCE loss).

    Applied to both directions (A → B and B → A) and summed.

    Parameters
    ----------
    weight : float, optional
        Scalar multiplier for the total loss (default 1.0).
    """

    required_inputs = ["overlap_ab", "overlap_ba", "mask_a", "mask_b"]

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    @staticmethod
    def _weighted_bce(pred, target):
        """Class-frequency-weighted BCE matching the original EchoMatch formula.

        When gt is all-positive or all-negative, falls back to unweighted BCE
        (reweighting would zero-out one class entirely).
        """
        # Clamp to open interval: sigmoid can saturate to 0 or 1 in float32,
        # which triggers a CUDA device-side assert in binary_cross_entropy.
        pred = pred.clamp(min=1e-6, max=1.0 - 1e-6)
        class_loss = torch.nn.functional.binary_cross_entropy(
            pred, target, reduction="none"
        )
        n_positive = target.sum()
        total = float(target.size(0))
        if n_positive == 0 or n_positive == total:
            # Degenerate: fallback to unweighted
            return class_loss.mean()
        weights = torch.ones_like(target)
        w_negative = n_positive / total
        w_positive = 1.0 - w_negative
        weights[target >= 0.5] = w_positive
        weights[target < 0.5] = w_negative
        return (weights * class_loss).mean()

    def forward(self, overlap_ab, overlap_ba, mask_a, mask_b):
        """Compute overlap BCE loss.

        Parameters
        ----------
        overlap_ab : Tensor[n_a]
            Predicted overlap scores for shape A (output of sigmoid).
        overlap_ba : Tensor[n_b]
            Predicted overlap scores for shape B (output of sigmoid).
        mask_a : Tensor[n_a]
            GT binary mask for shape A.
        mask_b : Tensor[n_b]
            GT binary mask for shape B.

        Returns
        -------
        Tensor (scalar)
        """
        loss_a = self._weighted_bce(overlap_ab.float(), mask_a.float())
        loss_b = self._weighted_bce(overlap_ba.float(), mask_b.float())
        return self.weight * (loss_a + loss_b)


class CrossNCELoss(nn.Module):
    """Cross-shape normalised contrastive embedding (NCE) loss.

    For each ground-truth correspondence pair (i in A, j in B),
    the feature of vertex i should be closer to the feature of vertex j
    than to all other vertices in B (InfoNCE / NT-Xent style).

    Parameters
    ----------
    temperature : float, optional
        Softmax temperature τ (default 0.07).
    weight : float, optional
        Scalar multiplier (default 1.0).
    """

    required_inputs = ["descr_a", "descr_b", "corr_a", "corr_b"]

    def __init__(self, temperature=0.07, weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(self, descr_a, descr_b, corr_a, corr_b):
        """Compute cross-NCE loss (both directions).

        Parameters
        ----------
        descr_a : Tensor[n_feat, n_a]
            Features for shape A (columns = vertices).
        descr_b : Tensor[n_feat, n_b]
        corr_a : Tensor[n_corr]
            GT source correspondence indices into A.
        corr_b : Tensor[n_corr]
            GT target correspondence indices into B.

        Returns
        -------
        Tensor (scalar)
        """
        # descr_a: [n_feat, n_a]  → transpose to [n_a, n_feat]
        feat_a = descr_a.T.float()  # [n_a, n_feat]
        feat_b = descr_b.T.float()  # [n_b, n_feat]

        # Normalise
        feat_a = feat_a / (feat_a.norm(dim=-1, keepdim=True) + 1e-8)
        feat_b = feat_b / (feat_b.norm(dim=-1, keepdim=True) + 1e-8)

        # Full cross-similarity matrix [n_a, n_b]
        logits = (feat_a @ feat_b.T) / self.temperature

        # Direction A→B: for each corr_a vertex, predict corr_b
        loss_ab = torch.nn.functional.cross_entropy(logits[corr_a], corr_b.long())

        # Direction B→A: for each corr_b vertex, predict corr_a
        loss_ba = torch.nn.functional.cross_entropy(logits.T[corr_b], corr_a.long())

        return self.weight * (loss_ab + loss_ba)


class SelfNCELoss(nn.Module):
    """Within-shape identity NCE loss.

    Encourages each vertex's feature to be most similar to itself
    (diagonal of the self-similarity matrix should be the maximum).
    Applied independently to both shapes A and B.

    Parameters
    ----------
    temperature : float, optional
        Softmax temperature τ (default 0.07).
    weight : float, optional
        Scalar multiplier (default 1.0).
    max_vertices : int, optional
        Maximum number of vertices to use per shape (random subsample) to
        avoid O(V²) memory cost on large shapes.  Default 2048.
    """

    required_inputs = ["descr_a", "descr_b"]

    def __init__(self, temperature=0.07, weight=1.0, max_vertices=2048):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.max_vertices = max_vertices

    @staticmethod
    def _self_nce(feat, temperature, max_v):
        """InfoNCE loss where every vertex is its own positive pair."""
        n = feat.shape[0]
        if n > max_v:
            idx = torch.randperm(n, device=feat.device)[:max_v]
            feat = feat[idx]
            n = max_v
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        logits = (feat @ feat.T) / temperature  # [n, n]
        labels = torch.arange(n, device=feat.device)
        return torch.nn.functional.cross_entropy(logits, labels)

    def forward(self, descr_a, descr_b):
        """Compute self-NCE loss.

        Parameters
        ----------
        descr_a : Tensor[n_feat, n_a]
        descr_b : Tensor[n_feat, n_b]

        Returns
        -------
        Tensor (scalar)
        """
        feat_a = descr_a.T.float()
        feat_b = descr_b.T.float()
        loss_a = self._self_nce(feat_a, self.temperature, self.max_vertices)
        loss_b = self._self_nce(feat_b, self.temperature, self.max_vertices)
        return self.weight * (loss_a + loss_b)


class GeodesicError(nn.Module):
    """
    Computes the accuracy of a correspondence by measuring the mean of the geodesic distances between points of the predicted permuted target and the ground truth target.

    Parameters
    ----------
    None
    """

    def __init__(self):
        super().__init__()

    required_inputs = [
        "p2p12",
        "dist_b",
        "corr_a",
        "corr_b",
    ]

    def _compute_geodesic_loss(self, p2p, target_dist, source_corr, target_corr):
        """
        Compute the geodesic loss for batched inputs.

        Parameters
        ----------
        p2p : torch.Tensor
            Predicted point-to-point map.
        target_dist : torch.Tensor
            Geodesic distance matrix for the target shape.
        source_corr : torch.Tensor
            Indices of source correspondences.
        target_corr : torch.Tensor
            Indices of target correspondences.

        Returns
        -------
        torch.Tensor
            Mean geodesic distance error.
        """
        return torch.mean(target_dist[p2p[source_corr], target_corr])

    def forward(self, p2p12, dist_b, corr_a, corr_b):
        """
        Forward pass.

        Parameters
        ----------
        p2p12 : torch.Tensor
            Predicted point-to-point map.
        dist_b : torch.Tensor
            Geodesic distance matrix for the target shape.
        corr_a : torch.Tensor
            Indices of source correspondences.
        corr_b : torch.Tensor
            Indices of target correspondences.

        Returns
        -------
        torch.Tensor
            Mean geodesic distance error.
        """
        loss = self._compute_geodesic_loss(p2p12, dist_b, corr_a, corr_b)
        return loss


######################PARTIAL SHAPE EVALUATION METRICS ############################


class PartialGeodesicError(nn.Module):
    """Geodesic error restricted to the ground-truth overlap region.

    Only correspondences where the ground-truth mask marks the vertex as
    *present* in shape A (``mask_a[corr_a] == 1``) are evaluated.  This
    matches the filtered evaluation protocol of EchoMatch / SHREC16.

    Parameters
    ----------
    weight : float
        Scalar weight applied to the metric value (for use in LossManager).
        Default 1.0.
    """

    required_inputs = ["p2p21", "dist_a", "corr_a", "corr_b", "mask_a"]

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, p2p21, dist_a, corr_a, corr_b, mask_a):
        """Compute filtered geodesic error.

        Parameters
        ----------
        p2p21 : Tensor[n_b]
            Predicted p2p: for each vertex in B, its match in A.
        dist_a : Tensor[n_a, n_a]
            Geodesic distance matrix of shape A.
        corr_a : Tensor[n_corr]
            GT correspondence indices into shape A.
        corr_b : Tensor[n_corr]
            GT correspondence indices into shape B.
        mask_a : Tensor[n_a]
            Binary mask: 1 if vertex in A is in the overlap region.

        Returns
        -------
        Tensor (scalar)
        """
        valid = mask_a[corr_a] > 0.5
        if valid.sum() == 0:
            return torch.tensor(0.0, device=dist_a.device, requires_grad=False)
        corr_a_valid = corr_a[valid]
        corr_b_valid = corr_b[valid]
        predicted_in_a = p2p21[corr_b_valid]
        geo_err = dist_a[corr_a_valid, predicted_in_a]
        return self.weight * geo_err.mean()


class OverlapIoU(nn.Module):
    """Intersection-over-Union between predicted and ground-truth overlap masks.

    The predicted overlap ``overlap_ab`` is thresholded at ``threshold``
    to produce a binary prediction.

    Returns ``weight * (1 - IoU)`` so that minimising this value maximises
    IoU, consistent with ``mode="min"`` in the trainer.

    Parameters
    ----------
    threshold : float
        Sigmoid threshold to binarise predicted overlap scores.  Default 0.5.
    weight : float
        Scalar multiplier.  Default 1.0.
    """

    required_inputs = ["overlap_ab", "mask_a"]

    def __init__(self, threshold=0.5, weight=1.0):
        super().__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, overlap_ab, mask_a):
        """Compute overlap IoU cost.

        Parameters
        ----------
        overlap_ab : Tensor[n_a]
            Predicted overlap scores in [0, 1].
        mask_a : Tensor[n_a]
            Ground-truth binary overlap mask.

        Returns
        -------
        Tensor (scalar)
            ``weight * (1 - IoU)``.
        """
        pred = (overlap_ab >= self.threshold).float()
        gt = (mask_a >= 0.5).float()
        intersection = (pred * gt).sum()
        union = ((pred + gt) >= 1.0).float().sum()
        if union == 0:
            iou = torch.tensor(1.0, device=overlap_ab.device)
        else:
            iou = intersection / union
        return self.weight * (1.0 - iou)


class PCKMetric(nn.Module):
    """Area under the PCK (Percentage of Correct Keypoints) curve.

    The PCK curve plots the fraction of valid correspondences with geodesic
    error below threshold *t*, for *t* ranging from 0 to ``t_max``.  The
    AUC is approximated by the trapezoidal rule over ``n_steps`` thresholds.

    Only correspondences where ``mask_a[corr_a] == 1`` are evaluated.

    Returns ``weight * (1 - AUC)`` so that minimising this value maximises
    the AUC, consistent with ``mode="min"`` in the trainer.

    Parameters
    ----------
    t_max : float
        Maximum geodesic threshold (normalised by diameter).  Default 0.20.
    n_steps : int
        Number of threshold steps for AUC integration.  Default 100.
    weight : float
        Scalar multiplier.  Default 1.0.
    """

    required_inputs = ["p2p21", "dist_a", "corr_a", "corr_b", "mask_a"]

    def __init__(self, t_max=0.20, n_steps=100, weight=1.0):
        super().__init__()
        self.t_max = t_max
        self.n_steps = n_steps
        self.weight = weight

    def forward(self, p2p21, dist_a, corr_a, corr_b, mask_a):
        """Compute 1 - AUC of the PCK curve.

        Parameters
        ----------
        p2p21 : Tensor[n_b]
        dist_a : Tensor[n_a, n_a]
        corr_a : Tensor[n_corr]
        corr_b : Tensor[n_corr]
        mask_a : Tensor[n_a]

        Returns
        -------
        Tensor (scalar)
            ``weight * (1 - AUC)``.
        """
        valid = mask_a[corr_a] > 0.5
        if valid.sum() == 0:
            return torch.tensor(0.0, device=dist_a.device, requires_grad=False)
        corr_a_valid = corr_a[valid]
        corr_b_valid = corr_b[valid]
        predicted_in_a = p2p21[corr_b_valid]
        geo_err = dist_a[corr_a_valid, predicted_in_a]
        diam = dist_a.max()
        geo_err_norm = geo_err / diam if diam > 0 else geo_err
        thresholds = torch.linspace(0.0, self.t_max, self.n_steps, device=dist_a.device)
        pck_values = torch.stack(
            [(geo_err_norm <= t).float().mean() for t in thresholds]
        )
        auc = torch.trapezoid(pck_values, thresholds) / self.t_max
        return self.weight * (1.0 - auc)
