"""Evaluation metrics for shape correspondences.

Notation Convention
-------------------
Following the functional maps convention used throughout the library:
- `fmap12` = functional map from shape_a to shape_b, shape = [spectrum_size_b, spectrum_size_a]
- `p2p21` = point-to-point map from shape_b to shape_a (derived from fmap12)
    - For each vertex i in shape_b, p2p21[i] gives the corresponding vertex in shape_a
    - shape = [n_vertices_b]

The evaluation functions expect:
- `shape_a` = target shape (where correspondences land)
- `shape_b` = source shape (where correspondences originate)
- `p2p21` = map from B to A
- `corr_b` = ground truth indices on source (B)
- `corr_a` = ground truth indices on target (A)

The error is computed on the target shape (shape_a).
"""

import gsops.backend as gs
import numpy as np

from geomfum.metric import VertexEuclideanMetric


def normalized_geodesic_error(
    dist_a,
    p2p21,
    corr_a=None,
    corr_b=None,
):
    """Compute normalized geodesic error of a correspondence.

    The geodesic error measures the mean geodesic distance between predicted
    correspondences and ground truth correspondences on the target shape,
    normalized by the geodesic diameter of the target shape.

    Parameters
    ----------
    dist_a : array-like, shape=[n_vertices_a, n_vertices_a]
        Geodesic distance matrix on the target shape (A).
    p2p21 : array-like, shape=[n_vertices_b] or shape=[n_corr]
        Point-to-point map from shape_b to shape_a. For each vertex index i in
        shape_b, p2p21[i] gives the corresponding vertex index in shape_a.
    corr_a : array-like, shape=[n_correspondences], optional
        Indices of ground truth correspondences on target shape (A).
        If None, assumes identity correspondence.
    corr_b : array-like, shape=[n_correspondences], optional
        Indices of ground truth correspondences on source shape (B).
        If None, assumes identity correspondence.

    Returns
    -------
    error : float
        Normalized mean geodesic distance error in [0, 1].

    Notes
    -----
    The geodesic error is computed as:
        error = mean(dist_a[p2p21[corr_b], corr_a]) / diameter_a

    where dist_a is the geodesic distance matrix on the target shape.
    """
    if corr_a is None or corr_b is None:
        # Assume identity correspondence (same mesh topology)
        p2p_gt = gs.arange(len(p2p21))
        geodesic_error = gs.mean(dist_a[p2p21, p2p_gt])
    else:
        # p2p21[corr_b] gives predicted vertices in A for source points in B
        # corr_a gives ground truth vertices in A
        geodesic_error = gs.mean(dist_a[p2p21[corr_b], corr_a])

    return geodesic_error / dist_a.max()


def normalized_euclidean_error(
    shape_a,
    p2p21,
    corr_a=None,
    corr_b=None,
):
    """Compute normalized Euclidean error of a correspondence.

    The Euclidean error measures the mean Euclidean distance between predicted
    correspondences and ground truth correspondences on the target shape,
    normalized by the Euclidean diameter of the target shape.

    Parameters
    ----------
    shape_a : TriangleMesh or PointCloud
        Target shape (where correspondences land).
    shape_b : TriangleMesh or PointCloud
        Source shape (where correspondences originate).
    p2p21 : array-like, shape=[n_vertices_b] or shape=[n_corr]
        Point-to-point map from shape_b to shape_a.
    corr_a : array-like, shape=[n_correspondences], optional
        Indices of ground truth correspondences on target shape (A).
        If None, assumes identity correspondence.
    corr_b : array-like, shape=[n_correspondences], optional
        Indices of ground truth correspondences on source shape (B).
        If None, assumes identity correspondence.

    Returns
    -------
    error : float
        Normalized mean Euclidean distance error in [0, 1].
    """
    vertices_a = shape_a.vertices

    if corr_a is None or corr_b is None:
        # Assume identity correspondence
        p2p_gt = gs.arange(len(p2p21))
        predicted_positions = vertices_a[p2p21]
        gt_positions = vertices_a[p2p_gt]
    else:
        predicted_positions = vertices_a[p2p21[corr_b]]
        gt_positions = vertices_a[corr_a]

    # Compute Euclidean distances
    euclidean_distances = gs.linalg.norm(predicted_positions - gt_positions, axis=-1)
    euclidean_error = gs.mean(euclidean_distances)

    # Normalize by Euclidean diameter
    eucl_metric = VertexEuclideanMetric(shape_a)
    eucl_diam = eucl_metric.dist_matrix().max()

    return euclidean_error / eucl_diam


def dirichlet_energy(shape_a, shape_b, p2p21):
    """Compute Dirichlet energy of a correspondence.

    The Dirichlet energy measures the smoothness of the mapping by computing
    the sum of squared gradient magnitudes of the mapped coordinates.
    A lower Dirichlet energy indicates a smoother, more continuous mapping.

    Parameters
    ----------
    shape_a : TriangleMesh or PointCloud
        Target shape (where correspondences land).
    shape_b : TriangleMesh or PointCloud
        Source shape (where correspondences originate).
    p2p21 : array-like, shape=[n_vertices_b]
        Point-to-point map from shape_b to shape_a.

    Returns
    -------
    energy : float
        Normalized Dirichlet energy (divided by number of source vertices).

    Notes
    -----
    The Dirichlet energy is computed as:
        E = sum_i (v_a[p2p21]^T @ L_b @ v_a[p2p21])

    where v_a[p2p21] are the target vertices indexed by the correspondence,
    and L_b is the Laplacian of the source shape.
    """
    # Get target vertex positions mapped by correspondence
    # For each vertex in B, we get the position of the corresponding vertex in A
    mapping = gs.array(shape_a.vertices[p2p21])

    # Get Laplacian of source mesh (B)
    L, _ = shape_b.laplacian.find()

    # Convert to proper format for matrix operations
    if hasattr(L, "tocsr"):
        L = L.tocsr()

    # Compute Dirichlet energy for each coordinate
    energy_x = mapping[:, 0].T @ L @ mapping[:, 0]
    energy_y = mapping[:, 1].T @ L @ mapping[:, 1]
    energy_z = mapping[:, 2].T @ L @ mapping[:, 2]

    total_energy = energy_x + energy_y + energy_z

    return total_energy / shape_b.n_vertices


def coverage(shape_a, shape_b, p2p21):
    """Compute coverage of a correspondence.

    Coverage measures what fraction of the target shape's area is covered
    by the mapping. A coverage of 1.0 means every vertex in the target
    is mapped to at least once.

    Parameters
    ----------
    shape_a : TriangleMesh or PointCloud
        Target shape (where correspondences land).
    shape_b : TriangleMesh or PointCloud
        Source shape (where correspondences originate).
    p2p21 : array-like, shape=[n_vertices_b]
        Point-to-point map from shape_b to shape_a. For each vertex index i in
        shape_b, p2p21[i] gives the corresponding vertex index in shape_a.

    Returns
    -------
    coverage : float
        Coverage ratio in [0, 1]. The fraction of target area that is
        covered by the mapping.

    Notes
    -----
    The coverage is area-weighted, so vertices with larger associated
    areas contribute more to the coverage metric.
    """
    vertex_areas = shape_a.vertex_areas

    # Get unique target vertices that are mapped to
    unique_targets = gs.unique(p2p21)

    # Compute area-weighted coverage
    covered_area = vertex_areas[unique_targets].sum()
    total_area = vertex_areas.sum()

    return covered_area / total_area


def partial_geodesic_error(dist_a, p2p21, corr_a, corr_b, mask_a):
    """Geodesic error filtered to the ground-truth overlap region.

    Only correspondences where ``mask_a[corr_a] == 1`` are evaluated,
    following the filtered protocol of EchoMatch / SHREC16.

    Parameters
    ----------
    dist_a : array-like, shape=[n_vertices_a, n_vertices_a]
        Geodesic distance matrix on shape A (normalised or raw).
    p2p21 : array-like, shape=[n_vertices_b]
        Predicted p2p map: for each vertex in B, its match in A.
    corr_a : array-like, shape=[n_corr]
        GT correspondence indices into shape A.
    corr_b : array-like, shape=[n_corr]
        GT correspondence indices into shape B.
    mask_a : array-like, shape=[n_vertices_a]
        Binary mask: 1 if vertex in A is in the overlap region.

    Returns
    -------
    error : float
        Mean geodesic error over valid (overlap-masked) correspondence pairs.
    """
    mask_a = np.asarray(mask_a)
    corr_a = np.asarray(corr_a)
    corr_b = np.asarray(corr_b)
    p2p21 = np.asarray(p2p21)

    valid = mask_a[corr_a] > 0.5
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(dist_a[corr_a[valid], p2p21[corr_b[valid]]]))


def overlap_iou(overlap_ab, mask_a, threshold=0.5):
    """Intersection-over-Union between predicted and GT overlap masks.

    Parameters
    ----------
    overlap_ab : array-like, shape=[n_vertices_a]
        Predicted overlap scores in [0, 1].
    mask_a : array-like, shape=[n_vertices_a]
        Ground-truth binary overlap mask.
    threshold : float
        Binarisation threshold for predicted scores.  Default 0.5.

    Returns
    -------
    iou : float
        IoU score in [0, 1].
    """
    pred = np.asarray(overlap_ab) >= threshold
    gt = np.asarray(mask_a) >= 0.5
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return 1.0 if union == 0 else float(intersection / union)


def pck_auc(dist_a, p2p21, corr_a, corr_b, mask_a, t_max=0.20, n_steps=100):
    """Area under the PCK (Percentage of Correct Keypoints) curve.

    Parameters
    ----------
    dist_a : array-like, shape=[n_vertices_a, n_vertices_a]
        Geodesic distance matrix on shape A.
    p2p21 : array-like, shape=[n_vertices_b]
        Predicted p2p map.
    corr_a : array-like, shape=[n_corr]
        GT correspondence indices into A.
    corr_b : array-like, shape=[n_corr]
        GT correspondence indices into B.
    mask_a : array-like, shape=[n_vertices_a]
        Binary overlap mask.
    t_max : float
        Maximum normalised geodesic threshold.  Default 0.20.
    n_steps : int
        Number of threshold steps.  Default 100.

    Returns
    -------
    auc : float
        AUC of the PCK curve in [0, 1].
    """
    mask_a = np.asarray(mask_a)
    corr_a = np.asarray(corr_a)
    corr_b = np.asarray(corr_b)
    p2p21 = np.asarray(p2p21)

    valid = mask_a[corr_a] > 0.5
    if valid.sum() == 0:
        return 0.0

    geo_err = dist_a[corr_a[valid], p2p21[corr_b[valid]]]
    diam = dist_a.max()
    geo_err_norm = geo_err / diam if diam > 0 else geo_err

    thresholds = np.linspace(0.0, t_max, n_steps)
    pck_values = np.array([(geo_err_norm <= t).mean() for t in thresholds])
    return float(np.trapezoid(pck_values, thresholds) / t_max)


def coverage_count(shape_a, shape_b, p2p21):
    """Compute count-based coverage of a correspondence.

    This is a simpler coverage metric that just counts the fraction
    of unique target vertices that are mapped to.

    Parameters
    ----------
    shape_a : TriangleMesh or PointCloud
        Target shape (where correspondences land).
    shape_b : TriangleMesh or PointCloud
        Source shape (where correspondences originate).
    p2p21 : array-like, shape=[n_vertices_b]
        Point-to-point map from shape_b to shape_a.

    Returns
    -------
    coverage : float
        Coverage ratio in [0, 1]. The fraction of target vertices
        that are mapped to at least once.
    """
    p2p_np = gs.to_numpy(p2p21)
    unique_targets = np.unique(p2p_np)

    return len(unique_targets) / shape_a.n_vertices


def evaluate_correspondence(
    shape_a,
    shape_b,
    p2p21,
    corr_a=None,
    corr_b=None,
    dist_a=None,
    metrics=None,
):
    """Compute all evaluation metrics for a correspondence.

    This is a convenience function that computes all available metrics
    and returns them in a dictionary.

    Parameters
    ----------
    shape_a : TriangleMesh or PointCloud
        Target shape (where correspondences land).
    shape_b : TriangleMesh or PointCloud
        Source shape (where correspondences originate).
    p2p21 : array-like, shape=[n_vertices_b]
        Point-to-point map from shape_b to shape_a.
    corr_a : array-like, shape=[n_correspondences], optional
        Indices of ground truth correspondences on target shape (A).
    corr_b : array-like, shape=[n_correspondences], optional
        Indices of ground truth correspondences on source shape (B).
    dist_a : array-like, shape=[n_vertices_a, n_vertices_a], optional
        Geodesic distance matrix on shape A. If None and geodesic_error
        is requested, it will be computed from shape_a.
    metrics : list[str], optional
        Subset of metrics to compute. If None, computes all available metrics.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'geodesic_error': Normalized geodesic error (if dist_a provided)
        - 'euclidean_error': Normalized Euclidean error (if corr available)
        - 'dirichlet_energy': Dirichlet energy of the mapping
        - 'coverage': Area-weighted coverage
        - 'coverage_count': Count-based coverage
    """
    requested_metrics = set(metrics) if metrics is not None else None
    output_metrics = {}

    # Geodesic error requires distance matrix
    if (
        requested_metrics is None or "geodesic_error" in requested_metrics
    ) and dist_a is not None:
        output_metrics["geodesic_error"] = float(
            normalized_geodesic_error(dist_a, p2p21, corr_a, corr_b)
        )

    # Euclidean error requires ground truth correspondences
    if (
        (requested_metrics is None or "euclidean_error" in requested_metrics)
        and corr_a is not None
        and corr_b is not None
    ):
        output_metrics["euclidean_error"] = float(
            normalized_euclidean_error(shape_a, p2p21, corr_a, corr_b)
        )

    # Metrics that don't require ground truth
    if requested_metrics is None or "dirichlet_energy" in requested_metrics:
        output_metrics["dirichlet_energy"] = float(
            dirichlet_energy(shape_a, shape_b, p2p21)
        )
    if requested_metrics is None or "coverage" in requested_metrics:
        output_metrics["coverage"] = float(coverage(shape_a, shape_b, p2p21))
    if requested_metrics is None or "coverage_count" in requested_metrics:
        output_metrics["coverage_count"] = float(
            coverage_count(shape_a, shape_b, p2p21)
        )

    return output_metrics
