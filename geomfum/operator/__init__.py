"""Spectral and differential operators on shapes."""

from .base import (
    FaceDivergenceOperator,
    FaceOrientationOperator,
    FaceValuedGradient,
    FunctionalOperator,
    Gradient,
    Laplacian,
    VectorFieldOperator,
)
from .connection import (
    ConnectionLaplacian,
    ConnectionLaplacianFinder,
    ConnectionSpectrumFinder,
)
from .elastic import (
    ElasticShellHessian,
    ElasticShellHessianFinder,
    ElasticSpectrumFinder,
)
