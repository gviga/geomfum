"""Models for learning features for functional maps.

References
----------
.. "Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence" by Nicolas Donati, Abhishek Sharma, Maks Ovsjanikov.
.. "Deep Functional Maps: Structured Prediction for Dense Shape Correspondence" by O. Litany, T. Remez, E. Rodola, A. Bronstein, M. Bronstein.
.. "EchoMatch: Partial-to-Partial Shape Matching via Correspondence Reflection" by Xie et al., CVPR 2025.
"""

from .echomatch import EchoMatchNet, EchoScorer, OverlapRefiner
from .fmnet import FMNet
from .robust import RobustFMNet
