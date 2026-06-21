"""Extensions for advanced geographical detector analysis.

Modules
-------
- ``_lesh`` : Shapley decomposition (LESH)
- ``_opgd`` : Optimal Parameter Geographical Detector (OPGD)
- ``_gozh`` : Geographically Optimal Zones-based Heterogeneity (GOZH)
- ``_geometric`` : Geometric interval discretization
- ``_rgd`` : Robust Geographical Detector (RGD)
"""

from ._geometric import discretize_geometric, geometric_breaks
from ._gozh import GOZH, rpart_discretize
from ._lesh import LESH, shapley_decompose
from ._opgd import OPGD
from ._rgd import RGD, robust_discretize

__all__ = [
    "LESH",
    "shapley_decompose",
    "OPGD",
    "GOZH",
    "rpart_discretize",
    "geometric_breaks",
    "discretize_geometric",
    "RGD",
    "robust_discretize",
]
