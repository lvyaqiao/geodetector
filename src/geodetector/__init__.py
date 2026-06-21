"""GeoDetector — Spatial Stratified Heterogeneity Analysis Toolkit.

Main APIs
---------
- ``GeoDetector`` — master class, runs all four detectors at once
- ``FactorDetector`` — single-factor q-statistic (sklearn-compatible)
- ``InteractionDetector`` — pairwise factor interactions
- ``RiskDetector`` — pairwise stratum t-tests
- ``EcologicalDetector`` — factor comparison via F-test
- ``Discretizer``, ``OptimalDiscretizer`` — sklearn-compatible transformers
- ``discretize()`` — convert continuous variables into strata
- ``q_statistic`` — core q-value computation (for advanced use)
-

Extensions
----------
- ``OPGD`` — optimal parameter geographical detector
- ``GOZH`` — geographically optimal zones-based heterogeneity
- ``RGD`` — robust geographical detector
- ``LESH`` — locally explained stratified heterogeneity (Shapley)
"""

from . import dataset, extensions
from ._stats import q_statistic
from .detectors import (
    EcologicalDetector,
    FactorDetector,
    InteractionDetector,
    RiskDetector,
)
from .discretize import Discretizer, OptimalDiscretizer, discretize, should_discretize
from .geodetector import GeoDetector

__all__ = [
    # Main orchestrator
    "GeoDetector",
    # Detectors
    "FactorDetector",
    "InteractionDetector",
    "RiskDetector",
    "EcologicalDetector",
    # Discretization
    "discretize",
    "Discretizer",
    "OptimalDiscretizer",
    "should_discretize",
    # Core stats (advanced)
    "q_statistic",
    # Sub-modules
    "dataset",
    "extensions",
]
__version__ = "0.2.0"
