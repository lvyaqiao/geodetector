"""Type definitions for geodetector results."""

from dataclasses import dataclass


@dataclass
class FactorResult:
    """Single factor detector result."""
    variable: str
    q_value: float
    p_value: float
    significant: bool
    n_strata: int


@dataclass
class InteractionPair:
    """Single interaction pair result."""
    factor_1: str
    factor_2: str
    q_1: float
    q_2: float
    q_12: float
    interaction_type: int
    interaction_label: str


# Interaction type labels matching GD package convention.
INTERACTION_TYPES = {
    0: "Weaken, nonlinear",
    1: "Weaken, uni-variable",
    2: "Enhance, bi-variable",
    3: "Independent",
    4: "Enhance, nonlinear",
}
