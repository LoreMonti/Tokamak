"""Tokamak: simulatore di trasporto e bilancio di potenza di un plasma da fusione."""

from __future__ import annotations

from .burn import BurnState, simulate_burn, z_effective
from .control import PIDController, simulate_controlled
from .engineering import (
    TokamakConfig,
    check_operational_limits,
    greenwald_density,
    greenwald_fraction,
    plasma_beta,
    troyon_beta_limit,
)
from .equilibrium import GradShafranovSolver
from .optimization import OptimalPoint, optimize_operating_point
from .power_balance import (
    alpha_power_density,
    bremsstrahlung_power_density,
    fusion_gain_Q,
    fusion_power_density,
    heating_power_required,
    loss_power_density,
    stored_energy_density,
    triple_product_ignition,
)
from .radiation import (
    cooling_function,
    max_impurity_fraction,
    total_radiated_power_density,
    z_eff_from_fractions,
)
from .reactivity import reactivity_dt
from .stability import (
    closed_loop_is_stable,
    simulate_vertical_control,
    vertical_growth_rate,
)
from .transport import TransportSolver1D

__version__ = "0.1.0"

__all__ = [
    "reactivity_dt",
    "fusion_power_density",
    "alpha_power_density",
    "bremsstrahlung_power_density",
    "stored_energy_density",
    "loss_power_density",
    "heating_power_required",
    "fusion_gain_Q",
    "triple_product_ignition",
    "TransportSolver1D",
    "TokamakConfig",
    "check_operational_limits",
    "greenwald_density",
    "greenwald_fraction",
    "plasma_beta",
    "troyon_beta_limit",
    "PIDController",
    "simulate_controlled",
    "GradShafranovSolver",
    "simulate_burn",
    "BurnState",
    "z_effective",
    "cooling_function",
    "z_eff_from_fractions",
    "total_radiated_power_density",
    "max_impurity_fraction",
    "optimize_operating_point",
    "OptimalPoint",
    "vertical_growth_rate",
    "closed_loop_is_stable",
    "simulate_vertical_control",
]
