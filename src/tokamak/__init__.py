"""Tokamak: simulatore di trasporto e bilancio di potenza di un plasma da fusione."""

from __future__ import annotations

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
from .reactivity import reactivity_dt
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
]
