r"""Fase 8 — Ottimizzazione del punto operativo sotto vincoli.

Cerca la coppia (densita', temperatura) che MASSIMIZZA la densita' di potenza di
fusione rispettando i limiti ingegneristici (Greenwald, Troyon). Lega insieme la
fisica (fusione) e l'ingegneria (vincoli) in un unico risultato.

    max_{n_e, T}  P_fus(n_e, T)
    soggetto a:   n_e <= n_Greenwald            (densita')
                  beta(n_e, T) <= beta_Troyon   (pressione)

Poiche' P_fus ~ n^2 <sigma v>(T) cresce con densita' e temperatura, l'ottimo
giace sul bordo dei vincoli: tipicamente all'incrocio Greenwald-Troyon.
Risolviamo con SLSQP (gestisce vincoli di disuguaglianza).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .engineering import (
    TokamakConfig,
    greenwald_density,
    greenwald_fraction,
    plasma_beta,
    troyon_beta_limit,
)
from .power_balance import fusion_gain_Q, fusion_power_density


@dataclass
class OptimalPoint:
    """Punto operativo ottimo trovato e relative diagnostiche."""

    n_e: float
    T_keV: float
    fusion_power_density: float
    Q: float
    greenwald_fraction: float
    beta: float
    beta_limit: float
    success: bool

    @property
    def greenwald_active(self) -> bool:
        """True se il limite di Greenwald e' (quasi) saturato all'ottimo."""
        return self.greenwald_fraction > 0.98

    @property
    def troyon_active(self) -> bool:
        """True se il limite di Troyon e' (quasi) saturato all'ottimo."""
        return self.beta > 0.98 * self.beta_limit


def optimize_operating_point(
    config: TokamakConfig,
    *,
    tau_E: float = 3.0,
    T_min: float = 5.0,
    T_max: float = 40.0,
) -> OptimalPoint:
    """Massimizza la densita' di potenza di fusione sotto Greenwald e Troyon.

    tau_E serve solo per riportare Q all'ottimo (non vincola l'ottimizzazione,
    che e' sulla densita' di potenza di fusione).
    """
    n_G = greenwald_density(config.plasma_current_MA, config.minor_radius_m)
    beta_max = troyon_beta_limit(
        config.beta_N,
        config.plasma_current_MA,
        config.minor_radius_m,
        config.B_toroidal_T,
    )
    B = config.B_toroidal_T

    # Variabili scalate per condizionamento: x = [n in 1e20 m^-3, T in keV].
    # Anche l'obiettivo e' scalato (P_fus ~ 1e6 W/m^3) per gradienti ben condizionati.
    def neg_pfus(x: np.ndarray) -> float:
        n_e, T = x[0] * 1e20, x[1]
        return -float(fusion_power_density(n_e, T)) / 1e6

    constraints = [
        # Greenwald: n_G - n_e >= 0 (in unita' 1e20).
        {"type": "ineq", "fun": lambda x: n_G / 1e20 - x[0]},
        # Troyon: beta_max - beta(n,T) >= 0.
        {"type": "ineq", "fun": lambda x: beta_max - plasma_beta(x[0] * 1e20, x[1], B)},
    ]
    bounds = [(0.05, n_G / 1e20), (T_min, T_max)]

    # Punto iniziale AMMISSIBILE: scegliamo n e T cosi' che beta sia ben sotto
    # il limite (altrimenti SLSQP puo' partire da una regione non ammissibile).
    T0 = 0.5 * (T_min + T_max)
    n0_beta = beta_max * B**2 / (2.0 * 4e-7 * np.pi) / (2.0 * T0 * 1.602e-16)  # n a beta=beta_max
    n0 = min(0.5 * n_G, 0.5 * n0_beta) / 1e20
    x0 = np.array([max(n0, 0.05), T0])

    res = minimize(
        neg_pfus, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )

    n_opt, T_opt = res.x[0] * 1e20, res.x[1]
    return OptimalPoint(
        n_e=n_opt,
        T_keV=T_opt,
        fusion_power_density=float(fusion_power_density(n_opt, T_opt)),
        Q=float(fusion_gain_Q(n_opt, T_opt, tau_E)),
        greenwald_fraction=greenwald_fraction(
            n_opt, config.plasma_current_MA, config.minor_radius_m
        ),
        beta=plasma_beta(n_opt, T_opt, B),
        beta_limit=beta_max,
        success=bool(res.success),
    )
