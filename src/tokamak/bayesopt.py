r"""Fase 14 — Ottimizzazione bayesiana del punto operativo.

Quando l'obiettivo e' una "scatola nera" costosa, l'ottimizzazione bayesiana (BO)
trova l'ottimo con pochissime valutazioni:

1. modella la funzione con un processo gaussiano (GP) dai punti gia' valutati;
2. sceglie il prossimo punto massimizzando l'Expected Improvement (EI), che
   bilancia sfruttamento (dove il GP predice valori alti) ed esplorazione (dove
   e' incerto);
3. valuta, aggiorna il GP, ripete.

La applichiamo allo STESSO problema della Fase 8 (massimizzare P_fus sotto i
vincoli di Greenwald e Troyon), trattandolo come scatola nera, per mostrare che
BO converge all'ottimo con poche valutazioni. Implementata da zero col GP di
scikit-learn; internamente si lavora in coordinate normalizzate [0,1]^d.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .engineering import (
    greenwald_density,
    plasma_beta,
    troyon_beta_limit,
)
from .power_balance import fusion_power_density


def expected_improvement(
    mu: NDArray[np.float64], sigma: NDArray[np.float64], best: float, xi: float = 0.01
) -> NDArray[np.float64]:
    """Expected Improvement per la massimizzazione.

    EI(x) = (mu - best - xi) Phi(z) + sigma phi(z),  z = (mu - best - xi)/sigma.
    xi favorisce un po' l'esplorazione. EI = 0 dove l'incertezza e' nulla.
    """
    sigma = np.maximum(sigma, 1e-12)
    improvement = mu - best - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


@dataclass
class BOResult:
    """Esito dell'ottimizzazione bayesiana."""

    best_x: NDArray[np.float64]
    best_y: float
    best_history: NDArray[np.float64]  # miglior valore trovato vs iterazione
    X: NDArray[np.float64]  # tutti i punti valutati (coordinate reali)
    y: NDArray[np.float64]


def bayesian_optimize(
    objective: Callable[[NDArray[np.float64]], float],
    bounds: list[tuple[float, float]],
    *,
    n_init: int = 5,
    n_iter: int = 25,
    n_candidates: int = 2000,
    seed: int = 0,
) -> BOResult:
    """Massimizza `objective` su `bounds` con ottimizzazione bayesiana (EI)."""
    bounds_arr = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
    d = len(bounds)
    rng = np.random.default_rng(seed)

    def to_real(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return lo + u * (hi - lo)

    # Design iniziale casuale in [0,1]^d.
    U = rng.random((n_init, d))
    y = np.array([objective(to_real(u)) for u in U])

    # Matern (nu=2.5): adatto a funzioni meno lisce (l'obiettivo e' discontinuo
    # al bordo dei vincoli). Length-scale limitati per evitare collassi del fit.
    kernel = ConstantKernel(1.0) * Matern(
        length_scale=[0.2] * d, length_scale_bounds=(0.02, 2.0), nu=2.5
    ) + WhiteKernel(1e-4, noise_level_bounds=(1e-7, 1e-1))

    for _ in range(n_iter):
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0
        )
        # I warning di convergenza degli iperparametri sono attesi (obiettivo
        # discontinuo al bordo dei vincoli) e non actionable: li silenziamo.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(U, y)

        # Massimizza EI su un insieme denso di candidati casuali.
        cand = rng.random((n_candidates, d))
        mu, sigma = gp.predict(cand, return_std=True)
        ei = expected_improvement(mu, sigma, best=float(y.max()))
        u_next = cand[int(np.argmax(ei))]

        y_next = objective(to_real(u_next))
        U = np.vstack([U, u_next])
        y = np.append(y, y_next)

    best_history = np.maximum.accumulate(y[n_init:]) if n_iter > 0 else y.copy()
    i_best = int(np.argmax(y))
    return BOResult(
        best_x=to_real(U[i_best]),
        best_y=float(y[i_best]),
        best_history=best_history,
        X=np.array([to_real(u) for u in U]),
        y=y,
    )


def make_operating_point_objective(
    config,
    *,
    T_min: float = 5.0,
    T_max: float = 40.0,
) -> tuple[Callable[[NDArray[np.float64]], float], list[tuple[float, float]]]:
    """Obiettivo "scatola nera": densita' di potenza di fusione [MW/m^3] sotto i
    vincoli di Greenwald e Troyon (penalizzata a 0 se infeasibile).

    Restituisce (objective, bounds) con x = [n_e in 1e20, T in keV].
    """
    n_G = greenwald_density(config.plasma_current_MA, config.minor_radius_m)
    beta_max = troyon_beta_limit(
        config.beta_N, config.plasma_current_MA, config.minor_radius_m,
        config.B_toroidal_T,
    )
    B = config.B_toroidal_T

    def objective(x: NDArray[np.float64]) -> float:
        n_e, T = x[0] * 1e20, x[1]
        feasible = (n_e <= n_G) and (plasma_beta(n_e, T, B) <= beta_max)
        if not feasible:
            return 0.0
        return float(fusion_power_density(n_e, T)) / 1e6

    bounds = [(0.1, 1.1 * n_G / 1e20), (T_min, T_max)]
    return objective, bounds
