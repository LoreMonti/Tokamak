"""Validazione dell'ottimizzazione bayesiana."""

from __future__ import annotations

import numpy as np

from tokamak.bayesopt import (
    bayesian_optimize,
    expected_improvement,
    make_operating_point_objective,
)
from tokamak.engineering import TokamakConfig
from tokamak.optimization import optimize_operating_point


def test_expected_improvement_non_negative():
    mu = np.array([1.0, 2.0, 0.5])
    sigma = np.array([0.1, 0.5, 0.0])
    ei = expected_improvement(mu, sigma, best=1.0)
    assert np.all(ei >= 0.0)


def test_bo_finds_maximum_of_simple_function():
    """BO massimizza -(x-2)^2 su [0,5]: deve avvicinarsi a x=2, valore ~0."""
    res = bayesian_optimize(
        lambda x: -((x[0] - 2.0) ** 2), bounds=[(0.0, 5.0)],
        n_init=4, n_iter=20, seed=1,
    )
    assert abs(res.best_x[0] - 2.0) < 0.15
    assert res.best_y > -0.05


def test_best_history_is_monotonic():
    """Il miglior valore trovato non puo' che migliorare nel tempo."""
    res = bayesian_optimize(
        lambda x: -((x[0] - 2.0) ** 2), bounds=[(0.0, 5.0)],
        n_init=4, n_iter=15, seed=2,
    )
    assert np.all(np.diff(res.best_history) >= 0.0)


def test_bo_matches_slsqp_optimum():
    """BO deve trovare circa lo stesso P_fus dell'ottimizzatore SLSQP (Fase 8)."""
    cfg = TokamakConfig()
    objective, bounds = make_operating_point_objective(cfg)
    res = bayesian_optimize(objective, bounds, n_init=6, n_iter=30, seed=3)

    slsqp = optimize_operating_point(cfg)
    p_slsqp = slsqp.fusion_power_density / 1e6  # MW/m^3
    # BO entro il 5% dell'ottimo SLSQP.
    assert res.best_y > 0.95 * p_slsqp
