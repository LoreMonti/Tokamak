"""Validazione del controllo di stabilita' verticale."""

from __future__ import annotations

import numpy as np

from tokamak.control import PIDController
from tokamak.stability import (
    closed_loop_is_stable,
    simulate_vertical_control,
    vertical_growth_rate,
)


def test_circular_plasma_is_stable():
    """Un plasma circolare (kappa=1) non e' verticalmente instabile."""
    assert vertical_growth_rate(1.0) == 0.0


def test_growth_rate_increases_with_elongation():
    """Piu' elongazione -> instabilita' piu' rapida."""
    assert vertical_growth_rate(1.8) > vertical_growth_rate(1.4) > 0.0


def test_open_loop_runs_away():
    """Senza controllo, un plasma instabile si allontana esponenzialmente."""
    h = simulate_vertical_control(
        gamma=200.0, b=1.0, controller=None, z0=0.01, dt=1e-4, n_steps=2000
    )
    assert abs(h.z[-1]) > 10.0 * abs(h.z[0])  # cresce di molto


def _pd(kp: float, kd: float, limit: float = np.inf) -> PIDController:
    # PD = PID con ki=0, setpoint z=0, uscita simmetrica.
    return PIDController(kp=kp, ki=0.0, kd=kd, setpoint=0.0,
                         output_min=-limit, output_max=limit)


def test_closed_loop_stabilizes_with_adequate_gains():
    """Con guadagni adeguati (b*kp > gamma^2) il plasma torna a z=0."""
    gamma, b = 200.0, 1.0
    kp, kd = 1.5 * gamma**2, 200.0
    assert closed_loop_is_stable(gamma, b, kp, kd)
    h = simulate_vertical_control(
        gamma=gamma, b=b, controller=_pd(kp, kd), z0=0.01, dt=1e-4, n_steps=4000
    )
    assert abs(h.z[-1]) < 0.05 * abs(h.z[0])  # riportato vicino a zero


def test_insufficient_proportional_gain_fails():
    """Se b*kp < gamma^2 il controllo NON basta: resta instabile."""
    gamma, b = 200.0, 1.0
    kp, kd = 0.5 * gamma**2, 200.0  # troppo debole
    assert not closed_loop_is_stable(gamma, b, kp, kd)
    h = simulate_vertical_control(
        gamma=gamma, b=b, controller=_pd(kp, kd), z0=0.01, dt=1e-4, n_steps=3000
    )
    assert abs(h.z[-1]) > abs(h.z[0])  # diverge comunque


def test_disturbance_is_rejected():
    """Dopo un calcio impulsivo, il controllore riporta z a zero."""
    gamma, b = 200.0, 1.0
    kp, kd = 1.5 * gamma**2, 250.0
    h = simulate_vertical_control(
        gamma=gamma, b=b, controller=_pd(kp, kd), z0=0.0, dt=1e-4, n_steps=5000,
        kicks={1000: 5.0},  # calcio in velocita' a t=0.1 s
    )
    # Dopo il calcio la posizione si allontana, ma alla fine torna ~0.
    assert abs(h.z[-1]) < 1e-3
    assert np.max(np.abs(h.z)) > 1e-3  # il calcio ha avuto effetto
