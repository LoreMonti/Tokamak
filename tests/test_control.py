"""Validazione del regolatore PID e della simulazione controllata."""

from __future__ import annotations

import numpy as np

from tokamak.control import PIDController, simulate_controlled
from tokamak.transport import TransportSolver1D


def test_pid_drives_first_order_plant_to_setpoint():
    """Il PID porta a regime un impianto del primo ordine al setpoint.

    Impianto giocattolo: dy/dt = -leak*y + gain*u (es. serbatoio con perdita).
    A regime l'errore deve annullarsi grazie al termine integrale.
    """
    pid = PIDController(kp=2.0, ki=5.0, kd=0.1, setpoint=1.0, output_max=100.0)
    y, dt, leak, gain = 0.0, 0.01, 1.0, 1.0
    for _ in range(20000):
        u = pid.update(y, dt)
        y += (-leak * y + gain * u) * dt
    assert abs(y - pid.setpoint) < 1e-3


def test_pid_output_respects_saturation():
    """L'uscita resta nei limiti [output_min, output_max]."""
    pid = PIDController(kp=1e9, ki=0.0, kd=0.0, setpoint=10.0, output_min=0.0, output_max=5.0)
    assert pid.update(0.0, 0.1) == 5.0  # errore enorme -> saturazione superiore
    pid2 = PIDController(kp=1e9, ki=0.0, kd=0.0, setpoint=0.0, output_min=2.0, output_max=9.0)
    assert pid2.update(100.0, 0.1) == 2.0  # errore negativo -> saturazione inferiore


def test_pid_anti_windup_limits_integral():
    """In saturazione prolungata l'integrale non deve caricarsi senza limite."""
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0, setpoint=1.0, output_max=0.5)
    for _ in range(1000):
        pid.update(0.0, 0.1)  # errore costante positivo, uscita satura a 0.5
    # Senza anti-windup l'integrale sarebbe ~100; con anti-windup resta contenuto.
    assert pid._integral < 1.0


def _make_solver() -> TransportSolver1D:
    s = TransportSolver1D(a=2.0, R0=6.2, n_e=1.0e20, chi=0.5, T_edge=0.1, n_cells=80)
    s.T = s.T_edge + 0.5 * (1.0 - (s.r / s.a) ** 2)
    return s


def test_controlled_simulation_reaches_setpoint():
    """La simulazione controllata porta T0 vicino al target a regime."""
    solver = _make_solver()
    shape = np.exp(-((solver.r / 0.6) ** 2))
    pid = PIDController(kp=8e6, ki=6e6, kd=5e5, setpoint=9.0, output_max=1.5e8)
    hist = simulate_controlled(solver, pid, shape, dt=5e-3, n_steps=1600)
    assert abs(hist.measurement[-1] - 9.0) < 0.2


def test_controller_rejects_confinement_disturbance():
    """Dopo un degrado di confinamento (chi x2) il PID aumenta la potenza
    e riporta T0 al target."""
    solver = _make_solver()
    shape = np.exp(-((solver.r / 0.6) ** 2))
    pid = PIDController(kp=8e6, ki=6e6, kd=5e5, setpoint=9.0, output_max=1.5e8)
    n = 2400
    hist = simulate_controlled(
        solver, pid, shape, dt=5e-3, n_steps=n, chi_disturbance=(n // 2, 1.0)
    )
    p_before = hist.power[n // 2 - 1]
    p_after = hist.power[-1]
    assert p_after > p_before  # piu' potenza per compensare le perdite maggiori
    assert abs(hist.measurement[-1] - 9.0) < 0.3  # target recuperato
