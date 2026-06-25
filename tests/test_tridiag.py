"""Validazione del solutore tridiagonale (kernel C++ vs scipy).

Se il kernel C++ non e' compilato, i test relativi vengono saltati: il pacchetto
resta valido in puro Python.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokamak._tridiag import CPP_AVAILABLE, solve_tridiagonal
from tokamak.transport import TransportSolver1D


def _random_system(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    lower = -rng.random(n)
    upper = -rng.random(n)
    diag = 3.0 + rng.random(n)  # diagonale dominante -> ben condizionato
    rhs = rng.random(n)
    return lower, diag, upper, rhs


def test_scipy_backend_solves_correctly():
    """Verifica diretta: A x = rhs ricostruito dalla soluzione scipy."""
    lower, diag, upper, rhs = _random_system(50)
    x = solve_tridiagonal(lower, diag, upper, rhs, backend="scipy")
    # Ricostruisce A x e confronta con rhs.
    Ax = diag * x
    Ax[1:] += lower[1:] * x[:-1]
    Ax[:-1] += upper[:-1] * x[1:]
    assert np.allclose(Ax, rhs)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="kernel C++ non compilato")
def test_cpp_matches_scipy():
    """Il kernel C++ riproduce scipy a precisione macchina."""
    lower, diag, upper, rhs = _random_system(400, seed=3)
    xs = solve_tridiagonal(lower, diag, upper, rhs, backend="scipy")
    xc = solve_tridiagonal(lower, diag, upper, rhs, backend="cpp")
    assert np.allclose(xs, xc, atol=1e-12)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="kernel C++ non compilato")
def test_transport_backends_agree():
    """Il profilo stazionario e' identico con backend scipy e C++."""
    kw = dict(a=1.0, R0=3.0, n_e=1e20, chi=1.0, T_edge=0.1, n_cells=60)
    s_sci = TransportSolver1D(**kw, backend="scipy")
    s_cpp = TransportSolver1D(**kw, backend="cpp")
    p = 4e5 * np.exp(-((s_sci.r / 0.3) ** 2))
    s_sci.solve_steady_state(p_ext=p, dt=1e-2, tol=1e-6)
    s_cpp.solve_steady_state(p_ext=p, dt=1e-2, tol=1e-6)
    assert np.allclose(s_sci.T, s_cpp.T, rtol=1e-9)


def test_unknown_backend_raises():
    lower, diag, upper, rhs = _random_system(10)
    with pytest.raises(ValueError):
        solve_tridiagonal(lower, diag, upper, rhs, backend="quantum")
