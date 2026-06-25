r"""Validazione del solver di Grad-Shafranov.

Test chiave: il solver ellittico deve riprodurre una soluzione ANALITICA nota.
Usiamo un polinomio (tipo Solov'ev) di cui conosciamo esattamente Delta*:

    psi_ex = a R^4 + b Z^2 + c R^2 Z^2
    Delta* psi_ex = (8a + 2c) R^2 + 2b

Imponendo quel termine di destra e psi_ex come condizione al bordo, il solver
deve ricostruire psi_ex all'interno (entro l'errore di discretizzazione).
"""

from __future__ import annotations

import numpy as np

from tokamak.equilibrium import GradShafranovSolver


def test_delta_star_recovers_analytic_polynomial():
    solver = GradShafranovSolver(
        R_min=4.0, R_max=8.0, Z_min=-3.0, Z_max=3.0, nR=81, nZ=121
    )
    a, b, c = 0.3, -1.5, 0.2
    psi_exact = a * solver.RR**4 + b * solver.ZZ**2 + c * solver.RR**2 * solver.ZZ**2
    source = (8.0 * a + 2.0 * c) * solver.RR**2 + 2.0 * b

    psi = solver.solve_linear(source, boundary=psi_exact)

    interior = (slice(1, -1), slice(1, -1))
    err = np.max(np.abs(psi[interior] - psi_exact[interior]))
    scale = np.max(np.abs(psi_exact))
    assert err / scale < 1e-3


def test_picard_converges_and_produces_peaked_psi():
    """L'iterazione non lineare converge e da' un psi piccato (asse magnetico)."""
    solver = GradShafranovSolver(
        R_min=4.2, R_max=8.2, Z_min=-3.0, Z_max=3.0, nR=65, nZ=97
    )
    R0 = 6.2

    def rhs(psi, RR):
        # Sorgente tipo Solov'ev: parte costante (soluzione non banale) + parte
        # non lineare in psi (rende Picard una vera itearazione).
        psi_n = np.clip(psi / (np.max(psi) + 1e-30), 0.0, None)
        return -120.0 * (0.6 * (RR / R0) ** 2 + 0.4) * (1.0 + 1.8 * psi_n)

    iters = solver.solve_picard(rhs, max_iter=200, tol=1e-7, relax=0.4)
    assert 1 < iters < 200  # iterazione non lineare reale, ma converge

    # psi e' nullo al bordo, con un massimo interno (asse magnetico).
    R_ax, Z_ax, psi_ax = solver.magnetic_axis()
    assert psi_ax > 0.0
    assert solver.R_min < R_ax < solver.R_max
    assert abs(Z_ax) < 1.0  # asse vicino al piano mediano per simmetria


def test_d_shaped_boundary_gives_elongated_plasma():
    """Con un bordo a D (kappa>1), il plasma e' nullo fuori dalla D e la regione
    con psi>0 e' piu' estesa in Z che in R (elongazione)."""
    solver = GradShafranovSolver(
        R_min=3.8, R_max=8.6, Z_min=-3.6, Z_max=3.6, nR=71, nZ=101
    )
    kappa = 1.8
    solver.set_d_shaped_boundary(R0=6.2, a=2.0, kappa=kappa, delta=0.3)

    def rhs(psi, RR):
        psi_n = np.clip(psi / (np.max(psi) + 1e-30), 0.0, None)
        return -120.0 * (0.7 * (RR / 6.2) ** 2 + 0.3) * (1.0 + 1.5 * psi_n)

    solver.solve_picard(rhs, max_iter=200, tol=1e-7, relax=0.4)

    # Fuori dalla D (nodi fissati) psi e' esattamente nullo.
    assert np.allclose(solver.psi[solver._fixed], 0.0, atol=1e-9)

    # La regione di plasma (psi>0) e' elongata verticalmente.
    plasma = solver.psi > 0.05 * solver.psi.max()
    R_extent = np.ptp(solver.RR[plasma])
    Z_extent = np.ptp(solver.ZZ[plasma])
    assert Z_extent / R_extent > 1.3  # riflette kappa > 1


def test_boundary_condition_is_enforced():
    """Con sorgente nulla e bordo nullo, psi deve essere identicamente zero."""
    solver = GradShafranovSolver(
        R_min=4.0, R_max=8.0, Z_min=-3.0, Z_max=3.0, nR=41, nZ=61
    )
    psi = solver.solve_linear(np.zeros((solver.nR, solver.nZ)), boundary=0.0)
    assert np.allclose(psi, 0.0, atol=1e-10)
