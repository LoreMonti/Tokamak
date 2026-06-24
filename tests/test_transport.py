"""Validazione del solver di trasporto 1D.

Due test numerici fondamentali per qualunque solver di diffusione:
1. Riproduce la soluzione ANALITICA stazionaria (sorgente e D costanti).
2. CONSERVA l'energia quando il dominio e' isolato e non ci sono sorgenti.
Piu' un test fisico sul tempo di confinamento emergente.
"""

from __future__ import annotations

import numpy as np

from tokamak.transport import TransportSolver1D


class _ConstantSourceSolver(TransportSolver1D):
    """Solver con sorgente uniforme imposta, per il confronto analitico."""

    S0: float = 1e5  # W/m^3 costante

    def source_density(self, T, p_ext=0.0):  # type: ignore[override]
        return np.full_like(self.r, self.S0)


def test_steady_state_matches_analytic_parabola():
    r"""Con S e D=n*chi costanti, lo stazionario e' parabolico:

        T(r) = T_edge + S0 (a^2 - r^2) / (4 D),   D = n*chi

    Deriva da (1/r) d/dr(r D dT/dr) = -S0 con regolarita' in r=0.
    """
    solver = _ConstantSourceSolver(
        a=1.0, R0=3.0, n_e=1e20, chi=1.0, T_edge=0.5, n_cells=200
    )
    solver.solve_steady_state(dt=1e-3, tol=1e-10)

    D = solver.n_e * solver.chi
    # S0 in keV: la PDE e' risolta in keV, quindi convertiamo come nel solver.
    from tokamak.constants import KEV_TO_JOULE

    s0_keV = solver.S0 / KEV_TO_JOULE
    T_analytic = solver.T_edge + s0_keV * (solver.a**2 - solver.r**2) / (4.0 * D)

    # Accordo entro l'1% (errore di discretizzazione).
    rel_err = np.max(np.abs(solver.T - T_analytic) / T_analytic)
    assert rel_err < 1e-2


def test_energy_is_conserved_when_insulated_and_no_sources():
    """Dominio isolato (Neumann) e nessuna sorgente: l'energia totale e' costante.

    La diffusione redistribuisce il calore ma non lo crea ne' distrugge.
    """

    class _NoSource(TransportSolver1D):
        def source_density(self, T, p_ext=0.0):  # type: ignore[override]
            return np.zeros_like(self.r)

    solver = _NoSource(
        a=1.0, R0=3.0, n_e=1e20, chi=2.0, T_edge=0.0, n_cells=100, insulated_edge=True
    )
    # Profilo iniziale piccato al centro.
    solver.T = 1.0 + 10.0 * np.exp(-((solver.r / 0.2) ** 2))

    e0 = solver.stored_energy()
    for _ in range(500):
        solver.step(dt=1e-3)
    e1 = solver.stored_energy()

    assert abs(e1 - e0) / e0 < 1e-6


def test_profile_is_peaked_and_emergent_tau_E_positive():
    """Con riscaldamento al centro, T(r) e' piccata e tau_E e' finito e positivo."""
    solver = TransportSolver1D(
        a=1.0, R0=3.0, n_e=1e20, chi=1.0, T_edge=0.1, n_cells=120
    )
    # Riscaldamento esterno concentrato al centro (gaussiana), in W/m^3.
    p_ext = 5e5 * np.exp(-((solver.r / 0.3) ** 2))
    solver.solve_steady_state(p_ext=p_ext, dt=1e-3)

    # Profilo decrescente dal centro al bordo.
    assert solver.T[0] > solver.T[-1]
    tau = solver.tau_E(p_ext=p_ext)
    assert 0.0 < tau < 100.0
