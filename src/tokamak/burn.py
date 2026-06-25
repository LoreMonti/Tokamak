r"""Fase 6 — Combustione D-T auto-consistente (0D, dipendente dal tempo).

Estende il bilancio 0D rendendo DINAMICHE le densita': il combustibile D-T si
consuma, l'elio (cenere) si accumula, e l'energia evolve. Le tre quantita' sono
accoppiate dalla quasi-neutralita' e dalla dipendenza della reattivita' da T.

Stato del sistema (per metro cubo):
    n_DT  densita' di combustibile (n_D + n_T, con n_D = n_T = n_DT/2)
    n_He  densita' di cenere di elio (He-4)
    U     densita' di energia termica = (3/2)(n_e + n_i) T   [J/m^3]

Equazioni:
    dn_DT/dt = S_fuel - 2 R                 (ogni fusione toglie un D e un T)
    dn_He/dt = R - n_He / tau_p             (cenere prodotta e rimossa)
    dU/dt    = P_alpha + P_ext - P_brem - U/tau_E

con R = (n_DT/2)^2 <sigma v>(T) il tasso di reazioni.

Accoppiamenti fisici chiave:
- quasi-neutralita': n_e = n_DT + 2 n_He (l'elio ha carica +2);
- Z_eff cresce con la cenere -> piu' radiazione di Bremsstrahlung;
- la cenere diluisce il combustibile a densita' elettronica fissata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .constants import E_ALPHA_JOULE, E_FUSION_DT_JOULE, KEV_TO_JOULE
from .power_balance import C_BREM
from .reactivity import reactivity_dt


def _temperature_keV(n_e: float, n_i: float, U: float) -> float:
    """Temperatura [keV] dall'energia U = (3/2)(n_e+n_i) T."""
    denom = 1.5 * (n_e + n_i) * KEV_TO_JOULE
    return U / denom if denom > 0 else 0.0


def z_effective(n_DT: float, n_He: float, n_e: float) -> float:
    """Carica efficace Z_eff = sum(n_j Z_j^2) / n_e (D,T con Z=1; He con Z=2)."""
    return (n_DT * 1.0 + n_He * 4.0) / n_e if n_e > 0 else 1.0


@dataclass
class BurnState:
    """Serie temporali della combustione."""

    time: NDArray[np.float64]
    n_DT: NDArray[np.float64]
    n_He: NDArray[np.float64]
    T_keV: NDArray[np.float64]
    P_alpha: NDArray[np.float64]
    P_ext: NDArray[np.float64]


def simulate_burn(
    *,
    n_DT0: float = 1.0e20,
    n_He0: float = 0.0,
    T0_keV: float = 5.0,
    tau_E: float = 2.0,
    tau_p: float = 6.0,
    fueling: float = 0.0,
    p_ext: float = 0.0,
    p_ext_off_time: float | None = None,
    t_end: float = 30.0,
    n_points: int = 600,
) -> BurnState:
    """Integra nel tempo la combustione 0D.

    Parameters
    ----------
    tau_E, tau_p:
        tempi di confinamento di energia e particelle [s].
    fueling:
        sorgente di combustibile S_fuel [m^-3 s^-1].
    p_ext:
        riscaldamento esterno [W/m^3]; spento dopo p_ext_off_time (se dato),
        per mostrare se la combustione si AUTO-sostiene (ignition).
    """
    def p_ext_of_t(t: float) -> float:
        if p_ext_off_time is not None and t >= p_ext_off_time:
            return 0.0
        return p_ext

    def rhs(t: float, y: NDArray[np.float64]) -> list[float]:
        n_DT, n_He, U = y
        n_DT = max(n_DT, 0.0)
        n_He = max(n_He, 0.0)
        n_e = n_DT + 2.0 * n_He
        n_i = n_DT + n_He
        T = _temperature_keV(n_e, n_i, U)
        T = max(T, 1e-3)

        sigv = float(reactivity_dt(T))
        R = (0.5 * n_DT) ** 2 * sigv  # reazioni / m^3 / s

        p_alpha = R * E_ALPHA_JOULE
        z_eff = z_effective(n_DT, n_He, n_e)
        p_brem = C_BREM * z_eff * n_e**2 * np.sqrt(T)
        p_loss = U / tau_E
        p_e = p_ext_of_t(t)

        dn_DT = fueling - 2.0 * R
        dn_He = R - n_He / tau_p
        dU = p_alpha + p_e - p_brem - p_loss
        return [dn_DT, dn_He, dU]

    n_e0 = n_DT0 + 2.0 * n_He0
    n_i0 = n_DT0 + n_He0
    U0 = 1.5 * (n_e0 + n_i0) * T0_keV * KEV_TO_JOULE

    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        rhs, (0.0, t_end), [n_DT0, n_He0, U0], t_eval=t_eval, method="LSODA",
        rtol=1e-7, atol=1e-3,
    )

    # Ricostruisce T e le potenze dalle serie integrate.
    n_DT = np.clip(sol.y[0], 0.0, None)
    n_He = np.clip(sol.y[1], 0.0, None)
    U = sol.y[2]
    n_e = n_DT + 2.0 * n_He
    n_i = n_DT + n_He
    T = U / (1.5 * (n_e + n_i) * KEV_TO_JOULE)
    R = (0.5 * n_DT) ** 2 * reactivity_dt(np.clip(T, 1e-3, None))
    p_alpha = R * E_ALPHA_JOULE
    p_ext_series = np.array([p_ext_of_t(t) for t in sol.t])

    return BurnState(
        time=sol.t, n_DT=n_DT, n_He=n_He, T_keV=T, P_alpha=p_alpha, P_ext=p_ext_series
    )


def fusion_power_density_burn(state: BurnState) -> NDArray[np.float64]:
    """Densita' di potenza di fusione totale [W/m^3] lungo l'evoluzione."""
    return state.P_alpha * (E_FUSION_DT_JOULE / E_ALPHA_JOULE)
