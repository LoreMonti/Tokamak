r"""Fase 9 — Stabilita' verticale del plasma e suo controllo.

I tokamak allungano il plasma (elongazione kappa > 1, forma a "D") perche'
migliora il confinamento. Il prezzo: un plasma allungato e' VERTICALMENTE
INSTABILE, tende a muoversi su/giu' come un pendolo inverso, accelerando
esponenzialmente verso la parete. Serve un controllo attivo in retroazione.

Modello ridotto ("rigid plasma"), per la posizione verticale z del baricentro:

    d2z/dt2 = gamma^2 z + b u

- gamma  = tasso di crescita dell'instabilita' [1/s], cresce con l'elongazione;
- b      = efficacia di controllo (accel. per unita' di comando alle bobine);
- u      = comando di controllo (corrente nelle bobine di campo poloidale).

Con retroazione PD u = -(kp z + kd dz/dt) il sistema in anello chiuso diventa

    z'' + (b kd) z' + (b kp - gamma^2) z = 0

stabile se  b kp > gamma^2  (vince l'instabilita') e  kd > 0 (smorza).

Il controllore PD e' il PID della Fase 4A con ki=0: lo RIUSIAMO qui, con setpoint
z=0. Il sensore misura z; la velocita' e' ricostruita dalla derivata (come nei
sistemi reali).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .control import PIDController


def vertical_growth_rate(elongation: float, gamma_scale: float = 300.0) -> float:
    """Tasso di crescita dell'instabilita' verticale [1/s] (modello semplice).

    Un plasma circolare (kappa=1) e' verticalmente stabile (gamma=0); l'instabilita'
    cresce con l'elongazione. Usiamo gamma = gamma_scale * (kappa - 1), clampato
    a zero. gamma_scale fissa la scala temporale (ms) tipica delle macchine reali.
    """
    return max(0.0, gamma_scale * (elongation - 1.0))


def closed_loop_is_stable(gamma: float, b: float, kp: float, kd: float) -> bool:
    """Criterio di stabilita' dell'anello chiuso: b*kp > gamma^2 e b*kd > 0."""
    return (b * kp > gamma**2) and (b * kd > 0.0)


@dataclass
class VerticalHistory:
    """Serie temporali della posizione verticale e del comando di controllo."""

    time: NDArray[np.float64]
    z: NDArray[np.float64]
    u: NDArray[np.float64]


def simulate_vertical_control(
    *,
    gamma: float,
    b: float,
    controller: PIDController | None,
    z0: float = 0.01,
    zdot0: float = 0.0,
    dt: float = 1e-4,
    n_steps: int = 4000,
    actuator_limit: float = np.inf,
    kicks: dict[int, float] | None = None,
) -> VerticalHistory:
    """Integra la dinamica verticale con (o senza) controllo in retroazione.

    Parameters
    ----------
    controller:
        PIDController (usato come PD, ki=0, setpoint=0). Se None, anello APERTO
        (nessun controllo): il plasma instabile scappa via.
    z0, zdot0:
        spostamento e velocita' iniziali [m, m/s].
    actuator_limit:
        saturazione del comando |u| (un alimentatore reale e' limitato).
    kicks:
        {passo: delta_v} perturbazioni impulsive sulla velocita' (disturbi).

    Integrazione RK4 con comando u mantenuto costante sul passo (zero-order hold).
    """
    kicks = kicks or {}
    time = np.zeros(n_steps)
    z_hist = np.zeros(n_steps)
    u_hist = np.zeros(n_steps)

    def accel(z: float, u: float) -> float:
        return gamma**2 * z + b * u

    z, zd = z0, zdot0
    for k in range(n_steps):
        if controller is not None:
            u = controller.update(z, dt)
        else:
            u = 0.0
        u = float(np.clip(u, -actuator_limit, actuator_limit))

        # RK4 sul sistema [z, zd] con u costante sul passo.
        def deriv(state: tuple[float, float], u: float = u) -> tuple[float, float]:
            zz, zzd = state
            return zzd, accel(zz, u)

        k1 = deriv((z, zd))
        k2 = deriv((z + 0.5 * dt * k1[0], zd + 0.5 * dt * k1[1]))
        k3 = deriv((z + 0.5 * dt * k2[0], zd + 0.5 * dt * k2[1]))
        k4 = deriv((z + dt * k3[0], zd + dt * k3[1]))
        z += dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        zd += dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

        if k in kicks:
            zd += kicks[k]

        time[k] = k * dt
        z_hist[k] = z
        u_hist[k] = u

    return VerticalHistory(time=time, z=z_hist, u=u_hist)
