r"""Controllo in retroazione del plasma: regolatore PID sul riscaldamento.

Un reattore e' un sistema dinamico: la temperatura va mantenuta a un valore
target agendo sull'unico attuatore rapido disponibile, la potenza di
riscaldamento esterno P_ext (fasci di neutri / onde a radiofrequenza).

Il regolatore PID calcola P_ext dall'errore e(t) = T_target - T_misurata:

    P_ext(t) = Kp e(t) + Ki integral(e dt) + Kd de/dt

- Kp (proporzionale): risposta immediata, proporzionale all'errore.
- Ki (integrale):      annulla l'errore residuo a regime (un plasma scaldato
                       solo in proporzione si assesta sotto il target).
- Kd (derivativo):     smorza la dinamica reagendo alla velocita' dell'errore.

Includiamo due accorgimenti pratici di ogni PID reale:
- SATURAZIONE: P_ext e' limitata a [0, P_max] (un riscaldatore non eroga
  potenza negativa ne' infinita).
- ANTI-WINDUP: quando l'uscita e' saturata non accumuliamo l'integrale, per
  evitare che "carichi" e provochi grandi sovraelongazioni.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .transport import TransportSolver1D


@dataclass
class PIDController:
    """Regolatore PID con saturazione dell'uscita e anti-windup."""

    kp: float
    ki: float
    kd: float
    setpoint: float
    output_min: float = 0.0
    output_max: float = np.inf

    _integral: float = field(default=0.0, init=False)
    _prev_error: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def update(self, measurement: float, dt: float) -> float:
        """Calcola l'uscita di controllo dato il valore misurato e il passo dt."""
        error = self.setpoint - measurement
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt

        # Uscita provvisoria col valore corrente dell'integrale.
        candidate_integral = self._integral + error * dt
        output = self.kp * error + self.ki * candidate_integral + self.kd * derivative

        # Saturazione + anti-windup: accumuliamo l'integrale solo se non
        # spinge ulteriormente l'uscita oltre il limite raggiunto.
        if output > self.output_max:
            output = self.output_max
            if error < 0:
                self._integral = candidate_integral
        elif output < self.output_min:
            output = self.output_min
            if error > 0:
                self._integral = candidate_integral
        else:
            self._integral = candidate_integral

        self._prev_error = error
        return output


@dataclass
class ControlHistory:
    """Serie temporali registrate durante una simulazione controllata."""

    time: NDArray[np.float64]
    measurement: NDArray[np.float64]
    power: NDArray[np.float64]
    setpoint: float


def simulate_controlled(
    solver: TransportSolver1D,
    pid: PIDController,
    heating_shape: NDArray[np.float64],
    *,
    dt: float = 5e-3,
    n_steps: int = 2000,
    measure: str = "T0",
    chi_disturbance: tuple[int, float] | None = None,
) -> ControlHistory:
    """Simula l'evoluzione del plasma con riscaldamento regolato dal PID.

    A ogni passo: si misura la grandezza controllata, il PID decide la potenza
    totale, la si distribuisce col profilo `heating_shape` e si avanza il
    trasporto di un passo dt.

    Parameters
    ----------
    measure:
        "T0" (temperatura centrale) o "T_avg" (media sul volume).
    chi_disturbance:
        (passo, nuovo_chi): a quel passo la diffusivita' cambia di colpo,
        simulando un degrado (o miglioramento) del confinamento. Serve a
        mostrare la reiezione del disturbo da parte del controllore.

    Returns
    -------
    ControlHistory con time, grandezza misurata, potenza erogata.
    """
    time = np.zeros(n_steps)
    meas = np.zeros(n_steps)
    power = np.zeros(n_steps)
    dV = solver._volume_element()
    vol = float(np.sum(dV))

    for k in range(n_steps):
        if chi_disturbance is not None and k == chi_disturbance[0]:
            solver.chi = chi_disturbance[1]

        if measure == "T0":
            y = float(solver.T[0])
        elif measure == "T_avg":
            y = float(np.sum(solver.T * dV) / vol)
        else:  # pragma: no cover
            raise ValueError(f"grandezza di misura sconosciuta: {measure}")

        p_total = pid.update(y, dt)
        p_ext = solver.heating_density_for_power(p_total, heating_shape)
        solver.step(dt, p_ext)

        time[k] = k * dt
        meas[k] = y
        power[k] = p_total

    return ControlHistory(time=time, measurement=meas, power=power, setpoint=pid.setpoint)
