"""Demo di controllo: il PID mantiene la temperatura centrale a un target.

Scenario:
1. Il plasma parte freddo; il PID alza il riscaldamento per portare T_0 al target.
2. A meta' simulazione il confinamento PEGGIORA (chi aumenta di colpo: come una
   transizione di modo o un degrado). La temperatura cala...
3. ...e il controllore reagisce aumentando la potenza per recuperare il target.

Mostra la reiezione del disturbo, cuore del controllo di un tokamak.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak.control import PIDController, simulate_controlled
from tokamak.transport import TransportSolver1D

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    solver = TransportSolver1D(
        a=2.0, R0=6.2, n_e=1.0e20, chi=0.5, T_edge=0.1, n_cells=120
    )
    solver.T = solver.T_edge + 0.5 * (1.0 - (solver.r / solver.a) ** 2)  # partenza fredda

    shape = np.exp(-((solver.r / 0.6) ** 2))  # deposizione centrale
    target_T0 = 10.0  # keV

    pid = PIDController(
        kp=8e6, ki=6e6, kd=5e5, setpoint=target_T0, output_min=0.0, output_max=1.5e8
    )

    n_steps = 3000
    disturb_step = n_steps // 2
    hist = simulate_controlled(
        solver,
        pid,
        shape,
        dt=5e-3,
        n_steps=n_steps,
        measure="T0",
        chi_disturbance=(disturb_step, 1.0),  # chi 0.5 -> 1.0: confinamento dimezzato
    )

    t_dist = hist.time[disturb_step]
    print(f"T0 finale     = {hist.measurement[-1]:.2f} keV (target {target_T0})")
    print(f"P_ext finale  = {hist.power[-1] / 1e6:.1f} MW")
    print(f"P_ext pre-dist= {hist.power[disturb_step - 1] / 1e6:.1f} MW")
    print(f"Disturbo (chi x2) a t = {t_dist:.1f} s")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True)

    ax1.plot(hist.time, hist.measurement, color="crimson", lw=2, label="$T_0$ (measured)")
    ax1.axhline(target_T0, color="black", ls="--", lw=1, label="target")
    ax1.axvline(t_dist, color="gray", ls=":", lw=1)
    ax1.set_ylabel("$T_0$ [keV]")
    ax1.set_title("PID control of core temperature")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(hist.time, hist.power / 1e6, color="navy", lw=2)
    ax2.axvline(t_dist, color="gray", ls=":", lw=1, label="confinement degradation")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("$P_{ext}$ [MW]")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "control_demo.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
