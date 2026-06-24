"""Profilo radiale di temperatura T(r) allo stato stazionario.

Risolve il trasporto del calore 1D per un plasma tipo tokamak con riscaldamento
esterno concentrato al centro, e mostra il profilo T(r) emergente insieme al
tempo di confinamento dell'energia tau_E calcolato (non imposto).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak.transport import TransportSolver1D

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    solver = TransportSolver1D(
        a=2.0,  # raggio minore [m]
        R0=6.2,  # raggio maggiore [m] (scala ITER)
        n_e=1.0e20,  # densita' [m^-3]
        chi=1.0,  # diffusivita' termica [m^2/s]
        T_edge=0.1,  # bordo freddo [keV]
        n_cells=200,
    )

    # Riscaldamento esterno: gaussiana centrata (es. iniezione di neutri / RF).
    p_ext = 3e5 * np.exp(-((solver.r / 0.6) ** 2))  # W/m^3

    steps = solver.solve_steady_state(p_ext=p_ext, dt=1e-3)
    tau = solver.tau_E(p_ext=p_ext)
    p_fus_alpha = solver.total_power("fusion_alpha")
    p_ext_tot = solver.total_power("external", p_ext)

    print(f"Convergenza in {steps} passi")
    print(f"T centrale  = {solver.T[0]:.1f} keV")
    print(f"tau_E       = {tau:.2f} s  (EMERGENTE dal profilo, non imposto)")
    print(f"P_alpha tot = {p_fus_alpha / 1e6:.2f} MW")
    print(f"P_ext   tot = {p_ext_tot / 1e6:.2f} MW")
    print(f"Volume      = {solver.plasma_volume():.1f} m^3")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(solver.r, solver.T, color="crimson", lw=2)
    ax.fill_between(solver.r, solver.T, alpha=0.1, color="crimson")
    ax.set_xlabel("Raggio minore r [m]")
    ax.set_ylabel("Temperatura T [keV]")
    ax.set_title(
        f"Profilo radiale stazionario  —  $T_0$={solver.T[0]:.1f} keV, "
        rf"$\tau_E$={tau:.2f} s"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "radial_profile.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
