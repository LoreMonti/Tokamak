"""Superfici magnetiche di flusso da un equilibrio di Grad-Shafranov.

Risolve l'equazione di Grad-Shafranov non lineare (iterazione di Picard) e
disegna le curve di livello della funzione di flusso psi(R,Z): sono le superfici
magnetiche annidate su cui e' confinato il plasma. Il massimo di psi e' l'asse
magnetico, spostato verso l'esterno (shift di Shafranov) rispetto al centro
geometrico: e' una firma reale dell'equilibrio toroidale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak.equilibrium import GradShafranovSolver

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    R0 = 6.2  # raggio maggiore [m]
    a = 2.0   # raggio minore [m]
    kappa = 1.7  # elongazione (forma a D)
    delta = 0.33  # triangolarita'
    solver = GradShafranovSolver(
        R_min=3.6, R_max=8.8, Z_min=-3.8, Z_max=3.8, nR=121, nZ=181
    )
    # Bordo del plasma a forma di D (imposto dalle bobine di sagomatura).
    solver.set_d_shaped_boundary(R0=R0, a=a, kappa=kappa, delta=delta)

    def rhs(psi, RR):
        # Termine di destra di Grad-Shafranov tipo Solov'ev: una parte COSTANTE
        # (profili p', FF' di base -> garantisce una soluzione non banale) piu'
        # una parte NON lineare in psi (rende l'iterazione di Picard reale). Il
        # peso radiale ~R^2 produce lo shift di Shafranov verso l'esterno.
        psi_n = np.clip(psi / (np.max(psi) + 1e-30), 0.0, None)
        return -120.0 * (0.7 * (RR / R0) ** 2 + 0.3) * (1.0 + 1.8 * psi_n)

    iters = solver.solve_picard(rhs, max_iter=300, tol=1e-8, relax=0.4)
    R_ax, Z_ax, psi_ax = solver.magnetic_axis()

    print(f"Forma a D: kappa={kappa}, delta={delta}")
    print(f"Picard convergente in {iters} iterazioni")
    print(f"Asse magnetico: R = {R_ax:.2f} m, Z = {Z_ax:.2f} m")
    print(f"Shift di Shafranov (asse oltre R0): {R_ax - R0:+.2f} m")

    fig, ax = plt.subplots(figsize=(6.2, 8))
    levels = np.linspace(0.02 * psi_ax, psi_ax, 14)
    cf = ax.contourf(solver.RR, solver.ZZ, solver.psi, levels=30, cmap="plasma")
    ax.contour(solver.RR, solver.ZZ, solver.psi, levels=levels, colors="white",
               linewidths=0.7, alpha=0.8)
    # Ultima superficie chiusa (separatrice approssimata) e asse magnetico.
    ax.contour(solver.RR, solver.ZZ, solver.psi, levels=[0.02 * psi_ax],
               colors="cyan", linewidths=2)
    ax.plot(R_ax, Z_ax, "w+", markersize=14, markeredgewidth=2.5)

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Magnetic flux surfaces (Grad-Shafranov)")
    fig.colorbar(cf, ax=ax, label=r"$\psi$ (poloidal flux)", shrink=0.8)
    fig.tight_layout()

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "flux_surfaces.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
