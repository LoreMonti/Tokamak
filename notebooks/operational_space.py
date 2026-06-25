"""Diagramma dello spazio operativo: densita' vs temperatura con i limiti.

Mostra, sul piano (T, n_e), la regione in cui un tokamak puo' operare in modo
sicuro e con guadagno. Sovrappone:
- il limite di densita' di Greenwald (orizzontale: n_e < n_G);
- il limite di beta di Troyon (curva: n*T < cost, perche' beta ~ n*T);
- la curva di break-even Q=1 dal modello 0D (serve confinamento adeguato).

La regione utile e' la "finestra" che soddisfa tutti i vincoli. E' la sintesi
visiva di fisica (Q) e ingegneria (Greenwald, Troyon) del progetto.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tokamak.engineering import (
    TokamakConfig,
    greenwald_density,
    plasma_beta,
    troyon_beta_limit,
)
from tokamak.power_balance import fusion_gain_Q

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    cfg = TokamakConfig()
    T = np.linspace(2.0, 30.0, 400)  # keV
    n = np.linspace(0.1e20, 2.0e20, 400)  # m^-3
    TT, NN = np.meshgrid(T, n)

    # Limite di Greenwald (densita' massima, indipendente da T).
    n_G = greenwald_density(cfg.plasma_current_MA, cfg.minor_radius_m)

    # Limite di Troyon: beta(n,T) < beta_max  ->  curva n(T) limite.
    beta_max = troyon_beta_limit(
        cfg.beta_N, cfg.plasma_current_MA, cfg.minor_radius_m, cfg.B_toroidal_T
    )
    beta_grid = plasma_beta(NN, TT, cfg.B_toroidal_T)

    # Q dal modello 0D (con tau_E rappresentativo per disegnare il break-even).
    tau_E = 3.0  # s, scala ITER
    Q_grid = np.vectorize(lambda nn, tt: fusion_gain_Q(nn, tt, tau_E))(NN, TT)

    fig, ax = plt.subplots(figsize=(8.5, 6))

    # Greenwald: regione proibita sopra n_G.
    ax.axhline(n_G / 1e20, color="darkgreen", lw=2, ls="--", label="Limite di Greenwald")
    ax.fill_between(T, n_G / 1e20, 2.0, color="green", alpha=0.08)

    # Troyon: contorno beta = beta_max.
    ax.contour(
        TT, NN / 1e20, beta_grid, levels=[beta_max], colors="purple", linewidths=2
    )

    # Break-even e Q=10.
    cs_q = ax.contour(
        TT, NN / 1e20, Q_grid, levels=[1.0, 10.0], colors=["orange", "crimson"],
        linewidths=2,
    )
    ax.clabel(cs_q, fmt={1.0: "Q=1", 10.0: "Q=10"}, inline=True, fontsize=9)

    ax.set_xlabel("Temperatura T [keV]")
    ax.set_ylabel(r"Densita' $n_e$ [$10^{20}\,$m$^{-3}$]")
    ax.set_title("Spazio operativo del tokamak (parametri ~ ITER)")
    ax.set_ylim(0, 2.0)

    # Legenda con handle proxy (i contour non espongono piu' .collections).
    handles = [
        Line2D([], [], color="darkgreen", ls="--", lw=2, label="Limite di Greenwald"),
        Line2D([], [], color="purple", lw=2, label="Limite di Troyon (beta)"),
        Line2D([], [], color="orange", lw=2, label="Break-even Q=1"),
        Line2D([], [], color="crimson", lw=2, label="Q=10 (target ITER)"),
    ]
    ax.legend(handles=handles, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "operational_space.png"
    fig.savefig(out, dpi=130)
    print(f"n_Greenwald = {n_G:.2e} m^-3")
    print(f"beta_max (Troyon) = {beta_max * 100:.2f}%")
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
