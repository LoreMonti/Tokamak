"""Ottimizzazione bayesiana del punto operativo: convergenza e punti campionati.

1. Convergenza: miglior P_fus trovato vs numero di valutazioni, confrontato con
   l'ottimo dell'ottimizzatore SLSQP (Fase 8).
2. Dove BO ha campionato sul piano (T, n_e): si concentra vicino all'ottimo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak.bayesopt import bayesian_optimize, make_operating_point_objective
from tokamak.engineering import (
    TokamakConfig,
    greenwald_density,
    plasma_beta,
    troyon_beta_limit,
)
from tokamak.optimization import optimize_operating_point
from tokamak.power_balance import fusion_power_density

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    cfg = TokamakConfig()
    objective, bounds = make_operating_point_objective(cfg)

    n_init, n_iter = 6, 30
    res = bayesian_optimize(objective, bounds, n_init=n_init, n_iter=n_iter, seed=3)
    slsqp = optimize_operating_point(cfg)
    p_slsqp = slsqp.fusion_power_density / 1e6

    print(f"BO:    n_e={res.best_x[0]*1e20:.2e} m^-3, T={res.best_x[1]:.1f} keV, "
          f"P_fus={res.best_y:.3f} MW/m^3")
    print(f"SLSQP: n_e={slsqp.n_e:.2e} m^-3, T={slsqp.T_keV:.1f} keV, "
          f"P_fus={p_slsqp:.3f} MW/m^3")
    print(f"BO raggiunge il {100*res.best_y/p_slsqp:.1f}% dell'ottimo SLSQP "
          f"in {n_init + n_iter} valutazioni")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # --- Convergenza ---
    evals = np.arange(n_init + 1, n_init + n_iter + 1)
    a1.plot(evals, res.best_history, "o-", color="crimson", ms=4,
            label="ottimizzazione bayesiana")
    a1.axhline(p_slsqp, color="navy", ls="--", lw=2, label="ottimo SLSQP (Fase 8)")
    a1.set_xlabel("numero di valutazioni")
    a1.set_ylabel(r"miglior $P_{fus}$ trovato [MW/m$^3$]")
    a1.set_title("Convergenza dell'ottimizzazione bayesiana")
    a1.legend(loc="lower right")
    a1.grid(True, alpha=0.3)

    # --- Punti campionati ---
    T = np.linspace(2.0, 35.0, 200)
    n = np.linspace(0.05e20, 1.5e20, 200)
    TT, NN = np.meshgrid(T, n)
    cf = a2.contourf(TT, NN / 1e20, fusion_power_density(NN, TT) / 1e6,
                     levels=20, cmap="viridis")
    fig.colorbar(cf, ax=a2, label=r"$P_{fus}$ [MW/m$^3$]")
    n_G = greenwald_density(cfg.plasma_current_MA, cfg.minor_radius_m)
    beta_max = troyon_beta_limit(cfg.beta_N, cfg.plasma_current_MA,
                                 cfg.minor_radius_m, cfg.B_toroidal_T)
    a2.axhline(n_G / 1e20, color="white", ls="--", lw=2)
    a2.contour(TT, NN / 1e20, plasma_beta(NN, TT, cfg.B_toroidal_T),
               levels=[beta_max], colors="white", linewidths=2)
    a2.scatter(res.X[:, 1], res.X[:, 0], c="orange", s=25, edgecolor="k",
               linewidth=0.4, label="punti valutati da BO")
    a2.plot(res.best_x[1], res.best_x[0], "r*", ms=20, markeredgecolor="white",
            label="ottimo BO")
    a2.set_xlabel("T [keV]")
    a2.set_ylabel(r"$n_e$ [$10^{20}$ m$^{-3}$]")
    a2.set_title("Punti campionati (Greenwald/Troyon in bianco)")
    a2.legend(loc="upper right", fontsize=8)
    a2.set_ylim(0, 1.5)

    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "bayesopt.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
