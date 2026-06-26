"""Ottimizzazione bayesiana del punto operativo: convergenza e punti campionati.

1. Convergenza a parita' di valutazioni: miglior P_fus trovato vs numero di
   valutazioni dell'obiettivo, per BO E per SLSQP (Fase 8). Entrambe le curve
   crescono fino al loro plateau.
2. Dove BO ha campionato sul piano (T, n_e): si concentra vicino all'ottimo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from tokamak.bayesopt import bayesian_optimize, make_operating_point_objective
from tokamak.engineering import (
    TokamakConfig,
    greenwald_density,
    plasma_beta,
    troyon_beta_limit,
)
from tokamak.power_balance import fusion_power_density

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _slsqp_trace(cfg: TokamakConfig, T_min: float, T_max: float):
    """Esegue SLSQP (come la Fase 8) tracciando il miglior P_fus FEASIBLE
    a ogni valutazione dell'obiettivo, per confronto con BO."""
    n_G = greenwald_density(cfg.plasma_current_MA, cfg.minor_radius_m)
    beta_max = troyon_beta_limit(cfg.beta_N, cfg.plasma_current_MA,
                                 cfg.minor_radius_m, cfg.B_toroidal_T)
    B = cfg.B_toroidal_T
    feasible_pfus: list[float] = []

    def neg_pfus(x):
        n_e, T = x[0] * 1e20, x[1]
        p = float(fusion_power_density(n_e, T)) / 1e6
        feasible = (n_e <= n_G) and (plasma_beta(n_e, T, B) <= beta_max)
        feasible_pfus.append(p if feasible else -np.inf)
        return -p

    constraints = [
        {"type": "ineq", "fun": lambda x: n_G / 1e20 - x[0]},
        {"type": "ineq", "fun": lambda x: beta_max - plasma_beta(x[0] * 1e20, x[1], B)},
    ]
    bounds = [(0.05, n_G / 1e20), (T_min, T_max)]
    x0 = np.array([0.4 * n_G / 1e20, 0.5 * (T_min + T_max)])
    minimize(neg_pfus, x0, method="SLSQP", bounds=bounds, constraints=constraints,
             options={"ftol": 1e-10, "maxiter": 300})

    best = np.maximum.accumulate(np.array(feasible_pfus))
    best[~np.isfinite(best)] = np.nan  # prima di trovare un punto feasible
    return best


def main() -> None:
    cfg = TokamakConfig()
    T_min, T_max = 5.0, 40.0
    objective, bounds = make_operating_point_objective(cfg, T_min=T_min, T_max=T_max)

    n_init, n_iter = 6, 30
    res = bayesian_optimize(objective, bounds, n_init=n_init, n_iter=n_iter, seed=3)
    bo_best = np.maximum.accumulate(res.y)  # miglior valore vs valutazione (da 1)

    slsqp_best = _slsqp_trace(cfg, T_min, T_max)
    p_opt = float(np.nanmax([np.nanmax(slsqp_best), bo_best.max()]))

    print(f"BO:    n_e={res.best_x[0]*1e20:.2e} m^-3, T={res.best_x[1]:.1f} keV, "
          f"P_fus={res.best_y:.3f} MW/m^3 in {len(res.y)} valutazioni")
    print(f"SLSQP: P_fus={np.nanmax(slsqp_best):.3f} MW/m^3 in {len(slsqp_best)} "
          f"valutazioni")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # --- Convergenza (entrambe vs numero di valutazioni) ---
    a1.plot(np.arange(1, len(bo_best) + 1), bo_best, "o-", color="crimson", ms=4,
            label="ottimizzazione bayesiana")
    a1.plot(np.arange(1, len(slsqp_best) + 1), slsqp_best, "s-", color="navy", ms=4,
            label="SLSQP (Fase 8)")
    a1.set_xlabel("numero di valutazioni dell'obiettivo")
    a1.set_ylabel(r"miglior $P_{fus}$ trovato [MW/m$^3$]")
    a1.set_title("Convergenza: BO vs SLSQP")
    a1.legend(loc="lower right")
    a1.grid(True, alpha=0.3)

    # --- Punti campionati (range esteso a tutto il dominio di BO) ---
    n_G = greenwald_density(cfg.plasma_current_MA, cfg.minor_radius_m)
    T = np.linspace(2.0, T_max, 220)
    n = np.linspace(0.0, 1.1 * n_G, 220)
    TT, NN = np.meshgrid(T, n)
    cf = a2.contourf(TT, NN / 1e20, fusion_power_density(NN, TT) / 1e6,
                     levels=20, cmap="viridis")
    fig.colorbar(cf, ax=a2, label=r"$P_{fus}$ [MW/m$^3$]")
    beta_max = troyon_beta_limit(cfg.beta_N, cfg.plasma_current_MA,
                                 cfg.minor_radius_m, cfg.B_toroidal_T)
    a2.axhline(n_G / 1e20, color="white", ls="--", lw=2)
    a2.contour(TT, NN / 1e20, plasma_beta(NN, TT, cfg.B_toroidal_T),
               levels=[beta_max], colors="white", linewidths=2)
    a2.scatter(res.X[:, 1], res.X[:, 0], c="orange", s=22, edgecolor="k",
               linewidth=0.4, label="punti valutati da BO")
    a2.plot(res.best_x[1], res.best_x[0], marker="D", color="red", ms=7,
            markeredgecolor="white", markeredgewidth=1.2, ls="", label="ottimo BO")
    a2.set_xlabel("T [keV]")
    a2.set_ylabel(r"$n_e$ [$10^{20}$ m$^{-3}$]")
    a2.set_title("Punti campionati (Greenwald/Troyon in bianco)")
    a2.legend(loc="upper right", fontsize=8)
    a2.set_xlim(2.0, T_max)
    a2.set_ylim(0.0, 1.1 * n_G / 1e20)

    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "bayesopt.png"
    fig.savefig(out, dpi=130)
    print(f"P_fus ottimo (riferimento): {p_opt:.3f} MW/m^3")
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
