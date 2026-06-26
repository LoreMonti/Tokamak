"""Punto operativo ottimo nello spazio (T, n_e) con i vincoli.

Massimizza la densita' di potenza di fusione sotto i limiti di Greenwald e
Troyon, e segna l'ottimo sul diagramma dello spazio operativo. L'ottimo cade sul
bordo dei vincoli: e' la sintesi quantitativa di fisica + ingegneria.
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
from tokamak.optimization import optimize_operating_point
from tokamak.power_balance import fusion_power_density

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    cfg = TokamakConfig()
    opt = optimize_operating_point(cfg)

    print(f"Punto ottimo: n_e = {opt.n_e:.3e} m^-3, T = {opt.T_keV:.1f} keV")
    print(f"P_fus = {opt.fusion_power_density / 1e6:.3f} MW/m^3, Q = {opt.Q:.1f}")
    print(f"Greenwald attivo: {opt.greenwald_active} | Troyon attivo: {opt.troyon_active}")

    T = np.linspace(2.0, 35.0, 400)
    n = np.linspace(0.05e20, 1.5e20, 400)
    TT, NN = np.meshgrid(T, n)

    n_G = greenwald_density(cfg.plasma_current_MA, cfg.minor_radius_m)
    beta_max = troyon_beta_limit(
        cfg.beta_N, cfg.plasma_current_MA, cfg.minor_radius_m, cfg.B_toroidal_T
    )
    beta_grid = plasma_beta(NN, TT, cfg.B_toroidal_T)
    pfus_grid = fusion_power_density(NN, TT) / 1e6  # MW/m^3

    fig, ax = plt.subplots(figsize=(8.5, 6))
    cf = ax.contourf(TT, NN / 1e20, pfus_grid, levels=25, cmap="viridis")
    fig.colorbar(cf, ax=ax, label=r"$P_{fus}$ [MW/m$^3$]")

    ax.axhline(n_G / 1e20, color="white", lw=2, ls="--")
    ax.contour(TT, NN / 1e20, beta_grid, levels=[beta_max], colors="white", linewidths=2)
    ax.plot(opt.T_keV, opt.n_e / 1e20, "r*", markersize=22, markeredgecolor="white")

    ax.set_xlabel("Temperature T [keV]")
    ax.set_ylabel(r"Density $n_e$ [$10^{20}$ m$^{-3}$]")
    ax.set_title("Optimal operating point (max $P_{fus}$ under constraints)")
    handles = [
        Line2D([], [], color="white", ls="--", lw=2, label="Greenwald"),
        Line2D([], [], color="white", lw=2, label="Troyon (beta)"),
        Line2D([], [], color="red", marker="*", ls="", markersize=14, label="optimum"),
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "optimum_point.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
