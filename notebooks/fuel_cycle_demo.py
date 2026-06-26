"""Ciclo del combustibile: autosufficienza del trizio.

Tre messaggi:
1. Consumo di trizio vs potenza di fusione (~0.5 kg/giorno): le scorte mondiali
   (pochi kg) basterebbero pochi giorni -> il breeding e' obbligatorio.
2. Inventory evolution per TBR < 1, = 1, > 1.
3. Doubling time vs TBR: quanto surplus serve per avviare nuovi reattori.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak.fuel_cycle import (
    doubling_time_years,
    simulate_inventory,
    tritium_burn_rate_kg_per_day,
)

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    P_fus = 3e9  # 3 GW di potenza di fusione (scala reattore)
    burn = tritium_burn_rate_kg_per_day(P_fus)
    print(f"Consumo di trizio a {P_fus/1e9:.0f} GW: {burn:.2f} kg/giorno "
          f"({burn * 365:.0f} kg/anno)")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # 1) Consumo vs potenza.
    P = np.linspace(0.5e9, 4e9, 100)
    axes[0].plot(P / 1e9, [tritium_burn_rate_kg_per_day(p) for p in P], color="navy", lw=2)
    axes[0].set_xlabel("fusion power [GW]")
    axes[0].set_ylabel("tritium consumption [kg/day]")
    axes[0].set_title("Tritium consumption")
    axes[0].grid(True, alpha=0.3)

    # 2) Inventario nel tempo per diversi TBR.
    for tbr, col in [(0.95, "crimson"), (1.0, "gray"), (1.10, "green")]:
        h = simulate_inventory(N0_kg=2.0, P_fusion_W=P_fus, TBR=tbr, t_end_years=8.0)
        axes[1].plot(h.time_years, h.inventory_kg, color=col, lw=2, label=f"TBR={tbr}")
    axes[1].set_xlabel("time [years]")
    axes[1].set_ylabel("tritium inventory [kg]")
    axes[1].set_title("Inventory evolution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3) Doubling time vs TBR.
    tbr = np.linspace(1.01, 1.30, 100)
    t2 = [doubling_time_years(b, P_fus, startup_inventory_kg=10.0) for b in tbr]
    axes[2].plot(tbr, t2, color="purple", lw=2)
    axes[2].set_xlabel("TBR")
    axes[2].set_ylabel("doubling time [years]")
    axes[2].set_title("Doubling time (starting new reactors)")
    axes[2].set_ylim(0, 40)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "fuel_cycle.png"
    fig.savefig(out, dpi=130)
    print(f"Doubling time a TBR=1.10: {doubling_time_years(1.10, P_fus, 10.0):.1f} anni")
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
