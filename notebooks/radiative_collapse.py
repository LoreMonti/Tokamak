"""Collasso radiativo: quanta impurità tollera il plasma prima di spegnersi.

Per ogni impurità, all'aumentare della concentrazione la potenza irraggiata
cresce. Quando supera il riscaldamento disponibile, la temperatura collassa.
Le impurità ad alto Z (tungsteno) sono catastrofiche anche in tracce, per via
dello scaling ~Z^3 della radiazione: e' il motivo per cui il controllo delle
impurità e' cruciale nei reattori con divertore in tungsteno.

(Modello di funzione di raffreddamento schematico: vedi radiation.py.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak.radiation import (
    ATOMIC_NUMBER,
    max_impurity_fraction,
    total_radiated_power_density,
)

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    n_e, T = 1.0e20, 10.0  # m^-3, keV
    heating = 5e5  # W/m^3 disponibili (alfa + esterno)
    species_list = ["C", "Ne", "Ar", "Fe", "W"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # Pannello 1: radiazione vs concentrazione per ogni impurità.
    # Il range si ferma al limite di quasi-neutralità (~0.9/Z) di ogni specie.
    for s in species_list:
        c_hi = min(0.1, 0.9 / ATOMIC_NUMBER[s])
        c = np.logspace(-5, np.log10(c_hi), 200)
        p_rad = np.array([total_radiated_power_density(n_e, T, {s: ci}) for ci in c])
        ax1.loglog(c, p_rad / 1e6, lw=2, label=f"{s} (Z={ATOMIC_NUMBER[s]})")
    ax1.axhline(heating / 1e6, color="black", ls="--", lw=1.5, label="heating")
    ax1.set_xlabel("impurity fraction  $n_z/n_e$")
    ax1.set_ylabel("radiated power [MW/m$^3$]")
    ax1.set_title(f"Radiation vs impurity (T={T:.0f} keV)")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # Pannello 2: concentrazione massima tollerabile vs Z (soglia di collasso).
    Z = [ATOMIC_NUMBER[s] for s in species_list]
    c_max = [max_impurity_fraction(s, n_e, T, heating) for s in species_list]
    ax2.loglog(Z, c_max, "o-", color="crimson", lw=2, markersize=8)
    for s, zz, cm in zip(species_list, Z, c_max, strict=True):
        ax2.annotate(s, (zz, cm), textcoords="offset points", xytext=(6, 6))
    ax2.set_xlabel("atomic number Z")
    ax2.set_ylabel("max fraction before collapse")
    ax2.set_title("Impurity tolerance vs Z")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "radiative_collapse.png"
    fig.savefig(out, dpi=130)

    print("Frazione massima tollerabile prima del collasso radiativo:")
    for s, cm in zip(species_list, c_max, strict=True):
        print(f"  {s:2s} (Z={ATOMIC_NUMBER[s]:2d}): {cm:.2e}")
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
