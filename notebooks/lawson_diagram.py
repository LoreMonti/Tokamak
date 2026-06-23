"""Genera il diagramma di Lawson: triplo prodotto n*T*tau_E richiesto vs T.

Il diagramma di Lawson e' il grafico-firma della fisica della fusione. Sull'asse
x la temperatura, sull'asse y il triplo prodotto n*T*tau_E necessario per
l'ignition. La curva ha un MINIMO: e' la "finestra" piu' facile in cui far
funzionare un reattore.

Fisica del minimo (~14 keV per il D-T)
--------------------------------------
- A bassa T la reattivita' <sigma v> e' minuscola e il Bremsstrahlung domina:
  serve un confinamento enorme (la curva schizza verso l'alto, fino a infinito
  sotto la "ideal ignition temperature" dove la fusione non batte la radiazione).
- Ad alta T <sigma v> e' grande ma cresce piu' lentamente di T^2, quindi il
  requisito n*T*tau torna a salire dolcemente.
Il compromesso ottimo per il D-T cade intorno a 14 keV: per questo i reattori
puntano a ~10-20 keV, NON al picco di reattivita' (~66 keV).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak.power_balance import triple_product_ignition

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    T = np.linspace(2.0, 100.0, 500)
    ntt_with_brem = triple_product_ignition(T, include_bremsstrahlung=True)
    ntt_ideal = triple_product_ignition(T, include_bremsstrahlung=False)

    # Punto operativo rappresentativo di ITER (ordine di grandezza).
    iter_T, iter_ntt = 14.0, 3e21

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(T, ntt_ideal, "--", color="gray", label="Ignition ideale (no Bremsstrahlung)")
    ax.plot(T, ntt_with_brem, color="crimson", lw=2, label="Ignition reale (con Bremsstrahlung)")
    ax.scatter([iter_T], [iter_ntt], color="navy", zorder=5, s=60,
               label="Obiettivo ITER (~3e21)")

    # Minimo della curva reale.
    finite = np.isfinite(ntt_with_brem)
    i_min = np.argmin(np.where(finite, ntt_with_brem, np.inf))
    ax.annotate(
        f"minimo ~{T[i_min]:.0f} keV",
        xy=(T[i_min], ntt_with_brem[i_min]),
        xytext=(T[i_min] + 18, ntt_with_brem[i_min] * 2.5),
        arrowprops={"arrowstyle": "->"},
    )

    ax.set_yscale("log")
    ax.set_xlabel("Temperatura T [keV]")
    ax.set_ylabel(r"Triplo prodotto richiesto  $n\,T\,\tau_E$  [keV·s·m$^{-3}$]")
    ax.set_title("Criterio di Lawson per il plasma D-T")
    ax.set_ylim(1e21, 1e23)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "lawson_diagram.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")
    print(f"Minimo del triplo prodotto a T = {T[i_min]:.1f} keV")


if __name__ == "__main__":
    main()
