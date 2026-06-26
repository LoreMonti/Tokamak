"""Emulatore deep-learning del profilo T(r): accuratezza e speed-up.

Genera un dataset col solver, addestra una rete neurale (PyTorch) che predice
l'INTERO profilo radiale T(r) dai parametri, e mostra:
1. profili predetti vs veri su casi mai visti;
2. R^2/RMSE e speed-up rispetto al solver.

A differenza dell'emulatore GP scalare (Fase 11), qui l'output e' un vettore
(il profilo completo): regressione funzionale con deep learning.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak.profile_emulator import (
    RADIAL_GRID,
    generate_profile_dataset,
    profile_r2,
    profile_rmse,
    run_profile,
    train_profile_emulator,
)

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    X, Y = generate_profile_dataset(300, seed=11, n_cells=40)
    n_tr = 240
    emu = train_profile_emulator(X[:n_tr], Y[:n_tr], epochs=2500, seed=0)
    X_te, Y_te = X[n_tr:], Y[n_tr:]
    r2 = profile_r2(emu, X_te, Y_te)
    rmse = profile_rmse(emu, X_te, Y_te)
    print(f"R^2 = {r2:.4f}, RMSE = {rmse:.3f} keV su {len(X_te)} profili di test")

    # Speed-up: tempo per profilo.
    t0 = time.perf_counter()
    for _ in range(3):
        run_profile(1.0, 1.5, 18.0, n_cells=40)
    t_solver = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for _ in range(100):
        emu.predict(X_te[:1])
    t_emu = (time.perf_counter() - t0) / 100
    print(f"Solver: {t_solver*1e3:.0f} ms | Emulatore: {t_emu*1e3:.2f} ms "
          f"-> speed-up ~{t_solver/t_emu:.0f}x")

    pred = emu.predict(X_te)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Profili: alcuni esempi (vero vs predetto).
    idx = np.argsort(Y_te[:, 0])[:: max(1, len(X_te) // 6)][:6]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(idx)))
    for c, i in zip(colors, idx, strict=False):
        a1.plot(RADIAL_GRID, Y_te[i], color=c, lw=2)
        a1.plot(RADIAL_GRID, pred[i], color=c, ls="--", lw=1.5)
    a1.plot([], [], "k-", label="solver (vero)")
    a1.plot([], [], "k--", label="rete neurale")
    a1.set_xlabel("r / a")
    a1.set_ylabel("T [keV]")
    a1.set_title("Profili: solver vs emulatore NN")
    a1.legend()
    a1.grid(True, alpha=0.3)

    # Parity plot su tutti i punti dei profili di test.
    a2.scatter(Y_te.ravel(), pred.ravel(), s=6, alpha=0.3, color="crimson",
               edgecolor="none")
    lim = [0, Y_te.max() * 1.05]
    a2.plot(lim, lim, "k--", lw=1)
    a2.set_xlabel("T vero [keV]")
    a2.set_ylabel("T predetto [keV]")
    a2.set_title(f"Parity ($R^2$={r2:.3f}, RMSE={rmse:.2f} keV)")
    a2.grid(True, alpha=0.3)

    fig.suptitle(f"Emulatore NN del profilo T(r) — speed-up ~{t_solver/t_emu:.0f}x")
    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "profile_emulator.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
