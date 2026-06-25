"""Emulatore ML del solver di trasporto: accuratezza e speed-up.

Genera un dataset col solver 1D, addestra un emulatore (processo gaussiano) e:
1. mostra il parity plot (predetto vs vero) su dati mai visti, con R^2;
2. misura lo speed-up rispetto al solver.

Il dataset viene messo in cache su disco: la prima esecuzione e' lenta (esegue
il solver), le successive sono immediate.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak.surrogate import (
    TARGET_NAMES,
    generate_dataset,
    r2_per_target,
    run_transport_point,
    train_surrogate,
)

DOCS = Path(__file__).resolve().parent.parent / "docs"
DATA = Path(__file__).resolve().parent.parent / "data"


def _load_or_generate(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    DATA.mkdir(exist_ok=True)
    cache = DATA / f"surrogate_dataset_{n_samples}.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"Dataset caricato da cache: {cache.name}")
        return d["X"], d["Y"]
    print(f"Genero {n_samples} campioni col solver (una tantum)...")
    X, Y = generate_dataset(n_samples, seed=42, n_cells=40)
    np.savez(cache, X=X, Y=Y)
    return X, Y


def main() -> None:
    X, Y = _load_or_generate(300)
    n_train = 240
    model = train_surrogate(X[:n_train], Y[:n_train])
    X_te, Y_te = X[n_train:], Y[n_train:]
    pred = model.predict(X_te)
    score = r2_per_target(model, X_te, Y_te)
    print(f"R^2 su test: tau_E = {score.r2_tau_E:.3f}, T0 = {score.r2_T0:.3f}")

    # Speed-up: tempo medio per chiamata.
    t0 = time.perf_counter()
    for _ in range(5):
        run_transport_point(1.0, 1.0, 20.0, n_cells=40)
    t_solver = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(200):
        model.predict([[1.0, 1.0, 20.0]])
    t_surr = (time.perf_counter() - t0) / 200
    print(f"Solver: {t_solver*1e3:.1f} ms/chiamata | Emulatore: {t_surr*1e3:.3f} ms "
          f"-> speed-up ~{t_solver/t_surr:.0f}x")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for k, (ax, name, unit) in enumerate(
        zip(axes, TARGET_NAMES, ["s", "keV"], strict=True)
    ):
        true, p = Y_te[:, k], pred[:, k]
        ax.scatter(true, p, alpha=0.6, color="crimson", edgecolor="k", linewidth=0.3)
        lim = [min(true.min(), p.min()), max(true.max(), p.max())]
        ax.plot(lim, lim, "k--", lw=1)
        r2 = score.r2_tau_E if k == 0 else score.r2_T0
        ax.set_xlabel(f"{name} vero [{unit}]")
        ax.set_ylabel(f"{name} predetto [{unit}]")
        ax.set_title(f"{name}  ($R^2$={r2:.3f})")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Emulatore GP vs solver — speed-up ~{t_solver/t_surr:.0f}x")
    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "surrogate.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
