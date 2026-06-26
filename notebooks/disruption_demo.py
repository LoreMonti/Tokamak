"""Predizione di disruption: ROC e regione sicura appresa dal classificatore.

1. Curva ROC su dati mai visti (quanto bene separa stabile da disrupt).
2. Mappa della probabilita' di disruption appresa sul piano (T, n_e), con
   sovrapposti i limiti FISICI di Greenwald e Troyon: il classificatore deve
   "riscoprire" la regione sicura.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from tokamak.disruption import (
    generate_labeled_dataset,
    train_disruption_classifier,
)
from tokamak.engineering import greenwald_density, plasma_beta, troyon_beta_limit

DOCS = Path(__file__).resolve().parent.parent / "docs"
A_MINOR, BETA_N = 2.0, 2.5


def main() -> None:
    X, y = generate_labeled_dataset(3000, seed=7)
    X_tr, y_tr = X[:2200], y[:2200]
    X_te, y_te = X[2200:], y[2200:]
    clf = train_disruption_classifier(X_tr, y_tr)

    proba_te = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba_te)
    fpr, tpr, _ = roc_curve(y_te, proba_te)
    print(f"ROC-AUC su test: {auc:.3f}  ({y.sum()}/{len(y)} disrupt nel dataset)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # --- ROC ---
    a1.plot(fpr, tpr, color="crimson", lw=2, label=f"classificatore (AUC={auc:.3f})")
    a1.plot([0, 1], [0, 1], "k--", lw=1, label="caso casuale")
    a1.set_xlabel("tasso falsi positivi")
    a1.set_ylabel("tasso veri positivi")
    a1.set_title("Curva ROC (predizione disruption)")
    a1.legend(loc="lower right")
    a1.grid(True, alpha=0.3)

    # --- Regione sicura appresa, a Ip, B fissi ---
    Ip, B = 15.0, 5.3
    T = np.linspace(3.0, 30.0, 200)
    n = np.linspace(0.2e20, 1.6e20, 200)
    TT, NN = np.meshgrid(T, n)
    grid = np.column_stack([
        (NN / 1e20).ravel(), TT.ravel(),
        np.full(TT.size, Ip), np.full(TT.size, B),
    ])
    prob = clf.predict_proba(grid)[:, 1].reshape(TT.shape)

    cf = a2.contourf(TT, NN / 1e20, prob, levels=20, cmap="RdYlGn_r", vmin=0, vmax=1)
    fig.colorbar(cf, ax=a2, label="prob. di disruption (appresa)")
    # Limiti fisici sovrapposti.
    n_G = greenwald_density(Ip, A_MINOR)
    beta_max = troyon_beta_limit(BETA_N, Ip, A_MINOR, B)
    a2.axhline(n_G / 1e20, color="blue", lw=2, ls="--", label="Greenwald")
    a2.contour(TT, NN / 1e20, plasma_beta(NN, TT, B), levels=[beta_max],
               colors="black", linewidths=2)
    a2.plot([], [], color="black", lw=2, label="Troyon")
    a2.set_xlabel("T [keV]")
    a2.set_ylabel(r"$n_e$ [$10^{20}$ m$^{-3}$]")
    a2.set_title(f"Regione sicura appresa (Ip={Ip} MA, B={B} T)")
    a2.legend(loc="upper right")

    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "disruption.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
