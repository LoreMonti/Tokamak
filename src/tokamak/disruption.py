r"""Fase 13 — Predizione di disruption (classificazione ML).

Una disruption e' la perdita improvvisa del confinamento del plasma. Nei tokamak
reali la probabilita' di disruption cresce avvicinandosi ai limiti operativi
(densita' di Greenwald, beta di Troyon): non e' una soglia netta ma un rischio
crescente. La predizione delle disruzioni con ML e' un'applicazione molto reale.

Qui:
1. generiamo un dataset etichettato (stabile / disrupt) con una probabilita' di
   disruption FISICAMENTE motivata: cresce con la vicinanza ai limiti;
2. addestriamo un classificatore a predire la disruption dai parametri operativi;
3. mostriamo che la "regione sicura" appresa coincide con i limiti fisici.

Le etichette sono CAMPIONATE dalla probabilita' (rumorose), quindi e' un vero
problema di classificazione, non una frontiera banalmente separabile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier

from .engineering import greenwald_density, plasma_beta, troyon_beta_limit

# Parametri di macchina fissi (raggio minore, beta normalizzato).
_A_MINOR = 2.0
_BETA_N = 2.5

# Intervalli di campionamento delle feature.
PARAM_RANGES = {
    "n_e_1e20": (0.2, 1.6),  # densita' [10^20 m^-3]
    "T_keV": (3.0, 30.0),    # temperatura [keV]
    "Ip_MA": (8.0, 18.0),    # corrente di plasma [MA]
    "B_T": (4.0, 6.0),       # campo toroidale [T]
}
FEATURE_NAMES = list(PARAM_RANGES)

# Larghezza della transizione (soft margin) attorno al limite.
_MARGIN = 0.08


def limit_proximity(
    n_e: NDArray[np.float64],
    T_keV: NDArray[np.float64],
    Ip_MA: NDArray[np.float64],
    B_T: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vicinanza ai limiti: max tra frazione di Greenwald e di Troyon.

    Vale ~1 sul limite, <1 al sicuro, >1 oltre il limite.
    """
    n_G = greenwald_density(Ip_MA, _A_MINOR)
    f_greenwald = n_e / n_G
    beta = plasma_beta(n_e, T_keV, B_T)
    beta_max = troyon_beta_limit(_BETA_N, Ip_MA, _A_MINOR, B_T)
    f_troyon = beta / beta_max
    return np.maximum(f_greenwald, f_troyon)


def disruption_probability(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Probabilita' di disruption (sigmoide della vicinanza ai limiti).

    p = 1 / (1 + exp(-(proximity - 1)/margin)): bassa al sicuro, ~0.5 sul
    limite, alta oltre. X colonne = [n_e_1e20, T, Ip, B].
    """
    n_e = X[:, 0] * 1e20
    prox = limit_proximity(n_e, X[:, 1], X[:, 2], X[:, 3])
    return 1.0 / (1.0 + np.exp(-(prox - 1.0) / _MARGIN))


def sample_parameters(n_samples: int, seed: int = 0) -> NDArray[np.float64]:
    """Campiona uniformemente i parametri operativi negli intervalli definiti."""
    rng = np.random.default_rng(seed)
    cols = [rng.uniform(lo, hi, n_samples) for lo, hi in PARAM_RANGES.values()]
    return np.column_stack(cols)


def generate_labeled_dataset(
    n_samples: int, seed: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Genera (X, y): y campionato da Bernoulli(prob di disruption)."""
    X = sample_parameters(n_samples, seed)
    p = disruption_probability(X)
    rng = np.random.default_rng(seed + 1)
    y = (rng.random(n_samples) < p).astype(int)
    return X, y


def train_disruption_classifier(
    X: NDArray[np.float64], y: NDArray[np.int_]
) -> GradientBoostingClassifier:
    """Addestra un classificatore gradient-boosting (riproducibile)."""
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=0
    )
    clf.fit(X, y)
    return clf


@dataclass
class ClassifierScore:
    """Metriche di valutazione del classificatore."""

    accuracy: float
    roc_auc: float


def evaluate_classifier(
    clf: GradientBoostingClassifier, X: NDArray[np.float64], y: NDArray[np.int_]
) -> ClassifierScore:
    """Accuratezza e ROC-AUC sul set dato."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    pred = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]
    return ClassifierScore(
        accuracy=float(accuracy_score(y, pred)),
        roc_auc=float(roc_auc_score(y, proba)),
    )
