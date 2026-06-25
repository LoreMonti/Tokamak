r"""Fase 11 — Emulatore ML del solver di trasporto (surrogate model).

Il solver 1D e' accurato ma lento (decimi di secondo per punto). Per scan
massicci o controllo in tempo reale serve qualcosa di molto piu' rapido. Qui
addestriamo un modello di machine learning a riprodurre la mappa

    (n_e, chi, P_ext)  ->  (tau_E, T_centrale)

imparata dai dati del solver. Una volta addestrato, predice in microsecondi
(speed-up di ordini di grandezza), con accuratezza misurata su dati mai visti.

Pattern classico "physics + ML":
1. generare un dataset eseguendo il solver su parametri campionati;
2. addestrare un regressore (gradient boosting multi-output);
3. validare R^2 su un test set e misurare lo speed-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .transport import TransportSolver1D

# Intervalli di campionamento dei parametri di ingresso. Scelti per restare in
# un regime di confinamento "liscio": a chi troppo basso + potenza alta il
# plasma andrebbe in runaway termico (ignition), con T fuori dal range di
# validita' della reattivita' e una mappa molto non lineare da emulare.
PARAM_RANGES = {
    "n_e_1e20": (0.4, 1.2),  # densita' [10^20 m^-3]
    "chi": (0.6, 2.0),       # diffusivita' termica [m^2/s]
    "P_ext_MW": (5.0, 30.0),  # riscaldamento esterno totale [MW]
}
FEATURE_NAMES = list(PARAM_RANGES)
TARGET_NAMES = ["tau_E", "T0"]


def run_transport_point(
    n_e_1e20: float, chi: float, P_ext_MW: float, *, n_cells: int = 40
) -> tuple[float, float]:
    """Esegue il solver di trasporto a regime e restituisce (tau_E, T_centrale).

    Griglia/passo scelti per un buon compromesso accuratezza-velocita' (la
    generazione del dataset richiede molte chiamate).
    """
    s = TransportSolver1D(
        a=2.0, R0=6.2, n_e=n_e_1e20 * 1e20, chi=chi, T_edge=0.1, n_cells=n_cells
    )
    shape = np.exp(-((s.r / 0.6) ** 2))
    p_ext = s.heating_density_for_power(P_ext_MW * 1e6, shape)
    s.solve_steady_state(p_ext=p_ext, dt=2e-2, tol=1e-4, max_steps=5000)
    return s.tau_E(p_ext=p_ext), float(s.T[0])


def sample_parameters(n_samples: int, seed: int = 0) -> NDArray[np.float64]:
    """Campiona uniformemente i parametri di ingresso negli intervalli definiti."""
    rng = np.random.default_rng(seed)
    cols = [rng.uniform(lo, hi, n_samples) for lo, hi in PARAM_RANGES.values()]
    return np.column_stack(cols)


def generate_dataset(
    n_samples: int, seed: int = 0, n_cells: int = 40
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Genera (X, Y) eseguendo il solver sui parametri campionati.

    X: (n_samples, 3) parametri; Y: (n_samples, 2) [tau_E, T0].
    """
    X = sample_parameters(n_samples, seed)
    Y = np.array([run_transport_point(*row, n_cells=n_cells) for row in X])
    return X, Y


def train_surrogate(X: NDArray[np.float64], Y: NDArray[np.float64]) -> Pipeline:
    """Addestra un emulatore basato su processo gaussiano (GP).

    Per dataset piccoli e funzioni lisce a bassa dimensione il GP e' ideale:
    interpola accuratamente e quantifica l'incertezza. Usiamo un kernel RBF
    (regolarita') + WhiteKernel (rumore numerico residuo), con StandardScaler
    sulle feature e normalizzazione dei target (normalize_y). Il GP gestisce
    nativamente l'output multiplo (tau_E e T0).
    """
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0, 1.0]) + WhiteKernel(1e-3)
    # random_state fissato: rende l'addestramento riproducibile (i restart
    # dell'ottimizzatore degli iperparametri sono altrimenti casuali).
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0
    )
    model = Pipeline([("scaler", StandardScaler()), ("gp", gp)])
    model.fit(X, Y)
    return model


@dataclass
class SurrogateScore:
    """Accuratezza dell'emulatore su un test set."""

    r2_tau_E: float
    r2_T0: float


def r2_per_target(
    model: Pipeline, X: NDArray[np.float64], Y: NDArray[np.float64]
) -> SurrogateScore:
    """Coefficiente di determinazione R^2 per ciascun target sul set dato."""
    from sklearn.metrics import r2_score

    pred = model.predict(X)
    return SurrogateScore(
        r2_tau_E=float(r2_score(Y[:, 0], pred[:, 0])),
        r2_T0=float(r2_score(Y[:, 1], pred[:, 1])),
    )
