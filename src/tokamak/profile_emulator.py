r"""Fase 15 — Emulatore deep-learning del profilo radiale (PyTorch).

La Fase 11 emula con un processo gaussiano grandezze SCALARI (tau_E, T0). Qui
spingiamo oltre: una RETE NEURALE (PyTorch) che predice l'INTERO profilo
radiale T(r) dai parametri (n_e, chi, P_ext) — un output vettoriale (regressione
funzionale), task piu' difficile e che mostra il deep learning.

Pattern:
1. dataset: parametri -> profilo T(r) (ricampionato su una griglia normalizzata)
   eseguendo il solver di trasporto;
2. addestrare un MLP che mappa 3 parametri -> profilo (32 punti);
3. validare (RMSE sul profilo, R^2) e misurare lo speed-up.

NOTA: questo modulo richiede PyTorch (dipendenza opzionale [ml]) e NON viene
importato dal package __init__, cosi' `import tokamak` resta possibile senza torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .transport import TransportSolver1D

A_MINOR, R0 = 2.0, 6.2
N_PROFILE_POINTS = 32
RADIAL_GRID = np.linspace(0.0, 1.0, N_PROFILE_POINTS)  # r/a in [0,1]

# Intervalli scelti nel regime sub-ignition (profili lisci): chi e potenza tali
# da evitare la runaway termica, che produrrebbe profili estremi (T0 ~ decine di
# keV) impossibili da emulare bene.
PARAM_RANGES = {
    "n_e_1e20": (0.4, 1.2),
    "chi": (1.2, 2.2),
    "P_ext_MW": (8.0, 26.0),
}


def sample_parameters(n_samples: int, seed: int = 0) -> NDArray[np.float64]:
    """Campiona uniformemente i parametri negli intervalli (regime liscio)."""
    rng = np.random.default_rng(seed)
    cols = [rng.uniform(lo, hi, n_samples) for lo, hi in PARAM_RANGES.values()]
    return np.column_stack(cols)


def run_profile(
    n_e_1e20: float, chi: float, P_ext_MW: float, *, n_cells: int = 40
) -> NDArray[np.float64]:
    """Profilo T(r) a regime, ricampionato su RADIAL_GRID (r/a in [0,1])."""
    s = TransportSolver1D(
        a=A_MINOR, R0=R0, n_e=n_e_1e20 * 1e20, chi=chi, T_edge=0.1, n_cells=n_cells
    )
    shape = np.exp(-((s.r / 0.6) ** 2))
    p_ext = s.heating_density_for_power(P_ext_MW * 1e6, shape)
    s.solve_steady_state(p_ext=p_ext, dt=2e-2, tol=1e-4, max_steps=5000)
    return np.interp(RADIAL_GRID * A_MINOR, s.r, s.T)


def generate_profile_dataset(
    n_samples: int, seed: int = 0, n_cells: int = 40
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Genera (X, Y): X (n,3) parametri, Y (n, 32) profili T(r)."""
    X = sample_parameters(n_samples, seed)
    Y = np.array([run_profile(*row, n_cells=n_cells) for row in X])
    return X, Y


def _to_tensor(arr: NDArray[np.float64]) -> torch.Tensor:
    """numpy -> tensor passando per liste (evita il ponte numpy<->torch, che e'
    incompatibile tra torch 2.2 e numpy 2.x su questa piattaforma)."""
    return torch.tensor(np.asarray(arr, dtype=np.float64).tolist(), dtype=torch.float32)


def _to_numpy(t: torch.Tensor) -> NDArray[np.float64]:
    """tensor -> numpy passando per liste (stesso motivo)."""
    return np.array(t.detach().tolist(), dtype=np.float64)


class _ProfileNet(nn.Module):
    """MLP: 3 parametri -> profilo (N_PROFILE_POINTS)."""

    def __init__(self, n_in: int = 3, n_out: int = N_PROFILE_POINTS, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ProfileEmulator:
    """Rete addestrata + statistiche di normalizzazione."""

    net: _ProfileNet
    x_mean: NDArray[np.float64]
    x_std: NDArray[np.float64]
    y_mean: NDArray[np.float64]
    y_std: NDArray[np.float64]
    radial_grid: NDArray[np.float64]

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predice i profili T(r) per i parametri X (n,3) -> (n, 32)."""
        Xs = (np.atleast_2d(X) - self.x_mean) / self.x_std
        self.net.eval()
        with torch.no_grad():
            ys = _to_numpy(self.net(_to_tensor(Xs)))
        return ys * self.y_std + self.y_mean


def train_profile_emulator(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    *,
    epochs: int = 1500,
    lr: float = 5e-3,
    seed: int = 0,
) -> ProfileEmulator:
    """Addestra l'MLP (input e output normalizzati, Adam, loss MSE)."""
    torch.manual_seed(seed)
    x_mean, x_std = X.mean(0), X.std(0)
    y_mean, y_std = Y.mean(0), Y.std(0)
    Xs = _to_tensor((X - x_mean) / x_std)
    Ys = _to_tensor((Y - y_mean) / y_std)

    net = _ProfileNet(n_in=X.shape[1], n_out=Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(Xs), Ys)
        loss.backward()
        opt.step()

    return ProfileEmulator(net, x_mean, x_std, y_mean, y_std, RADIAL_GRID.copy())


def profile_rmse(
    emulator: ProfileEmulator, X: NDArray[np.float64], Y: NDArray[np.float64]
) -> float:
    """RMSE sul profilo [keV] (radice della media degli errori quadratici)."""
    pred = emulator.predict(X)
    return float(np.sqrt(np.mean((pred - Y) ** 2)))


def profile_r2(
    emulator: ProfileEmulator, X: NDArray[np.float64], Y: NDArray[np.float64]
) -> float:
    """R^2 globale sui profili appiattiti."""
    pred = emulator.predict(X)
    ss_res = np.sum((Y - pred) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)
