"""Controllo del plasma con RL: agente PPO vs PID, con reiezione del disturbo.

Addestra un agente PPO a regolare il riscaldamento per tenere T al target, poi
lo confronta col PID (Fase 4A) sulla STESSA dinamica e sullo stesso scenario
(con un degrado di confinamento a meta' episodio). Mostra T(t) e l'azione per
entrambi, e l'errore quadratico medio.

Risultato riportato onestamente: che l'RL batta o pareggi il PID, e' un
confronto istruttivo (il PID e' gia' molto buono su un sistema cosi' semplice).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from tokamak.rl_control import (  # noqa: E402
    FusionControlEnv,
    pid_policy,
    rl_policy,
    run_episode,
    train_ppo,
)

DOCS = Path(__file__).resolve().parent.parent / "docs"
PID_GAINS = dict(kp=2e5, ki=1e5, kd=2e3)


def _rmse_to_target(T: np.ndarray, target: float) -> float:
    return float(np.sqrt(np.mean((T - target) ** 2)))


def main() -> None:
    t0 = time.perf_counter()
    print("Addestramento PPO (puo' richiedere qualche minuto)...")
    # action_cost basso: l'agente non "baratta" precisione per risparmio di
    # potenza, quindi insegue il target piu' da vicino (vedi README).
    model = train_ppo(timesteps=80000, seed=0, disturb_step=100, action_cost=1e-4)
    print(f"Addestrato in {time.perf_counter() - t0:.0f} s")

    # Scenario di valutazione identico per entrambi (stesso seed -> stessa T0).
    seed = 7
    env_rl = FusionControlEnv(disturb_step=100, n_steps=200)
    env_pid = FusionControlEnv(disturb_step=100, n_steps=200)
    res_rl = run_episode(env_rl, rl_policy(model), seed=seed)
    res_pid = run_episode(env_pid, pid_policy(env_pid, **PID_GAINS), seed=seed)
    target = env_rl.T_target
    t_axis = np.arange(env_rl.n_steps) * env_rl.dt
    t_dist = 100 * env_rl.dt

    rmse_rl = _rmse_to_target(res_rl["T"], target)
    rmse_pid = _rmse_to_target(res_pid["T"], target)
    print(f"RMSE al target  ->  RL: {rmse_rl:.3f} keV  |  PID: {rmse_pid:.3f} keV")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    a1.plot(t_axis, res_pid["T"], color="navy", lw=2, label=f"PID (RMSE {rmse_pid:.2f})")
    a1.plot(t_axis, res_rl["T"], color="crimson", lw=2, label=f"RL/PPO (RMSE {rmse_rl:.2f})")
    a1.axhline(target, color="black", ls="--", lw=1, label="target")
    a1.axvline(t_dist, color="gray", ls=":", lw=1)
    a1.set_ylabel("T [keV]")
    a1.set_title("Controllo della temperatura: RL (PPO) vs PID")
    a1.legend(loc="lower right")
    a1.grid(True, alpha=0.3)

    a2.plot(t_axis, res_pid["action"], color="navy", lw=1.5, label="PID")
    a2.plot(t_axis, res_rl["action"], color="crimson", lw=1.5, label="RL/PPO")
    a2.axvline(t_dist, color="gray", ls=":", lw=1, label="disturbo (τ_E ↓)")
    a2.set_xlabel("tempo [s]")
    a2.set_ylabel("azione (P_ext / P_max)")
    a2.legend(loc="upper right")
    a2.grid(True, alpha=0.3)

    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "rl_control.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
