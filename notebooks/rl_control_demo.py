"""Controllo del plasma con RL: due agenti PPO vs PID, con reiezione del disturbo.

Esperimento controllato su tre controllori, sulla STESSA dinamica e scenario
(disturbo: degrado di confinamento a meta' episodio):

- PID (Fase 4A): regolatore classico, con termine integrale;
- RL "energy-aware" (action_cost alto): premiato anche per risparmiare potenza;
- RL "precisione" (action_cost ~ 0): premiato quasi solo per stare sul target.

Domanda: senza dargli la fisica, l'RL "precisione" batte il PID? E quanto
conta il reward shaping (confronto coi due RL)?

Risultato riportato onestamente, qualunque sia.
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


def _rmse(T: np.ndarray, target: float) -> float:
    return float(np.sqrt(np.mean((T - target) ** 2)))


def main() -> None:
    seed_eval = 7
    timesteps = 80000

    t0 = time.perf_counter()
    print("Addestramento RL 'energy-aware' (action_cost alto)...")
    model_eco = train_ppo(timesteps=timesteps, seed=0, disturb_step=100,
                          action_cost=1e-3, progress=True)
    print("Addestramento RL 'precisione' (action_cost ~ 0)...")
    model_pre = train_ppo(timesteps=timesteps, seed=0, disturb_step=100,
                          action_cost=1e-6, progress=True)
    print(f"Due agenti addestrati in {time.perf_counter() - t0:.0f} s")

    # Valutazione: stesso scenario per tutti (stesso seed -> stessa T0).
    def episode(make_act):
        env = FusionControlEnv(disturb_step=100, n_steps=200)
        return run_episode(env, make_act(env), seed=seed_eval)

    res_pid = episode(lambda env: pid_policy(env, **PID_GAINS))
    res_eco = episode(lambda env: rl_policy(model_eco))
    res_pre = episode(lambda env: rl_policy(model_pre))
    target = 10.0
    dt = FusionControlEnv().dt
    t_axis = np.arange(200) * dt
    t_dist = 100 * dt

    rmse = {k: _rmse(r["T"], target) for k, r in
            [("PID", res_pid), ("RL eco", res_eco), ("RL precisione", res_pre)]}
    print("RMSE al target [keV]:", {k: round(v, 3) for k, v in rmse.items()})

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)

    a1.plot(t_axis, res_pid["T"], color="navy", lw=2,
            label=f"PID (RMSE {rmse['PID']:.2f})")
    a1.plot(t_axis, res_eco["T"], color="crimson", lw=2,
            label=f"RL energy-aware (RMSE {rmse['RL eco']:.2f})")
    a1.plot(t_axis, res_pre["T"], color="green", lw=2,
            label=f"RL precision (RMSE {rmse['RL precisione']:.2f})")
    a1.axhline(target, color="black", ls="--", lw=1, label="target")
    a1.axvline(t_dist, color="gray", ls=":", lw=1)
    a1.set_ylabel("T [keV]")
    a1.set_title("Temperature control: two RL agents vs PID")
    a1.legend(loc="lower right", fontsize=8)
    a1.grid(True, alpha=0.3)

    a2.plot(t_axis, res_pid["action"], color="navy", lw=1.3)
    a2.plot(t_axis, res_eco["action"], color="crimson", lw=1.3)
    a2.plot(t_axis, res_pre["action"], color="green", lw=1.3)
    a2.axvline(t_dist, color="gray", ls=":", lw=1, label="disturbance (τ_E ↓)")
    a2.set_xlabel("time [s]")
    a2.set_ylabel("action (P_ext / P_max)")
    a2.legend(loc="upper right", fontsize=8)
    a2.grid(True, alpha=0.3)

    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "rl_control.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
