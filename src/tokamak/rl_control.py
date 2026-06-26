r"""Fase 16 — Controllo del plasma con Reinforcement Learning.

Sulla scia di DeepMind/TCV (Nature 2022), addestriamo un agente RL a regolare la
potenza di riscaldamento per mantenere la temperatura a un target, e lo
confrontiamo col PID fatto a mano (Fase 4A) sulla STESSA dinamica.

Dinamica (0D, scalare) dell'energia del plasma:

    dU/dt = P_alpha(T) + P_ext - P_brem(T) - U/tau_E,   U = 3 n T

L'ambiente (gymnasium) espone: stato = (T normalizzata, errore al target),
azione = potenza di riscaldamento in [0, P_max], reward = -errore^2 - costo
azione. Una perturbazione (degrado di tau_E) a meta' episodio mette alla prova
la reiezione del disturbo.

Richiede `gymnasium` (e `stable-baselines3` per PPO): dipendenze opzionali [rl].
Il modulo NON e' importato da __init__, cosi' `import tokamak` resta possibile.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .constants import KEV_TO_JOULE
from .power_balance import (
    alpha_power_density,
    bremsstrahlung_power_density,
    stored_energy_density,
)


class PlasmaTempDynamics:
    """Dinamica 0D scalare della temperatura (controllabile via P_ext)."""

    def __init__(self, n_e: float = 1.0e20, tau_E: float = 2.0, z_eff: float = 1.0):
        self.n_e = n_e
        self.tau_E = tau_E
        self.z_eff = z_eff

    def _dTdt(self, T_keV: float, p_ext_W: float) -> float:
        T = max(T_keV, 1e-3)
        p_alpha = float(alpha_power_density(self.n_e, T))
        p_brem = float(bremsstrahlung_power_density(self.n_e, T, self.z_eff))
        U = float(stored_energy_density(self.n_e, T))  # 3 n T  [J/m^3]
        dU = p_alpha + p_ext_W - p_brem - U / self.tau_E
        return dU / (3.0 * self.n_e * KEV_TO_JOULE)

    def step(self, T_keV: float, p_ext_W: float, dt: float, substeps: int = 5) -> float:
        """Avanza T di dt con RK4 (P_ext costante sul passo)."""
        h = dt / substeps
        T = T_keV
        for _ in range(substeps):
            k1 = self._dTdt(T, p_ext_W)
            k2 = self._dTdt(T + 0.5 * h * k1, p_ext_W)
            k3 = self._dTdt(T + 0.5 * h * k2, p_ext_W)
            k4 = self._dTdt(T + h * k3, p_ext_W)
            T += h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return max(T, 1e-3)


class FusionControlEnv(gym.Env):
    """Ambiente RL: regolare P_ext per tenere T al target.

    Stato:  [T/T_ref, (T_target - T)/T_ref]   (T_ref = 20 keV per normalizzare)
    Azione: a in [0,1] -> P_ext = a * p_max
    Reward: -((T - T_target)/T_target)^2 - action_cost * a^2
    """

    metadata: dict = {}

    def __init__(
        self,
        *,
        T_target: float = 10.0,
        p_max: float = 1.0e6,
        dt: float = 0.05,
        n_steps: int = 200,
        action_cost: float = 1e-3,
        disturb_step: int | None = 100,
        seed: int | None = None,
    ):
        super().__init__()
        self.T_target = T_target
        self.p_max = p_max
        self.dt = dt
        self.n_steps = n_steps
        self.action_cost = action_cost
        self.disturb_step = disturb_step
        self._T_ref = 20.0
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.dyn = PlasmaTempDynamics()
        self.T = T_target
        self.k = 0

    def _obs(self) -> NDArray[np.float32]:
        return np.array(
            [self.T / self._T_ref, (self.T_target - self.T) / self._T_ref],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.T = float(self._rng.uniform(3.0, 15.0))  # partenza casuale
        self.dyn = PlasmaTempDynamics(tau_E=2.0)
        self.k = 0
        return self._obs(), {}

    def step(self, action: NDArray[np.float32]):
        a = float(np.clip(action[0], 0.0, 1.0))
        # Disturbo: degrado del confinamento a meta' episodio.
        if self.disturb_step is not None and self.k == self.disturb_step:
            self.dyn.tau_E = 1.0  # tau_E 2.0 -> 1.0 (confinamento dimezzato)

        self.T = self.dyn.step(self.T, a * self.p_max, self.dt)
        self.k += 1

        err = (self.T - self.T_target) / self.T_target
        reward = -(err**2) - self.action_cost * a**2
        terminated = False
        truncated = self.k >= self.n_steps
        return self._obs(), float(reward), terminated, truncated, {}


# --- Esecuzione di un episodio con un controllore qualsiasi ----------------
def run_episode(env: FusionControlEnv, act_fn, seed: int = 0) -> dict:
    """Esegue un episodio applicando act_fn(obs) -> azione. Registra T e azione."""
    obs, _ = env.reset(seed=seed)
    T_hist: list[float] = []
    a_hist: list[float] = []
    total_reward = 0.0
    done = False
    while not done:
        action = np.asarray(act_fn(obs), dtype=np.float32).reshape(1)
        obs, reward, terminated, truncated, _ = env.step(action)
        T_hist.append(env.T)
        a_hist.append(float(action[0]))
        total_reward += reward
        done = terminated or truncated
    return {
        "T": np.array(T_hist),
        "action": np.array(a_hist),
        "total_reward": total_reward,
    }


def pid_policy(env: FusionControlEnv, kp: float, ki: float, kd: float):
    """Crea un act_fn basato sul PID della Fase 4A (riuso del PIDController)."""
    from .control import PIDController

    pid = PIDController(kp=kp, ki=ki, kd=kd, setpoint=env.T_target,
                        output_min=0.0, output_max=env.p_max)
    pid.reset()

    def act_fn(obs: NDArray[np.float32]) -> NDArray[np.float32]:
        T = float(obs[0]) * env._T_ref
        p_ext = pid.update(T, env.dt)
        return np.array([p_ext / env.p_max], dtype=np.float32)

    return act_fn


def rl_policy(model):
    """Crea un act_fn dalla policy RL addestrata (azione deterministica)."""
    def act_fn(obs: NDArray[np.float32]) -> NDArray[np.float32]:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(1)

    return act_fn


def train_ppo(timesteps: int = 60000, seed: int = 0, **env_kwargs):
    """Addestra un agente PPO (stable-baselines3) sull'ambiente di controllo."""
    from stable_baselines3 import PPO

    env = FusionControlEnv(**env_kwargs)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model
