"""Validazione del controllo RL (ambiente, baseline PID, smoke test PPO).

Saltati se gymnasium/stable-baselines3 non sono installati (dipendenze [rl]).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from tokamak.rl_control import (  # noqa: E402
    FusionControlEnv,
    pid_policy,
    run_episode,
)

PID_GAINS = dict(kp=2e5, ki=1e5, kd=2e3)


def test_env_reset_and_step_api():
    env = FusionControlEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (2,)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    assert obs.shape == (2,)
    assert isinstance(reward, float)
    assert not terminated


def test_pid_stabilizes_to_target():
    """Il PID porta la temperatura al target (senza disturbo)."""
    env = FusionControlEnv(disturb_step=None)
    res = run_episode(env, pid_policy(env, **PID_GAINS), seed=1)
    assert abs(res["T"][-1] - env.T_target) < 0.3


def test_pid_beats_no_control():
    """Il PID ottiene un reward molto migliore del non-controllo (azione nulla)."""
    env_pid = FusionControlEnv(disturb_step=None)
    r_pid = run_episode(env_pid, pid_policy(env_pid, **PID_GAINS), seed=2)
    env_zero = FusionControlEnv(disturb_step=None)
    r_zero = run_episode(env_zero, lambda obs: np.array([0.0]), seed=2)
    assert r_pid["total_reward"] > r_zero["total_reward"]


def test_pid_rejects_disturbance():
    """Con il disturbo (tau_E dimezzato a meta' episodio) il PID recupera il target."""
    env = FusionControlEnv(disturb_step=100, n_steps=200)
    res = run_episode(env, pid_policy(env, **PID_GAINS), seed=3)
    assert abs(res["T"][-1] - env.T_target) < 0.5


def test_ppo_trains_and_predicts():
    """Smoke test: PPO si addestra brevemente e predice azioni valide."""
    sb3 = pytest.importorskip("stable_baselines3")
    env = FusionControlEnv()
    model = sb3.PPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=0, verbose=0)
    model.learn(total_timesteps=200)
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1,)
    assert 0.0 <= float(np.clip(action[0], 0, 1)) <= 1.0
