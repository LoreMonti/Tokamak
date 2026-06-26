"""Validazione dell'emulatore deep-learning dei profili (PyTorch).

Se PyTorch non e' installato i test vengono saltati (dipendenza opzionale [ml]).
Dataset piccoli e pochi epoch per restare veloci.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from tokamak.profile_emulator import (  # noqa: E402
    N_PROFILE_POINTS,
    generate_profile_dataset,
    profile_r2,
    profile_rmse,
    run_profile,
    train_profile_emulator,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_profile_dataset(140, seed=3, n_cells=30)


@pytest.fixture(scope="module")
def trained(dataset):
    X, Y = dataset
    emu = train_profile_emulator(X[:110], Y[:110], epochs=1500, seed=0)
    return emu, X[110:], Y[110:]


def test_profile_is_peaked_and_positive():
    """Un profilo del solver e' positivo e piccato al centro (r=0)."""
    prof = run_profile(1.0, 1.0, 20.0, n_cells=30)
    assert prof.shape == (N_PROFILE_POINTS,)
    assert np.all(prof > 0)
    assert prof[0] > prof[-1]  # centro piu' caldo del bordo


def test_dataset_shapes(dataset):
    X, Y = dataset
    assert X.shape == (140, 3)
    assert Y.shape == (140, N_PROFILE_POINTS)


def test_emulator_accurate_on_holdout(trained):
    """L'emulatore riproduce i profili su dati mai visti (R^2 alto, RMSE basso)."""
    emu, X_te, Y_te = trained
    assert profile_r2(emu, X_te, Y_te) > 0.98
    assert profile_rmse(emu, X_te, Y_te) < 0.5  # keV


def test_predict_shape(trained):
    emu, X_te, _ = trained
    pred = emu.predict(X_te[:5])
    assert pred.shape == (5, N_PROFILE_POINTS)


def test_emulator_monotonic_in_heating(trained):
    """Piu' riscaldamento -> profilo previsto piu' caldo (in media)."""
    emu, _, _ = trained
    low = emu.predict(np.array([[1.0, 1.0, 8.0]]))[0].mean()
    high = emu.predict(np.array([[1.0, 1.0, 28.0]]))[0].mean()
    assert high > low
