"""Validazione dell'emulatore ML del solver di trasporto.

Il solver e' lento, quindi generiamo UN dataset condiviso (fixture di modulo) e
lo riusiamo in tutti i test: verifica shape, accuratezza, coerenza fisica e
speed-up senza rieseguire il solver piu' volte.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tokamak.surrogate import (
    generate_dataset,
    r2_per_target,
    run_transport_point,
    sample_parameters,
    train_surrogate,
)


@pytest.fixture(scope="module")
def dataset():
    """Dataset condiviso, generato una sola volta per tutto il modulo."""
    return generate_dataset(120, seed=3, n_cells=30)


@pytest.fixture(scope="module")
def trained(dataset):
    X, Y = dataset
    return train_surrogate(X[:95], Y[:95]), X[95:], Y[95:]


def test_sampling_within_ranges():
    X = sample_parameters(50, seed=1)
    assert X.shape == (50, 3)
    assert (X[:, 0] >= 0.4).all() and (X[:, 0] <= 1.2).all()


def test_dataset_shapes(dataset):
    X, Y = dataset
    assert X.shape == (120, 3)
    assert Y.shape == (120, 2)
    assert np.all(Y > 0)  # tau_E e T0 positivi


def test_surrogate_is_accurate_on_holdout(trained):
    """L'emulatore deve riprodurre il solver su dati mai visti (R^2 alto)."""
    model, X_te, Y_te = trained
    score = r2_per_target(model, X_te, Y_te)
    assert score.r2_tau_E > 0.85
    assert score.r2_T0 > 0.85


def test_surrogate_is_much_faster_than_solver(trained):
    """La predizione dell'emulatore deve essere ordini di grandezza piu' rapida."""
    model, _, _ = trained

    t0 = time.perf_counter()
    run_transport_point(1.0, 1.0, 20.0, n_cells=30)
    t_solver = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.predict([[1.0, 1.0, 20.0]])
    t_surrogate = time.perf_counter() - t0

    assert t_surrogate < t_solver / 10.0


def test_surrogate_respects_monotonicity(trained):
    """Piu' riscaldamento esterno -> T centrale prevista piu' alta."""
    model, _, _ = trained
    low = model.predict([[1.0, 1.0, 8.0]])[0, 1]
    high = model.predict([[1.0, 1.0, 28.0]])[0, 1]
    assert high > low
