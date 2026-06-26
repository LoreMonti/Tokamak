"""Validazione del classificatore di disruption."""

from __future__ import annotations

import numpy as np

from tokamak.disruption import (
    disruption_probability,
    evaluate_classifier,
    generate_labeled_dataset,
    sample_parameters,
    train_disruption_classifier,
)


def test_probability_in_unit_interval():
    X = sample_parameters(200, seed=1)
    p = disruption_probability(X)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_high_density_is_more_dangerous():
    """A parita' del resto, densita' vicina/oltre Greenwald -> disruption piu'
    probabile (un punto a bassa densita' e' molto piu' sicuro)."""
    safe = np.array([[0.4, 12.0, 15.0, 5.3]])      # bassa densita'
    risky = np.array([[1.5, 12.0, 15.0, 5.3]])     # alta densita' (vicino n_G)
    assert disruption_probability(risky)[0] > disruption_probability(safe)[0]


def test_dataset_has_both_classes():
    X, y = generate_labeled_dataset(400, seed=2)
    assert X.shape == (400, 4)
    assert set(np.unique(y)).issubset({0, 1})
    assert 0 < y.sum() < len(y)  # presenti sia stabili sia disrupt


def test_classifier_separates_well_on_holdout():
    """Il classificatore deve avere ROC-AUC alto su dati mai visti."""
    X, y = generate_labeled_dataset(1500, seed=3)
    clf = train_disruption_classifier(X[:1100], y[:1100])
    score = evaluate_classifier(clf, X[1100:], y[1100:])
    assert score.roc_auc > 0.9
    assert score.accuracy > 0.85


def test_classifier_flags_clearly_unsafe_point():
    """Un punto ben oltre i limiti e' classificato come disruption quasi certa."""
    X, y = generate_labeled_dataset(1200, seed=4)
    clf = train_disruption_classifier(X, y)
    unsafe = np.array([[1.6, 28.0, 8.0, 4.0]])  # densita' e beta entrambi alti
    assert clf.predict_proba(unsafe)[0, 1] > 0.8
