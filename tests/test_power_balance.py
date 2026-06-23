"""Validazione del bilancio di potenza 0D.

Oltre ai test unitari, includiamo test che verificano comportamenti FISICI noti
e una validazione con parametri tipo ITER.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokamak.constants import ALPHA_HEATING_FRACTION
from tokamak.power_balance import (
    alpha_power_density,
    fusion_gain_Q,
    fusion_power_density,
    heating_power_required,
    triple_product_ignition,
)


def test_alpha_is_one_fifth_of_fusion():
    """Le alfa portano ~1/5 dell'energia di fusione (3.52 / 17.59 MeV)."""
    n, T = 1e20, 15.0
    ratio = alpha_power_density(n, T) / fusion_power_density(n, T)
    assert ratio == pytest.approx(ALPHA_HEATING_FRACTION)
    assert 0.19 < ratio < 0.21


def test_fusion_power_scales_as_density_squared():
    """P_fus ~ n^2 (prodotto n_D * n_T): raddoppiando n, P_fus quadruplica."""
    T = 15.0
    p1 = fusion_power_density(1e20, T)
    p2 = fusion_power_density(2e20, T)
    assert p2 / p1 == np.float64(4.0)


def test_ignition_gives_infinite_Q():
    """Con confinamento eccellente il plasma e' ignito: P_heat<=0 => Q=inf."""
    q = fusion_gain_Q(n_e=1.5e20, T_keV=20.0, tau_e=10.0)
    assert np.isinf(q)


def test_bremsstrahlung_prevents_ignition_at_low_T():
    """Sotto una T minima il Bremsstrahlung domina: nessuna ignition possibile."""
    # A 3 keV il termine radiativo supera quello di fusione -> triplo prodotto inf.
    assert np.isinf(triple_product_ignition(3.0))
    # A 15 keV l'ignition e' invece possibile (valore finito).
    assert np.isfinite(triple_product_ignition(15.0))


def test_iter_like_parameters_give_high_gain():
    """Validazione: parametri rappresentativi di ITER devono dare Q elevato.

    ITER punta a Q ~ 10 con n ~ 1e20 m^-3, T ~ 15 keV, tau_E ~ pochi secondi.
    Un modello 0D ottimistico (profili piatti) tende a sovrastimare, quindi
    richiediamo solo che il guadagno sia chiaramente alto (Q > 5).
    """
    q = fusion_gain_Q(n_e=1.0e20, T_keV=15.0, tau_e=2.0)
    assert q > 5.0


def test_triple_product_order_of_magnitude_at_15_keV():
    """Il triplo prodotto di ignition a ~15 keV e' ~ qualche 1e21 keV*s/m^3."""
    ntt = triple_product_ignition(15.0)
    assert 1e21 < ntt < 1e22


def test_heating_required_positive_for_marginal_confinement():
    """Con confinamento scarso serve riscaldamento esterno (P_heat > 0)."""
    assert heating_power_required(n_e=1e20, T_keV=10.0, tau_e=0.5) > 0
