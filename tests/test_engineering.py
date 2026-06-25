"""Validazione dei vincoli ingegneristici contro i parametri noti di ITER.

ITER (riferimento): a=2.0 m, R0=6.2 m, I_p=15 MA, B_t=5.3 T, n_e~1e20,
T~8-15 keV (media), beta~2.5%. Verifichiamo che il modello collochi questi
valori dentro i limiti, come deve essere per una macchina progettata bene.
"""

from __future__ import annotations

import pytest

from tokamak.engineering import (
    TokamakConfig,
    check_operational_limits,
    greenwald_density,
    greenwald_fraction,
    plasma_beta,
    troyon_beta_limit,
)

ITER = TokamakConfig()


def test_greenwald_density_iter_value():
    """n_G = 15 / (pi * 2^2) ~ 1.19e20 m^-3 per ITER."""
    n_G = greenwald_density(ITER.plasma_current_MA, ITER.minor_radius_m)
    assert n_G == pytest.approx(1.19e20, rel=0.02)


def test_iter_operates_below_greenwald():
    """A n_e=1e20 ITER e' sotto Greenwald (frazione ~0.84 < 1)."""
    f_G = greenwald_fraction(1.0e20, ITER.plasma_current_MA, ITER.minor_radius_m)
    assert 0.8 < f_G < 0.9


def test_troyon_limit_iter_order_of_magnitude():
    """beta_max = 2.5 * 15 / (2 * 5.3) ~ 3.5%."""
    beta_max = troyon_beta_limit(
        ITER.beta_N, ITER.plasma_current_MA, ITER.minor_radius_m, ITER.B_toroidal_T
    )
    assert beta_max == pytest.approx(0.0354, rel=0.02)


def test_iter_beta_below_troyon_limit():
    """Con T MEDIA sul volume (~8 keV) e n=1e20, beta ITER ~2.3% < Troyon ~3.5%.

    Nota: il beta di Troyon usa la pressione mediata sul volume, quindi qui si
    usa la temperatura media (non quella di picco al centro, piu' alta).
    """
    beta = plasma_beta(1.0e20, 8.0, ITER.B_toroidal_T)
    beta_max = troyon_beta_limit(
        ITER.beta_N, ITER.plasma_current_MA, ITER.minor_radius_m, ITER.B_toroidal_T
    )
    assert beta < beta_max
    assert 0.015 < beta < 0.030


def test_beta_scales_with_pressure():
    """beta ~ n*T: raddoppiando la temperatura, beta raddoppia."""
    b1 = plasma_beta(1e20, 10.0, 5.0)
    b2 = plasma_beta(1e20, 20.0, 5.0)
    assert b2 == pytest.approx(2.0 * b1)


def test_operational_check_iter_all_within_limits():
    """Punto operativo realistico (T media, divertore mitigato): tutto ok.

    La potenza al divertore (10 MW) e' quella RESIDUA dopo mitigazione
    radiativa; il carico non mitigato sarebbe molto piu' alto.
    """
    check = check_operational_limits(
        ITER, n_e=1.0e20, T_keV=8.0, power_to_divertor_W=10e6
    )
    assert check.greenwald_ok
    assert check.beta_ok
    assert check.divertor_ok
    assert check.all_ok


def test_operational_check_flags_density_violation():
    """A densita' eccessiva il limite di Greenwald deve scattare."""
    check = check_operational_limits(
        ITER, n_e=3.0e20, T_keV=8.0, power_to_divertor_W=10e6
    )
    assert not check.greenwald_ok
    assert not check.all_ok


def test_operational_check_flags_divertor_overload():
    """Carico al divertore non mitigato (100 MW): il limite termico scatta."""
    check = check_operational_limits(
        ITER, n_e=1.0e20, T_keV=8.0, power_to_divertor_W=100e6
    )
    assert not check.divertor_ok
    assert not check.all_ok
