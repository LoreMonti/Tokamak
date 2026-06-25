"""Validazione del modello di radiazione da impurità.

I valori ASSOLUTI della funzione di raffreddamento sono schematici, quindi
testiamo soprattutto comportamenti RELATIVI e formule esatte (Z_eff).
"""

from __future__ import annotations

import pytest

from tokamak.radiation import (
    cooling_function,
    max_impurity_fraction,
    total_radiated_power_density,
    z_eff_from_fractions,
)


def test_zeff_pure_plasma_is_one():
    """Senza impurità Z_eff = 1."""
    assert z_eff_from_fractions({}) == 1.0


def test_zeff_known_mixture():
    r"""Z_eff esatto per una frazione nota di carbonio.

    c_C = 0.01, Z=6: n_main/n_e = 1 - 0.06 = 0.94;
    Z_eff = 0.94 + 0.01*36 = 0.94 + 0.36 = 1.30.
    """
    assert z_eff_from_fractions({"C": 0.01}) == pytest.approx(1.30, rel=1e-9)


def test_zeff_high_z_raises_fast():
    """A parità di frazione, il tungsteno alza Z_eff molto più del carbonio."""
    assert z_eff_from_fractions({"W": 1e-3}) > z_eff_from_fractions({"C": 1e-3})


def test_cooling_function_scales_as_Z_cubed():
    """L_z ~ Z^3: il rapporto W/C deve essere ~ (74/6)^3."""
    ratio = cooling_function(1.0, "W") / cooling_function(1.0, "C")
    assert ratio == pytest.approx((74 / 6) ** 3, rel=1e-6)


def test_cooling_function_decreases_with_temperature():
    """La radiazione di linea (schematica) cala con T."""
    assert cooling_function(0.5, "Ne") > cooling_function(5.0, "Ne")


def test_radiated_power_increases_with_impurity():
    """Più impurità -> più radiazione."""
    p0 = total_radiated_power_density(1e20, 10.0, {"Ar": 1e-3})
    p1 = total_radiated_power_density(1e20, 10.0, {"Ar": 5e-3})
    assert p1 > p0


def test_high_z_collapses_at_lower_concentration():
    """Il collasso radiativo arriva a concentrazione MOLTO più bassa per il
    tungsteno che per il carbonio: meno tungsteno basta a spegnere il plasma."""
    n_e, T, heating = 1e20, 10.0, 5e5  # W/m^3
    c_W = max_impurity_fraction("W", n_e, T, heating)
    c_C = max_impurity_fraction("C", n_e, T, heating)
    assert 0.0 < c_W < c_C
