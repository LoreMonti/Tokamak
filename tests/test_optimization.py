"""Validazione dell'ottimizzazione del punto operativo."""

from __future__ import annotations

from dataclasses import replace

from tokamak.engineering import TokamakConfig
from tokamak.optimization import optimize_operating_point


def test_optimum_succeeds_and_respects_constraints():
    """L'ottimizzazione converge e rispetta i vincoli (con piccola tolleranza)."""
    opt = optimize_operating_point(TokamakConfig())
    assert opt.success
    assert opt.greenwald_fraction <= 1.001
    assert opt.beta <= opt.beta_limit * 1.001


def test_optimum_sits_on_a_constraint_boundary():
    """L'ottimo deve saturare almeno un vincolo (P_fus cresce con n e T)."""
    opt = optimize_operating_point(TokamakConfig())
    assert opt.greenwald_active or opt.troyon_active


def test_higher_current_allows_more_fusion_power():
    """Aumentando I_p si alzano sia il limite di Greenwald sia quello di Troyon,
    quindi il punto ottimo raggiunge una potenza di fusione maggiore."""
    base = TokamakConfig(plasma_current_MA=12.0)
    high = replace(base, plasma_current_MA=18.0)
    p_base = optimize_operating_point(base).fusion_power_density
    p_high = optimize_operating_point(high).fusion_power_density
    assert p_high > p_base


def test_stronger_field_allows_more_fusion_power():
    """Un campo toroidale piu' forte alza il limite di beta (a parita' di
    pressione) -> consente piu' densita'/temperatura -> piu' fusione."""
    base = TokamakConfig(B_toroidal_T=4.0)
    strong = replace(base, B_toroidal_T=6.0)
    p_base = optimize_operating_point(base).fusion_power_density
    p_strong = optimize_operating_point(strong).fusion_power_density
    assert p_strong > p_base
