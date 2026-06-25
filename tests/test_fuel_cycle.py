"""Validazione del ciclo del combustibile (trizio)."""

from __future__ import annotations

import numpy as np
import pytest

from tokamak.fuel_cycle import (
    doubling_time_years,
    required_TBR,
    simulate_inventory,
    tritium_burn_rate_atoms,
    tritium_burn_rate_kg_per_day,
)


def test_burn_rate_scales_linearly_with_power():
    assert tritium_burn_rate_atoms(2e9) == pytest.approx(2 * tritium_burn_rate_atoms(1e9))


def test_burn_rate_magnitude_half_kg_per_day():
    """Un reattore da ~3 GW di fusione brucia ~0.46 kg/giorno di trizio."""
    assert tritium_burn_rate_kg_per_day(3e9) == pytest.approx(0.46, abs=0.03)


def test_required_TBR_above_one():
    """Il TBR richiesto supera 1 per via di decadimento e perdite."""
    assert required_TBR(3e9, inventory_kg=2.0) > 1.0


def test_breeding_above_one_grows_inventory():
    """TBR > 1 -> l'inventario di trizio cresce nel tempo."""
    h = simulate_inventory(N0_kg=1.0, P_fusion_W=3e9, TBR=1.10, t_end_years=5.0)
    assert h.inventory_kg[-1] > h.inventory_kg[0]


def test_breeding_below_one_depletes_inventory():
    """TBR < 1 -> l'inventario cala (verso l'esaurimento)."""
    h = simulate_inventory(N0_kg=2.0, P_fusion_W=3e9, TBR=0.95, t_end_years=5.0)
    assert h.inventory_kg[-1] < h.inventory_kg[0]


def test_doubling_time_infinite_without_surplus():
    """Senza surplus (TBR <= 1) non si accumula mai l'inventario di avvio."""
    assert np.isinf(doubling_time_years(1.0, 3e9, startup_inventory_kg=5.0))


def test_doubling_time_decreases_with_breeding():
    """Piu' alto il TBR, piu' breve il doubling time."""
    t_low = doubling_time_years(1.05, 3e9, startup_inventory_kg=3.0)
    t_high = doubling_time_years(1.20, 3e9, startup_inventory_kg=3.0)
    assert t_high < t_low
