"""Validazione del modello di combustione auto-consistente."""

from __future__ import annotations

import numpy as np

from tokamak.burn import simulate_burn, z_effective


def test_zeff_pure_dt_is_one():
    """Plasma D-T puro (niente elio) -> Z_eff = 1."""
    assert z_effective(n_DT=1e20, n_He=0.0, n_e=1e20) == 1.0


def test_zeff_increases_with_helium():
    """La cenere di elio (Z=2) alza Z_eff sopra 1."""
    n_He = 1e19
    n_DT = 1e20
    n_e = n_DT + 2 * n_He
    assert z_effective(n_DT, n_He, n_e) > 1.0


def test_ash_production_conserves_reaction_count():
    """Senza rifornimento e con cenere intrappolata (tau_p enorme), ogni
    reazione produce 1 elio e consuma 2 nuclei di combustibile:

        Delta n_He = -0.5 * Delta n_DT
    """
    s = simulate_burn(
        n_DT0=1e20, n_He0=0.0, T0_keV=15.0, tau_E=3.0, tau_p=1e12,
        fueling=0.0, p_ext=0.0, t_end=5.0,
    )
    d_He = s.n_He[-1] - s.n_He[0]
    d_DT = s.n_DT[-1] - s.n_DT[0]
    assert d_He > 0 and d_DT < 0
    assert np.isclose(d_He, -0.5 * d_DT, rtol=1e-3)


def test_ignition_self_sustains_after_heating_off():
    """Con buon confinamento, dopo aver spento il riscaldamento esterno la
    temperatura si MANTIENE alta grazie al solo self-heating delle alfa."""
    s = simulate_burn(
        n_DT0=1.2e20, T0_keV=6.0, tau_E=4.0, tau_p=8.0,
        fueling=2.0e18, p_ext=3e5, p_ext_off_time=4.0, t_end=20.0,
    )
    # Dopo lo spegnimento (t>4 s) la T resta in regime di fusione (> 5 keV).
    after = s.T_keV[s.time > 6.0]
    assert after.min() > 5.0


def test_no_reactions_when_cold_and_unheated():
    """Plasma freddo senza riscaldamento: si raffredda, niente accensione."""
    s = simulate_burn(
        n_DT0=1e20, T0_keV=2.0, tau_E=1.0, tau_p=5.0,
        fueling=0.0, p_ext=0.0, t_end=10.0,
    )
    assert s.T_keV[-1] < s.T_keV[0]  # solo perdite -> si raffredda
