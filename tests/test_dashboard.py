"""Validazione della logica della dashboard (funzioni pure, senza Streamlit).

Importare `dashboard` non esegue l'interfaccia (e' sotto if __name__),
quindi possiamo testare le funzioni che costruiscono le figure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")  # backend non interattivo per i test

# Carica dashboard.py (nella radice del progetto, non nel pacchetto).
_spec = importlib.util.spec_from_file_location(
    "dashboard", Path(__file__).resolve().parent.parent / "dashboard.py"
)
dashboard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dashboard)

from tokamak.engineering import TokamakConfig  # noqa: E402


def test_operating_space_figure_and_optimum():
    fig, info = dashboard.fig_operating_space(TokamakConfig())
    assert isinstance(fig, Figure)
    assert info["T_opt"] > 0 and info["n_e_opt"] > 0
    assert info["Pfus_opt"] > 0


def test_radial_profile_figure():
    fig, info = dashboard.fig_radial_profile(n_e=1e20, chi=1.0, P_ext_MW=20.0)
    assert isinstance(fig, Figure)
    assert info["tau_E"] > 0
    assert info["T0"] > info["tau_E"] * 0  # T0 positivo


def test_burn_figure():
    fig = dashboard.fig_burn(tau_E=4.0, tau_p=8.0, P_ext_MW_m3=4.0, t_off=5.0)
    assert isinstance(fig, Figure)


def test_fuel_cycle_figure_and_burn_rate():
    fig, info = dashboard.fig_fuel_cycle(P_fusion_GW=3.0, TBR=1.10)
    assert isinstance(fig, Figure)
    assert info["burn_kg_day"] == pytest.approx(0.46, abs=0.05)
