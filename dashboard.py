"""Fase 12 — Dashboard interattiva del simulatore Tokamak (capstone).

Integra tutte le fasi in un'unica interfaccia esplorabile. Lanciare con:

    streamlit run dashboard.py

La LOGICA (costruzione delle figure) e' in funzioni pure `fig_*`, testabili
senza il runtime di Streamlit; `main()` si limita a collegare gli slider.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from tokamak.burn import simulate_burn
from tokamak.engineering import (
    TokamakConfig,
    greenwald_density,
    plasma_beta,
    troyon_beta_limit,
)
from tokamak.fuel_cycle import simulate_inventory, tritium_burn_rate_kg_per_day
from tokamak.optimization import optimize_operating_point
from tokamak.power_balance import fusion_power_density
from tokamak.transport import TransportSolver1D


# --- Funzioni pure che costruiscono le figure (testabili) -------------------
def fig_operating_space(config: TokamakConfig) -> tuple[Figure, dict]:
    """Spazio operativo (T, n_e) con vincoli e punto ottimo."""
    T = np.linspace(2.0, 35.0, 250)
    n = np.linspace(0.05e20, 1.5e20, 250)
    TT, NN = np.meshgrid(T, n)
    n_G = greenwald_density(config.plasma_current_MA, config.minor_radius_m)
    beta_max = troyon_beta_limit(
        config.beta_N, config.plasma_current_MA, config.minor_radius_m, config.B_toroidal_T
    )
    pfus = fusion_power_density(NN, TT) / 1e6
    opt = optimize_operating_point(config)

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(TT, NN / 1e20, pfus, levels=25, cmap="viridis")
    fig.colorbar(cf, ax=ax, label=r"$P_{fus}$ [MW/m$^3$]")
    ax.axhline(n_G / 1e20, color="white", ls="--", lw=2)
    ax.contour(TT, NN / 1e20, plasma_beta(NN, TT, config.B_toroidal_T),
               levels=[beta_max], colors="white", linewidths=2)
    ax.plot(opt.T_keV, opt.n_e / 1e20, "r*", markersize=18, markeredgecolor="white")
    ax.set_xlabel("T [keV]")
    ax.set_ylabel(r"$n_e$ [$10^{20}$ m$^{-3}$]")
    ax.set_title("Spazio operativo + ottimo")
    return fig, {
        "n_e_opt": opt.n_e, "T_opt": opt.T_keV,
        "Pfus_opt": opt.fusion_power_density, "Q_opt": opt.Q,
    }


def fig_radial_profile(n_e: float, chi: float, P_ext_MW: float) -> tuple[Figure, dict]:
    """Profilo radiale T(r) allo stato stazionario e tau_E emergente."""
    # Griglia/passo scelti per reattivita' interattiva (~0.3 s per aggiornamento).
    s = TransportSolver1D(a=2.0, R0=6.2, n_e=n_e, chi=chi, T_edge=0.1, n_cells=60)
    shape = np.exp(-((s.r / 0.6) ** 2))
    p_ext = s.heating_density_for_power(P_ext_MW * 1e6, shape)
    s.solve_steady_state(p_ext=p_ext, dt=1e-2, tol=1e-4)
    tau = s.tau_E(p_ext=p_ext)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s.r, s.T, color="crimson", lw=2)
    ax.fill_between(s.r, s.T, alpha=0.1, color="crimson")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("T [keV]")
    ax.set_title(f"Profilo radiale — T0={s.T[0]:.1f} keV, tau_E={tau:.2f} s")
    ax.grid(True, alpha=0.3)
    return fig, {"T0": float(s.T[0]), "tau_E": tau}


def fig_burn(tau_E: float, tau_p: float, P_ext_MW_m3: float, t_off: float) -> Figure:
    """Combustione auto-consistente: temperatura e densita' nel tempo."""
    s = simulate_burn(
        n_DT0=1.1e20, T0_keV=5.0, tau_E=tau_E, tau_p=tau_p,
        fueling=2.5e18, p_ext=P_ext_MW_m3 * 1e5, p_ext_off_time=t_off, t_end=30.0,
    )
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    a1.plot(s.time, s.T_keV, color="crimson", lw=2)
    a1.axvline(t_off, color="gray", ls=":")
    a1.set_ylabel("T [keV]")
    a1.set_title("Combustione auto-consistente")
    a1.grid(True, alpha=0.3)
    a2.plot(s.time, s.n_DT / 1e20, color="navy", lw=2, label="D-T")
    a2.plot(s.time, s.n_He / 1e20, color="orange", lw=2, label="He")
    a2.set_xlabel("tempo [s]")
    a2.set_ylabel(r"$n$ [$10^{20}$ m$^{-3}$]")
    a2.legend()
    a2.grid(True, alpha=0.3)
    return fig


def fig_fuel_cycle(P_fusion_GW: float, TBR: float) -> tuple[Figure, dict]:
    """Inventario di trizio nel tempo per il TBR scelto."""
    P = P_fusion_GW * 1e9
    h = simulate_inventory(N0_kg=2.0, P_fusion_W=P, TBR=TBR, t_end_years=10.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(h.time_years, h.inventory_kg, color="green", lw=2)
    ax.set_xlabel("tempo [anni]")
    ax.set_ylabel("inventario trizio [kg]")
    ax.set_title(f"Ciclo del trizio (TBR={TBR:.2f})")
    ax.grid(True, alpha=0.3)
    return fig, {"burn_kg_day": tritium_burn_rate_kg_per_day(P)}


# --- Interfaccia Streamlit --------------------------------------------------
def main() -> None:  # pragma: no cover (richiede il runtime di Streamlit)
    import streamlit as st

    st.set_page_config(page_title="Tokamak Simulator", layout="wide")
    st.title("⚛️ Simulatore di reattore a fusione (Tokamak)")
    st.caption("Esplora dal vivo fisica, ingegneria e controllo del plasma.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Spazio operativo", "Profilo & confinamento", "Combustione", "Ciclo del trizio"]
    )

    with tab1:
        c1, c2, c3 = st.columns(3)
        Ip = c1.slider("Corrente di plasma I_p [MA]", 5.0, 20.0, 15.0)
        B = c2.slider("Campo toroidale B [T]", 3.0, 8.0, 5.3)
        beta_N = c3.slider("Beta normalizzato β_N", 1.5, 4.0, 2.5)
        cfg = TokamakConfig(plasma_current_MA=Ip, B_toroidal_T=B, beta_N=beta_N)
        fig, info = fig_operating_space(cfg)
        st.pyplot(fig)
        st.metric("Punto ottimo", f"n={info['n_e_opt']:.2e} m⁻³, T={info['T_opt']:.1f} keV")
        st.metric("P_fus ottimo", f"{info['Pfus_opt']/1e6:.2f} MW/m³")

    with tab2:
        c1, c2, c3 = st.columns(3)
        n_e = c1.slider("Densità n_e [10²⁰ m⁻³]", 0.4, 1.4, 1.0) * 1e20
        chi = c2.slider("Diffusività χ [m²/s]", 0.3, 2.0, 1.0)
        P_ext = c3.slider("Riscaldamento P_ext [MW]", 5.0, 40.0, 20.0)
        fig, info = fig_radial_profile(n_e, chi, P_ext)
        st.pyplot(fig)
        st.metric("τ_E emergente", f"{info['tau_E']:.2f} s")

    with tab3:
        c1, c2, c3, c4 = st.columns(4)
        tau_E = c1.slider("τ_E [s]", 1.0, 6.0, 4.0)
        tau_p = c2.slider("τ_p (cenere) [s]", 2.0, 15.0, 8.0)
        P_heat = c3.slider("Riscaldamento [×10⁵ W/m³]", 1.0, 8.0, 4.0)
        t_off = c4.slider("Spegnimento a t [s]", 1.0, 15.0, 5.0)
        st.pyplot(fig_burn(tau_E, tau_p, P_heat, t_off))

    with tab4:
        c1, c2 = st.columns(2)
        P_fus = c1.slider("Potenza di fusione [GW]", 0.5, 4.0, 3.0)
        TBR = c2.slider("Tritium Breeding Ratio", 0.90, 1.30, 1.10)
        fig, info = fig_fuel_cycle(P_fus, TBR)
        st.pyplot(fig)
        st.metric("Consumo di trizio", f"{info['burn_kg_day']:.2f} kg/giorno")


if __name__ == "__main__":
    main()
