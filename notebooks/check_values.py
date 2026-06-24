"""Stampa i valori fisici chiave del modello 0D, per verifica rapida.

Lancia con:  python notebooks/check_values.py
"""

from __future__ import annotations

import numpy as np

from tokamak import fusion_gain_Q, reactivity_dt

print("=== Reattivita' <sigma v>(T) ===")
print(f"  T = 10 keV : {float(reactivity_dt(10.0)):.3e} m^3/s   (atteso ~1.14e-22)")
print(f"  T = 100 keV: {float(reactivity_dt(100.0)):.3e} m^3/s   (atteso ~8.42e-22)")

T = np.linspace(20.0, 200.0, 400)
T_peak = float(T[reactivity_dt(T).argmax()])
print(f"  Picco a    : {T_peak:.1f} keV                 (atteso ~66 keV)")

print("\n=== Fattore di guadagno Q ===")
# Confinamento scarso (tau_E breve) -> serve riscaldamento esterno -> Q finito.
q_marginal = float(fusion_gain_Q(n_e=1.0e20, T_keV=15.0, tau_e=0.6))
print(f"  n=1e20, T=15 keV, tau_E=0.6 s : Q = {q_marginal:.2f}   (finito)")
# Confinamento eccellente (tau_E lungo) -> le alfa bastano -> ignition, Q=inf.
q_ignited = float(fusion_gain_Q(n_e=1.0e20, T_keV=15.0, tau_e=2.0))
print(f"  n=1e20, T=15 keV, tau_E=2.0 s : Q = {q_ignited}   (ignition: P_heat<=0)")
