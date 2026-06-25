r"""Fase 7 — Radiazione da impurità e carica efficace Z_eff.

Le impurità (carbonio dalle pareti, tungsteno dal divertore, gas iniettati come
neon/argon) irraggiano molta potenza per RADIAZIONE DI LINEA: gli elettroni
legati degli ioni non completamente ionizzati emettono fotoni. Questo è il
meccanismo che puo' portare al COLLASSO RADIATIVO (la radiazione supera il
riscaldamento, la temperatura crolla).

Due grandezze:
- Z_eff = sum_j n_j Z_j^2 / n_e : carica efficace, pesa le impurità per Z^2.
  Entra nel Bremsstrahlung (gia' nel modello) amplificandolo.
- L_z(T): "funzione di raffreddamento" dell'impurità [W m^3]; la potenza di
  radiazione di linea e' P_line = n_e * n_imp * L_z(T).

ATTENZIONE — modello schematico
--------------------------------
Le funzioni di raffreddamento reali provengono da database atomici (ADAS) e
dipendono dal modello (equilibrio coronale o meno). Qui usiamo un modello
SCHEMATICO con lo scaling noto L_z ~ Z^3 (vedi letteratura sui fit coronali),
calibrato sul carbonio, con una debole dipendenza da T. Serve a mostrare il
FENOMENO (collasso radiativo, impurità ad alto Z molto piu' pericolose), NON per
valori quantitativi: per quelli usare i fit di Mavrin (2018) / dati ADAS.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .power_balance import C_BREM

# Numero atomico di alcune impurità tipiche dei tokamak.
ATOMIC_NUMBER: dict[str, int] = {
    "C": 6,    # carbonio (pareti)
    "Ne": 10,  # neon (gas iniettato)
    "Ar": 18,  # argon (gas iniettato)
    "Fe": 26,  # ferro (acciaio)
    "W": 74,   # tungsteno (divertore)
}

# Calibrazione schematica: per il carbonio (Z=6) la funzione di raffreddamento
# di picco e' ~1.5e-31 W m^3 a ~1 keV. Imponiamo lo scaling ~Z^3.
_LZ_REF_CARBON = 1.5e-31  # W m^3
_T_REF_KEV = 1.0


def cooling_function(T_keV: ArrayLike, species: str) -> NDArray[np.float64]:
    """Funzione di raffreddamento L_z(T) dell'impurità [W m^3] (SCHEMATICA).

    Modello: L_z = L_ref * (Z/Z_C)^3 * (T_ref/T)^(1/2), con L_ref e Z_C riferiti
    al carbonio. Lo scaling Z^3 e' coerente con la letteratura coronale; la
    dipendenza ~T^-1/2 cattura la radiazione di linea che si attenua quando gli
    ioni si spogliano a temperatura piu' alta. Valori assoluti solo indicativi.
    """
    if species not in ATOMIC_NUMBER:
        raise ValueError(f"impurità sconosciuta: {species}")
    Z = ATOMIC_NUMBER[species]
    T = np.asarray(T_keV, dtype=np.float64)
    z_scale = (Z / ATOMIC_NUMBER["C"]) ** 3
    t_scale = np.sqrt(_T_REF_KEV / np.clip(T, 1e-3, None))
    return _LZ_REF_CARBON * z_scale * t_scale


def z_eff_from_fractions(impurity_fractions: dict[str, float]) -> float:
    """Z_eff dato l'insieme delle frazioni di impurità c_z = n_z / n_e.

    Lo ione principale (idrogenico, Z=1) riempie la quasi-neutralità:
        n_DT/n_e = 1 - sum_z c_z Z_z
    quindi
        Z_eff = (1 - sum c_z Z_z) * 1 + sum c_z Z_z^2.
    """
    sum_cZ = sum(c * ATOMIC_NUMBER[s] for s, c in impurity_fractions.items())
    sum_cZ2 = sum(c * ATOMIC_NUMBER[s] ** 2 for s, c in impurity_fractions.items())
    n_main_frac = 1.0 - sum_cZ
    if n_main_frac < 0:
        raise ValueError("frazioni di impurità troppo alte: quasi-neutralità violata")
    return n_main_frac * 1.0 + sum_cZ2


def line_radiation_density(
    n_e: float, T_keV: float, impurity_fractions: dict[str, float]
) -> float:
    """Densità di potenza di radiazione di linea P_line [W/m^3].

    P_line = n_e * sum_z n_z * L_z(T) = n_e^2 * sum_z c_z * L_z(T).
    """
    total = 0.0
    for species, c in impurity_fractions.items():
        total += c * float(cooling_function(T_keV, species))
    return n_e**2 * total


def total_radiated_power_density(
    n_e: float, T_keV: float, impurity_fractions: dict[str, float]
) -> float:
    """Radiazione totale = Bremsstrahlung (con Z_eff) + radiazione di linea."""
    z_eff = z_eff_from_fractions(impurity_fractions)
    p_brem = C_BREM * z_eff * n_e**2 * np.sqrt(T_keV)
    p_line = line_radiation_density(n_e, T_keV, impurity_fractions)
    return p_brem + p_line


def max_impurity_fraction(
    species: str, n_e: float, T_keV: float, heating_density_W: float
) -> float:
    """Frazione massima di impurità prima del collasso radiativo.

    Cerca la concentrazione c per cui la radiazione totale eguaglia la potenza di
    riscaldamento disponibile: oltre, la radiazione vince e la T collassa.
    Restituisce c (= n_imp/n_e); ricerca per bisezione.
    """
    def excess(c: float) -> float:
        return total_radiated_power_density(n_e, T_keV, {species: c}) - heating_density_W

    if excess(0.0) > 0:
        return 0.0  # gia' in collasso senza impurità (radiazione di fondo)
    lo, hi = 0.0, 1.0 / ATOMIC_NUMBER[species]  # limite di quasi-neutralità
    if excess(hi) < 0:
        return hi  # nessun collasso nel range fisico
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
