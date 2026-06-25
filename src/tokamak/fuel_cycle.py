r"""Fase 10 — Ciclo del combustibile e produzione di trizio (breeding).

Il trizio non esiste in natura (emivita 12.3 anni) e va PRODOTTO nel reattore,
catturando i neutroni di fusione in un mantello di litio:

    n + Li-6 -> T + He-4 + 4.8 MeV

Ogni fusione consuma un trizio ma libera un neutrone, quindi in linea di
principio il trizio bruciato puo' essere rigenerato. La grandezza chiave e' il
Tritium Breeding Ratio:

    TBR = trizio prodotto / trizio consumato

Per l'autosufficienza serve TBR > 1, con margine per compensare il decadimento
radioattivo dell'inventario, le perdite di processo, e per generare un surplus
con cui avviare nuovi reattori (doubling time).

Bilancio dell'inventario di trizio (in atomi):

    dN/dt = (TBR - 1) * burn_rate - lambda * N + S_ext
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import E_FUSION_DT_JOULE

# Massa del trizio e costante di decadimento.
_U = 1.660_539_066_60e-27  # unita' di massa atomica [kg]
M_TRITIUM = 3.016_049 * _U  # kg
TRITIUM_HALF_LIFE_S = 12.32 * 365.25 * 86400.0  # s
_LAMBDA = np.log(2.0) / TRITIUM_HALF_LIFE_S  # costante di decadimento [1/s]

_SECONDS_PER_DAY = 86400.0
_SECONDS_PER_YEAR = 365.25 * _SECONDS_PER_DAY


def tritium_burn_rate_atoms(P_fusion_W: float) -> float:
    """Tasso di consumo di trizio [atomi/s] per una data potenza di fusione.

    Ogni reazione libera E_fus e consuma un atomo di trizio, quindi il numero di
    reazioni al secondo e' P_fus / E_fus.
    """
    return P_fusion_W / E_FUSION_DT_JOULE


def tritium_burn_rate_kg_per_day(P_fusion_W: float) -> float:
    """Consumo di trizio in kg/giorno (la celebre scala del ~mezzo kg/giorno)."""
    return tritium_burn_rate_atoms(P_fusion_W) * M_TRITIUM * _SECONDS_PER_DAY


def required_TBR(
    P_fusion_W: float,
    inventory_kg: float,
    loss_fraction: float = 0.02,
) -> float:
    """TBR minimo per mantenere COSTANTE l'inventario di trizio.

    A regime: produzione = consumo + decadimento + perdite, da cui

        TBR = 1 + loss_fraction + lambda * N / burn_rate.

    Il termine di decadimento mostra perche' serve TBR > 1 anche senza perdite.
    """
    burn = tritium_burn_rate_atoms(P_fusion_W)
    n_atoms = inventory_kg / M_TRITIUM
    return 1.0 + loss_fraction + _LAMBDA * n_atoms / burn


def doubling_time_years(
    TBR: float, P_fusion_W: float, startup_inventory_kg: float
) -> float:
    """Tempo per accumulare un inventario di avvio per un secondo reattore [anni].

    Accumulo con decadimento: dN/dt = (TBR-1) burn - lambda N, N(0)=0, da cui
    N(t) = ((TBR-1) burn / lambda)(1 - e^{-lambda t}). L'inventario massimo
    accumulabile e' N_max = (TBR-1) burn / lambda: se l'obiettivo lo supera, NON
    e' mai raggiungibile (restituiamo inf) perche' il decadimento lo limita.
    """
    if TBR <= 1.0:
        return np.inf
    burn = tritium_burn_rate_atoms(P_fusion_W)
    target = startup_inventory_kg / M_TRITIUM
    n_max = (TBR - 1.0) * burn / _LAMBDA
    if target >= n_max:
        return np.inf
    t = -np.log(1.0 - target / n_max) / _LAMBDA
    return t / _SECONDS_PER_YEAR


@dataclass
class InventoryHistory:
    """Evoluzione dell'inventario di trizio."""

    time_years: NDArray[np.float64]
    inventory_kg: NDArray[np.float64]
    TBR: float


def simulate_inventory(
    *,
    N0_kg: float,
    P_fusion_W: float,
    TBR: float,
    external_supply_kg_per_year: float = 0.0,
    t_end_years: float = 10.0,
    n_points: int = 400,
) -> InventoryHistory:
    """Integra l'inventario di trizio nel tempo (soluzione analitica esatta).

    Il bilancio dN/dt = (TBR-1) burn + S_ext - lambda N e' lineare, quindi lo
    risolviamo in forma chiusa (piu' accurato dell'integrazione numerica):

        N(t) = N_eq + (N0 - N_eq) e^{-lambda t},  N_eq = source / lambda
    """
    burn = tritium_burn_rate_atoms(P_fusion_W)
    s_ext = external_supply_kg_per_year / M_TRITIUM / _SECONDS_PER_YEAR
    source = (TBR - 1.0) * burn + s_ext  # atomi/s netti prima del decadimento

    n0 = N0_kg / M_TRITIUM
    n_eq = source / _LAMBDA
    t_s = np.linspace(0.0, t_end_years * _SECONDS_PER_YEAR, n_points)
    n_t = n_eq + (n0 - n_eq) * np.exp(-_LAMBDA * t_s)
    n_t = np.clip(n_t, 0.0, None)  # l'inventario non puo' essere negativo

    return InventoryHistory(
        time_years=t_s / _SECONDS_PER_YEAR,
        inventory_kg=n_t * M_TRITIUM,
        TBR=TBR,
    )
