r"""Modello 0D: bilancio di potenza, fattore Q e criterio di Lawson.

"0D" significa che trattiamo il plasma come un punto: un solo valore di densita'
e temperatura, niente profili spaziali (quelli arrivano nella Fase 2, 1D).
Confrontiamo densita' di potenza (W/m^3) prodotte e perse. E' il modello piu'
semplice che cattura la domanda fondamentale: *questo plasma produce piu'
energia di quanta ne richiede per restare caldo?*

Ipotesi del modello
--------------------
- Plasma puro D-T con miscela 50:50, quasi-neutralita' n_e = n_i = n.
  Quindi n_D = n_T = n/2 e il prodotto n_D * n_T = n^2 / 4.
- Temperatura unica T_e = T_i = T (ioni ed elettroni termalizzati insieme).
- Energia immagazzinata W = (3/2) n_e T + (3/2) n_i T = 3 n T
  (3/2 kT per particella per i gradi di liberta' traslazionali, sommato su
  elettroni e ioni).

I quattro termini di potenza
----------------------------
1. P_fus   - potenza di fusione totale prodotta (alfa + neutrone).
2. P_alpha - quota che resta nel plasma (alfa carico) -> self-heating. ~P_fus/5.
3. P_brem  - perdita per radiazione di Bremsstrahlung (frenamento degli
             elettroni nel campo dei nuclei). Inevitabile in un plasma caldo.
4. P_loss  - perdita per trasporto/conduzione, P_loss = W / tau_E, dove tau_E
             e' il tempo di confinamento dell'energia: quanto a lungo il plasma
             "trattiene" la sua energia prima di disperderla. E' IL parametro
             ingegneristico chiave del confinamento.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import ALPHA_HEATING_FRACTION, E_FUSION_DT_JOULE, KEV_TO_JOULE
from .reactivity import reactivity_dt

# Costante di Bremsstrahlung per T in keV, n in m^-3 -> W/m^3.
# Forma standard: P_brem = C_BREM * Z_eff * n_e^2 * sqrt(T_keV).
# La radiazione cresce solo come sqrt(T), mentre la fusione cresce molto piu'
# rapidamente: per questo esiste una T minima sotto la quale il Bremsstrahlung
# vince sempre e l'ignition e' impossibile.
C_BREM = 5.35e-37  # W * m^3 / sqrt(keV)


def fusion_power_density(n_e: ArrayLike, T_keV: ArrayLike) -> NDArray[np.float64]:
    """Densita' di potenza di fusione TOTALE (alfa + neutrone), in W/m^3.

    P_fus = n_D * n_T * <sigma v> * E_fus = (n_e^2 / 4) * <sigma v> * E_fus
    """
    n_e = np.asarray(n_e, dtype=np.float64)
    sigv = reactivity_dt(T_keV)
    return 0.25 * n_e**2 * sigv * E_FUSION_DT_JOULE


def alpha_power_density(n_e: ArrayLike, T_keV: ArrayLike) -> NDArray[np.float64]:
    """Densita' di potenza di self-heating delle alfa, in W/m^3 (~P_fus / 5)."""
    return ALPHA_HEATING_FRACTION * fusion_power_density(n_e, T_keV)


def bremsstrahlung_power_density(
    n_e: ArrayLike, T_keV: ArrayLike, z_eff: float = 1.0
) -> NDArray[np.float64]:
    """Perdita per radiazione di Bremsstrahlung, in W/m^3.

    z_eff (carica efficace) = 1 per un plasma D-T puro; >1 in presenza di
    impurita', che aumentano molto le perdite radiative.
    """
    n_e = np.asarray(n_e, dtype=np.float64)
    T = np.asarray(T_keV, dtype=np.float64)
    return C_BREM * z_eff * n_e**2 * np.sqrt(T)


def stored_energy_density(n_e: ArrayLike, T_keV: ArrayLike) -> NDArray[np.float64]:
    """Densita' di energia termica immagazzinata W = 3 n T, in J/m^3."""
    n_e = np.asarray(n_e, dtype=np.float64)
    T = np.asarray(T_keV, dtype=np.float64)
    return 3.0 * n_e * T * KEV_TO_JOULE


def loss_power_density(
    n_e: ArrayLike, T_keV: ArrayLike, tau_e: float
) -> NDArray[np.float64]:
    """Perdita per trasporto P_loss = W / tau_E, in W/m^3."""
    return stored_energy_density(n_e, T_keV) / tau_e


def heating_power_required(
    n_e: ArrayLike, T_keV: ArrayLike, tau_e: float, z_eff: float = 1.0
) -> NDArray[np.float64]:
    """Potenza esterna di riscaldamento per mantenere lo stato stazionario.

    In equilibrio la potenza entrante eguaglia quella uscente:

        P_alpha + P_heat = P_loss + P_brem
        => P_heat = P_loss + P_brem - P_alpha

    Se P_heat <= 0 il self-heating delle alfa basta da solo a coprire le
    perdite: il plasma e' IGNITO e non serve riscaldamento esterno.
    """
    p_loss = loss_power_density(n_e, T_keV, tau_e)
    p_brem = bremsstrahlung_power_density(n_e, T_keV, z_eff)
    p_alpha = alpha_power_density(n_e, T_keV)
    return p_loss + p_brem - p_alpha


def fusion_gain_Q(
    n_e: ArrayLike, T_keV: ArrayLike, tau_e: float, z_eff: float = 1.0
) -> NDArray[np.float64]:
    """Fattore di guadagno di fusione Q = P_fus / P_heat in stato stazionario.

    Q misura quanta potenza di fusione otteniamo per ogni watt esterno immesso.
    - Q = 1  : break-even (fusione = riscaldamento esterno).
    - Q = 10 : obiettivo di ITER.
    - Q -> inf: ignition (P_heat -> 0), restituito come np.inf.
    """
    p_heat = heating_power_required(n_e, T_keV, tau_e, z_eff)
    p_fus = fusion_power_density(n_e, T_keV)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(p_heat > 0.0, p_fus / p_heat, np.inf)
    return q


def triple_product_ignition(
    T_keV: ArrayLike, z_eff: float = 1.0, include_bremsstrahlung: bool = True
) -> NDArray[np.float64]:
    r"""Triplo prodotto n * T * tau_E minimo per l'ignition, in keV*s*m^-3.

    Condizione di ignition: il self-heating delle alfa copre le perdite,

        P_alpha = P_loss + P_brem

    Sia le alfa che il Bremsstrahlung scalano come n^2, mentre le perdite di
    trasporto come n. Risolvendo per n * tau_E:

        n * tau_E = 12 * T_J / ( <sigma v> * E_alpha - 4 * C_brem * z_eff * sqrt(T) )

    e il triplo prodotto e' n * T_keV * tau_E. Il denominatore mostra la fisica
    chiave: a T troppo bassa il termine di Bremsstrahlung supera quello di
    fusione, il denominatore diventa <= 0 e l'ignition e' IMPOSSIBILE a
    qualunque densita' (restituiamo np.inf).
    """
    T = np.asarray(T_keV, dtype=np.float64)
    sigv = reactivity_dt(T)
    e_alpha = ALPHA_HEATING_FRACTION * E_FUSION_DT_JOULE
    T_joule = T * KEV_TO_JOULE

    brem_term = 4.0 * C_BREM * z_eff * np.sqrt(T) if include_bremsstrahlung else 0.0
    denom = sigv * e_alpha - brem_term

    with np.errstate(divide="ignore", invalid="ignore"):
        n_tau = np.where(denom > 0.0, 12.0 * T_joule / denom, np.inf)
    return n_tau * T  # n * tau * T_keV
