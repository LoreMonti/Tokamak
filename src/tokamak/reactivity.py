r"""Reattivita' termica D-T: <sigma v>(T), calcolata dalla fisica di base.

Cos'e' la reattivita' e perche' serve
-------------------------------------
Il tasso di reazioni di fusione per unita' di volume e'

    R = n_D * n_T * <sigma v>

dove sigma(E) e' la sezione d'urto (probabilita' di reazione a una data energia
di collisione) e v la velocita' relativa. In un plasma le particelle NON hanno
tutte la stessa energia: seguono una distribuzione di Maxwell-Boltzmann a
temperatura T. La grandezza fisicamente rilevante e' quindi la MEDIA del
prodotto sigma*v su quella distribuzione.

Per una Maxwelliana, integrando sull'energia del centro di massa E si ottiene:

    <sigma v>(T) = sqrt(8 / (pi * m_r)) * (k_B T)^(-3/2)
                   * integrale_0^inf [ sigma(E) * E * exp(-E / k_B T) ] dE

dove m_r e' la massa ridotta della coppia D-T. Questo integrale e' il "ponte"
tra la fisica microscopica (sezione d'urto, tunneling quantistico attraverso la
barriera coulombiana) e la fisica macroscopica del reattore (quanta potenza
produce un metro cubo di plasma a una data T).

Perche' <sigma v> cresce e poi raggiunge un massimo
---------------------------------------------------
Due nuclei carichi si respingono (barriera di Coulomb). Fondere richiede di
attraversarla per effetto tunnel: la probabilita' e' dominata dal fattore di
Gamow exp(-B_G / sqrt(E)). Solo le particelle nella coda ad alta energia della
Maxwelliana riescono a fondere, percio' <sigma v> cresce rapidamente con T fino
a un massimo intorno a ~64 keV; oltre, l'integrale e' dominato da energie sopra
il picco della sezione d'urto e <sigma v> torna a calare lentamente. Il regime
operativo dei reattori (~10-20 keV) e' molto sotto il picco: la scelta non e'
dettata dalla reattivita' ma dal bilancio con le perdite (vedi power_balance).

Implementazione
---------------
Usiamo la sezione d'urto parametrizzata di Bosch & Hale (Nucl. Fusion 32, 1992)
e ne facciamo la media maxwelliana per integrazione numerica. Il risultato e'
validato contro i valori di letteratura (entro ~2% su 1-200 keV). Per efficienza
il calcolo viene fatto una volta su una griglia e poi interpolato con una spline.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate
from scipy.interpolate import CubicSpline

from .constants import KEV_TO_JOULE

# --- Sezione d'urto D-T (Bosch-Hale, canale D(t,n)He-4), valida 0.5-550 keV ---
# sigma(E) [millibarn] = S(E) / (E * exp(B_G / sqrt(E))),  E = energia CM in keV
# con S(E) fattore astrofisico (Pade'), che assorbe la fisica nucleare residua
# una volta tolta la dipendenza dominante (Gamow + 1/E) dalla sezione d'urto.
_BG = 34.3827  # costante di Gamow, sqrt(keV)
_A = (6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0.0)
_B = (6.38e1, -9.95e-1, 6.981e-5, 1.728e-4)

_MB_TO_M2 = 1e-31  # 1 millibarn = 1e-31 m^2

# Massa ridotta della coppia D-T (kg).
_PROTON_MASS = 1.672_621_924e-27
_M_D = 2.013_553 * _PROTON_MASS
_M_T = 3.015_501 * _PROTON_MASS
_M_REDUCED = _M_D * _M_T / (_M_D + _M_T)

_T_MIN_KEV = 0.5
_T_MAX_KEV = 200.0


def cross_section_dt(E_keV: ArrayLike) -> NDArray[np.float64]:
    """Sezione d'urto D-T in m^2, energia E nel centro di massa (keV)."""
    E = np.asarray(E_keV, dtype=np.float64)
    a0, a1, a2, a3, a4 = _A
    b1, b2, b3, b4 = _B
    s_factor = (a0 + E * (a1 + E * (a2 + E * (a3 + E * a4)))) / (
        1.0 + E * (b1 + E * (b2 + E * (b3 + E * b4)))
    )
    sigma_mb = s_factor / (E * np.exp(_BG / np.sqrt(E)))
    return sigma_mb * _MB_TO_M2


def _reactivity_single(T_keV: float) -> float:
    """<sigma v> [m^3/s] a una T, per integrazione maxwelliana diretta."""
    kT = T_keV * KEV_TO_JOULE  # energia termica in joule

    def integrand(E_keV: float) -> float:
        E_joule = E_keV * KEV_TO_JOULE
        # dE espresso in joule: per questo moltiplichiamo per KEV_TO_JOULE
        # (l'integrazione avviene sulla variabile E_keV).
        return (
            cross_section_dt(E_keV)
            * E_joule
            * np.exp(-E_joule / kT)
            * KEV_TO_JOULE
        )

    integral, _ = integrate.quad(integrand, _T_MIN_KEV, 1000.0, limit=200)
    prefactor = np.sqrt(8.0 / (np.pi * _M_REDUCED)) * kT ** (-1.5)
    return float(prefactor * integral)


@lru_cache(maxsize=1)
def _spline() -> CubicSpline:
    """Spline di log10(<sigma v>) vs log10(T), costruita una sola volta.

    Interpoliamo in scala logaritmica perche' <sigma v> varia di molti ordini di
    grandezza: lo spline log-log e' molto piu' accurato di uno lineare.
    """
    T = np.logspace(np.log10(_T_MIN_KEV), np.log10(_T_MAX_KEV), 120)
    sigv = np.array([_reactivity_single(t) for t in T])
    return CubicSpline(np.log10(T), np.log10(sigv))


def reactivity_dt(T_keV: ArrayLike) -> NDArray[np.float64]:
    """Reattivita' termica <sigma v> per la reazione D-T, in **m^3 / s**.

    Parameters
    ----------
    T_keV:
        Temperatura del plasma in keV (scalare o array). Si assume T_i = T_e.

    Note
    ----
    Valida ~0.5-200 keV. Fuori range emettiamo un avviso (extrapolazione spline),
    senza interrompere scan di temperatura che sconfinano ai bordi.
    """
    T = np.asarray(T_keV, dtype=np.float64)
    if np.any((T < _T_MIN_KEV) | (T > _T_MAX_KEV)):
        import warnings

        warnings.warn(
            f"Temperatura fuori dal range di validita' "
            f"[{_T_MIN_KEV}, {_T_MAX_KEV}] keV; risultato extrapolato.",
            stacklevel=2,
        )
    return np.power(10.0, _spline()(np.log10(T)))
