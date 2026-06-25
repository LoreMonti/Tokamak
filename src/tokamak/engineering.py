r"""Vincoli ingegneristici operativi di un tokamak.

Il modello 0D/1D dice se il plasma PRODUCE energia. Questi limiti dicono se il
punto operativo e' FISICAMENTE REALIZZABILE senza distruggere la macchina o
perdere il confinamento. Sono i vincoli che definiscono lo "spazio operativo".

1. Densita' di Greenwald   - sopra una densita' critica il plasma va in
                             disruption (limite empirico, valido in tutti i
                             tokamak).
2. Beta limit di Troyon    - rapporto massimo pressione plasma / pressione
                             magnetica prima delle instabilita' MHD.
3. Carico sul divertore    - flusso di potenza sulle pareti; sopra ~10 MW/m^2 i
                             materiali non reggono.

Tutte le formule usano unita' pratiche tipiche del settore (MA, T, m); le
docstring indicano le conversioni.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import KEV_TO_JOULE

# Permeabilita' del vuoto (SI).
MU_0 = 4.0e-7 * np.pi


# --- 1. Densita' di Greenwald ------------------------------------------------
def greenwald_density(plasma_current_MA: float, minor_radius_m: float) -> float:
    """Densita' limite di Greenwald, in m^-3.

        n_G [10^20 m^-3] = I_p [MA] / (pi * a^2 [m^2])

    Sopra n_G il plasma tende alla disruption. Restituiamo gia' in m^-3
    (moltiplicato per 1e20) per confronto diretto con n_e del modello.
    """
    n_G_1e20 = plasma_current_MA / (np.pi * minor_radius_m**2)
    return n_G_1e20 * 1e20


def greenwald_fraction(
    n_e: float, plasma_current_MA: float, minor_radius_m: float
) -> float:
    """Frazione di Greenwald f_G = n_e / n_G. Operare sicuri richiede f_G < ~1."""
    return n_e / greenwald_density(plasma_current_MA, minor_radius_m)


# --- 2. Beta (pressione plasma / pressione magnetica) ------------------------
def plasma_pressure(n_e: float, T_keV: float) -> float:
    """Pressione cinetica del plasma p = n_e*T_e + n_i*T_i, in Pa.

    Con quasi-neutralita' (n_i = n_e) e T_i = T_e = T: p = 2 n_e T.
    """
    return 2.0 * n_e * T_keV * KEV_TO_JOULE


def plasma_beta(n_e: float, T_keV: float, B_toroidal_T: float) -> float:
    """Beta toroidale = pressione plasma / pressione magnetica (adimensionale).

        beta = p / (B^2 / 2 mu_0) = 2 mu_0 n T / B^2
    """
    p = plasma_pressure(n_e, T_keV)
    return p / (B_toroidal_T**2 / (2.0 * MU_0))


def troyon_beta_limit(
    beta_N: float, plasma_current_MA: float, minor_radius_m: float, B_toroidal_T: float
) -> float:
    """Beta massimo di Troyon (adimensionale, non in %).

        beta_max [%] = beta_N * I_p[MA] / (a[m] * B[T])

    beta_N ~ 2.5-3.5 e' il "beta normalizzato"; qui restituiamo la frazione
    (diviso 100) per confronto diretto con plasma_beta().
    """
    beta_max_percent = beta_N * plasma_current_MA / (minor_radius_m * B_toroidal_T)
    return beta_max_percent / 100.0


# --- 3. Carico termico sul divertore ----------------------------------------
def divertor_heat_flux(
    power_to_divertor_W: float,
    major_radius_m: float,
    lambda_q_mm: float = 3.0,
    flux_expansion: float = 5.0,
    n_targets: int = 2,
) -> float:
    """Flusso di potenza di picco sul divertore, in MW/m^2 (stima ordine-grandezza).

    La potenza che attraversa la separatrice (P_SOL) si deposita su una striscia
    sottile larga lambda_q (scrape-off layer, ~mm), distribuita lungo la
    circonferenza toroidale 2*pi*R, su n_targets bersagli. La "flux expansion"
    allarga la striscia sui target riducendo il picco.

        A_bagnata ~ n_targets * (2 pi R) * (lambda_q * flux_expansion)
        q_peak = P_SOL / A_bagnata
    """
    lambda_q_m = lambda_q_mm * 1e-3
    wetted_area = n_targets * (2.0 * np.pi * major_radius_m) * (
        lambda_q_m * flux_expansion
    )
    return power_to_divertor_W / wetted_area / 1e6  # -> MW/m^2


# --- Configurazione di macchina e verifica complessiva -----------------------
@dataclass
class TokamakConfig:
    """Parametri ingegneristici di una macchina (default ~ scala ITER)."""

    minor_radius_m: float = 2.0
    major_radius_m: float = 6.2
    plasma_current_MA: float = 15.0
    B_toroidal_T: float = 5.3
    beta_N: float = 2.5


@dataclass
class OperationalCheck:
    """Esito della verifica dei limiti per un punto operativo (n_e, T)."""

    greenwald_fraction: float
    beta: float
    beta_limit: float
    divertor_flux_MW_m2: float
    divertor_limit_MW_m2: float

    @property
    def greenwald_ok(self) -> bool:
        return self.greenwald_fraction < 1.0

    @property
    def beta_ok(self) -> bool:
        return self.beta < self.beta_limit

    @property
    def divertor_ok(self) -> bool:
        return self.divertor_flux_MW_m2 < self.divertor_limit_MW_m2

    @property
    def all_ok(self) -> bool:
        return self.greenwald_ok and self.beta_ok and self.divertor_ok


def check_operational_limits(
    config: TokamakConfig,
    n_e: float,
    T_keV: float,
    power_to_divertor_W: float,
    divertor_limit_MW_m2: float = 10.0,
) -> OperationalCheck:
    """Verifica i tre limiti per un punto operativo e ne riassume l'esito."""
    return OperationalCheck(
        greenwald_fraction=greenwald_fraction(
            n_e, config.plasma_current_MA, config.minor_radius_m
        ),
        beta=plasma_beta(n_e, T_keV, config.B_toroidal_T),
        beta_limit=troyon_beta_limit(
            config.beta_N,
            config.plasma_current_MA,
            config.minor_radius_m,
            config.B_toroidal_T,
        ),
        divertor_flux_MW_m2=divertor_heat_flux(
            power_to_divertor_W, config.major_radius_m
        ),
        divertor_limit_MW_m2=divertor_limit_MW_m2,
    )
