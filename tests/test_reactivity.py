"""Validazione della reattivita' D-T contro valori di letteratura.

Verifichiamo non solo che il codice "giri", ma che riproduca la FISICA nota:
valori tabulati di Bosch-Hale e l'andamento qualitativo atteso.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokamak.reactivity import reactivity_dt


@pytest.mark.parametrize(
    "T_keV, expected_m3s, rtol",
    [
        # Valori di riferimento (media maxwelliana della sezione d'urto
        # Bosch-Hale, Nucl. Fusion 32, 1992), in m^3/s. Validati entro ~3%.
        (10.0, 1.13e-22, 0.04),
        (20.0, 4.33e-22, 0.04),
        (100.0, 8.4e-22, 0.04),
    ],
)
def test_reactivity_literature_values(T_keV, expected_m3s, rtol):
    assert reactivity_dt(T_keV) == pytest.approx(expected_m3s, rel=rtol)


def test_reactivity_increases_with_temperature_below_peak():
    """<sigma v> deve crescere monotonamente con T fino al picco (~64 keV).

    Riflette il fatto che a T piu' alta una frazione maggiore di particelle
    supera la barriera coulombiana per tunneling.
    """
    T = np.linspace(2.0, 60.0, 50)
    sigv = reactivity_dt(T)
    assert np.all(np.diff(sigv) > 0)


def test_reactivity_peaks_near_64_keV():
    """Il picco della reattivita' D-T e' noto essere intorno a 64 keV."""
    T = np.linspace(10.0, 100.0, 200)
    T_peak = T[np.argmax(reactivity_dt(T))]
    assert 55.0 < T_peak < 75.0
