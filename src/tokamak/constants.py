"""Costanti fisiche e parametri di reazione usati nel modello.

Convenzione di unità del progetto
----------------------------------
In fisica del plasma le temperature si esprimono in **keV**, non in kelvin.
Il motivo è che ciò che conta nelle reazioni è l'energia termica delle
particelle, k_B * T. Misurare T in unità di energia (eV/keV) significa
assorbire k_B nella definizione: "T = 15 keV" vuol dire che l'energia termica
tipica delle particelle e' 15 keV. Per riferimento, 1 keV ~ 11.6 milioni di K.

Le densita' sono in m^-3 e le potenze in W/m^3 (densita' di potenza), perche'
il bilancio energetico e' locale: confrontiamo potenza prodotta e persa per
unita' di volume di plasma.
"""

from __future__ import annotations

# --- Costanti universali (SI) ---
ELEMENTARY_CHARGE = 1.602_176_634e-19  # C  (anche: 1 eV in joule)
BOLTZMANN = 1.380_649e-23  # J/K

# --- Conversioni di unita' ---
KEV_TO_JOULE = 1e3 * ELEMENTARY_CHARGE  # 1 keV in joule
KEV_TO_KELVIN = KEV_TO_JOULE / BOLTZMANN  # ~1.16e7 K per keV

# --- Energetica della reazione D-T ---
# D + T -> He-4 + n  libera 17.59 MeV totali, ripartiti per conservazione
# di energia e quantita' di moto inversamente alla massa dei prodotti:
#   - nucleo di He-4 (particella alfa): 3.52 MeV
#   - neutrone:                         14.07 MeV
# Distinzione FISICAMENTE cruciale per il bilancio di potenza:
#   * l'alfa e' carico -> resta confinato dal campo magnetico e RISCALDA il
#     plasma (self-heating). E' il termine che rende possibile l'ignition.
#   * il neutrone e' neutro -> sfugge al confinamento, deposita la sua energia
#     nel mantello (blanket) esterno. E' la potenza che useremmo per produrre
#     elettricita', ma NON contribuisce a sostenere il plasma.
E_FUSION_DT_MEV = 17.59  # energia totale per reazione
E_ALPHA_MEV = 3.52  # quota che riscalda il plasma
E_NEUTRON_MEV = 14.07  # quota che sfugge

MEV_TO_JOULE = 1e6 * ELEMENTARY_CHARGE
E_FUSION_DT_JOULE = E_FUSION_DT_MEV * MEV_TO_JOULE
E_ALPHA_JOULE = E_ALPHA_MEV * MEV_TO_JOULE

# Frazione dell'energia di fusione che resta nel plasma (~1/5).
# Deriva direttamente dal rapporto 3.52 / 17.59.
ALPHA_HEATING_FRACTION = E_ALPHA_MEV / E_FUSION_DT_MEV
