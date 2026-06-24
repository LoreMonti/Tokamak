# Tokamak — Simulatore di bilancio di potenza e trasporto di un plasma da fusione

[![CI](https://github.com/lorenzomonti/Tokamak/actions/workflows/ci.yml/badge.svg)](https://github.com/lorenzomonti/Tokamak/actions/workflows/ci.yml)

Simulatore della fisica di confinamento di un reattore a fusione (tokamak),
costruito dai principi primi: dalla reattività nucleare al bilancio di potenza
0D, fino (nelle fasi successive) al trasporto radiale 1D e ai vincoli
ingegneristici reali.

**Domanda guida:** un plasma D-T a una data densità, temperatura e qualità di
confinamento, produce più energia di quanta ne serva per restare caldo?

## Stato del progetto

- ✅ **Fase 1 — Modello 0D**: reattività D-T, bilancio di potenza, fattore di
  guadagno *Q* e criterio di Lawson.
- ✅ **Fase 2 — Trasporto radiale 1D**: equazione di diffusione del calore,
  solver implicito a volumi finiti, profilo *T(r)* e *τ_E* emergente.
- ⏳ Fase 3 — Vincoli ingegneristici (Greenwald, Troyon, divertore).

Vedi [ROADMAP.md](ROADMAP.md) per il piano completo.

## La fisica in breve

Il modello confronta densità di potenza (W/m³) prodotte e perse in un plasma
D-T 50:50:

| Termine | Significato | Scaling |
|---|---|---|
| `P_fus` | Potenza di fusione D-T → ⁴He + n | ∝ n²⟨σv⟩ |
| `P_α` | Self-heating delle particelle alfa (resta confinato) | ≈ P_fus / 5 |
| `P_brem` | Perdita per radiazione di Bremsstrahlung | ∝ n²√T |
| `P_loss` | Perdita per trasporto, `W/τ_E` | ∝ nT/τ_E |

La **reattività** ⟨σv⟩(T) è calcolata come media maxwelliana della sezione
d'urto di Bosch-Hale, validata contro i valori di letteratura (entro ~2% tra 1 e
200 keV, picco a ~66 keV).

### Criterio di Lawson

![Diagramma di Lawson](docs/lawson_diagram.png)

Il triplo prodotto `n·T·τ_E` richiesto per l'ignition ha un **minimo a ~14 keV**:
è la finestra operativa ottimale del D-T. Sotto una temperatura minima il
Bremsstrahlung domina la fusione e l'ignition diventa impossibile a qualunque
densità (la curva rossa diverge).

### Trasporto radiale 1D

![Profilo radiale](docs/radial_profile.png)

Risolviamo l'equazione di diffusione del calore lungo il raggio minore con uno
schema implicito a volumi finiti (sistema tridiagonale, algoritmo di Thomas):

```
(3/2) n ∂T/∂t = (1/r) ∂/∂r( r·nχ·∂T/∂r ) + S(r)
```

A differenza del modello 0D, il tempo di confinamento `τ_E` non è imposto ma
**emerge** dal profilo calcolato, dalla diffusività `χ` e dalla geometria.
Validazione numerica: confronto con la soluzione analitica parabolica (sorgente
e `χ` costanti) e conservazione dell'energia a dominio isolato.

## Validazione

| Grandezza | Modello | Riferimento |
|---|---|---|
| ⟨σv⟩ a 10 keV | 1.14e-22 m³/s | ~1.1e-22 m³/s |
| Picco di ⟨σv⟩ | 66 keV | ~64 keV |
| Minimo del triplo prodotto | 14.4 keV | ~14 keV |
| Frazione alfa | 0.200 | 3.52/17.59 = 0.200 |

## Uso

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Test (include validazioni fisiche)
pytest

# Genera il diagramma di Lawson e il profilo radiale 1D
python notebooks/lawson_diagram.py
python notebooks/radial_profile.py
```

```python
from tokamak import fusion_gain_Q

# Q in stato stazionario per parametri tipo ITER
Q = fusion_gain_Q(n_e=1.0e20, T_keV=15.0, tau_e=2.0)
```

## Riferimenti

- H.-S. Bosch & G.M. Hale, *Improved formulas for fusion cross-sections and
  thermal reactivities*, Nucl. Fusion **32** (1992) 611.
- J. Wesson, *Tokamaks*, Oxford University Press.
- J. Freidberg, *Plasma Physics and Fusion Energy*, Cambridge University Press.

## Licenza

MIT
