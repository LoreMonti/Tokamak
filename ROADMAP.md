# Tokamak — Simulatore di trasporto e bilancio di potenza di un plasma da fusione

Simulatore della fisica di confinamento di un reattore a fusione (tokamak), dal
bilancio di potenza 0D al trasporto radiale 1D, con vincoli ingegneristici reali.

**Obiettivo del progetto:** calcolare il fattore di guadagno di fusione *Q* e le
condizioni di *ignition* (criterio di Lawson) per configurazioni realistiche tipo
ITER, partendo dai principi fisici.

---

## Glossario rapido dei concetti chiave

| Simbolo | Significato | Ordine di grandezza (ITER) |
|---|---|---|
| `T` | Temperatura del plasma | ~15 keV (~150 milioni K) |
| `n` | Densità (ioni/elettroni) | ~10²⁰ m⁻³ |
| `τ_E` | Tempo di confinamento dell'energia | ~3–4 s |
| `Q` | Potenza fusione / potenza immessa | ~10 (target ITER) |
| `nTτ_E` | Triplo prodotto (criterio di Lawson) | ~3×10²¹ keV·s·m⁻³ |

**Reazione di riferimento:** D + T → ⁴He (3.5 MeV) + n (14.1 MeV).
Solo l'alfa (3.5 MeV) resta confinato e riscalda il plasma; il neutrone scappa.

---

## Fase 0 — Setup del repository (giorno 1)

Obiettivo: repo professionale fin dal primo commit.

- [ ] `git init`, licenza (MIT), `.gitignore` Python
- [ ] Struttura a pacchetto:
  ```
  tokamak/
  ├── src/tokamak/
  │   ├── __init__.py
  │   ├── constants.py        # costanti fisiche, masse, energie di reazione
  │   ├── reactivity.py       # <σv> per D-T in funzione di T
  │   ├── power_balance.py    # modello 0D
  │   └── transport.py        # modello 1D (fase 2)
  ├── tests/
  ├── notebooks/              # esplorazione + figure per il README
  ├── docs/
  ├── pyproject.toml
  ├── README.md
  └── ROADMAP.md
  ```
- [ ] `pyproject.toml` con dipendenze: `numpy`, `scipy`, `matplotlib`
- [ ] Tooling: `ruff` (lint+format), `pytest`, GitHub Actions CI (lint + test)
- [ ] README iniziale con scopo del progetto

**Commit:** `chore: project scaffold and CI`

---

## Fase 1 — Modello 0D: bilancio di potenza e criterio di Lawson (settimana 1)

Il cuore fisico-ingegneristico. Nessuna PDE: solo algebra e una reattività.

### 1.1 Reattività D-T `<σv>(T)`
- [ ] Implementare la parametrizzazione di **Bosch-Hale** per `<σv>` D-T
      (formula analitica standard, valida 0.2–100 keV)
- [ ] Test: confronto con valori tabulati di letteratura a 10 e 20 keV

### 1.2 Termini di potenza (densità di potenza, W/m³)
- [ ] **Fusione:** `P_fus = n_D · n_T · <σv> · E_fus`
- [ ] **Riscaldamento alfa:** `P_α = (1/5) · P_fus` (solo 3.5/17.6 MeV resta)
- [ ] **Radiazione di Bremsstrahlung:** `P_brem ∝ Z_eff · n_e² · √T`
- [ ] **Perdite per confinamento:** `P_loss = W / τ_E`, con `W = 3 n T`

### 1.3 Equilibrio e figure di merito
- [ ] Calcolo del fattore **Q = P_fus / P_heat**
- [ ] Condizione di **ignition**: `P_α ≥ P_loss + P_brem`
- [ ] **Criterio di Lawson**: ricavare la curva `n·τ_E` vs `T` minima

### 1.4 Deliverable visivo (fondamentale per il CV)
- [ ] **Diagramma di Lawson**: piano `T` vs `n·τ_E` con curve di break-even,
      Q=10 e ignition; segnare il punto operativo di ITER
- [ ] Salvare le figure in `docs/` e inserirle nel README

**Validazione:** con parametri ITER (T≈15 keV, n≈10²⁰, τ_E≈3.5 s) devi
ritrovare Q≈10. Documentalo nel README come prova di correttezza.

**Commit:** `feat: 0D power balance, Lawson criterion and Q factor`

---

## Fase 2 — Modello di trasporto 1D radiale (settimane 2–3)

Qui dimostri competenza numerica seria (PDE).

### 2.1 Equazione di diffusione del calore
Risolvere lungo il raggio minore `r ∈ [0, a]`:
```
∂(3/2 nT)/∂t = (1/r) ∂/∂r ( r · n·χ · ∂T/∂r ) + S(r)
```
- [ ] Discretizzazione spaziale a volumi finiti (geometria cilindrica/toroidale)
- [ ] Integrazione temporale **implicita** (Crank-Nicolson o Eulero implicito)
      per stabilità — risoluzione di sistema tridiagonale (Thomas)
- [ ] Sorgenti `S(r)`: riscaldamento alfa + riscaldamento esterno; pozzi: radiazione
- [ ] Condizioni al contorno: simmetria in `r=0`, `T` fissata al bordo

### 2.2 Profili e accoppiamento
- [ ] Profili radiali di `T(r)`, `n(r)` → evoluzione fino allo stato stazionario
- [ ] Calcolo di `τ_E` *emergente* dal profilo (non più imposto)
- [ ] Integrazione dei termini di potenza sul volume → Q "vero" dal profilo

### 2.3 Validazione numerica
- [ ] Test di **conservazione dell'energia** (senza sorgenti/perdite)
- [ ] Confronto con **soluzione analitica** in regime semplice (χ costante, stazionario)
- [ ] Test di convergenza in griglia e in passo temporale

**Commit:** `feat: 1D radial heat transport solver (implicit)`

---

## Fase 3 — Vincoli ingegneristici (settimana 4)

La parte che trasforma "fisica" in "progetto di reattore".

- [ ] **Limite di Greenwald** sulla densità: `n_G = I_p / (π a²)`
- [ ] **Beta limit** (Troyon): rapporto pressione plasma / pressione magnetica
- [ ] **Carico termico sul divertore**: potenza per unità di superficie sui bersagli
- [ ] **Scaling law ITER** per `τ_E` (es. IPB98(y,2)) come confronto al τ_E simulato
- [ ] Diagramma operativo (operational space) con tutti i limiti tracciati

**Commit:** `feat: engineering operational limits (Greenwald, Troyon, divertor)`

---

## Fase 4 — Controllo e tocco finale (opzionale, settimana 5+)

Per chi vuole spingere su control engineering / HPC.

- [ ] **Loop di controllo**: regolatore (PID) sulla potenza di riscaldamento
      esterno per mantenere un Q o una T target
- [ ] Simulazione di un transitorio (es. rampa di densità) con risposta del controllore
- [ ] *(Opzionale HPC)* riscrivere il solver tridiagonale 1D in **C++** e
      richiamarlo da Python con **pybind11**; benchmark Python vs C++ nel README
- [ ] *(Opzionale)* modulo equilibrio **Grad-Shafranov** 2D per geometria realistica

**Commit:** `feat: feedback control of fusion gain`

---

## Standard di qualità (trasversali — è ciò che impressiona i recruiter)

- ✅ **Test fisici significativi**, non solo unitari: conservazione, limiti noti,
  validazione contro letteratura/ITER
- ✅ **Type hints** ovunque + `ruff`/`mypy` puliti
- ✅ **CI verde** su GitHub Actions (badge nel README)
- ✅ **README narrativo**: il problema fisico, le equazioni in LaTeX, i grafici,
  la tabella di validazione contro ITER, le istruzioni d'uso
- ✅ **Notebook dimostrativo** che riproduce le figure principali
- ✅ Commit atomici con messaggi convenzionali (Conventional Commits)

---

## Riferimenti utili

- Bosch & Hale (1992), *Improved formulas for fusion cross-sections and thermal reactivities* — per `<σv>`
- Wesson, *Tokamaks* — testo di riferimento sulla fisica
- Freidberg, *Plasma Physics and Fusion Energy* — bilancio di potenza, ingegneria
- ITER Physics Basis (1999/2007) — scaling laws e parametri di riferimento

---

## Ordine consigliato di esecuzione

**Fase 0 → 1** ti dà già un repo presentabile con un risultato fisico forte (il
diagramma di Lawson + validazione ITER). Fermati lì se hai poco tempo: è già un
ottimo progetto da CV. Le fasi 2–4 lo trasformano in un progetto *notevole*.
