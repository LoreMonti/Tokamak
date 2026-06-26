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

# Parte II — Estensioni (fasi 5–12)

> **Stato:** le fasi 0–3 e 4A (controllo PID) sono completate. Questa seconda
> parte trasforma la raccolta di moduli in un **simulatore integrato e
> auto-consistente**, con un ordine guidato dalle DIPENDENZE tra i moduli (non
> filone per filone): un modulo si fa quando quelli da cui dipende sono pronti.

Ordine complessivo: **5 → 6 → 7 → 8 → 9 → 10 → 11 → 12**, con la 4B (kernel C++)
inseribile in qualsiasi momento perché ortogonale al resto.

## Fase 5 — Equilibrio magnetico di Grad-Shafranov (2D)

Il pezzo "iconico" della fisica del plasma: le superfici di flusso a forma di D.

- [ ] Risolvere l'equazione di Grad-Shafranov (PDE ellittica 2D nella sezione
      poloidale R-Z): `Δ*ψ = -μ0 R² dp/dψ - F dF/dψ`
- [ ] Profili sorgente `p(ψ)` e `FF'(ψ)` parametrizzati
- [ ] Solver: differenze finite + iterazione di punto fisso (Picard) sul termine
      non lineare di destra
- [ ] Visualizzazione delle superfici di flusso annidate + ultima superficie chiusa
- [ ] Validazione: confronto con la soluzione analitica di Solov'ev

**Dipendenze:** nessuna (modulo autonomo). **Sblocca:** Fase 9 (controllo forma).
**Commit:** `feat: 2D Grad-Shafranov equilibrium solver`

## Fase 6 — Combustione auto-consistente (burn dynamics)

Chiude il bilancio: oggi `τ_E` emerge ma fuel e cenere sono statici.

- [ ] Evoluzione accoppiata di temperatura, densità di combustibile D-T e
      densità di cenere di elio (He-4 prodotto dalle reazioni)
- [ ] Consumo del combustibile (burn-up) e accumulo di cenere che diluisce il plasma
- [ ] Sorgente di rifornimento (fueling) e tempo di confinamento delle particelle
- [ ] Dinamica di accensione: dimostrare l'ignition come stato auto-sostenuto
- [ ] Validazione: conservazione del numero di particelle, stato stazionario coerente

**Dipendenze:** Fasi 1–2. **Sblocca:** Fasi 8, 11.
**Commit:** `feat: self-consistent D-T burn with helium ash`

## Fase 7 — Radiazione da impurità e Z_eff

- [ ] Modello di radiazione di linea da impurità (es. funzione di raffreddamento)
- [ ] Calcolo di `Z_eff` da una miscela di impurità data
- [ ] Effetto sul bilancio di potenza e possibile collasso radiativo

**Dipendenze:** Fase 6. **Commit:** `feat: impurity radiation and Z_eff`

## Fase 8 — Ottimizzazione del punto operativo

Lega insieme fisica e vincoli in un unico risultato.

- [ ] Funzione obiettivo: massimizzare Q (o densità di potenza di fusione)
- [ ] Vincoli: limiti di Greenwald, Troyon e divertore (Fase 3)
- [ ] Ottimizzatore con vincoli (`scipy.optimize`, eventualmente globale)
- [ ] Mappa del punto operativo ottimo sul diagramma dello spazio operativo

**Dipendenze:** Fasi 3, 6. **Commit:** `feat: constrained operating-point optimization`

## Fase 9 — Controllo di forma e posizione del plasma

Estende il PID della 4A al problema (instabile!) della posizione verticale.

- [ ] Modello ridotto della dinamica verticale del plasma (instabilità intrinseca)
- [ ] Controllore di stabilizzazione su correnti delle bobine di campo poloidale
- [ ] Uso dell'equilibrio di Grad-Shafranov per definire la forma di riferimento

**Dipendenze:** Fasi 4A, 5. **Commit:** `feat: vertical position/shape control`

## Fase 10 — Ciclo del combustibile e tritium breeding

- [ ] Bilancio del trizio: produzione nel mantello (breeding ratio) vs consumo
- [ ] Inventario di trizio e condizione di autosufficienza (TBR > 1)
- [ ] Collegamento con la potenza neutronica del modello (Fase 1)

**Dipendenze:** Fase 1. **Commit:** `feat: tritium fuel cycle and breeding`

## Fase 11 — Emulatore ML del solver (surrogate model)

Fisica + machine learning: predire i risultati del solver, ma ~1000× più veloce.

- [ ] Generare un dataset campionando i parametri ed eseguendo il solver 1D/burn
- [ ] Addestrare un modello (rete neurale o gradient boosting) a predire
      `T(r)`, `τ_E` o Q dai parametri di input
- [ ] Validare accuratezza e speed-up; usarlo per scan/ottimizzazione rapidi

**Dipendenze:** Fase 6. **Commit:** `feat: ML surrogate model of the transport solver`

## Fase 12 — Dashboard interattiva (capstone)

- [ ] App (Streamlit/Plotly) per esplorare dal vivo spazio operativo, profili,
      transitori controllati ed equilibrio
- [ ] Slider sui parametri macchina con aggiornamento in tempo reale (via emulatore)

**Dipendenze:** tutte. **Commit:** `feat: interactive exploration dashboard`

## Fase 4B — Kernel C++ ad alte prestazioni (inseribile quando si vuole)

- [ ] Riscrivere il solver tridiagonale 1D in C++, esposto con **pybind11**
- [ ] Validazione numerica contro la versione Python (stessi risultati)
- [ ] Benchmark Python vs C++ documentato nel README
- [ ] Integrazione nella build e nella CI

**Dipendenze:** Fase 2. **Commit:** `perf: C++ tridiagonal kernel via pybind11`

---

# Parte III — Machine learning avanzato (fasi 13–16)

> Quattro tecniche ML distinte, ciascuna ancorata alla fisica del progetto:
> classificazione, ottimizzazione bayesiana, deep learning di regressione e
> reinforcement learning. Ordine guidato da dipendenze e rischio crescente:
> **13 → 14 → 15 → 16**.

## Fase 13 — Predizione di disruption (classificazione)

Una delle applicazioni ML piu' reali in fusione: predire se un punto operativo
e' stabile o va in disruption.

- [ ] Generare un dataset etichettato (stabile / disrupt) dai vincoli operativi
      (Greenwald, Troyon) ed eventuale runaway termico
- [ ] Addestrare un classificatore (gradient boosting / logistic) e valutarlo
      (accuratezza, ROC-AUC, matrice di confusione) su un test set
- [ ] Visualizzare la "regione sicura" appresa sul piano (T, n_e)

**Dipendenze:** Fasi 3, 6. **Commit:** `feat: ML disruption prediction`

## Fase 14 — Ottimizzazione bayesiana del punto operativo

- [ ] Ottimizzazione bayesiana (GP della Fase 11 + funzione di acquisizione,
      es. Expected Improvement) per massimizzare Q/P_fus con poche valutazioni
- [ ] Confronto col risultato dell'ottimizzatore della Fase 8 e con la
      convergenza al variare del numero di valutazioni

**Dipendenze:** Fasi 8, 11. **Commit:** `feat: Bayesian optimization of operating point`

## Fase 15 — Emulatore deep-learning dei profili (PyTorch)

Estende l'emulatore scalare (Fase 11) all'intero profilo radiale.

- [ ] Rete neurale (PyTorch) che predice il profilo completo T(r) dai parametri
      (output vettoriale), addestrata sui dati del solver
- [ ] Validazione: errore sul profilo, parity plot, speed-up vs solver
- [ ] Confronto col surrogate GP scalare (tecnica e task diversi)

**Dipendenze:** Fasi 2, 11. **Commit:** `feat: PyTorch neural-network profile emulator`

## Fase 16 — Controllo con Reinforcement Learning

Sostituire il PID fatto a mano con un agente RL (sulla scia di DeepMind/TCV).

- [ ] Ambiente in stile Gym che avvolge la dinamica (combustione/trasporto):
      stato = (T, ...), azione = potenza di riscaldamento, reward = vicinanza
      al target meno il costo del controllo
- [ ] Addestrare un agente (PPO, stable-baselines3) con seed fissati
- [ ] Confronto RL vs PID su inseguimento del target e reiezione del disturbo

**Dipendenze:** Fasi 4A, 6. **Commit:** `feat: reinforcement-learning plasma control`

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
