# Tokamak — Transport and power-balance simulator of a fusion plasma

A simulator of the confinement physics of a fusion reactor (tokamak), from the
0D power balance to 1D radial transport, with real engineering constraints.

**Project goal:** compute the fusion gain factor *Q* and the *ignition*
conditions (Lawson criterion) for realistic ITER-like configurations, starting
from physical first principles.

---

## Quick glossary of key concepts

| Symbol | Meaning | Order of magnitude (ITER) |
|---|---|---|
| `T` | Plasma temperature | ~15 keV (~150 million K) |
| `n` | Density (ions/electrons) | ~10²⁰ m⁻³ |
| `τ_E` | Energy confinement time | ~3–4 s |
| `Q` | Fusion power / injected power | ~10 (ITER target) |
| `nTτ_E` | Triple product (Lawson criterion) | ~3×10²¹ keV·s·m⁻³ |

**Reference reaction:** D + T → ⁴He (3.5 MeV) + n (14.1 MeV).
Only the alpha (3.5 MeV) stays confined and heats the plasma; the neutron escapes.

---

## Phase 0 — Repository setup (day 1)

Goal: a professional repo from the very first commit.

- [ ] `git init`, license (MIT), Python `.gitignore`
- [ ] Package structure:
  ```
  tokamak/
  ├── src/tokamak/
  │   ├── __init__.py
  │   ├── constants.py        # physical constants, masses, reaction energies
  │   ├── reactivity.py       # <σv> for D-T as a function of T
  │   ├── power_balance.py    # 0D model
  │   └── transport.py        # 1D model (phase 2)
  ├── tests/
  ├── notebooks/              # exploration + figures for the README
  ├── docs/
  ├── pyproject.toml
  ├── README.md
  └── ROADMAP.md
  ```
- [ ] `pyproject.toml` with dependencies: `numpy`, `scipy`, `matplotlib`
- [ ] Tooling: `ruff` (lint+format), `pytest`, GitHub Actions CI (lint + test)
- [ ] Initial README with the project's purpose

**Commit:** `chore: project scaffold and CI`

---

## Phase 1 — 0D model: power balance and Lawson criterion (week 1)

The physics/engineering core. No PDE: just algebra and a reactivity.

### 1.1 D-T reactivity `<σv>(T)`
- [ ] Implement the **Bosch-Hale** parametrization for D-T `<σv>`
      (standard analytic formula, valid 0.2–100 keV)
- [ ] Test: comparison with tabulated literature values at 10 and 20 keV

### 1.2 Power terms (power density, W/m³)
- [ ] **Fusion:** `P_fus = n_D · n_T · <σv> · E_fus`
- [ ] **Alpha heating:** `P_α = (1/5) · P_fus` (only 3.5/17.6 MeV stays)
- [ ] **Bremsstrahlung radiation:** `P_brem ∝ Z_eff · n_e² · √T`
- [ ] **Confinement losses:** `P_loss = W / τ_E`, with `W = 3 n T`

### 1.3 Equilibrium and figures of merit
- [ ] Compute the **Q = P_fus / P_heat** factor
- [ ] **Ignition** condition: `P_α ≥ P_loss + P_brem`
- [ ] **Lawson criterion**: derive the minimum `n·τ_E` vs `T` curve

### 1.4 Visual deliverable (key for the CV)
- [ ] **Lawson diagram**: `T` vs `n·τ_E` plane with break-even, Q=10 and
      ignition curves; mark the ITER operating point
- [ ] Save the figures in `docs/` and include them in the README

**Validation:** with ITER parameters (T≈15 keV, n≈10²⁰, τ_E≈3.5 s) you must
recover Q≈10. Document it in the README as a correctness check.

**Commit:** `feat: 0D power balance, Lawson criterion and Q factor`

---

## Phase 2 — 1D radial transport model (weeks 2–3)

Here you demonstrate serious numerical skill (PDE).

### 2.1 Heat diffusion equation
Solve along the minor radius `r ∈ [0, a]`:
```
∂(3/2 nT)/∂t = (1/r) ∂/∂r ( r · n·χ · ∂T/∂r ) + S(r)
```
- [ ] Finite-volume spatial discretization (cylindrical/toroidal geometry)
- [ ] **Implicit** time integration (Crank-Nicolson or backward Euler) for
      stability — solving a tridiagonal system (Thomas)
- [ ] Sources `S(r)`: alpha heating + external heating; sinks: radiation
- [ ] Boundary conditions: symmetry at `r=0`, `T` fixed at the edge

### 2.2 Profiles and coupling
- [ ] Radial profiles of `T(r)`, `n(r)` → evolution to steady state
- [ ] Compute the `τ_E` *emerging* from the profile (no longer imposed)
- [ ] Integrate the power terms over the volume → "true" Q from the profile

### 2.3 Numerical validation
- [ ] **Energy conservation** test (no sources/losses)
- [ ] Comparison with the **analytic solution** in a simple regime (constant χ, steady)
- [ ] Grid and time-step convergence tests

**Commit:** `feat: 1D radial heat transport solver (implicit)`

---

## Phase 3 — Engineering constraints (week 4)

The part that turns "physics" into "reactor design".

- [ ] **Greenwald limit** on density: `n_G = I_p / (π a²)`
- [ ] **Beta limit** (Troyon): plasma pressure / magnetic pressure ratio
- [ ] **Divertor heat load**: power per unit area on the targets
- [ ] **ITER scaling law** for `τ_E` (e.g. IPB98(y,2)) as a comparison to the simulated τ_E
- [ ] Operational-space diagram with all the limits drawn

**Commit:** `feat: engineering operational limits (Greenwald, Troyon, divertor)`

---

## Phase 4 — Control and finishing touches (optional, week 5+)

For pushing into control engineering / HPC.

- [ ] **Control loop**: a (PID) regulator on the external heating power to hold
      a target Q or T
- [ ] Simulation of a transient (e.g. a density ramp) with the controller's response
- [ ] *(Optional HPC)* rewrite the 1D tridiagonal solver in **C++** and call it
      from Python with **pybind11**; Python vs C++ benchmark in the README
- [ ] *(Optional)* 2D **Grad-Shafranov** equilibrium module for realistic geometry

**Commit:** `feat: feedback control of fusion gain`

---

# Part II — Extensions (phases 5–12)

> **Status:** phases 0–3 and 4A (PID control) are complete. This second part
> turns the collection of modules into an **integrated, self-consistent
> simulator**, with an order driven by the DEPENDENCIES between modules (not
> theme by theme): a module is done once those it depends on are ready.

Overall order: **5 → 6 → 7 → 8 → 9 → 10 → 11 → 12**, with 4B (C++ kernel)
insertable at any time since it is orthogonal to the rest.

## Phase 5 — Grad-Shafranov magnetic equilibrium (2D)

The "iconic" piece of plasma physics: the D-shaped flux surfaces.

- [ ] Solve the Grad-Shafranov equation (2D elliptic PDE in the R-Z poloidal
      cross-section): `Δ*ψ = -μ0 R² dp/dψ - F dF/dψ`
- [ ] Parametrized source profiles `p(ψ)` and `FF'(ψ)`
- [ ] Solver: finite differences + fixed-point iteration (Picard) on the
      nonlinear right-hand side
- [ ] Visualization of the nested flux surfaces + last closed surface
- [ ] Validation: comparison with the analytic Solov'ev solution

**Dependencies:** none (standalone module). **Unlocks:** Phase 9 (shape control).
**Commit:** `feat: 2D Grad-Shafranov equilibrium solver`

## Phase 6 — Self-consistent burn (burn dynamics)

Closes the balance: today `τ_E` emerges but fuel and ash are static.

- [ ] Coupled evolution of temperature, D-T fuel density and helium ash density
      (He-4 produced by the reactions)
- [ ] Fuel consumption (burn-up) and ash accumulation that dilutes the plasma
- [ ] Refueling source (fueling) and particle confinement time
- [ ] Ignition dynamics: demonstrate ignition as a self-sustained state
- [ ] Validation: particle-number conservation, consistent steady state

**Dependencies:** Phases 1–2. **Unlocks:** Phases 8, 11.
**Commit:** `feat: self-consistent D-T burn with helium ash`

## Phase 7 — Impurity radiation and Z_eff

- [ ] Impurity line-radiation model (e.g. cooling function)
- [ ] Compute `Z_eff` from a given impurity mixture
- [ ] Effect on the power balance and possible radiative collapse

**Dependencies:** Phase 6. **Commit:** `feat: impurity radiation and Z_eff`

## Phase 8 — Operating-point optimization

Ties physics and constraints together into a single result.

- [ ] Objective function: maximize Q (or fusion power density)
- [ ] Constraints: Greenwald, Troyon and divertor limits (Phase 3)
- [ ] Constrained optimizer (`scipy.optimize`, possibly global)
- [ ] Map the optimal operating point on the operational-space diagram

**Dependencies:** Phases 3, 6. **Commit:** `feat: constrained operating-point optimization`

## Phase 9 — Plasma shape and position control

Extends the Phase-4A PID to the (unstable!) vertical-position problem.

- [ ] Reduced model of the plasma's vertical dynamics (intrinsic instability)
- [ ] Stabilizing controller on the poloidal-field coil currents
- [ ] Use the Grad-Shafranov equilibrium to define the reference shape

**Dependencies:** Phases 4A, 5. **Commit:** `feat: vertical position/shape control`

## Phase 10 — Fuel cycle and tritium breeding

- [ ] Tritium balance: production in the blanket (breeding ratio) vs consumption
- [ ] Tritium inventory and self-sufficiency condition (TBR > 1)
- [ ] Link with the model's neutron power (Phase 1)

**Dependencies:** Phase 1. **Commit:** `feat: tritium fuel cycle and breeding`

## Phase 11 — ML solver emulator (surrogate model)

Physics + machine learning: predict the solver's results, but ~1000× faster.

- [ ] Generate a dataset by sampling parameters and running the 1D/burn solver
- [ ] Train a model (neural network or gradient boosting) to predict
      `T(r)`, `τ_E` or Q from the input parameters
- [ ] Validate accuracy and speed-up; use it for fast scans/optimization

**Dependencies:** Phase 6. **Commit:** `feat: ML surrogate model of the transport solver`

## Phase 12 — Interactive dashboard (capstone)

- [ ] App (Streamlit/Plotly) to explore operational space, profiles, controlled
      transients and equilibrium live
- [ ] Sliders on the machine parameters with real-time updates (via the emulator)

**Dependencies:** all. **Commit:** `feat: interactive exploration dashboard`

## Phase 4B — High-performance C++ kernel (insertable anytime)

- [ ] Rewrite the 1D tridiagonal solver in C++, exposed with **pybind11**
- [ ] Numerical validation against the Python version (identical results)
- [ ] Python vs C++ benchmark documented in the README
- [ ] Integration into the build and CI

**Dependencies:** Phase 2. **Commit:** `perf: C++ tridiagonal kernel via pybind11`

---

# Part III — Advanced machine learning (phases 13–16)

> Four distinct ML techniques, each anchored to the project's physics:
> classification, Bayesian optimization, regression deep learning and
> reinforcement learning. Order driven by dependencies and increasing risk:
> **13 → 14 → 15 → 16**.

## Phase 13 — Disruption prediction (classification)

One of the most real ML applications in fusion: predict whether an operating
point is stable or disrupts.

- [ ] Generate a labeled dataset (stable / disrupt) from the operational limits
      (Greenwald, Troyon) and possible thermal runaway
- [ ] Train a classifier (gradient boosting / logistic) and evaluate it
      (accuracy, ROC-AUC, confusion matrix) on a test set
- [ ] Visualize the learned "safe region" on the (T, n_e) plane

**Dependencies:** Phases 3, 6. **Commit:** `feat: ML disruption prediction`

## Phase 14 — Bayesian optimization of the operating point

- [ ] Bayesian optimization (Phase-11 GP + acquisition function, e.g. Expected
      Improvement) to maximize Q/P_fus with few evaluations
- [ ] Comparison with the Phase-8 optimizer's result and with the convergence as
      the number of evaluations varies

**Dependencies:** Phases 8, 11. **Commit:** `feat: Bayesian optimization of operating point`

## Phase 15 — Deep-learning profile emulator (PyTorch)

Extends the scalar emulator (Phase 11) to the full radial profile.

- [ ] Neural network (PyTorch) that predicts the full profile T(r) from the
      parameters (vector output), trained on solver data
- [ ] Validation: profile error, parity plot, speed-up vs solver
- [ ] Comparison with the scalar GP surrogate (different technique and task)

**Dependencies:** Phases 2, 11. **Commit:** `feat: PyTorch neural-network profile emulator`

## Phase 16 — Reinforcement-learning control

Replace the hand-tuned PID with an RL agent (following DeepMind/TCV).

- [ ] Gym-style environment wrapping the dynamics (burn/transport):
      state = (T, ...), action = heating power, reward = closeness to the target
      minus the control cost
- [ ] Train an agent (PPO, stable-baselines3) with fixed seeds
- [ ] Compare RL vs PID on target tracking and disturbance rejection

**Dependencies:** Phases 4A, 6. **Commit:** `feat: reinforcement-learning plasma control`

---

## Quality standards (cross-cutting — this is what impresses recruiters)

- ✅ **Meaningful physics tests**, not just unit tests: conservation, known
  limits, validation against literature/ITER
- ✅ **Type hints** everywhere + clean `ruff`/`mypy`
- ✅ **Green CI** on GitHub Actions (badge in the README)
- ✅ **Narrative README**: the physical problem, the equations in LaTeX, the
  plots, the ITER validation table, the usage instructions
- ✅ **Demo notebooks** that reproduce the main figures
- ✅ Atomic commits with conventional messages (Conventional Commits)

---

## Useful references

- Bosch & Hale (1992), *Improved formulas for fusion cross-sections and thermal reactivities* — for `<σv>`
- Wesson, *Tokamaks* — reference physics textbook
- Freidberg, *Plasma Physics and Fusion Energy* — power balance, engineering
- ITER Physics Basis (1999/2007) — scaling laws and reference parameters

---

## Recommended order of execution

**Phase 0 → 1** already gives you a presentable repo with a strong physics result
(the Lawson diagram + ITER validation). Stop there if you are short on time: it
is already a great CV project. Phases 2–4 turn it into a *remarkable* one.
