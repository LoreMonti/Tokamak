"""Animazione: il plasma che si accende nella sezione a D del tokamak.

Combina due fasi gia' esistenti, senza nuova fisica:
- Fase 5 (Grad-Shafranov): geometria 2D, la funzione di flusso psi(R,Z) a D;
- Fase 2 (trasporto): la temperatura T(r, t) che evolve nel tempo.

La temperatura e' costante su ogni superficie magnetica, quindi mappiamo il
profilo 1D T(r) sulle superfici 2D usando psi come coordinata radiale:
ottenuto un campo T(R,Z) a forma di D che animiamo mentre il plasma si scalda
dal bordo freddo al centro caldo.

NOTA: modello RIDOTTO (trasporto 1D mappato su un equilibrio 2D statico), non
una simulazione di trasporto 2D completa. Rappresenta fedelmente l'accensione
nella sezione poloidale del tokamak.

Genera docs/plasma_heating.gif.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from tokamak.equilibrium import GradShafranovSolver
from tokamak.transport import TransportSolver1D

DOCS = Path(__file__).resolve().parent.parent / "docs"

R0, A_MINOR = 6.2, 2.0
KAPPA, DELTA = 1.7, 0.33


def _equilibrium() -> tuple[GradShafranovSolver, np.ndarray, np.ndarray]:
    """Risolve l'equilibrio a D e restituisce il solver, la coordinata di
    flusso normalizzata s in [0,1] (0=asse, 1=bordo) e la maschera del plasma."""
    eq = GradShafranovSolver(
        R_min=3.6, R_max=8.8, Z_min=-3.8, Z_max=3.8, nR=120, nZ=170
    )
    eq.set_d_shaped_boundary(R0=R0, a=A_MINOR, kappa=KAPPA, delta=DELTA)

    def rhs(psi, RR):
        psi_n = np.clip(psi / (np.max(psi) + 1e-30), 0.0, None)
        return -120.0 * (0.7 * (RR / R0) ** 2 + 0.3) * (1.0 + 1.8 * psi_n)

    eq.solve_picard(rhs, max_iter=300, tol=1e-8, relax=0.4)

    psi_n = np.clip(eq.psi / eq.psi.max(), 0.0, 1.0)
    inside = (~eq._fixed) & (eq.psi > 0.0)
    s = np.sqrt(np.clip(1.0 - psi_n, 0.0, 1.0))  # 0 all'asse, 1 al bordo
    return eq, s, inside


def _heating_snapshots(
    n_frames: int, steps_per_frame: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """Evolve il trasporto da freddo e raccoglie istantanee di T(r)."""
    # chi e potenza scelti per un punto operativo stabile (~16 keV, sub-ignition):
    # evita la runaway termica che renderebbe la scala colore illeggibile.
    s = TransportSolver1D(a=A_MINOR, R0=R0, n_e=1.0e20, chi=2.0, T_edge=0.1, n_cells=80)
    s.T = s.T_edge + 0.3 * (1.0 - (s.r / s.a) ** 2)  # partenza fredda
    shape = np.exp(-((s.r / 0.6) ** 2))
    p_ext = s.heating_density_for_power(60e6, shape)

    snaps = [s.T.copy()]
    for _ in range(n_frames - 1):
        for _ in range(steps_per_frame):
            s.step(dt=1e-2, p_ext=p_ext)
        snaps.append(s.T.copy())
    return snaps, s.r


def main() -> None:
    eq, s_coord, inside = _equilibrium()
    n_frames, steps_per_frame = 60, 15
    snaps, r_grid = _heating_snapshots(n_frames, steps_per_frame)
    T_vmax = float(snaps[-1][0]) * 1.05  # T centrale finale

    def temperature_field(T_profile: np.ndarray) -> np.ndarray:
        """Mappa T(r) sulla sezione 2D via la coordinata di flusso s."""
        T2d = np.interp(s_coord * A_MINOR, r_grid, T_profile)
        return np.where(inside, T2d, np.nan)

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color="#0a0a14")  # sfondo (fuori dal plasma)

    fig, ax = plt.subplots(figsize=(5.0, 6.4))
    fig.patch.set_facecolor("#0a0a14")
    extent = [eq.R_min, eq.R_max, eq.Z_min, eq.Z_max]
    im = ax.imshow(
        temperature_field(snaps[0]).T, origin="lower", extent=extent,
        cmap=cmap, vmin=0.0, vmax=T_vmax, aspect="equal",
    )
    ax.set_xlabel("R [m]", color="white")
    ax.set_ylabel("Z [m]", color="white")
    ax.tick_params(colors="white")
    title = ax.set_title("Accensione del plasma  —  t = 0.0 s", color="white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("T [keV]", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    dt_frame = steps_per_frame * 1e-2

    def update(k: int):
        im.set_data(temperature_field(snaps[k]).T)
        title.set_text(f"Accensione del plasma  —  t = {k * dt_frame:.1f} s")
        return im, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "plasma_heating.gif"
    anim.save(out, writer=PillowWriter(fps=12), dpi=80)
    print(f"T centrale finale: {snaps[-1][0]:.1f} keV")
    print(f"Frame: {n_frames}, durata simulata: {n_frames * dt_frame:.1f} s")
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
