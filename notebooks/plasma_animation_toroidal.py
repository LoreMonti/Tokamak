"""Animazione: il plasma visto DALL'ALTO (piano toroidale) mentre si accende.

Complementare a plasma_animation.py (sezione poloidale). Qui si vede la "ciambella":
il plasma e' un ANELLO tra R0-a e R0+a, con il BUCO centrale (la colonna
centrale del tokamak) a R < R0-a. Per simmetria toroidale e' il profilo di
temperatura del piano mediano T(R) ruotato attorno all'asse della macchina.

Stesso scenario di accensione (sub-ignition, ~16 keV). Genera
docs/plasma_heating_toroidal.gif.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from tokamak.transport import TransportSolver1D

DOCS = Path(__file__).resolve().parent.parent / "docs"
R0, A_MINOR = 6.2, 2.0


def _heating_snapshots(n_frames: int, steps_per_frame: int):
    s = TransportSolver1D(a=A_MINOR, R0=R0, n_e=1.0e20, chi=2.0, T_edge=0.1, n_cells=80)
    s.T = s.T_edge + 0.3 * (1.0 - (s.r / s.a) ** 2)
    shape = np.exp(-((s.r / 0.6) ** 2))
    p_ext = s.heating_density_for_power(60e6, shape)
    snaps = [s.T.copy()]
    for _ in range(n_frames - 1):
        for _ in range(steps_per_frame):
            s.step(dt=1e-2, p_ext=p_ext)
        snaps.append(s.T.copy())
    return snaps, s.r


def main() -> None:
    n_frames, steps_per_frame = 60, 15
    snaps, r_grid = _heating_snapshots(n_frames, steps_per_frame)
    T_vmax = float(snaps[-1][0]) * 1.05

    # Griglia cartesiana della vista dall'alto.
    lim = R0 + A_MINOR + 0.6
    xy = np.linspace(-lim, lim, 320)
    X, Y = np.meshgrid(xy, xy)
    R = np.sqrt(X**2 + Y**2)
    r_minor = np.abs(R - R0)  # distanza dall'asse magnetico (~R0) nel piano mediano
    inside = (R >= R0 - A_MINOR) & (R <= R0 + A_MINOR)  # l'anello di plasma

    def temperature_field(T_profile: np.ndarray) -> np.ndarray:
        T2d = np.interp(r_minor, r_grid, T_profile)
        return np.where(inside, T2d, np.nan)

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color="#0a0a14")

    fig, ax = plt.subplots(figsize=(6.0, 6.2))
    fig.patch.set_facecolor("#0a0a14")
    extent = [-lim, lim, -lim, lim]
    im = ax.imshow(temperature_field(snaps[0]), origin="lower", extent=extent,
                   cmap=cmap, vmin=0.0, vmax=T_vmax, aspect="equal")
    ax.set_xlabel("x [m]", color="white")
    ax.set_ylabel("y [m]", color="white")
    ax.tick_params(colors="white")
    title = ax.set_title("Accensione del plasma (vista dall'alto)  —  t = 0.0 s",
                         color="white", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("T [keV]", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    dt_frame = steps_per_frame * 1e-2

    def update(k: int):
        im.set_data(temperature_field(snaps[k]))
        title.set_text(f"Accensione del plasma (vista dall'alto)  —  t = {k*dt_frame:.1f} s")
        return im, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "plasma_heating_toroidal.gif"
    anim.save(out, writer=PillowWriter(fps=12), dpi=80)
    print(f"T centrale (asse) finale: {snaps[-1][0]:.1f} keV")
    print(f"Anello di plasma: R = {R0 - A_MINOR:.1f}–{R0 + A_MINOR:.1f} m, "
          f"buco centrale per R < {R0 - A_MINOR:.1f} m")
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
