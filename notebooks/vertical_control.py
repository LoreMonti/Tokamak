"""Stabilizzazione verticale di un plasma allungato.

Confronta:
- ANELLO APERTO: senza controllo, il plasma instabile fugge esponenzialmente.
- ANELLO CHIUSO: il controllore PD lo tiene centrato e respinge un disturbo
  impulsivo (un "calcio" verticale).

Mostra perche' i tokamak allungati richiedono un controllo attivo veloce.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from tokamak.control import PIDController
from tokamak.stability import simulate_vertical_control, vertical_growth_rate

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    elongation = 1.7
    gamma = vertical_growth_rate(elongation)
    b = 1.0
    print(f"Elongazione kappa = {elongation}  ->  gamma = {gamma:.0f} 1/s "
          f"(tempo di crescita ~ {1000 / gamma:.1f} ms)")

    dt, n = 1e-4, 6000
    z0 = 0.01  # 1 cm di spostamento iniziale

    # Anello aperto: nessun controllo.
    open_loop = simulate_vertical_control(
        gamma=gamma, b=b, controller=None, z0=z0, dt=dt, n_steps=n
    )

    # Anello chiuso: PD (= PID con ki=0), con saturazione dell'attuatore, e un
    # calcio di disturbo a meta' simulazione.
    # Il comando di picco al transitorio e' ~ kp*z0; il limite deve superarlo.
    limit = 1500.0
    pd = PIDController(
        kp=1.8 * gamma**2, ki=0.0, kd=300.0, setpoint=0.0,
        output_min=-limit, output_max=limit,
    )
    closed_loop = simulate_vertical_control(
        gamma=gamma, b=b, controller=pd, z0=z0, dt=dt, n_steps=n,
        actuator_limit=limit, kicks={n // 2: 0.5},
    )
    t_kick = (n // 2) * dt
    print(f"Anello aperto:  z finale = {open_loop.z[-1]:.3f} m (fuga)")
    print(f"Anello chiuso:  z finale = {closed_loop.z[-1]:.2e} m (stabilizzato)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)

    ax1.plot(open_loop.time * 1e3, open_loop.z * 100, color="gray", lw=2,
             label="open loop (no control)")
    ax1.plot(closed_loop.time * 1e3, closed_loop.z * 100, color="crimson", lw=2,
             label="closed loop (PD)")
    ax1.axvline(t_kick * 1e3, color="black", ls=":", lw=1, label="disturbance")
    ax1.set_ylabel("z position [cm]")
    ax1.set_ylim(-5, 5)
    ax1.set_title(f"Vertical stability control (kappa={elongation})")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(closed_loop.time * 1e3, closed_loop.u, color="navy", lw=1.5)
    ax2.axvline(t_kick * 1e3, color="black", ls=":", lw=1)
    ax2.set_xlabel("time [ms]")
    ax2.set_ylabel("coil command u")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "vertical_control.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
