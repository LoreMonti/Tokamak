"""Demo di combustione auto-consistente: accensione e accumulo di cenere.

Scenario:
1. Un breve riscaldamento esterno porta il plasma in temperatura.
2. A t = t_off il riscaldamento si SPEGNE: se le alfa bastano, la combustione
   si auto-sostiene (ignition).
3. Nel frattempo il combustibile si consuma e la cenere di elio si accumula,
   alzando Z_eff e le perdite radiative.

Mostra l'interazione tra accensione, consumo del combustibile e avvelenamento
da cenere, che nessun modello statico puo' catturare.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from tokamak.burn import fusion_power_density_burn, simulate_burn

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    t_off = 5.0
    s = simulate_burn(
        n_DT0=1.1e20,
        n_He0=0.0,
        T0_keV=5.0,
        tau_E=4.0,
        tau_p=8.0,
        fueling=2.5e18,
        p_ext=4e5,
        p_ext_off_time=t_off,
        t_end=30.0,
    )

    print(f"T iniziale       = {s.T_keV[0]:.1f} keV")
    print(f"T a regime       = {s.T_keV[-1]:.1f} keV")
    print(f"Riscaldamento spento a t = {t_off} s")
    print(f"P_alpha finale   = {s.P_alpha[-1] / 1e3:.1f} kW/m^3")
    print(f"Cenere He finale = {s.n_He[-1]:.2e} m^-3 "
          f"({100 * s.n_He[-1] / (s.n_DT[-1] + s.n_He[-1]):.1f}% dei ioni)")

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8), sharex=True)

    axes[0].plot(s.time, s.T_keV, color="crimson", lw=2)
    axes[0].axvline(t_off, color="gray", ls=":", lw=1)
    axes[0].set_ylabel("T [keV]")
    axes[0].set_title("Combustione auto-consistente D-T")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(s.time, s.n_DT / 1e20, color="navy", lw=2, label="combustibile D-T")
    axes[1].plot(s.time, s.n_He / 1e20, color="orange", lw=2, label="cenere He")
    axes[1].axvline(t_off, color="gray", ls=":", lw=1)
    axes[1].set_ylabel(r"densita' [$10^{20}$ m$^{-3}$]")
    axes[1].legend(loc="center right")
    axes[1].grid(True, alpha=0.3)

    p_fus = fusion_power_density_burn(s)
    axes[2].plot(s.time, s.P_alpha / 1e3, color="crimson", lw=2, label=r"$P_\alpha$ (self-heating)")
    axes[2].plot(s.time, s.P_ext / 1e3, color="green", lw=2, ls="--", label=r"$P_{ext}$")
    axes[2].plot(s.time, p_fus / 1e3, color="black", lw=1, alpha=0.6, label=r"$P_{fus}$ totale")
    axes[2].axvline(t_off, color="gray", ls=":", lw=1, label="spegnimento")
    axes[2].set_ylabel(r"potenza [kW/m$^3$]")
    axes[2].set_xlabel("tempo [s]")
    axes[2].legend(loc="center right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "burn_demo.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
