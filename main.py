"""Punto d'ingresso unico del progetto Tokamak.

Esegue, in un colpo solo, tutte le fasi (generando i grafici in docs/) e,
opzionalmente, la suite di test. Le fasi 15 (PyTorch) e 16 (RL) mostrano una
barra di avanzamento e sono saltate con un avviso se i rispettivi extra
([ml]/[rl]) non sono installati.

Esempi
------
    python main.py                 # esegue tutte le fasi (genera le figure)
    python main.py --test          # esegue prima i test, poi le fasi
    python main.py --only-test     # esegue soltanto i test
    python main.py --phase 15 16   # esegue solo le fasi indicate
"""

from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"

# (numero fase, titolo, script che la dimostra)
PHASES = {
    1: ("Modello 0D + criterio di Lawson", NOTEBOOKS / "lawson_diagram.py"),
    2: ("Trasporto radiale 1D", NOTEBOOKS / "radial_profile.py"),
    3: ("Vincoli ingegneristici / spazio operativo", NOTEBOOKS / "operational_space.py"),
    4: ("Controllo PID in retroazione", NOTEBOOKS / "control_demo.py"),
    5: ("Equilibrio di Grad-Shafranov (2D)", NOTEBOOKS / "flux_surfaces.py"),
    6: ("Combustione auto-consistente D-T", NOTEBOOKS / "burn_demo.py"),
    7: ("Radiazione da impurità / collasso radiativo", NOTEBOOKS / "radiative_collapse.py"),
    8: ("Ottimizzazione del punto operativo", NOTEBOOKS / "optimum_demo.py"),
    9: ("Controllo di stabilità verticale", NOTEBOOKS / "vertical_control.py"),
    10: ("Ciclo del combustibile / trizio", NOTEBOOKS / "fuel_cycle_demo.py"),
    11: ("Emulatore ML del solver (surrogate)", NOTEBOOKS / "surrogate_demo.py"),
    13: ("Predizione di disruption (ML)", NOTEBOOKS / "disruption_demo.py"),
    14: ("Ottimizzazione bayesiana", NOTEBOOKS / "bayesopt_demo.py"),
    15: ("Emulatore deep-learning dei profili (PyTorch)",
         NOTEBOOKS / "profile_emulator_demo.py"),
    16: ("Controllo con Reinforcement Learning (PPO)",
         NOTEBOOKS / "rl_control_demo.py"),
}

# Fasi che richiedono dipendenze opzionali (saltate con un avviso se mancano).
_OPTIONAL_PHASES = {15: "[ml] (torch)", 16: "[rl] (gymnasium + stable-baselines3)"}


def _banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def run_tests() -> int:
    """Esegue pytest e restituisce il codice di uscita."""
    _banner("TEST (pytest)")
    return subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)


def run_phase(num: int) -> None:
    """Esegue lo script dimostrativo di una fase.

    Le fasi con dipendenze opzionali (15, 16) vengono saltate con un avviso se i
    pacchetti non sono installati, invece di interrompere l'intera esecuzione.
    """
    title, script = PHASES[num]
    _banner(f"FASE {num} — {title}")
    try:
        runpy.run_path(str(script), run_name="__main__")
    except ImportError as exc:
        extra = _OPTIONAL_PHASES.get(num, "una dipendenza")
        print(f"[salto] Fase {num}: manca {extra}. Installa con "
              f'`pip install -e ".{extra.split()[0]}"`. ({exc})')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Esegue le fasi del progetto Tokamak.")
    parser.add_argument("--test", action="store_true", help="esegue anche i test")
    parser.add_argument("--only-test", action="store_true", help="esegue solo i test")
    parser.add_argument(
        "--phase",
        type=int,
        nargs="+",
        choices=sorted(PHASES),
        metavar="N",
        help="esegue solo le fasi indicate",
    )
    args = parser.parse_args(argv)

    if args.only_test:
        return run_tests()

    if args.test:
        code = run_tests()
        if code != 0:
            print("\n[!] Test falliti: interrompo prima di generare le figure.")
            return code

    for num in args.phase or sorted(PHASES):
        run_phase(num)

    _banner("FATTO — figure salvate in docs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
