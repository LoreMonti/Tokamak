"""Benchmark del kernel C++ (Thomas) vs scipy (LAPACK) sul solver tridiagonale.

Misura ONESTAMENTE le prestazioni dei due backend:
1. sul singolo solve tridiagonale, al variare della dimensione del sistema;
2. sull'evoluzione completa del solver di trasporto fino allo stato stazionario.

Nota: scipy.solve_banded e' gia' LAPACK compilato, quindi non e' scontato che il
C++ "fatto a mano" vinca. Riportiamo i numeri reali, qualunque siano.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak._tridiag import CPP_AVAILABLE, solve_tridiagonal
from tokamak.transport import TransportSolver1D

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _time_solve(backend: str, n: int, repeats: int) -> float:
    rng = np.random.default_rng(0)
    lower = -rng.random(n)
    upper = -rng.random(n)
    diag = 3.0 + rng.random(n)
    rhs = rng.random(n)
    t0 = time.perf_counter()
    for _ in range(repeats):
        solve_tridiagonal(lower, diag, upper, rhs, backend=backend)
    return (time.perf_counter() - t0) / repeats


def _time_transport(backend: str) -> tuple[float, float]:
    s = TransportSolver1D(a=2.0, R0=6.2, n_e=1e20, chi=1.0, T_edge=0.1,
                          n_cells=200, backend=backend)
    p = 3e5 * np.exp(-((s.r / 0.6) ** 2))
    t0 = time.perf_counter()
    s.solve_steady_state(p_ext=p, dt=2e-3, tol=1e-6, max_steps=8000)
    return time.perf_counter() - t0, float(s.T[0])


def main() -> None:
    if not CPP_AVAILABLE:
        print("Kernel C++ non compilato: esegui `python setup_cpp.py build_ext --inplace`")
        return

    sizes = [50, 100, 200, 500, 1000, 2000]
    reps = 2000
    t_sci = [_time_solve("scipy", n, reps) * 1e6 for n in sizes]  # microsecondi
    t_cpp = [_time_solve("cpp", n, reps) * 1e6 for n in sizes]

    print("Singolo solve tridiagonale (microsecondi):")
    print(f"{'n':>6} {'scipy':>10} {'cpp':>10} {'speedup':>9}")
    for n, ts, tc in zip(sizes, t_sci, t_cpp, strict=True):
        print(f"{n:>6} {ts:>10.2f} {tc:>10.2f} {ts/tc:>8.2f}x")

    # Evoluzione completa del trasporto.
    time_sci, T0_sci = _time_transport("scipy")
    time_cpp, T0_cpp = _time_transport("cpp")
    print("\nEvoluzione completa (200 celle, fino a regime):")
    print(f"  scipy: {time_sci*1e3:.0f} ms (T0={T0_sci:.2f})")
    print(f"  cpp  : {time_cpp*1e3:.0f} ms (T0={T0_cpp:.2f})")
    print(f"  speedup ~{time_sci/time_cpp:.2f}x")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.plot(sizes, t_sci, "o-", label="scipy (LAPACK)", color="navy")
    a1.plot(sizes, t_cpp, "s-", label="C++ (Thomas)", color="crimson")
    a1.set_xlabel("dimensione del sistema n")
    a1.set_ylabel("tempo per solve [µs]")
    a1.set_title("Singolo solve tridiagonale")
    a1.legend()
    a1.grid(True, alpha=0.3)

    a2.bar(["scipy", "C++"], [time_sci * 1e3, time_cpp * 1e3],
           color=["navy", "crimson"])
    a2.set_ylabel("tempo [ms]")
    a2.set_title("Evoluzione completa (200 celle)")
    a2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Benchmark kernel C++ vs scipy")
    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "cpp_benchmark.png"
    fig.savefig(out, dpi=130)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
