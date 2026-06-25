"""Solutore tridiagonale: kernel C++ (se disponibile) o fallback scipy.

Espone `solve_tridiagonal(lower, diag, upper, rhs, backend)` usato dallo schema
implicito in transport.py. Il modulo C++ (`_tridiag_cpp`, compilato con
setup_cpp.py) e' OPZIONALE: se manca, si usa scipy.solve_banded, cosi' il
pacchetto resta installabile in puro Python.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

try:
    from . import _tridiag_cpp  # estensione compilata

    CPP_AVAILABLE = True
except ImportError:  # pragma: no cover - dipende dalla build
    _tridiag_cpp = None
    CPP_AVAILABLE = False


def solve_tridiagonal(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
    backend: str = "scipy",
) -> NDArray[np.float64]:
    """Risolve il sistema tridiagonale A x = rhs.

    Parameters
    ----------
    lower, diag, upper:
        sotto-diagonale (lower[0] inutilizzato), diagonale, sopra-diagonale
        (upper[-1] inutilizzato).
    backend:
        "scipy" (LAPACK via solve_banded) o "cpp" (kernel C++ di Thomas).
    """
    if backend == "cpp":
        if not CPP_AVAILABLE:
            raise RuntimeError(
                "kernel C++ non disponibile: compila con "
                "`python setup_cpp.py build_ext --inplace`"
            )
        return _tridiag_cpp.solve_tridiagonal(lower, diag, upper, rhs)

    if backend == "scipy":
        n = diag.size
        ab = np.zeros((3, n))
        ab[0, 1:] = upper[:-1]  # sopra-diagonale
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]  # sotto-diagonale
        return solve_banded((1, 1), ab, rhs)

    raise ValueError(f"backend sconosciuto: {backend!r}")
