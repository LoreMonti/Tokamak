r"""Fase 5 — Equilibrio magnetico di Grad-Shafranov (2D).

L'equilibrio MHD assialsimmetrico di un plasma di tokamak e' descritto dalla
funzione di flusso poloidale psi(R, Z), soluzione dell'equazione di
Grad-Shafranov:

    Delta* psi = -mu0 R^2 dp/dpsi - F dF/dpsi

con l'operatore ellittico (in coordinate cilindriche R, Z):

    Delta* psi = d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2

Le CURVE DI LIVELLO di psi sono le superfici magnetiche: il plasma vive su
superfici annidate, e il massimo di psi e' l'asse magnetico.

Numerica
--------
- Discretizziamo Delta* a differenze finite su una griglia rettangolare (R, Z).
- Condizione al contorno di Dirichlet: psi assegnata sul bordo (psi=0 = bordo
  del plasma, "fixed boundary").
- L'operatore lineare diventa una grande matrice SPARSA, risolta con spsolve.
- Il termine di destra dipende da psi (non lineare): iteriamo con PICARD,
  ricostruendo la sorgente dall'ultima psi finche' il profilo si stabilizza.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

MU_0 = 4.0e-7 * np.pi


@dataclass
class GradShafranovSolver:
    """Solver a differenze finite per l'equazione di Grad-Shafranov.

    La griglia copre [R_min, R_max] x [Z_min, Z_max]. Il bordo e' Dirichlet.
    """

    R_min: float
    R_max: float
    Z_min: float
    Z_max: float
    nR: int = 65
    nZ: int = 129

    R: NDArray[np.float64] = field(init=False)
    Z: NDArray[np.float64] = field(init=False)
    RR: NDArray[np.float64] = field(init=False)
    ZZ: NDArray[np.float64] = field(init=False)
    psi: NDArray[np.float64] = field(init=False)
    _A: csr_matrix = field(init=False)

    def __post_init__(self) -> None:
        self.R = np.linspace(self.R_min, self.R_max, self.nR)
        self.Z = np.linspace(self.Z_min, self.Z_max, self.nZ)
        # Indici: RR varia lungo le righe (R), ZZ lungo le colonne (Z).
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z, indexing="ij")
        self.psi = np.zeros((self.nR, self.nZ))
        self._A = self._build_operator()

    # --- Costruzione dell'operatore Delta* (matrice sparsa) ----------------
    def _build_operator(self) -> csr_matrix:
        """Assembla Delta* come matrice sparsa, con righe identita' sul bordo."""
        nR, nZ = self.nR, self.nZ
        dR = self.R[1] - self.R[0]
        dZ = self.Z[1] - self.Z[0]
        n = nR * nZ

        def idx(i: int, j: int) -> int:
            return i * nZ + j

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for i in range(nR):
            for j in range(nZ):
                k = idx(i, j)
                # Bordo: Dirichlet -> riga identita'.
                if i == 0 or i == nR - 1 or j == 0 or j == nZ - 1:
                    rows.append(k)
                    cols.append(k)
                    data.append(1.0)
                    continue
                Ri = self.R[i]
                # d2/dR2 - (1/R) d/dR
                c_ip = 1.0 / dR**2 - 1.0 / (Ri * 2.0 * dR)
                c_im = 1.0 / dR**2 + 1.0 / (Ri * 2.0 * dR)
                c_jp = 1.0 / dZ**2
                c_jm = 1.0 / dZ**2
                c_0 = -2.0 / dR**2 - 2.0 / dZ**2
                for kk, cc in (
                    (idx(i + 1, j), c_ip),
                    (idx(i - 1, j), c_im),
                    (idx(i, j + 1), c_jp),
                    (idx(i, j - 1), c_jm),
                    (k, c_0),
                ):
                    rows.append(k)
                    cols.append(kk)
                    data.append(cc)

        return csr_matrix((data, (rows, cols)), shape=(n, n))

    # --- Soluzione lineare (sorgente data) ---------------------------------
    def solve_linear(
        self,
        source: NDArray[np.float64],
        boundary: NDArray[np.float64] | float = 0.0,
    ) -> NDArray[np.float64]:
        """Risolve Delta* psi = source con psi=boundary sul bordo.

        `source` e' la densita' del termine di destra valutata sulla griglia.
        Restituisce psi (nR x nZ) e la memorizza in self.psi.
        """
        nR, nZ = self.nR, self.nZ
        b = source.copy().reshape(-1)
        bnd = (
            np.full((nR, nZ), boundary)
            if np.isscalar(boundary)
            else np.asarray(boundary)
        )
        # Sovrascrive i nodi di bordo col valore di Dirichlet.
        mask = np.zeros((nR, nZ), dtype=bool)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
        b_grid = b.reshape(nR, nZ)
        b_grid[mask] = bnd[mask]

        psi = spsolve(self._A, b_grid.reshape(-1))
        self.psi = psi.reshape(nR, nZ)
        return self.psi

    # --- Soluzione non lineare (Picard) ------------------------------------
    def solve_picard(
        self,
        rhs: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
        max_iter: int = 100,
        tol: float = 1e-6,
        relax: float = 0.5,
    ) -> int:
        """Risolve il problema NON lineare Delta* psi = rhs(psi, R) con Picard.

        `rhs(psi, R)` restituisce il termine di destra dato il psi corrente.
        `relax` e' il sotto-rilassamento (mescola vecchia e nuova soluzione) per
        stabilizzare l'iterazione. Restituisce il numero di iterazioni.
        """
        # Guess iniziale: parabolico, nullo al bordo.
        Rmid = 0.5 * (self.R_min + self.R_max)
        Zmid = 0.5 * (self.Z_min + self.Z_max)
        aR = 0.5 * (self.R_max - self.R_min)
        aZ = 0.5 * (self.Z_max - self.Z_min)
        guess = np.clip(
            1.0 - ((self.RR - Rmid) / aR) ** 2 - ((self.ZZ - Zmid) / aZ) ** 2, 0.0, None
        )
        self.psi = guess

        for it in range(1, max_iter + 1):
            psi_old = self.psi  # salviamo PRIMA: solve_linear sovrascrive self.psi
            source = rhs(psi_old, self.RR)
            psi_new = self.solve_linear(source, boundary=0.0)
            # Sotto-rilassamento per stabilizzare il punto fisso.
            psi_mixed = relax * psi_new + (1.0 - relax) * psi_old
            change = np.max(np.abs(psi_mixed - psi_old)) / (
                np.max(np.abs(psi_mixed)) + 1e-30
            )
            self.psi = psi_mixed
            if change < tol:
                return it
        return max_iter

    # --- Diagnostica --------------------------------------------------------
    def magnetic_axis(self) -> tuple[float, float, float]:
        """Posizione (R, Z) dell'asse magnetico (massimo di psi) e valore psi."""
        i, j = np.unravel_index(np.argmax(self.psi), self.psi.shape)
        return float(self.R[i]), float(self.Z[j]), float(self.psi[i, j])
