r"""Modello 1D: trasporto radiale del calore nel plasma.

Passiamo dal modello 0D (plasma = un punto) a un profilo radiale T(r) lungo il
raggio minore r in [0, a]. Risolviamo l'equazione di diffusione del calore in
geometria cilindrica (approssimazione di un toro "srotolato" in un cilindro):

    (3/2) n dT/dt = (1/r) d/dr( r n chi dT/dr ) + S(r)

dove:
- n        densita' (qui uniforme in r), m^-3
- chi      diffusivita' termica (trasporto anomalo turbolento), m^2/s
- S(r)     densita' di potenza netta (alfa + esterno - Bremsstrahlung), W/m^3
- T        temperatura, in keV

Schema numerico: volumi finiti + Eulero implicito
-------------------------------------------------
- VOLUMI FINITI: integriamo l'equazione su celle [r_{i-1/2}, r_{i+1/2}]. Il
  peso geometrico cilindrico e' r dr; la cella centrale ha automaticamente flusso
  nullo dal lato r=0 (simmetria), quindi la singolarita' 1/r non da' problemi.
- EULERO IMPLICITO: valutiamo la conduzione al tempo nuovo (t+dt). E'
  incondizionatamente stabile, quindi possiamo usare passi grandi per arrivare
  allo stato stazionario. Porta a un sistema lineare TRIDIAGONALE A T = b, che
  risolviamo in O(N) con solve_banded (algoritmo di Thomas).

I termini di sorgente S(r) dipendono da T (non linearita'): li trattiamo in modo
esplicito (valutati al tempo corrente), cioe' uno splitting diffusione-implicita
/ sorgenti-esplicite. Semplice e adeguato per raggiungere lo stato stazionario.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ._tridiag import solve_tridiagonal
from .constants import KEV_TO_JOULE
from .power_balance import (
    alpha_power_density,
    bremsstrahlung_power_density,
    stored_energy_density,
)


@dataclass
class TransportSolver1D:
    """Solver di diffusione del calore radiale su griglia a volumi finiti.

    Parameters
    ----------
    a:        raggio minore del plasma [m]
    R0:       raggio maggiore del toro [m] (serve per i volumi)
    n_e:      densita' elettronica, uniforme in r [m^-3]
    chi:      diffusivita' termica [m^2/s]
    T_edge:   temperatura imposta al bordo r=a [keV]
    n_cells:  numero di celle radiali
    z_eff:    carica efficace (perdite di Bremsstrahlung)
    backend:  "scipy" (LAPACK) o "cpp" (kernel C++ di Thomas) per il solver
              tridiagonale
    """

    a: float
    R0: float
    n_e: float
    chi: float
    T_edge: float = 0.1
    n_cells: int = 100
    z_eff: float = 1.0
    backend: str = "scipy"
    insulated_edge: bool = False  # True: flusso nullo al bordo (Neumann), per test

    # Stato interno (inizializzato in __post_init__).
    r: NDArray[np.float64] = field(init=False)
    T: NDArray[np.float64] = field(init=False)
    _dr: float = field(init=False)
    _r_face: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        N = self.n_cells
        self._dr = self.a / N
        # Centri cella: r_i = (i + 1/2) dr,  i = 0..N-1.
        self.r = (np.arange(N) + 0.5) * self._dr
        # Facce: r_{i+1/2} = (i+1) dr,  i = -1..N-1  (faccia -1/2 = 0).
        self._r_face = np.arange(N + 1) * self._dr
        # Profilo iniziale: parabolico di tentativo (verra' rilassato).
        self.T = self.T_edge + 5.0 * (1.0 - (self.r / self.a) ** 2)

    # --- Sorgenti fisiche ---------------------------------------------------
    def source_density(
        self, T: NDArray[np.float64], p_ext: NDArray[np.float64] | float = 0.0
    ) -> NDArray[np.float64]:
        """Densita' di potenza netta S(r) in W/m^3.

        S = P_alpha (self-heating) + P_ext (esterno) - P_brem (radiazione).
        Le perdite per conduzione NON sono qui: sono il termine diffusivo.
        """
        p_alpha = alpha_power_density(self.n_e, T)
        p_brem = bremsstrahlung_power_density(self.n_e, T, self.z_eff)
        return p_alpha + p_ext - p_brem

    # --- Avanzamento temporale ---------------------------------------------
    def step(self, dt: float, p_ext: NDArray[np.float64] | float = 0.0) -> None:
        """Avanza la temparatura di un passo dt con diffusione implicita.

        Costruisce e risolve il sistema tridiagonale A T^{new} = b.
        """
        N = self.n_cells
        dr = self._dr
        D = self.n_e * self.chi  # coefficiente di diffusione n*chi

        # Capacita' termica per cella: (3/2) n * volume_geom, con peso r dr.
        # Lavoriamo in keV, quindi le sorgenti (W/m^3) vanno divise per
        # KEV_TO_JOULE per coerenza dimensionale.
        vol = self.r * dr  # fattore di volume cilindrico della cella, integral r dr
        cap = 1.5 * self.n_e * vol / dt  # coefficiente di (T^new - T^old)

        # Conduttanze sulle facce: g_{i+1/2} = r_{i+1/2} * D / dr (lunghezza N+1).
        # g[0]=0 -> nessun flusso dal centro r=0 (simmetria, gestita gratis).
        g = self._r_face * D / dr
        g_left = g[:N].copy()  # facce i-1/2
        g_right = g[1:].copy()  # facce i+1/2

        # Bordo esterno: Dirichlet (T_edge alla faccia r=a, distanza dr/2 ->
        # conduttanza doppia) oppure Neumann isolato (flusso nullo).
        g_right[-1] = 0.0 if self.insulated_edge else self._r_face[N] * D / (dr / 2.0)

        diag = cap + g_left + g_right
        lower = -g_left  # lower[0] inutilizzato da solve_banded
        upper = -g_right  # upper[-1] inutilizzato da solve_banded

        S = self.source_density(self.T, p_ext) / KEV_TO_JOULE
        rhs = cap * self.T + S * vol
        if not self.insulated_edge:
            rhs[-1] += g_right[-1] * self.T_edge  # contributo Dirichlet

        # Risoluzione del sistema tridiagonale, backend scelto (scipy o C++).
        self.T = solve_tridiagonal(lower, diag, upper, rhs, backend=self.backend)

    def solve_steady_state(
        self,
        p_ext: NDArray[np.float64] | float = 0.0,
        dt: float = 1e-3,
        max_steps: int = 20000,
        tol: float = 1e-8,
    ) -> int:
        """Itera nel tempo finche' il profilo smette di cambiare (stazionario).

        Restituisce il numero di passi effettuati. La convergenza e' misurata
        sulla variazione relativa massima di T tra due passi.
        """
        for step in range(1, max_steps + 1):
            T_old = self.T.copy()
            self.step(dt, p_ext)
            rel_change = np.max(np.abs(self.T - T_old)) / np.max(np.abs(self.T))
            if rel_change < tol:
                return step
        return max_steps

    # --- Grandezze integrate (geometria toroidale) -------------------------
    def _volume_element(self) -> NDArray[np.float64]:
        """dV per cella in un toro: dV = (2 pi R0)(2 pi r dr) = 4 pi^2 R0 r dr."""
        return 4.0 * np.pi**2 * self.R0 * self.r * self._dr

    def plasma_volume(self) -> float:
        """Volume totale del plasma [m^3] (~ 2 pi^2 R0 a^2)."""
        return float(np.sum(self._volume_element()))

    def heating_density_for_power(
        self, total_power_W: float, shape: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Profilo di densita' di potenza [W/m^3] con forma `shape` data, tale
        che la potenza totale iniettata nel volume sia esattamente total_power_W.

        Serve al controllo: l'attuatore comanda una POTENZA (scalare), che qui
        viene distribuita radialmente secondo una forma fissa (es. gaussiana
        centrale) e normalizzata sul volume.
        """
        dV = self._volume_element()
        norm = float(np.sum(shape * dV))
        if norm <= 0.0:  # pragma: no cover
            raise ValueError("la forma di deposizione ha integrale di volume nullo")
        return total_power_W * shape / norm

    def stored_energy(self) -> float:
        """Energia termica totale immagazzinata W = integral (3 n T) dV [J]."""
        w_density = stored_energy_density(self.n_e, self.T)  # J/m^3
        return float(np.sum(w_density * self._volume_element()))

    def total_power(
        self, which: str, p_ext: NDArray[np.float64] | float = 0.0
    ) -> float:
        """Potenza totale integrata sul volume [W] per il termine richiesto.

        which in {"fusion_alpha", "bremsstrahlung", "external", "net_heating"}.
        """
        dV = self._volume_element()
        if which == "fusion_alpha":
            dens = alpha_power_density(self.n_e, self.T)
        elif which == "bremsstrahlung":
            dens = bremsstrahlung_power_density(self.n_e, self.T, self.z_eff)
        elif which == "external":
            dens = np.broadcast_to(p_ext, self.r.shape)
        elif which == "net_heating":
            dens = self.source_density(self.T, p_ext)
        else:  # pragma: no cover
            raise ValueError(f"termine sconosciuto: {which}")
        return float(np.sum(dens * dV))

    def tau_E(self, p_ext: NDArray[np.float64] | float = 0.0) -> float:
        """Tempo di confinamento dell'energia EMERGENTE dal profilo [s].

        Definizione: tau_E = W / P_loss, dove in stato stazionario la potenza
        persa per conduzione eguaglia la potenza netta depositata nel plasma
        (alfa + esterno - radiazione). A differenza del modello 0D, qui tau_E
        NON e' imposto: emerge da chi, dalla geometria e dal profilo calcolato.
        """
        w = self.stored_energy()
        p_loss = self.total_power("net_heating", p_ext)
        return w / p_loss if p_loss > 0 else np.inf
