// Kernel C++ ad alte prestazioni: solutore tridiagonale (algoritmo di Thomas).
//
// Fase 4B del progetto Tokamak. Risolve A x = d dove A e' tridiagonale, con
// l'algoritmo di Thomas in O(n): una spazzata in avanti (eliminazione) e una
// all'indietro (sostituzione). E' il cuore numerico dello schema implicito di
// diffusione del calore (transport.py). Esposto a Python con pybind11.
//
// Convenzione: a = sotto-diagonale (a[0] inutilizzato), b = diagonale,
//              c = sopra-diagonale (c[n-1] inutilizzato), d = termine noto.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

py::array_t<double> solve_tridiagonal(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<double, py::array::c_style | py::array::forcecast> c,
    py::array_t<double, py::array::c_style | py::array::forcecast> d) {

    const ssize_t n = b.size();
    if (a.size() != n || c.size() != n || d.size() != n) {
        throw std::invalid_argument("a, b, c, d devono avere la stessa lunghezza");
    }

    const double* ap = a.data();
    const double* bp = b.data();
    const double* cp_in = c.data();
    const double* dp_in = d.data();

    auto result = py::array_t<double>(n);
    double* x = result.mutable_data();

    std::vector<double> cprime(n), dprime(n);

    // Spazzata in avanti.
    cprime[0] = cp_in[0] / bp[0];
    dprime[0] = dp_in[0] / bp[0];
    for (ssize_t i = 1; i < n; ++i) {
        const double m = bp[i] - ap[i] * cprime[i - 1];
        cprime[i] = cp_in[i] / m;
        dprime[i] = (dp_in[i] - ap[i] * dprime[i - 1]) / m;
    }

    // Sostituzione all'indietro.
    x[n - 1] = dprime[n - 1];
    for (ssize_t i = n - 2; i >= 0; --i) {
        x[i] = dprime[i] - cprime[i] * x[i + 1];
    }

    return result;
}

PYBIND11_MODULE(_tridiag_cpp, m) {
    m.doc() = "Solutore tridiagonale (Thomas) in C++ per il solver di trasporto";
    m.def("solve_tridiagonal", &solve_tridiagonal,
          "Risolve A x = d (A tridiagonale) con l'algoritmo di Thomas",
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"));
}
