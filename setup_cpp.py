"""Build dell'estensione C++ (kernel tridiagonale) via pybind11.

Compila in-place il modulo `tokamak._tridiag_cpp`:

    python setup_cpp.py build_ext --inplace

L'estensione e' OPZIONALE: il pacchetto funziona in puro Python (con scipy) se
il modulo compilato non e' presente (vedi tokamak/_tridiag.py).
"""

from __future__ import annotations

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "tokamak._tridiag_cpp",
        ["src/tokamak/_tridiag.cpp"],
        cxx_std=14,
    )
]

setup(
    name="tokamak-cpp-kernel",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    package_dir={"": "src"},
)
