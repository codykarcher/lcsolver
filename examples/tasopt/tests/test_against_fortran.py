"""Verify the Python port against the original Fortran, to machine precision.

The reference CSVs in ``tests/data`` were produced by compiling the drivers in
``fortran_ref/`` against the TASOPT 2.16 sources with the project's own build
flags (``-O -fdefault-real-8 -fdollar-ok``, i.e. double precision), so these
are not hand-transcribed expectations — they are what the original code
computes.

Tolerance is 1e-13 relative rather than exact equality: the port evaluates
the same expressions but not necessarily in the same association order, so
the last bit or two can differ. Anything looser than 1e-13 would mean a real
difference in the formulas.

Regenerate the references with::

    gfortran -O -fdefault-real-8 -fdollar-ok -o drv_atmos \\
        fortran_ref/drv_atmos.f <TASOPT>/src/atmos.f
    ./drv_atmos > tests/data/atmos_ref.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.atmosphere import atmos
from tasopt_py.gas.properties import gasfun

DATA = Path(__file__).parent / "data"
RTOL = 1e-13


def _rows(name: str):
    with (DATA / name).open() as fh:
        for row in csv.DictReader(fh):
            yield {k.strip(): v for k, v in row.items()}


def _close(got: float, ref: float, what: str) -> None:
    denom = max(abs(ref), 1e-300)
    rel = abs(got - ref) / denom
    assert rel <= RTOL, f"{what}: got {got!r}, Fortran {ref!r}, rel {rel:.3e}"


def test_atmosphere_matches_fortran():
    n = 0
    for r in _rows("atmos_ref.csv"):
        h = float(r["h"])
        st = atmos(h)
        for key, got in (("T", st.T), ("p", st.p), ("rho", st.rho),
                         ("a", st.a), ("mu", st.mu)):
            _close(got, float(r[key]), f"atmos({h} km).{key}")
            n += 1
    assert n > 0


def test_gas_properties_match_fortran():
    n = 0
    for r in _rows("gasfun_ref.csv"):
        igas, t = int(r["igas"]), float(r["t"])
        g = gasfun(igas, t)
        got = {"s": g.s, "s_t": g.s_t, "h": g.h,
               "h_t": g.h_t, "cp": g.cp, "r": g.r}
        for key, val in got.items():
            _close(val, float(r[key]), f"gasfun({igas}, {t}).{key}")
            n += 1
    assert n > 0


def test_all_gases_covered():
    """Every gas index the Fortran implements must be exercised above."""
    seen = {int(r["igas"]) for r in _rows("gasfun_ref.csv")}
    assert seen == {1, 2, 3, 4, 5, 11, 12, 13, 14, 18, 24}


def test_undefined_gas_index_raises():
    with pytest.raises(ValueError, match="undefined gas index"):
        gasfun(99, 300.0)
