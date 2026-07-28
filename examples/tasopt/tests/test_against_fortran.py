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


# ---------------------------------------------------------------------------
# gascalc.f — gas mixture thermodynamics
# ---------------------------------------------------------------------------

# dry air by mass fraction: N2, O2, CO2, H2O, Ar  (as in drv_gascalc.f)
AIR = [0.7532, 0.2314, 0.0006, 0.0020, 0.0128]
FUEL = [0.0, 0.0, 0.0, 0.0, 1.0]
N = 5
IFUEL = 24  # C14H30


def _gascalc_rows():
    with (DATA / "gascalc_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            yield row[0].strip(), int(row[1]), [float(x) for x in row[2:]]


def test_gas_mixture_matches_fortran():
    from tasopt_py.gas.mixture import (
        gassum, gasfuel, gas_tset, gas_prat, gas_delh, gas_burn)

    gamma = gasfuel(IFUEL, N)
    to, po = 288.2, 1.0132e5
    st0 = gassum(AIR, N, to)
    n = 0

    for case, k, ref in _gascalc_rows():
        if case == "gassum":
            s = gassum(AIR, N, 250.0 + k * 150.0)
            got = [s.s, s.s_t, s.h, s.h_t, s.cp, s.r]
        elif case == "gasfuel":
            got = gamma
        elif case == "tset":
            h = gassum(AIR, N, 300.0 + k * 200.0).h
            got = [ref[0], gas_tset(AIR, N, h, 400.0)]
        elif case == "prat":
            got = list(gas_prat(AIR, N, po, to, st0.h, st0.s, st0.cp, st0.r,
                                1.0 + k * 3.0, 0.90)[:4])
        elif case == "delh":
            got = list(gas_delh(AIR, N, po, to, st0.h, st0.s, st0.cp, st0.r,
                                k * 5.0e4, 0.90)[:4])
        elif case == "burn":
            f, lam = gas_burn(AIR, FUEL, gamma, N, IFUEL,
                              700.0, 300.0, 1000.0 + k * 100.0)
            got = [f] + lam
        else:
            raise AssertionError(f"unknown case {case!r}")

        for j, (g, rf) in enumerate(zip(got, ref)):
            _close(g, rf, f"{case}[{k}][{j}]")
            n += 1
    assert n > 0


def test_gasfuel_conserves_mass():
    """Combustion mass fractions must balance: sum of changes is zero.

    Fuel mass in equals product mass out minus O2 consumed, so the gamma
    entries (positive for products, negative for O2) must sum to +1 per unit
    fuel mass. This is a property check on top of the value comparison.
    """
    from tasopt_py.gas.mixture import gasfuel
    for ifuel in (11, 12, 13, 14, 18, 24):
        gamma = gasfuel(ifuel, N)
        assert abs(sum(gamma) - 1.0) < 1e-9, (ifuel, sum(gamma))
