"""Combustion (gasburn.f) against the compiled Fortran.

Six cases: a heavy hydrocarbon at two burner temperatures, methane, propane,
a cold-compressor/hot-burner point, and a fuel carrying oxygen and nitrogen
so the full stoichiometry balance is exercised rather than just C and H.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_gasburn drv_gasburn.f gasburn.f
    ./drv_gasburn > tests/data/gasburn_ref.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.gas.burn import SPECIES, gasburn

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

AIR = [0.7570, 0.2320, 0.0006, 0.0104, 0.0]
BASE = dict(ttf=300.0, tt3=700.0, tt4=1400.0, rfuel=519.65, cpfuel=2240.0,
            hsfuel=-4.675e6, nhfuel=2, ncfuel=1, nnfuel=0, nofuel=0)

CASES = {
    1: {},
    2: dict(tt4=1700.0),
    3: dict(nhfuel=4, ncfuel=1),                      # methane
    4: dict(nhfuel=8, ncfuel=3),                      # propane
    5: dict(tt3=500.0, tt4=1800.0, ttf=435.0),
    6: dict(nhfuel=6, ncfuel=2, nnfuel=1, nofuel=1,   # oxygenated fuel
            cpfuel=1900.0, hsfuel=-6.20e6),
}


def _solved():
    return {ic: gasburn(alpha=list(AIR), **{**BASE, **kw})
            for ic, kw in CASES.items()}


def test_matches_fortran():
    got = _solved()
    n = 0
    with (DATA / "gasburn_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            r = got[ic]
            actual = (r.alpha[int(name[1]) - 1] if name.startswith("a")
                      else getattr(r, name))
            rel = abs(actual - ref) / max(abs(ref), 1e-300)
            assert rel <= RTOL, f"case {ic} {name}: {actual!r} vs {ref!r} rel {rel:.3e}"
            n += 1
    assert n == 60


def test_mass_fractions_still_sum_to_one():
    """Combustion moves mass between species; it does not create any."""
    for r in _solved().values():
        assert sum(r.alpha) == pytest.approx(sum(AIR), rel=1e-12)


def test_fuel_is_consumed():
    """The fuel slot ends empty: everything burnt."""
    for r in _solved().values():
        assert r.alpha[SPECIES.index("fuel")] == pytest.approx(0.0, abs=1e-12)


def test_hotter_burner_needs_more_fuel():
    got = _solved()
    assert got[2].f > got[1].f          # same fuel, tt4 1700 vs 1400


def test_oxygen_is_consumed_and_products_appear():
    r = _solved()[1]
    assert r.alpha[SPECIES.index("O2")] < AIR[SPECIES.index("O2")]
    assert r.alpha[SPECIES.index("CO2")] > AIR[SPECIES.index("CO2")]
    assert r.alpha[SPECIES.index("H2O")] > AIR[SPECIES.index("H2O")]


def test_gamma_drops_across_the_burner():
    """Hot combustion products have a lower ratio of specific heats."""
    for r in _solved().values():
        assert r.gam4 < r.gam3


def test_agrees_with_the_gasprop_variant():
    """TASOPT ships a second copy of this routine, in gasprop.f.

    It is the same physics without the composition argument: air is hardcoded
    at 78/21 N2/O2 with traces of CO2 and H2O. Running the port at that
    composition must reproduce it, which checks the port against a second
    independently written Fortran source rather than only the one it was
    transcribed from.

    Reference regenerated with::

        gfortran -fdefault-real-8 -O0 -o drv_gasprop drv_gasprop.f gasprop.f
        ./drv_gasprop > tests/data/gasprop_ref.csv
    """
    GASPROP_AIR = [0.78, 0.21, 0.00035, 0.00965, 0.0]
    r = gasburn(alpha=list(GASPROP_AIR), **BASE)
    n = 0
    with (DATA / "gasprop_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "name":
                continue
            name, ref = row[0].strip(), float(row[1])
            actual = getattr(r, name)
            rel = abs(actual - ref) / max(abs(ref), 1e-300)
            assert rel <= RTOL, f"{name}: {actual!r} vs {ref!r} rel {rel:.3e}"
            n += 1
    assert n == 5
