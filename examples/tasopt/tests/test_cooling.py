"""Turbine blade cooling (tfcool.f) against the compiled Fortran.

Covers both directions -- ``mcool`` sizing the coolant flow from a metal
temperature, and ``tmcalc`` recovering the metal temperature from the flow --
together with the sensitivities the Fortran returns alongside.

Five cases: a baseline, a hotter burner, a burner cool enough that fewer rows
need cooling at all, staggered metal temperatures down the turbine, and a
different film/Stanton combination.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_tfcool drv_tfcool.f tfcool.f
    ./drv_tfcool > tests/data/tfcool_ref.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.engine.cooling import mcool, tmcalc

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

BASE = dict(Tt3=700.0, Tt4=1600.0, dTstreak=200.0, Trrat=0.90,
            efilm=0.70, tfilm=0.30, StA=0.09)
CASES = {
    1: (dict(), [1200.0] * 4),
    2: (dict(Tt4=1900.0), [1200.0] * 4),
    3: (dict(Tt4=1250.0, dTstreak=50.0), [1200.0] * 4),
    4: (dict(Trrat=0.85), [1250.0, 1200.0, 1150.0, 1100.0]),
    5: (dict(efilm=0.50, tfilm=0.45, StA=0.15, Tt3=600.0), [1200.0] * 4),
}


def _solved():
    out = {}
    for ic, (kw, Tm) in CASES.items():
        args = {**BASE, **kw}
        flow = mcool(Tmrow=Tm, **args)
        back = tmcalc(ncrow=flow.ncrow, epsrow=flow.epsrow, **args)
        out[ic] = (flow, back)
    return out


def test_matches_fortran():
    got = _solved()
    n = 0
    with (DATA / "tfcool_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            flow, back = got[ic]
            if name == "ncrow":
                actual = float(flow.ncrow)
            else:
                k, idx = name[:3], int(name[3]) - 1
                actual = {"eps": flow.epsrow, "dT3": flow.epsrow_Tt3,
                          "dT4": flow.epsrow_Tt4, "dTr": flow.epsrow_Trr,
                          "Tm": back}[k.strip()][idx]
            rel = abs(actual - ref) / max(abs(ref), 1e-300)
            assert rel <= RTOL, f"case {ic} {name}: {actual!r} vs {ref!r} rel {rel:.3e}"
            n += 1
    assert n == 105


def test_round_trip_recovers_the_metal_temperature():
    """tmcalc inverts mcool on the rows that are actually cooled."""
    for ic, (flow, back) in _solved().items():
        target = CASES[ic][1]
        for i in range(flow.ncrow):
            assert back[i] == pytest.approx(target[i], rel=1e-9), (ic, i)


def test_hotter_burner_needs_more_coolant():
    got = _solved()
    assert got[2][0].epsrow[0] > got[1][0].epsrow[0]


def test_a_cool_burner_needs_fewer_cooled_rows():
    got = _solved()
    assert got[3][0].ncrow < got[1][0].ncrow


def test_uncooled_rows_stay_zero():
    for flow, _ in _solved().values():
        for i in range(flow.ncrow, len(flow.epsrow)):
            assert flow.epsrow[i] == 0.0


def test_first_row_sees_the_hot_streak():
    """Row 1 is sized on Tt4 + dTstreak, so removing the streak cools it."""
    hot = mcool(Tmrow=[1200.0] * 4, **BASE)
    nostreak = mcool(Tmrow=[1200.0] * 4, **{**BASE, "dTstreak": 0.0})
    assert hot.epsrow[0] > nostreak.epsrow[0]
