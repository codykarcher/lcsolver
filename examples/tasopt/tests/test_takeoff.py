"""Takeoff run and balanced field length (takeoff.f) vs the compiled Fortran.

Three cases: a normal takeoff, a heavier aircraft, and one with the thrust cut
far enough that the engine-out case is impossible -- which takes the source's
fallback branch rather than failing.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_takeoff drv_takeoff.f takeoff.f \\
        cdsum.f surfcd.f trefftz.f airfun.f airtable.f spline.f wingpo.f \\
        cfturb.f
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.aero.airfoil import airtable
from tasopt_py.model import Aircraft
from tasopt_py.model import indices as I
from tasopt_py.sizing.takeoff import CD_IVERT, takeoff

DATA = Path(__file__).parent / "data"
AIR = Path("/Users/codykarcher/Desktop/Tasopt2.16/air/C.air")
RTOL = 1e-12

pytestmark = pytest.mark.skipif(not AIR.exists(),
                                reason="airfoil database not present")

GEOM = {
    I.IGS: 124.0, I.IGB: 35.0, I.IGBS: 10.5, I.IGBO: 3.6, I.IGCO: 5.8,
    I.IGAR: 35.0 ** 2 / 124.0, I.IGSWEEP: 26.0, I.IGLAMBDAT: 0.18,
    I.IGLAMBDAS: 0.70, I.IGHBOXO: 0.140, I.IGHBOXS: 0.126, I.IGZWING: -1.2,
    I.IGFLO: -0.3, I.IGFLT: -0.05, I.IGSH: 31.0, I.IGBH: 13.0, I.IGBOH: 2.0,
    I.IGCOH: 3.1, I.IGSWEEPH: 30.0, I.IGLAMBDAH: 0.25, I.IGZHTAIL: 1.2,
    I.IGFCDHCEN: 0.1, I.IGSV: 27.0, I.IGBV: 7.0, I.IGBOV: 0.0, I.IGCOV: 4.6,
    I.IGSWEEPV: 25.0, I.IGLAMBDAV: 0.30, I.IGNVTAIL: 1.0, I.IGLNACE: 3.0,
    I.IGFSNACE: 8.5, I.IGRVNACE: 1.02, I.IGCOSLS: 1.0, I.IGRVSTRUT: 1.0,
    I.IGWMTO: 7.5e5, I.IGWFUEL: 1.6e5, I.IGWPAY: 1.7e5, I.IGDFAN: 1.7,
    I.IGHTRF: 0.30, I.IGNENG: 2.0, I.IGMUROLL: 0.025, I.IGMUBRAKE: 0.35,
    I.IGHOBST: 10.7, I.IGCDGEAR: 0.015, I.IGCDEFAN: 0.10, I.IGCDSPOIL: 0.10}

AERO = {I.IACL: 0.57, I.IACLH: -0.05, I.IAMACH: 0.20, I.IARCLS: 1.238,
        I.IARCLT: 0.90, I.IAFDUO: 0.018, I.IAFDUS: 0.014, I.IAFDUT: 0.0045,
        I.IAREUNIT: 2.5e6, I.IAREREFW: 2.0e7, I.IAREREFT: 1.0e7,
        I.IAAREXP: -0.15, I.IAFEXCDW: 1.02, I.IAFEXCDT: 1.02,
        I.IAFEXCDF: 1.03, I.IACDFW: 0.0050, I.IACDPW: 0.0035,
        I.IACDFT: 0.0060, I.IACDPT: 0.0035, I.IAPAFINF: 1.10}


@pytest.fixture(scope="module")
def table():
    return airtable(AIR)


def _build(**over):
    ac = Aircraft()
    for k, v in GEOM.items():
        ac.parg[k] = v
    ac.parm[I.IMWPAY] = 1.7e5
    ac.parm[I.IMWTO] = 7.5e5
    for ip in range(1, ac.npoint + 1):
        for k, v in AERO.items():
            ac.para[k, ip] = v
        ac.pare[I.IEM2, ip] = 0.60
        ac.pare[I.IERHO0, ip] = 1.225
    ac.para[I.IACD, I.IPCLIMB1] = 0.045
    ac.pare[I.IEU0, I.IPROTATE] = 68.0
    ac.pare[I.IEU0, I.IPTAKEOFF] = 78.0
    ac.pare[I.IEFE, I.IPSTATIC] = 1.10e5
    ac.pare[I.IEFE, I.IPROTATE] = 0.92e5
    ac.pare[I.IEMCORE, I.IPSTATIC] = 45.0
    ac.pare[I.IEMCORE, I.IPROTATE] = 43.0
    ac.pare[I.IEFF, I.IPSTATIC] = 0.030
    ac.pare[I.IEFF, I.IPROTATE] = 0.028
    for k, v in over.items():
        if k.startswith("m_"):
            ac.parm[getattr(I, k[2:])] = v
        elif k.startswith("g_"):
            ac.parg[getattr(I, k[2:])] = v
    return ac


def _case(table, ic):
    over = {}
    if ic == 2:
        over["m_IMWTO"] = 9.5e5
    ac = _build(**over)
    if ic == 3:
        ac.pare[I.IEFE, I.IPSTATIC] = 3.3e4
        ac.pare[I.IEFE, I.IPROTATE] = 2.8e4
    r = takeoff(ac.pari, ac.parg, ac.parm, ac.para, ac.pare, table)
    return ac, r


NAMES = {"V1": I.IMV1, "V2": I.IMV2, "lTO": I.IMLTO, "l1": I.IML1,
         "lBF": I.IMLBF, "tTO": I.IMTTO, "FTO": I.IMFTO,
         "gamTO": I.IMGAMVTO, "gamBF": I.IMGAMVBF}


def test_matches_fortran(table):
    got = {}
    for ic in (1, 2, 3):
        ac, _ = _case(table, ic)
        d = {k: ac.parm[v] for k, v in NAMES.items()}
        d["fracW"] = ac.para[I.IAFRACW, I.IPSTATIC]
        d["time"] = ac.para[I.IATIME, I.IPSTATIC]
        d["Range"] = ac.para[I.IARANGE, I.IPSTATIC]
        got[ic] = d
    n = 0
    with (DATA / "takeoff_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            assert got[ic][name] == pytest.approx(ref, rel=RTOL), \
                f"case {ic} {name}"
            n += 1
    assert n == 36


def test_decision_speed_is_below_v2(table):
    _, r = _case(table, 1)
    assert r.V1 < r.V2
    assert r.l1 < r.lBF


def test_heavier_aircraft_needs_more_runway(table):
    _, light = _case(table, 1)
    _, heavy = _case(table, 2)
    assert heavy.lTO > light.lTO
    assert heavy.lBF > light.lBF
    assert heavy.gamVTO < light.gamVTO      # and climbs more shallowly


def test_balanced_field_is_longer_than_normal_takeoff(table):
    _, r = _case(table, 1)
    assert r.lBF > r.lTO


def test_engine_out_impossible_takes_the_fallback(table):
    """With thrust cut far enough, the source substitutes lBF = 10 lTO and
    sets both speeds to stall rather than failing."""
    ac, r = _case(table, 3)
    assert r.normal_takeoff_possible
    assert not r.engine_out_takeoff_possible
    assert r.lBF == pytest.approx(10.0 * r.lTO, rel=1e-14)
    assert r.l1 == pytest.approx(0.7 * r.lTO, rel=1e-14)
    assert r.V1 == r.V2 == 68.0             # both set to Vstall
    assert r.gamVBF == 0.0


def test_rudder_drag_is_a_bare_constant():
    """CDivert = 0.002, hard-coded and not settable from input."""
    assert CD_IVERT == 0.002


def test_takeoff_fuel_is_backed_out_of_the_static_point(table):
    """Static-point weight fraction and time are set *backwards* from the
    takeoff point, since the run happens before it."""
    ac, r = _case(table, 1)
    assert ac.para[I.IATIME, I.IPSTATIC] == pytest.approx(-r.tTO, rel=1e-14)
    assert ac.para[I.IARANGE, I.IPSTATIC] == pytest.approx(-r.lTO, rel=1e-14)
    assert ac.para[I.IAFRACW, I.IPSTATIC] == pytest.approx(
        r.WfTO / GEOM[I.IGWMTO], rel=1e-14)
    assert r.WfTO > 0.0


def test_static_thrust_is_all_engines(table):
    ac, r = _case(table, 1)
    assert r.FTO == pytest.approx(1.10e5 * GEOM[I.IGNENG], rel=1e-15)


def test_climb_angles_are_clamped(table):
    """Both are clipped to [0.01, 0.99] in sine before the arcsin."""
    for ic in (1, 2, 3):
        _, r = _case(table, ic)
        for g in (r.gamVTO, r.gamVBF):
            if g != 0.0:
                assert math.asin(0.01) <= g <= math.asin(0.99)
