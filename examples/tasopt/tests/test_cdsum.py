"""Aircraft drag buildup (cdsum.f) against the compiled Fortran.

A 737-class aircraft at cruise, four ways: section drag taken from stored
values, section drag looked up from the airfoil database, the same with BLI
credits switched on, and at zero lift (which takes ``cditrp``'s early-out).

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_cdsum drv_cdsum.f cdsum.f \\
        surfcd.f trefftz.f airfun.f airtable.f spline.f wingpo.f cfturb.f

``cfturb`` has to be split out of ``wsize.f`` -- linking wsize.f whole pulls
in the entire program.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.aero.airfoil import airtable
from tasopt_py.aero.cdsum import WING_WAKE_FRACTION, cdsum, cditrp, cfturb
from tasopt_py.model import Aircraft
from tasopt_py.model import indices as I

DATA = Path(__file__).parent / "data"
AIR = Path("/Users/codykarcher/Desktop/Tasopt2.16/air/C.air")
RTOL = 1e-13

pytestmark = pytest.mark.skipif(not AIR.exists(),
                                reason="airfoil database not present")

GEOM = {I.IGS: 124.0, I.IGB: 35.0, I.IGBS: 10.5, I.IGBO: 3.6, I.IGCO: 5.8,
        I.IGAR: 35.0 ** 2 / 124.0, I.IGSWEEP: 26.0, I.IGLAMBDAT: 0.18,
        I.IGLAMBDAS: 0.70, I.IGLAMBDAC: 1.0, I.IGHBOXO: 0.140,
        I.IGHBOXS: 0.126, I.IGZWING: -1.2, I.IGFLO: -0.3, I.IGFLT: -0.05,
        I.IGSH: 31.0, I.IGBH: 13.0, I.IGBOH: 2.0, I.IGCOH: 3.1,
        I.IGARH: 13.0 ** 2 / 31.0, I.IGSWEEPH: 30.0, I.IGLAMBDAH: 0.25,
        I.IGZHTAIL: 1.2, I.IGFCDHCEN: 0.1,
        I.IGSV: 27.0, I.IGBV: 7.0, I.IGBOV: 0.0, I.IGCOV: 4.6,
        I.IGARV: 7.0 ** 2 / 27.0, I.IGSWEEPV: 25.0, I.IGLAMBDAV: 0.30,
        I.IGNVTAIL: 1.0, I.IGLNACE: 3.0, I.IGFSNACE: 8.5, I.IGRVNACE: 1.02,
        I.IGSSTRUT: 0.0, I.IGCOSLS: 1.0, I.IGRVSTRUT: 1.0,
        I.IGRFUSE: 1.9, I.IGXNOSE: 0.0, I.IGXEND: 37.0}

AERO = {I.IACL: 0.57, I.IACLH: -0.05, I.IAMACH: 0.80, I.IARCLS: 1.238,
        I.IARCLT: 0.90, I.IAFDUO: 0.018, I.IAFDUS: 0.014, I.IAFDUT: 0.0045,
        I.IAREUNIT: 1.30e6, I.IAREREFW: 2.0e7, I.IAREREFT: 1.0e7,
        I.IAAREXP: -0.15, I.IAFEXCDW: 1.02, I.IAFEXCDT: 1.02,
        I.IAFEXCDF: 1.03, I.IACDFW: 0.0050, I.IACDPW: 0.0035,
        I.IACDFT: 0.0060, I.IACDPT: 0.0035, I.IAPAFINF: 1.10,
        I.IADAFWAKE: 0.30}

NAMES = {"CD": I.IACD, "CDi": I.IACDI, "spanef": I.IASPANEFF,
         "CDwing": I.IACDWING, "CDover": I.IACDOVER, "CDhtai": I.IACDHTAIL,
         "CDvtai": I.IACDVTAIL, "CDfuse": I.IACDFUSE, "CDnace": I.IACDNACE,
         "CDstru": I.IACDSTRUT, "Cfnace": I.IACFNACE, "cdfw": I.IACDFW,
         "cdpw": I.IACDPW, "clpo": I.IACLPO, "clps": I.IACLPS,
         "clpt": I.IACLPT}


@pytest.fixture(scope="module")
def table():
    return airtable(AIR)


def _build(**over):
    ac = Aircraft(npoint=1)
    for k, v in GEOM.items():
        ac.parg[k] = v
    for k, v in AERO.items():
        ac.para[k, 1] = v
    ac.pare[I.IEM2, 1] = 0.60
    for k, v in over.items():
        if k.startswith("g_"):
            ac.parg[getattr(I, k[2:])] = v
        else:
            ac.para[getattr(I, k), 1] = v
    return ac


def _run(table, icdfun, **over):
    ac = _build(**over)
    para = ac.para.column(1)
    pare = ac.pare.column(1)
    cdsum(ac.pari, ac.parg, para, pare, icdfun, table)
    return para


def test_matches_fortran(table):
    got = {1: _run(table, 0),
           2: _run(table, 1),
           3: _run(table, 1, g_IGFBLIF=0.4, g_IGFBLIW=0.2),
           4: _run(table, 1, IACL=0.0, IACLH=0.0)}
    n = 0
    with (DATA / "cdsum_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            assert got[ic][NAMES[name]] == pytest.approx(
                ref, rel=RTOL, abs=1e-300), f"case {ic} {name}"
            n += 1
    assert n == 64


def test_components_sum_to_the_total(table):
    p = _run(table, 1, g_IGFBLIF=0.4, g_IGFBLIW=0.2)
    parts = (p[I.IACDI] + p[I.IACDFUSE] + p[I.IACDWING] + p[I.IACDOVER]
             + p[I.IACDHTAIL] + p[I.IACDVTAIL] + p[I.IACDSTRUT]
             + p[I.IACDNACE]
             - 0.4 * AERO[I.IADAFWAKE] / GEOM[I.IGS]
             - 0.2 * p[I.IACDWING] * WING_WAKE_FRACTION)
    assert p[I.IACD] == pytest.approx(parts, rel=1e-14)


def test_bli_credits_reduce_total_drag(table):
    without = _run(table, 1)
    with_bli = _run(table, 1, g_IGFBLIF=0.4, g_IGFBLIW=0.2)
    assert with_bli[I.IACD] < without[I.IACD]
    # Only the two BLI terms differ; every other component is untouched.
    for idx in (I.IACDI, I.IACDWING, I.IACDHTAIL, I.IACDVTAIL, I.IACDNACE):
        assert with_bli[idx] == pytest.approx(without[idx], rel=1e-14)


def test_fuselage_drag_comes_from_the_bl_solve_not_from_geometry(table):
    """CDfuse is PAfinf/S -- the wetted-area route through bodycd is dead."""
    p = _run(table, 1)
    assert p[I.IACDFUSE] == pytest.approx(AERO[I.IAPAFINF] / GEOM[I.IGS],
                                          rel=1e-14)
    # Change the fuselage radius and length: fuselage drag does not move.
    q = _run(table, 1, g_IGRFUSE=3.0, g_IGXEND=60.0)
    assert q[I.IACDFUSE] == pytest.approx(p[I.IACDFUSE], rel=1e-14)
    # But change the BL result and it does.
    r = _run(table, 1, IAPAFINF=2.20)
    assert r[I.IACDFUSE] == pytest.approx(2.0 * p[I.IACDFUSE], rel=1e-14)


def test_zero_lift_takes_the_cditrp_early_out(table):
    p = _run(table, 1, IACL=0.0, IACLH=0.0)
    assert p[I.IACDI] == 0.0
    assert p[I.IASPANEFF] == 1.0


def test_induced_drag_is_quadratic_in_lift(table):
    a = _run(table, 1, IACL=0.40)
    b = _run(table, 1, IACL=0.80)
    assert b[I.IACDI] / a[I.IACDI] == pytest.approx(4.0, rel=0.02)


def test_wing_gets_shock_unsweep_but_the_tails_do_not():
    """cdsum sets rkSunsw = 0.5 and both tail constants to zero."""
    import inspect

    from tasopt_py.aero import cdsum as mod
    src = inspect.getsource(mod.cdsum)
    assert "rkSunsw = 0.5" in src
    assert "rkSunsh = rkSunsv = 0.0" in src


def test_cfturb_is_whites_correlation():
    """0.523/ln(0.06 Re)^2 -- not the two Hoerner forms commented out above."""
    for re in (1e6, 1e7, 1e8):
        assert cfturb(re) == pytest.approx(0.523 / math.log(0.06 * re) ** 2,
                                           rel=1e-15)
    assert cfturb(1e8) < cfturb(1e6)          # falls with Reynolds number


def test_icdfun_one_needs_a_table():
    ac = _build()
    with pytest.raises(ValueError, match="airfoil table"):
        cdsum(ac.pari, ac.parg, ac.para.column(1), ac.pare.column(1), 1, None)


def test_cditrp_writes_only_its_two_outputs(table):
    ac = _build()
    para = ac.para.column(1)
    before = [para[i] for i in range(1, len(para) + 1)]
    cditrp(ac.pari, ac.parg, para)
    changed = {i for i in range(1, len(para) + 1)
               if para[i] != before[i - 1]}
    assert changed == {I.IACDI, I.IASPANEFF}
