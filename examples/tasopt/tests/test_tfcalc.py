"""Engine wrapper (tfcalc.f) against the compiled Fortran.

The **sizing** path is compared directly: two cases, uncooled and cooled with
offtakes, 74 values.

The **off-design** path cannot be compared against the Fortran *for this
engine*. The parameters here use an LPC/HPC pressure-ratio split
(1.935 / 9.369) well outside the shipped envelope, and on it ``tfoper`` fails
to converge; ``tfcalc`` then responds with a bare ``stop``, so the reference
program terminates rather than producing numbers. See DISCREPANCIES.md §22 --
on the shipped 737 engine ``tfoper`` converges at all 552 calls, so this is a
robustness limit, not a defect in the routine.

What is checked instead is that this port does the calculation, that
re-solving at the design point returns the design answer exactly, and that
solving for a specified thrust reproduces ``mcore = 19.70426376647826`` --
the value the Fortran ``tfoper`` produces for that case when driven directly
(see ``test_tfoper.py``), bypassing ``tfcalc``.

Reference regenerated with::

    gfortran -fdefault-real-8 -fdollar-ok -O0 -o drv_tfcalc drv_tfcalc.f \\
        tfcalc.f tfsize.f tfoper.f tfcool.f tfmap.f gascalc.f gasfun.f \\
        gaussn.f
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.aero.cdsum import WING_WAKE_FRACTION
from tasopt_py.engine.tfcalc import FD_WAKE, tfcalc
from tasopt_py.model import Aircraft
from tasopt_py.model import indices as I

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

A0 = math.sqrt(1.4 * 287.0 * 219.43)
GEOM = {I.IGGEARF: 1.0, I.IGNENG: 2.0, I.IGS: 124.0, I.IGTMETAL: 1200.0,
        I.IGHTRF: 0.30, I.IGHTRLC: 0.60, I.IGHTRHC: 0.80,
        I.IGWPAY: 1.7e5, I.IGWMTO: 7.5e5}
ENG = {I.IETFUEL: 280.0, I.IETT4: 1450.0, I.IEBPR: 5.1, I.IEPIF: 1.685,
       I.IEPILC: 1.935, I.IEPIHC: 9.369, I.IEPID: 0.998, I.IEPIB: 0.94,
       I.IEPIFN: 0.98, I.IEPITN: 0.989, I.IEEPOLF: 0.8948,
       I.IEEPOLLC: 0.88, I.IEEPOLHC: 0.87, I.IEEPOLHT: 0.889,
       I.IEEPOLLT: 0.899, I.IEETAB: 0.985, I.IEPIFK: 1.685, I.IEEPFK: 0.0,
       I.IEM2: 0.60, I.IEM25: 0.60, I.IEM0: 0.80, I.IET0: 219.43,
       I.IEP0: 23842.0, I.IEA0: A0, I.IERHO0: 23842.0 / (287.0 * 219.43),
       I.IEU0: 0.80 * A0, I.IEFE: 25000.0, I.IEDTSTRK: 200.0,
       I.IESTA: 0.09, I.IEMTEXIT: 1.0, I.IEM4A: 0.9, I.IERUC: 0.9,
       I.IEEFILM: 0.7, I.IETFILM: 0.3, I.IEEPSL: 0.0, I.IEEPSH: 0.0,
       I.IETT9: 300.0, I.IEPT9: 30000.0}

NAMES = {"TSFC": I.IETSFC, "Fsp": I.IEFSP, "ff": I.IEFF, "mcore": I.IEMCORE,
         "Fe": I.IEFE, "Tt4": I.IETT4, "BPR": I.IEBPR, "A2": I.IEA2,
         "A25": I.IEA25, "A5": I.IEA5, "A7": I.IEA7, "mbfD": I.IEMBFD,
         "mblcD": I.IEMBLCD, "mbhcD": I.IEMBHCD, "mbhtD": I.IEMBHTD,
         "mbltD": I.IEMBLTD, "NbfD": I.IENBFD, "pihtD": I.IEPIHTD,
         "piltD": I.IEPILTD, "mbf": I.IEMBF, "mblc": I.IEMBLC,
         "mbhc": I.IEMBHC, "Nf": I.IENF, "N1": I.IEN1, "N2": I.IEN2,
         "pif": I.IEPIF, "pilc": I.IEPILC, "pihc": I.IEPIHC,
         "Tt3": I.IETT3, "pt3": I.IEPT3, "Tt41": I.IETT41,
         "Tt45": I.IETT45, "fc": I.IEFC, "Phiin": I.IEPHIINL}
GNAMES = {"dfan": I.IGDFAN, "dlcom": I.IGDLCOMP, "dhcom": I.IGDHCOMP}


def _build(cooled=False):
    ac = Aircraft()
    ac.pari[I.IIFUEL] = 24
    ac.pari[I.IIBLIC] = 0
    for k, v in GEOM.items():
        ac.parg[k] = v
    ip = I.IPCRUISE1
    for k, v in ENG.items():
        ac.pare[k, ip] = v
    icool = 0
    if cooled:
        icool = 2
        ac.parg[I.IGMOFWMTO] = 0.5 / 7.5e5
        ac.parg[I.IGPOFWMTO] = 60000.0 / 7.5e5
    return ac, icool, ip


def _size(cooled=False):
    ac, icool, ip = _build(cooled)
    para = ac.para.column(ip)
    pare = ac.pare.column(ip)
    tfcalc(ac.pari, ac.parg, para, pare, ip, 0, icool, 0)
    return ac, para, pare, icool, ip


def test_sizing_matches_fortran():
    got = {}
    for ic, cooled in ((1, False), (2, True)):
        ac, _, pare, _, _ = _size(cooled)
        got[ic] = (pare, ac.parg)
    n = 0
    with (DATA / "tfcalc_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            pare, parg = got[ic]
            a = parg[GNAMES[name]] if name in GNAMES else pare[NAMES[name]]
            assert a == pytest.approx(ref, rel=RTOL, abs=1e-300), \
                f"case {ic} {name}"
            n += 1
    assert n == 74


def test_off_design_at_the_design_point_returns_the_design_answer():
    """The Fortran cannot do this -- tfoper diverges and tfcalc stops."""
    ac, para, pare, icool, ip = _size()
    Fe0, mcore0, TSFC0 = pare[I.IEFE], pare[I.IEMCORE], pare[I.IETSFC]
    tfcalc(ac.pari, ac.parg, para, pare, ip, 1, icool, 1)
    assert pare[I.IETT4] == pytest.approx(1450.0, rel=1e-12)
    assert pare[I.IEFE] == pytest.approx(Fe0, rel=1e-9)
    assert pare[I.IEMCORE] == pytest.approx(mcore0, rel=1e-9)
    assert pare[I.IETSFC] == pytest.approx(TSFC0, rel=1e-9)


def test_off_design_thrust_matches_the_fortran_tfoper():
    """Solving for Tt4 at 23 kN must give the value drv_tfoper produced.

    That reference comes from driving tfoper directly, which is the one
    off-design case the Fortran converges on. Routing the same case through
    tfcalc terminates the Fortran program, so this is the only way to check
    the wrapper's off-design path against it.
    """
    ac, para, pare, icool, ip = _size()
    pare[I.IEFE] = 23000.0
    tfcalc(ac.pari, ac.parg, para, pare, ip, 2, icool, 1)
    assert pare[I.IEMCORE] == pytest.approx(19.70426376647826, rel=1e-8)
    assert pare[I.IEFE] == pytest.approx(23000.0, rel=1e-9)
    assert pare[I.IETT4] == pytest.approx(1404.6022, rel=1e-5)


def test_turbine_design_ratios_are_the_temperature_form():
    """pihtD is Trh**gexh, not pt41/pt45 -- the source assigns the latter
    first and then overwrites it."""
    _, _, pare, _, _ = _size()
    assert pare[I.IEPIHTD] > 1.0        # temperature form is > 1
    assert pare[I.IEPILTD] > 1.0
    # pt41/pt45 would be > 1 too, but a different number entirely.
    assert pare[I.IEPIHTD] != pytest.approx(
        pare[I.IEPT41] / pare[I.IEPT45], rel=1e-3)


def test_design_spool_speeds_are_unity():
    """The design case is what defines 100% speed."""
    _, _, pare, _, _ = _size()
    assert pare[I.IENF] == pytest.approx(1.0, rel=1e-12)
    assert pare[I.IEN1] == pytest.approx(1.0, rel=1e-12)
    assert pare[I.IEN2] == pytest.approx(1.0, rel=1e-12)


def test_cooling_flow_is_recorded_when_metal_temperature_is_given():
    _, _, pare, _, _ = _size(cooled=True)
    assert pare[I.IEFC] > 0.0
    assert sum(pare[I.IEEPSC1 + k] for k in range(I.NCROWX)) > 0.0


def test_no_ingestion_at_zero_mach():
    """Standing still there is no wake to ingest, so both defects are zero.

    Note specific thrust comes back infinite here, as it does in the source:
    Fsp divides by flight speed with no guard. Sizing at zero speed is not a
    real use of tfsize -- the static point goes through tfoper -- but the
    ingestion branch is worth pinning.
    """
    ac, icool, ip = _build()
    ac.pare[I.IEM0, ip] = 0.0
    ac.pare[I.IEU0, ip] = 0.0
    ac.parg[I.IGFBLIF] = 0.5
    para = ac.para.column(ip)
    pare = ac.pare.column(ip)
    para[I.IADAFSURF] = 2.0
    tfcalc(ac.pari, ac.parg, para, pare, ip, 0, icool, 0)
    assert pare[I.IEPHIINL] == 0.0
    assert pare[I.IEKINL] == 0.0
    assert math.isinf(pare[I.IEFSP])


def test_ingestion_scales_with_the_ingested_fraction():
    ac, icool, ip = _build()
    para = ac.para.column(ip)
    pare = ac.pare.column(ip)
    para[I.IADAFSURF] = 2.0
    para[I.IAKAFTE] = 1.5
    ac.parg[I.IGFBLIF] = 0.4
    tfcalc(ac.pari, ac.parg, para, pare, ip, 0, icool, 0)
    half = pare[I.IEPHIINL]

    ac2, icool2, ip2 = _build()
    para2 = ac2.para.column(ip2)
    pare2 = ac2.pare.column(ip2)
    para2[I.IADAFSURF] = 2.0
    para2[I.IAKAFTE] = 1.5
    ac2.parg[I.IGFBLIF] = 0.8
    tfcalc(ac2.pari, ac2.parg, para2, pare2, ip2, 0, icool2, 0)
    assert pare2[I.IEPHIINL] == pytest.approx(2.0 * half, rel=1e-12)


def test_wake_fraction_matches_the_drag_buildup():
    """tfcalc and cdsum each hard-code 15%; they must stay in step or the
    ingested wing dissipation is double-counted or lost."""
    assert FD_WAKE == WING_WAKE_FRACTION == 0.15


def test_fan_diameter_follows_the_fan_face_area():
    ac, _, pare, _, _ = _size()
    A2 = pare[I.IEA2]
    expect = math.sqrt(4.0 * A2 / (math.pi * (1.0 - GEOM[I.IGHTRF] ** 2)))
    assert ac.parg[I.IGDFAN] == pytest.approx(expect, rel=1e-14)
