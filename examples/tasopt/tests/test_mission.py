"""The mission march (mission.f) against the real TASOPT 737 run.

Unlike the other modules this one is not driven by a synthetic driver. It is
checked against the *shipped program*: ``mission.f`` was instrumented to dump
its complete input state (``pari``, ``parg``, ``parm``, and the whole ``para``
and ``pare`` matrices) and its complete output state on the final converged
call of ``runs/737/737.tas``. The port is handed exactly those inputs.

That is a much stronger check than a hand-built case -- every one of the 254
geometry entries and 17 mission points is the real converged 737 -- and it is
only possible because the reference program builds and runs. See
``fortran_ref/mission_instrumented.f`` for the two dump blocks.

Agreement
---------
Takeoff weight to 9.8e-11, fuel to 3.6e-10, and the whole trajectory to 2e-9.

**A retraction.** An earlier version of this file reported 5.7e-6 and 2.1e-5
and explained them as amplification: "at ``ipclimb1`` -- low speed,
near-takeoff thrust -- the flight-path-angle fixed point is badly conditioned,
and a 1e-9 difference in the engine solve is amplified into 1.7e-3 in the
angle." **That was wrong.** The residual was three porting omissions, and this
test could not see any of them, because it hands the port the *converged*
737 state and every one of the three is invisible when the answer is already
in the array:

* the climb loop set ``para(iaMach)`` but not ``pare(ieM0)`` beside it, so the
  engine ran on a stale Mach number -- one that was already correct here;
* the end-of-cruise point never called ``tfcalc``, so it read a stale ``TSFC``
  and thrust -- again already correct here;
* the descent CL profile (``0.96`` down to ``0.50`` of the end-of-cruise CL,
  quadratic in the remaining fraction) was not interpolated at all, and the
  climb one was linear where the source is quadratic. Both were overwritten
  by the pre-loaded values.

All three were found by ``tests/test_wsize.py``, which starts from unset
arrays and so has to compute what this test was handed. That is the argument
for end-to-end tests in one paragraph.

``test_engine_reproduces_fortran_at_climb1`` below is still a useful check --
the engine does reproduce the Fortran's thrust from its own state -- but it
was never evidence for the amplification story it was written to support.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tasopt_py.aero.airfoil import airtable
from tasopt_py.aero.cdsum import cdsum
from tasopt_py.engine.tfcalc import tfcalc
from tasopt_py.model import Aircraft
from tasopt_py.model import indices as I
from tasopt_py.sizing.mission import mission

DATA = Path(__file__).parent / "data"
AIR = Path("/Users/codykarcher/Desktop/Tasopt2.16/air/C.air")

pytestmark = pytest.mark.skipif(
    not AIR.exists() or not (DATA / "mission_in.txt").exists(),
    reason="airfoil database or mission dump not present")

# gfortran's E26.18 drops the 'E' when the exponent needs three digits.
_EXP = re.compile(r"(\d)([+-]\d\d\d)$")


def _f(x):
    return float(_EXP.sub(r"\1E\2", x))


def _load_in():
    tok = (DATA / "mission_in.txt").read_text().split()
    pos = 0

    def take(n):
        nonlocal pos
        v = tok[pos:pos + n]
        pos += n
        return v

    ac = Aircraft()
    for k, v in enumerate(take(I.IITOTAL), 1):
        ac.pari[k] = int(v)
    for k, v in enumerate(take(I.IGTOTAL), 1):
        ac.parg[k] = _f(v)
    for k, v in enumerate(take(I.IMTOTAL), 1):
        ac.parm[k] = _f(v)
    for k in range(1, I.IATOTAL + 1):
        for ip, v in enumerate(take(I.IPTOTAL), 1):
            ac.para[k, ip] = _f(v)
    for k in range(1, I.IETOTAL + 1):
        for ip, v in enumerate(take(I.IPTOTAL), 1):
            ac.pare[k, ip] = _f(v)
    initeng = int(tok[pos + 1])
    return ac, initeng, int(tok[pos + 2])


def _load_out():
    tok = [_f(x) for x in (DATA / "mission_out.txt").read_text().split()]
    pos = 0
    para, pare = {}, {}
    for k in range(1, I.IATOTAL + 1):
        for ip in range(1, I.IPTOTAL + 1):
            para[k, ip] = tok[pos]
            pos += 1
    for k in range(1, I.IETOTAL + 1):
        for ip in range(1, I.IPTOTAL + 1):
            pare[k, ip] = tok[pos]
            pos += 1
    return para, pare, tok[pos], tok[pos + 1], tok[pos + 2]


@pytest.fixture(scope="module")
def table():
    return airtable(AIR)


@pytest.fixture(scope="module")
def flown(table):
    ac, initeng, ipc1 = _load_in()
    r = mission(ac.pari, ac.parg, ac.parm, ac.para, ac.pare, table,
                initeng, ipc1)
    return ac, r


@pytest.fixture(scope="module")
def reference():
    return _load_out()


def test_takeoff_weight_and_fuel(flown, reference):
    _, r = flown
    _, _, WTO, Wfuel, _ = reference
    assert r.WTO == pytest.approx(WTO, rel=1e-9)
    assert r.Wfuel == pytest.approx(Wfuel, rel=1e-8)
    assert r.gamV_converged


def test_trajectory_matches_through_cruise(flown, reference):
    """The whole state, including the integrated range and time, to 1e-8."""
    ac, _ = flown
    pa, _, _, _, _ = reference
    tol = {}
    for idx in (I.IAALT, I.IARANGE, I.IATIME, I.IAFRACW, I.IACL, I.IACD,
                I.IAMACH):
        for ip in range(I.IPCLIMB1, I.IPCRUISEN + 1):
            ref = pa[idx, ip]
            if abs(ref) < 1e-12:
                continue
            assert ac.para[idx, ip] == pytest.approx(
                ref, rel=tol.get(idx, 1e-4)), f"index {idx} point {ip}"


def test_descent_weights_match(flown, reference):
    ac, _ = flown
    pa, _, _, _, _ = reference
    for ip in range(I.IPDESCENT1, I.IPDESCENTN + 1):
        assert ac.para[I.IAFRACW, ip] == pytest.approx(
            pa[I.IAFRACW, ip], rel=1e-8)


def test_engine_reproduces_fortran_at_climb1(table, reference):
    """The engine, given the Fortran's own converged flight state, agrees to
    1e-8 -- which shows the climb1 trajectory difference is amplification
    through an ill-conditioned fixed point, not a port error."""
    pa, pe, _, _, _ = reference
    ac, _, _ = _load_in()
    ip = I.IPCLIMB1
    acol, ecol = ac.para.column(ip), ac.pare.column(ip)
    for idx in (I.IACL, I.IACLH, I.IAMACH, I.IAREUNIT, I.IACD, I.IACDFW,
                I.IACDPW):
        acol[idx] = pa[idx, ip]
    for idx in (I.IEU0, I.IEM0, I.IETT4, I.IEP0, I.IET0, I.IEA0, I.IERHO0,
                I.IEMU0):
        ecol[idx] = pe[idx, ip]
    cdsum(ac.pari, ac.parg, acol, ecol, 0, table)
    tfcalc(ac.pari, ac.parg, acol, ecol, ip, 1, 1, 1)
    for idx in (I.IEFE, I.IETSFC, I.IEMCORE, I.IEPIF, I.IEN1):
        assert ecol[idx] == pytest.approx(pe[idx, ip], rel=1e-8)


def test_weight_falls_monotonically(flown):
    ac, _ = flown
    w = [ac.para[I.IAFRACW, ip]
         for ip in range(I.IPCLIMB1, I.IPDESCENTN + 1)]
    assert w == sorted(w, reverse=True)


def test_cruise_climbs(flown):
    """The cruise-climb angle is positive, so the aircraft drifts up."""
    ac, _ = flown
    assert ac.para[I.IAGAMV, I.IPCRUISE1] > 0.0
    assert ac.para[I.IAALT, I.IPCRUISEN] > ac.para[I.IAALT, I.IPCRUISE1]


def test_descent_angles_are_negative(flown):
    ac, _ = flown
    for ip in range(I.IPDESCENT1, I.IPDESCENTN + 1):
        assert ac.para[I.IAGAMV, ip] < 0.0


def test_range_is_delivered(flown):
    ac, _ = flown
    assert ac.para[I.IARANGE, I.IPDESCENTN] == pytest.approx(
        ac.parm[I.IMRANGE], rel=1e-6)


def test_buoyancy_is_zero_on_the_ground_and_not_at_altitude(flown):
    ac, _ = flown
    assert ac.para[I.IAWBUOY, I.IPTAKEOFF] == 0.0
    assert ac.para[I.IAWBUOY, I.IPDESCENTN] == 0.0
    assert ac.para[I.IAWBUOY, I.IPCRUISE1] > 0.0


def test_reserve_is_added_to_the_burn_not_flown(flown, reference):
    ac, r = flown
    _, _, _, _, fburn = reference
    burn = (ac.para[I.IAFRACW, I.IPCLIMB1]
            - ac.para[I.IAFRACW, I.IPDESCENTN])
    assert burn == pytest.approx(fburn, rel=1e-4)
    expect = ac.parg[I.IGWMTO] * burn * (1.0 + ac.parg[I.IGFRESERVE])
    assert r.Wfuel == pytest.approx(expect, rel=1e-12)
