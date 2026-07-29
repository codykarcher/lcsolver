"""Off-design mission (woper.f) against the real TASOPT 737 run.

``woper.f`` was instrumented to dump its complete input state -- ``pari``,
``parg``, ``parm``, the working ``para``/``pare`` matrices *and* the design
mission's ``parad``/``pared`` -- on entry, and its complete output state on
exit, and the shipped ``runs/737/737.tas`` was run. This is the second
convergence table that case prints:

```
  iterw     errW           WTO        Wfuel       h_CR1     h_CR2       gam_BOC   gam_TOC
     1  1.0000000000  174979.1499   47487.2555   35000.00   39692.28    9.04160    1.24042
     2 -0.0000000000  174979.1499   47487.2555   35000.00   39692.28    9.04160    1.24042
     3  0.0000000000  174979.1499   47487.2555   35000.00   39692.28    9.04160    1.24042
```

The 737 case's one off-design mission has the same range and payload as the
design mission, so ``woper`` reproduces the sized aircraft exactly -- which
makes it a sharp test of the *scaling* logic rather than a weak one. Every
weight-fraction scale factor, the Breguet fuel guess and the ``pralt``
altitude rescale all have to come back to unity for the answer to land back on
174979.1499 lbf. A single sign or ratio wrong anywhere moves it.

Agreement: takeoff weight to 1.2e-10 and fuel to 4.6e-10, with the trajectory
and engine state to 5e-8 -- the same engine-Jacobian floor as ``wsize``
(``DISCREPANCIES.md`` §22).

``fortran_ref/woper_instrumented.f`` holds the two dump blocks. The shipped
file was restored and rebuilt afterwards, and the 737 still sizes to
WTO = 174979.1499 lbf in 18 iterations.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tasopt_py.aero.airfoil import airtable
from tasopt_py.model import Aircraft
from tasopt_py.model import indices as I
from tasopt_py.sizing.woper import woper

DATA = Path(__file__).parent / "data"
AIR = Path("/Users/codykarcher/Desktop/Tasopt2.16/air/C.air")

pytestmark = pytest.mark.skipif(
    not AIR.exists() or not (DATA / "woper_in.txt").exists(),
    reason="airfoil database or woper dump not present")

_EXP = re.compile(r"(\d)([+-]\d\d\d)$")


def _f(x):
    return float(_EXP.sub(r"\1E\2", x))


def _load_in():
    tok = (DATA / "woper_in.txt").read_text().split()
    pos = 0

    def take(n):
        nonlocal pos
        v = tok[pos:pos + n]
        pos += n
        return v

    ac = Aircraft()
    design = Aircraft()
    for k, v in enumerate(take(I.IITOTAL), 1):
        ac.pari[k] = int(v)
    for k, v in enumerate(take(I.IGTOTAL), 1):
        ac.parg[k] = _f(v)
    for k, v in enumerate(take(I.IMTOTAL), 1):
        ac.parm[k] = _f(v)
    for target, n in ((ac.para, I.IATOTAL), (ac.pare, I.IETOTAL),
                      (design.para, I.IATOTAL), (design.pare, I.IETOTAL)):
        for k in range(1, n + 1):
            for ip, v in enumerate(take(I.IPTOTAL), 1):
                target[k, ip] = _f(v)
    iterfmax, initeng, _iairf = (int(v) for v in take(3))
    return ac, design, dict(iterfmax=iterfmax, initeng=initeng)


def _load_out():
    tok = (DATA / "woper_out.txt").read_text().split()
    pos = 0

    def take(n):
        nonlocal pos
        v = [_f(x) for x in tok[pos:pos + n]]
        pos += n
        return v

    parm = {k: v for k, v in enumerate(take(I.IMTOTAL), 1)}
    para, pare = {}, {}
    for store, n in ((para, I.IATOTAL), (pare, I.IETOTAL)):
        for k in range(1, n + 1):
            for ip, v in enumerate(take(I.IPTOTAL), 1):
                store[k, ip] = v
    iterw, conv = int(tok[pos]), int(tok[pos + 1])
    return parm, para, pare, iterw, bool(conv), _f(tok[pos + 2])


@pytest.fixture(scope="module")
def flown():
    ac, design, kw = _load_in()
    r = woper(ac.pari, ac.parg, ac.parm, ac.para, ac.pare,
              design.para, design.pare, table=airtable(AIR), **kw)
    return ac, r


@pytest.fixture(scope="module")
def reference():
    return _load_out()


def test_converges_in_the_same_number_of_iterations(flown, reference):
    _, r = flown
    _, _, _, iterw, conv, _ = reference
    assert r.converged is conv is True
    assert r.iterations == iterw == 3


def test_takeoff_weight_and_fuel(flown, reference):
    ac, _ = flown
    parm, _, _, _, _, _ = reference
    LB_N = 1.0 / 4.44822
    assert ac.parm[I.IMWTO] * LB_N == pytest.approx(174979.1499, abs=0.05)
    assert ac.parm[I.IMWTO] == pytest.approx(parm[I.IMWTO], rel=1e-9)
    assert ac.parm[I.IMWFUEL] == pytest.approx(parm[I.IMWFUEL], rel=1e-9)


def test_the_off_design_mission_recovers_the_design_aircraft(flown):
    """This case's off-design mission is the design mission, so every scale
    factor in the routine has to come back to one."""
    ac, _ = flown
    assert ac.parm[I.IMWTO] == pytest.approx(ac.parg[I.IGWMTO], rel=1e-8)
    assert ac.parm[I.IMWFUEL] == pytest.approx(ac.parg[I.IGWFUEL], rel=1e-8)


def test_trajectory_matches(flown, reference):
    ac, _ = flown
    _, para, _, _, _, _ = reference
    for idx in (I.IAALT, I.IAFRACW, I.IACL, I.IACD, I.IAMACH, I.IARANGE,
                I.IATIME, I.IAGAMV):
        for ip in range(I.IPCLIMB1, I.IPDESCENTN + 1):
            ref = para[idx, ip]
            if abs(ref) < 1e-12:
                continue
            assert ac.para[idx, ip] == pytest.approx(ref, rel=1e-6), \
                f"index {idx} point {ip}"


def test_engine_state_matches(flown, reference):
    ac, _ = flown
    _, _, pare, _, _, _ = reference
    for idx in (I.IEFE, I.IETSFC, I.IEMCORE, I.IETT4, I.IEPIF, I.IEN1,
                I.IEU0, I.IEM0):
        for ip in range(I.IPSTATIC, I.IPDESCENTN + 1):
            ref = pare[idx, ip]
            if abs(ref) < 1e-12 or abs(ref) > 1e300:
                continue
            assert ac.pare[idx, ip] == pytest.approx(ref, rel=1e-6), \
                f"index {idx} point {ip}"


def test_iteration_history_matches_the_printed_table(flown):
    _, r = flown
    rows = []
    for line in (DATA / "woper_history.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 8 and parts[0].isdigit():
            rows.append([float(x) for x in parts])
    assert len(rows) == 3 and len(r.history) == 3
    # One unit in the last printed place -- weights at 4 decimals, altitudes
    # at 2, climb angles at 5 -- except errW, which is a difference of two
    # nearly equal takeoff weights and so shows the 1e-10 disagreement in
    # those weights at full size rather than relative to them.
    ulp = [None, 1e-9, 1e-4, 1e-4, 1e-2, 1e-2, 1e-5, 1e-5]
    for got, want in zip(r.history, rows):
        assert int(got[0]) == int(want[0])
        for col, (g, w) in enumerate(zip(got[1:], want[1:]), start=1):
            assert g == pytest.approx(w, abs=1.5 * ulp[col]), \
                f"iteration {int(want[0])} column {col}"


def test_takeoff_is_produced(flown):
    _, r = flown
    assert r.takeoff is not None and r.takeoff.lTO > 0.0


def test_nacelle_cf_starts_flat_not_from_the_design_mission():
    """woper.f copies the design mission's Cfnace over ipstatic..ipdescentn
    and then overwrites every point with 0.003 a few lines later, so the copy
    is dead."""
    import inspect

    from tasopt_py.sizing import woper as W
    src = inspect.getsource(W.woper)
    assert src.index("para[I.IACFNACE, jp] = parad[I.IACFNACE, jp]") \
        < src.index("para[I.IACFNACE, jp] = NACELLE_CF_GUESS")
    assert W.NACELLE_CF_GUESS == 0.003
