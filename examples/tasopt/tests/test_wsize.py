"""The outer sizing loop (wsize.f) against the real TASOPT 737 run.

Like ``mission``, this one is not driven by a synthetic case. ``wsize.f`` was
instrumented to dump its complete input state (``pari``, ``parg``, ``parm``
and the whole ``para``/``pare`` matrices, plus the relaxation factors and
flags) on entry and its complete output state on exit, and the shipped
``runs/737/737.tas`` was run. ``tasopt.f`` calls ``wsize`` exactly once for a
non-optimised case, so that dump is the whole sizing.

This is the end-to-end check the port has been building towards: the port is
handed the same inputs and has to size the same aircraft, running its own
fuselage BL, structures, engine and mission the whole way. It converges in the
same 18 iterations to the same WMTO = 174979.1499 lbf.

Agreement
---------
The converged aircraft matches to **1.5e-9** across ``parg``, 1.0e-7 across
``para`` and 5.2e-8 across ``pare``, and the printed takeoff weight is
174979.1500 lbf against the program's 174979.1499. The floor is the engine:
``tfoper`` differentiates numerically where the source differentiates
analytically, so the two Newtons stop at slightly different points inside the
same ball (``DISCREPANCIES.md`` §22), and 1e-9 is what that costs.

Writing this test found three bugs in ``mission`` that
``tests/test_mission.py`` could not, because that test hands the port a
*converged* 737 state and all three are invisible when the answer is already
in the array. Starting from the unset arrays a real sizing starts from, they
are not. See that file's docstring for the retraction; ``mission`` went from
5.7e-6 to 9.8e-11 once they were fixed.

``fortran_ref/wsize_instrumented.f`` holds the two dump blocks. The shipped
file was restored and rebuilt afterwards, and still sizes the 737 to
WTO = 174979.1499 lbf in 18 iterations.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tasopt_py.aero.airfoil import airtable
from tasopt_py.model import Aircraft
from tasopt_py.model import indices as I
from tasopt_py.sizing.wsize import wsize

DATA = Path(__file__).parent / "data"
AIR = Path("/Users/codykarcher/Desktop/Tasopt2.16/air/C.air")

pytestmark = pytest.mark.skipif(
    not AIR.exists() or not (DATA / "wsize_in.txt").exists(),
    reason="airfoil database or wsize dump not present")

# gfortran's E26.18 drops the 'E' when the exponent needs three digits.
_EXP = re.compile(r"(\d)([+-]\d\d\d)$")


def _f(x):
    return float(_EXP.sub(r"\1E\2", x))


def _load_in():
    tok = (DATA / "wsize_in.txt").read_text().split()
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
    iterwmax, initwgt, initeng, _iairf = (int(v) for v in take(4))
    wrlx1, wrlx2, wrlx3 = (_f(v) for v in take(3))
    return ac, dict(iterwmax=iterwmax, initwgt=initwgt, initeng=initeng,
                    wrlx1=wrlx1, wrlx2=wrlx2, wrlx3=wrlx3)


def _load_out():
    tok = (DATA / "wsize_out.txt").read_text().split()
    pos = 0

    def take(n):
        nonlocal pos
        v = [_f(x) for x in tok[pos:pos + n]]
        pos += n
        return v

    parg = {k: v for k, v in enumerate(take(I.IGTOTAL), 1)}
    parm = {k: v for k, v in enumerate(take(I.IMTOTAL), 1)}
    para, pare = {}, {}
    for k in range(1, I.IATOTAL + 1):
        for ip, v in enumerate(take(I.IPTOTAL), 1):
            para[k, ip] = v
    for k in range(1, I.IETOTAL + 1):
        for ip, v in enumerate(take(I.IPTOTAL), 1):
            pare[k, ip] = v
    iterw, conv = int(tok[pos]), int(tok[pos + 1])
    errw = _f(tok[pos + 2])
    return parg, parm, para, pare, iterw, bool(conv), errw


@pytest.fixture(scope="module")
def sized():
    ac, kw = _load_in()
    table = airtable(AIR)
    r = wsize(ac.pari, ac.parg, ac.parm, ac.para, ac.pare, table=table, **kw)
    return ac, r, kw


@pytest.fixture(scope="module")
def reference():
    return _load_out()


def test_converges_in_the_same_number_of_iterations(sized, reference):
    _, r, _ = sized
    _, _, _, _, iterw, conv, _ = reference
    assert r.converged is conv is True
    assert r.iterations == iterw == 18


def test_takeoff_weight_and_fuel(sized, reference):
    _, r, _ = sized
    ac, _, _ = sized
    parg, parm, _, _, _, _, _ = reference
    LB_N = 1.0 / 4.44822
    # The number the shipped program prints: WTO = 174979.1499 lbf.
    assert ac.parm[I.IMWTO] * LB_N == pytest.approx(174979.1499, abs=0.05)
    assert ac.parm[I.IMWTO] == pytest.approx(parm[I.IMWTO], rel=1e-8)
    assert ac.parg[I.IGWMTO] == pytest.approx(parg[I.IGWMTO], rel=1e-8)
    assert ac.parg[I.IGWFUEL] == pytest.approx(parg[I.IGWFUEL], rel=1e-8)
    assert r.errw < 1e-9


def test_component_weights_match(sized, reference):
    ac, _, _ = sized
    parg, _, _, _, _, _, _ = reference
    for name, idx in (("Wfuse", I.IGWFUSE), ("Wwing", I.IGWWING),
                      ("Whtail", I.IGWHTAIL), ("Wvtail", I.IGWVTAIL),
                      ("Weng", I.IGWENG), ("Wnace", I.IGWNACE),
                      ("Wshell", I.IGWSHELL), ("Wcone", I.IGWCONE),
                      ("Wfloor", I.IGWFLOOR), ("Wwindow", I.IGWWINDOW),
                      ("Winsul", I.IGWINSUL), ("Whbend", I.IGWHBEND),
                      ("Wvbend", I.IGWVBEND), ("Wcap", I.IGWCAP),
                      ("Wweb", I.IGWWEB), ("Wfmax", I.IGWFMAX)):
        assert ac.parg[idx] == pytest.approx(parg[idx], rel=1e-8), name


def test_geometry_matches(sized, reference):
    ac, _, _ = sized
    parg, _, _, _, _, _, _ = reference
    for name, idx in (("S", I.IGS), ("b", I.IGB), ("bs", I.IGBS),
                      ("co", I.IGCO), ("cma", I.IGCMA), ("Sh", I.IGSH),
                      ("Sv", I.IGSV), ("bh", I.IGBH), ("bv", I.IGBV),
                      ("coh", I.IGCOH), ("cov", I.IGCOV),
                      ("xwbox", I.IGXWBOX), ("xwing", I.IGXWING),
                      ("xhtail", I.IGXHTAIL), ("xvtail", I.IGXVTAIL),
                      ("xNP", I.IGXNP), ("Vh", I.IGVH), ("Vv", I.IGVV),
                      ("dfan", I.IGDFAN), ("fSnace", I.IGFSNACE)):
        assert ac.parg[idx] == pytest.approx(parg[idx], rel=1e-8), name


def test_fuselage_bl_areas_are_exact(sized, reference):
    """The BL solve sits outside the weight loop and feeds nothing back into
    it, so it carries none of the engine's 1e-9 and stays at test_fusebl's
    own agreement."""
    ac, _, _ = sized
    _, _, para, _, _, _, _ = reference
    for idx in (I.IADAFSURF, I.IADAFWAKE, I.IAKAFTE, I.IAPAFINF):
        for ip in (I.IPCRUISE1, I.IPCLIMB1, I.IPDESCENTN):
            assert ac.para[idx, ip] == pytest.approx(para[idx, ip],
                                                     rel=1e-13)


def test_mission_trajectory_matches(sized, reference):
    ac, _, _ = sized
    _, _, para, _, _, _, _ = reference
    for idx in (I.IAALT, I.IAFRACW, I.IACL, I.IACD, I.IAMACH):
        for ip in range(I.IPCLIMB1, I.IPDESCENTN + 1):
            ref = para[idx, ip]
            if abs(ref) < 1e-12:
                continue
            assert ac.para[idx, ip] == pytest.approx(ref, rel=1e-6), \
                f"index {idx} point {ip}"


def test_iteration_history_matches_the_printed_table(sized):
    """The port reproduces the 18-row convergence table the program prints,
    column for column, to the precision each column is printed at."""
    _, r, _ = sized
    rows = []
    for line in (DATA / "wsize_history.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 11 and parts[0].isdigit():
            rows.append([float(x) for x in parts])
    assert len(rows) == 18
    assert len(r.history) == 18
    # One unit in the last printed place: errW at 10 decimals, the five
    # weights at 4, span/area/HTarea at 3, xwbox at 5.
    ulp = [None, 1e-10] + [1e-4] * 5 + [1e-3] * 3 + [1e-5]
    for got, want in zip(r.history, rows):
        assert int(got[0]) == int(want[0])
        for col, (g, w) in enumerate(zip(got[1:], want[1:]), start=1):
            assert g == pytest.approx(w, abs=1.5 * ulp[col]), \
                f"iteration {int(want[0])} column {col}"


def test_takeoff_and_cg_limits_are_produced(sized):
    ac, r, _ = sized
    assert r.takeoff is not None and r.takeoff.lTO > 0.0
    assert ac.parg[I.IGXCGFWD] < ac.parg[I.IGXCGAFT]
    # Static margin: the neutral point must sit aft of the aft CG limit.
    assert ac.parg[I.IGXNP] > ac.parg[I.IGXCGAFT]


def test_the_fuselage_bl_runs_once_not_once_per_iteration(monkeypatch):
    """fusebl is hoisted out of the weight loop -- the fuselage geometry is an
    input and does not change -- so 18 iterations cost one BL solve."""
    import tasopt_py.sizing.wsize as W

    calls = []
    real = W.fusebl
    monkeypatch.setattr(W, "fusebl",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    ac, kw = _load_in()
    kw["iterwmax"] = 3          # enough to show it is not per-iteration
    W.wsize(ac.pari, ac.parg, ac.parm, ac.para, ac.pare,
            table=airtable(AIR), **kw)
    assert len(calls) == 1
