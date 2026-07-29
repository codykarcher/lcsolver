"""Design optimisation (fobj.f, simpop.f, hsort.f) against the real 737.

``fobj.f`` was instrumented to dump every objective evaluation -- the variable
vector, the raw and penalised objective, and the four constraint values -- and
the shipped 737 case was run with ``Lopt = T`` and ``istepmax = 4``. That gives
18 objective evaluations: a 12-vertex initial simplex over the 11 active
design variables, then four Nelder-Mead steps.

**The port reproduces all 18, in the same order, at the same simplex
vertices.** The vertices agree to 8.8e-16 -- machine precision, so the search
is taking identical moves, not merely similar ones -- and the objective values
to 1.2e-9, which is the ``tfoper`` numerical-Jacobian floor every
whole-aircraft number in this port sits at.

That full comparison takes about a hundred seconds, because each evaluation is
a complete aircraft: a sizing loop plus every weighted off-design mission. It
is therefore opt-in::

    TASOPT_SLOW=1 python -m pytest tests/test_optimise.py

Without the flag this file checks one objective evaluation against the
Fortran's first record, and everything cheap -- the simplex algebra, the
heapsort, the variable scatter/gather and the penalty arithmetic -- exactly.

``fortran_ref/fobj_instrumented.f`` holds the dump block. ``fobj.f`` was
restored and rebuilt afterwards, and the 737 still sizes to
WTO = 174979.1499 lbf in 18 iterations.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from tasopt_py.aero.airfoil import airtable
from tasopt_py.model import indices as I
from tasopt_py.optimise import (ALPHA, BETA, GAMMA, LAMT_CLAMP, fobj, hsort,
                                simpop, voptget, voptset)
from tasopt_py.tasfile import read_tas

DATA = Path(__file__).parent / "data"
TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas")
SLOW = os.environ.get("TASOPT_SLOW")

_EXP = re.compile(r"(\d)([+-]\d\d\d)$")


def _f(x):
    return float(_EXP.sub(r"\1E\2", x))


def _records():
    """Every fobj call the instrumented Fortran made."""
    tok = (DATA / "fobj_calls.txt").read_text().split()
    out, p = [], 0
    while p < len(tok):
        assert tok[p] == "CALL"
        istep, nv, nc = int(tok[p + 1]), int(tok[p + 2]), int(tok[p + 3])
        p += 4
        v = [_f(t) for t in tok[p:p + nv]]
        p += nv
        fun, func = _f(tok[p]), _f(tok[p + 1])
        p += 2
        con = [_f(t) for t in tok[p:p + nc]]
        p += nc
        PFEI, WMTO, Wfuel = (_f(t) for t in tok[p:p + 3])
        p += 3
        out.append(dict(istep=istep, v=v, fun=fun, func=func, con=con,
                        PFEI=PFEI, WMTO=WMTO, Wfuel=Wfuel))
    return out


# --- the cheap parts, tested exactly --------------------------------------

def test_hsort_matches_a_stable_ascending_sort():
    """hsort.f is ported literally rather than replaced by sorted(), because
    a different tie-break reorders equal simplex vertices and sends the
    search somewhere else."""
    for a in ([3.0, 1.0, 2.0], [1.0], [], [2.0, 2.0, 1.0],
              [5.0, 4.0, 3.0, 2.0, 1.0], [1.5, -2.0, 0.0, 1e6, -1e6]):
        idx = hsort(a)
        assert sorted(idx) == list(range(1, len(a) + 1))
        got = [a[k - 1] for k in idx]
        assert got == sorted(a)


def test_simplex_constants_are_the_live_ones():
    """Two other (alpha, beta, gamma) sets are commented out above these."""
    assert (ALPHA, BETA, GAMMA) == (1.0, 0.7, 1.4)


def test_simpop_minimises_a_quadratic():
    calls = []

    def f_of(v, tag, istep):
        calls.append(tag)
        return (v[0] - 3.0) ** 2 + (v[1] + 1.0) ** 2

    simplex = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    r = simpop(f_of, simplex, 1e-8, 0, 300)
    assert r.converged
    assert r.x[0] == pytest.approx(3.0, abs=1e-3)
    assert r.x[1] == pytest.approx(-1.0, abs=1e-3)
    # The three initial vertices are evaluated first, then reflections and
    # friends, and the best vertex is re-evaluated on the way out.
    assert calls[:3] == ["j", "j", "j"] and calls[-1] == "!"
    assert set(calls) <= {"j", "r", "e", "c", "m", "!"}


def test_simpop_reports_hitting_the_step_limit():
    r = simpop(lambda v, tag, istep: sum(x * x for x in v),
               [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], 1e-30, 0, 3)
    assert not r.converged and r.steps == 3


def test_simpop_rejects_a_simplex_of_the_wrong_size():
    with pytest.raises(ValueError, match="3 vertices"):
        simpop(lambda v, t, i: 0.0, [[0.0, 0.0], [1.0, 0.0]], 1e-6, 0, 1)


@pytest.mark.skipif(not TAS.exists(), reason="737.tas not present")
def test_variable_scatter_and_gather_round_trip():
    case = read_tas(TAS)
    iovar = [k for k in I.CPARO if case.dvarso.get(k, 0.0) != 0.0]
    assert iovar == ["CL", "AR", "sweep", "hboxo", "hboxs", "lambdat",
                     "rcls", "rclt", "alt", "Tt4CR", "Tt4TO"]
    v = voptget(case, iovar)
    bumped = [x * 1.01 for x in v]
    voptset(case, iovar, bumped)
    assert voptget(case, iovar) == pytest.approx(bumped)


@pytest.mark.skipif(not TAS.exists(), reason="737.tas not present")
def test_the_tip_taper_clamp_is_written_back_into_the_search_vector():
    """voptset clamps lambdat at 0.1 and *modifies vopt*, so the simplex
    vertex itself moves -- a clamp the search can see."""
    case = read_tas(TAS)
    iovar = ["lambdat"]
    vopt = [0.05]
    voptset(case, iovar, vopt)
    assert vopt[0] == LAMT_CLAMP == 0.1
    assert case.parg[I.IGLAMBDAT] == LAMT_CLAMP


# --- against the Fortran ---------------------------------------------------

@pytest.mark.skipif(not TAS.exists() or not (DATA / "fobj_calls.txt").exists(),
                    reason="737 case or fobj dump not present")
def test_the_dump_is_the_run_we_think_it_is():
    recs = _records()
    assert len(recs) == 18
    assert len(recs[0]["v"]) == 11 and len(recs[0]["con"]) == 4
    # The first vertex is the unperturbed 737: CL 0.57, AR 10.1, sweep 26.
    assert recs[0]["v"][:3] == pytest.approx([0.57, 10.1, 26.0])


@pytest.mark.skipif(not TAS.exists() or not (DATA / "fobj_calls.txt").exists(),
                    reason="737 case or fobj dump not present")
def test_one_objective_evaluation_matches_the_fortran():
    ref = _records()[0]
    case = read_tas(TAS)
    iovar = [k for k in I.CPARO if case.dvarso.get(k, 0.0) != 0.0]
    r = fobj(case, iovar, list(ref["v"]), table=airtable(case.airfoil_file),
             settings=case.settings, state={"initwgt": 0, "initeng": 0})

    assert r.fun == pytest.approx(ref["fun"], rel=1e-8)
    assert r.func == pytest.approx(ref["func"], rel=1e-8)
    assert r.PFEI == pytest.approx(ref["PFEI"], rel=1e-8)
    assert case.parg[I.IGWMTO] == pytest.approx(ref["WMTO"], rel=1e-8)
    # The constraints are differences of nearly equal numbers, so they carry
    # the objective's 1e-9 at full size.
    for k, (got, want) in enumerate(zip(r.con, ref["con"])):
        assert got == pytest.approx(want, rel=1e-7, abs=1e-9), f"con[{k}]"


@pytest.mark.skipif(not SLOW, reason="set TASOPT_SLOW=1 (about 100 seconds)")
def test_the_whole_search_path_matches_the_fortran():
    """Every one of the 18 objective evaluations, in order, at the same
    simplex vertices -- so the Nelder-Mead is taking identical moves."""
    recs = _records()
    case = read_tas(TAS)
    tab = airtable(case.airfoil_file)
    iovar = [k for k in I.CPARO if case.dvarso.get(k, 0.0) != 0.0]

    v0 = voptget(case, iovar)
    simplex = [list(v0)]
    for iv, name in enumerate(iovar):
        v = list(v0)
        v[iv] += case.dvarso[name]
        simplex.append(v)

    state = {"initwgt": 0, "initeng": 0}
    calls = []

    def f_of(vertex, tag, istep):
        r = fobj(case, iovar, vertex, table=tab, settings=case.settings,
                 state=state)
        calls.append((list(vertex), r))
        return r.func

    simpop(f_of, simplex, case.settings.Wftol, 0, 4)

    assert len(calls) == len(recs) == 18
    for k, ((v, r), ref) in enumerate(zip(calls, recs), start=1):
        for a, b in zip(v, ref["v"]):
            assert a == pytest.approx(b, rel=1e-13), f"call {k} vertex"
        assert r.func == pytest.approx(ref["func"], rel=1e-8), f"call {k}"
