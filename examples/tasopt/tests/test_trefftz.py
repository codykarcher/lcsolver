"""Trefftz-plane induced drag (trefftz1 in trefftz.f) vs the compiled Fortran.

Four cases: wing alone and wing plus a raised horizontal tail, each with the
loading free and with it rescaled to specified surface lifts.

Note ``trefftz.f`` holds a second routine, plain ``trefftz``, which takes
overspeed factors ``fduo``/``fdus``/``fdu1`` instead of a root loading. Its
only call site, in ``cdsum.f:471``, is commented out, so it is dead in the
shipped code and is not ported. Same situation as ``tfani.f``.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_trefftz drv_trefftz.f trefftz.f
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.aero.trefftz import BUNCH, TrefftzError, trefftz1

DATA = Path(__file__).parent / "data"
RTOL = 1e-12

WING = dict(nsurf=1, npout=[20], npinn=[6], npimg=[3], Sref=124.0, bref=35.0,
            b=[35.0], bs=[10.5], bo=[3.6], bop=[3.6], zcent=[0.0], po=[1.0],
            gammat=[0.28], gammas=[0.78], fLo=-0.3, ktip=16,
            CLsurfsp=[0.55], idim=360)
TAIL = dict(nsurf=2, npout=[20, 12], npinn=[6, 4], npimg=[3, 2],
            b=[35.0, 13.0], bs=[10.5, 4.0], bo=[3.6, 2.0], bop=[3.6, 2.0],
            zcent=[0.0, 2.5], po=[1.0, -0.18], gammat=[0.28, 0.30],
            gammas=[0.78, 0.80], CLsurfsp=[0.55, -0.08])


def _case(ic, **over):
    kw = dict(WING, Lspec=(ic in (2, 4)))
    if ic >= 3:
        kw.update(TAIL)
    kw.update(over)
    return trefftz1(**kw)


def test_matches_fortran():
    got = {ic: _case(ic) for ic in (1, 2, 3, 4)}
    n = 0
    with (DATA / "trefftz_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            g = got[ic]
            if name == "CL":
                a = g.CL
            elif name == "CD":
                a = g.CD
            elif name == "spane":
                a = g.spanef
            elif name.startswith("CLs"):
                a = g.CLsurf[int(name[3:]) - 1]
            elif name.startswith("gc"):
                a = g.gc[int(name[2:]) - 1]
            elif name.startswith("vnc"):
                a = g.vnc[int(name[3:]) - 1]
            elif name.startswith("wc"):
                a = g.wc[int(name[2:]) - 1]
            else:
                raise KeyError(name)
            assert a == pytest.approx(ref, rel=RTOL), f"case {ic} {name}"
            n += 1
    assert n == 90


def test_specified_lift_is_delivered():
    r = _case(4)
    assert r.CLsurf[0] == pytest.approx(0.55, rel=1e-10)
    assert r.CLsurf[1] == pytest.approx(-0.08, rel=1e-10)
    assert r.CL == pytest.approx(0.55 - 0.08, rel=1e-10)


def test_induced_drag_is_positive_and_below_elliptical_efficiency():
    """A real planform cannot beat the elliptical minimum, so spanef <= 1."""
    for ic in (1, 2, 3, 4):
        r = _case(ic)
        assert r.CD > 0.0
        assert 0.0 < r.spanef <= 1.0


def test_drag_scales_with_lift_squared():
    """Induced drag is quadratic in circulation, so doubling po quadruples CD."""
    base = _case(1)
    twice = _case(1, po=[2.0])
    assert twice.CL / base.CL == pytest.approx(2.0, rel=1e-10)
    assert twice.CD / base.CD == pytest.approx(4.0, rel=1e-10)


def test_refining_the_panelling_converges():
    """Halving the panel size must move CD by much less than the panels do."""
    coarse = _case(1, npout=[20], npinn=[6], npimg=[3])
    fine = _case(1, npout=[40], npinn=[12], npimg=[6])
    assert abs(fine.CD - coarse.CD) / coarse.CD < 0.01


def test_tail_downwash_couples_the_surfaces():
    """A downloaded tail in the wing's wake is not the same as the two apart.

    Solving them together must not give the sum of the isolated drags --
    if it did, the Trefftz plane would not be doing anything.
    """
    both = _case(3)
    wing = _case(1)
    tail_only = trefftz1(**dict(WING, Lspec=False, b=[13.0], bs=[4.0],
                                bo=[2.0], bop=[2.0], zcent=[2.5], po=[-0.18],
                                gammat=[0.30], gammas=[0.80],
                                npout=[12], npinn=[4], npimg=[2],
                                CLsurfsp=[-0.08]))
    assert both.CD != pytest.approx(wing.CD + tail_only.CD, rel=1e-3)


def test_tip_rolloff_is_enforced():
    """The loading must vanish at the tip or the downwash integral diverges."""
    r = _case(1)
    assert abs(r.gc[0]) > 0.5             # full loading at the root
    tip = [g for g in r.gc[:29] if g != 0.0][-1]
    assert abs(tip) < 0.2                 # rolled off by the tip


def test_bunching_constant_is_the_live_one():
    """trefftz.f carries three values for bunch; 0.5 is the uncommented one."""
    assert BUNCH == 0.5


def test_too_many_panels_raises():
    with pytest.raises(TrefftzError, match="overflow"):
        _case(1, npout=[400])
