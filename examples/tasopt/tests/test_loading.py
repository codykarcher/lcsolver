"""Wing root loading and section cl (wingpo.f) against the compiled Fortran.

Also cross-checks the section coefficients against ``surfcd2``, which
computes them independently from the same planform: two separately ported
routines agreeing is a stronger statement than either matching Fortran alone.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_wingpo drv_wingpo.f wingpo.f
    ./drv_wingpo > tests/data/wingpo_ref.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.aero.drag import surfcd2
from tasopt_py.aero.loading import planform_integrals, wingcl, wingpo

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

PLAN = dict(b=35.0, bs=12.0, bo=3.6, lambdat=0.25, lambdas=0.65,
            gammat=0.22, gammas=0.70, AR=11.667, fLo=-0.3, fLt=-0.05)
PO = dict(N=3.0, W=700000.0, Lhtail=-20000.0)
CLK = dict(sweep=26.0, CL=0.55, CLhtail=-0.05, duo=0.0, dus=0.0, dut=0.0)

CASES = {
    1: ({}, {}, {}),
    2: ({}, dict(Lhtail=0.0), dict(sweep=0.0)),
    3: (dict(lambdat=0.15, gammat=0.10, AR=7.5), dict(N=2.5), {}),
    4: ({}, {}, dict(duo=0.018, dus=0.014, dut=0.009, sweep=35.0)),
    5: (dict(fLo=0.0, fLt=0.0), dict(W=1200000.0), dict(CL=0.80)),
}


def _solved():
    out = {}
    for ic, (dp, dpo, dcl) in CASES.items():
        plan = {**PLAN, **dp}
        out[ic] = (wingpo(**plan, **{**PO, **dpo}),
                   wingcl(**plan, **{**CLK, **dcl}))
    return out


def test_matches_fortran():
    got = _solved()
    n = 0
    with (DATA / "wingpo_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            po, cl = got[ic]
            actual = po if name == "po" else getattr(cl, name)
            rel = abs(actual - ref) / max(abs(ref), 1e-300)
            assert rel <= RTOL, f"case {ic} {name}: {actual!r} vs {ref!r} rel {rel:.3e}"
            n += 1
    assert n == 20


def test_section_cl_agrees_with_surfcd2():
    """surfcd2 derives the same coefficients from the same planform."""
    plan = dict(PLAN)
    r = surfcd2(S=plan["b"] ** 2 / plan["AR"], b=plan["b"], bs=plan["bs"],
                bo=plan["bo"], lambdat=plan["lambdat"], lambdas=plan["lambdas"],
                gammat=plan["gammat"], gammas=plan["gammas"],
                toco=0.13, tocs=0.12, toct=0.10, Mach=0.8, sweep=26.0,
                co=plan["b"] ** 2 / plan["AR"] / (plan["b"] * 0.5),
                CL=0.55, CLhtail=-0.05, fLo=plan["fLo"], fLt=plan["fLt"],
                Reco=2.0e7, aRexp=-0.15, kSuns=0.5, fexcd=1.0, ARe=1.0e7,
                airfoil=lambda cl, toc, M: (0.005, 0.0025, 0.0, -0.1))
    c = wingcl(**plan, **CLK)
    # Both scale the same cl1 by gamma/lambda, so the ratios must match even
    # where the absolute level depends on the chord convention.
    assert r.clps / r.clpo == pytest.approx(c.cls / c.clo, rel=1e-12)
    assert r.clpt / r.clpo == pytest.approx(c.clt / c.clo, rel=1e-12)


def test_root_loading_balances_the_net_load():
    """Integrating the load distribution must return N*W - Lhtail."""
    plan = dict(PLAN)
    po = wingpo(**plan, **PO)
    _, _, Kp = planform_integrals(**plan)
    assert po * Kp * plan["b"] == pytest.approx(
        PO["N"] * PO["W"] - PO["Lhtail"], rel=1e-12)


def test_tail_downforce_raises_the_wing_load():
    """A download on the tail is carried by the wing, so po goes up."""
    plan = dict(PLAN)
    with_tail = wingpo(**plan, **PO)                      # Lhtail negative
    without = wingpo(**plan, **{**PO, "Lhtail": 0.0})
    assert with_tail > without


def test_overspeed_reduces_section_cl():
    plan = dict(PLAN)
    base = wingcl(**plan, **CLK)
    fast = wingcl(**plan, **{**CLK, "duo": 0.05})
    assert fast.clo < base.clo
