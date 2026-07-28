"""Wing moment (surfcm.f) and tail planform (tailpo.f) against the Fortran."""
from __future__ import annotations

import csv
from pathlib import Path

from tasopt_py.aero.moment import surfcm, tailpo

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

WING = dict(b=35.0, bs=12.0, bo=3.6, sweep=26.0, Xaxis=0.40,
            lambdat=0.25, lambdas=0.65, gammat=0.22, gammas=0.70,
            AR=10.1, fLo=-0.3, fLt=-0.05,
            cmpo=-0.20, cmps=-0.20, cmpt=-0.02)
WCASES = {1: {}, 2: dict(sweep=0.0, Xaxis=0.25),
          3: dict(sweep=35.0, lambdat=0.15, gammat=0.10, AR=7.5)}

TAIL = dict(S=42.0, AR=6.0, lambda_=0.25, qne=12000.0, CLmax=2.0)
TCASES = {1: {}, 2: dict(S=25.0, AR=1.8, lambda_=0.7),
          3: dict(S=60.0, AR=9.0, lambda_=0.4, qne=18000.0)}


def test_moment_and_tailpo_match_fortran():
    W = {ic: surfcm(**{**WING, **kw}) for ic, kw in WCASES.items()}
    T = {ic: tailpo(**{**TAIL, **kw}) for ic, kw in TCASES.items()}
    n = 0
    with (DATA / "moment_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            got = getattr(W[ic], name) if name.startswith("CM") else getattr(T[ic], name)
            rel = abs(got - ref) / max(abs(ref), 1e-300)
            assert rel <= RTOL, f"case {ic} {name}: {got!r} vs {ref!r} rel {rel:.3e}"
            n += 1
    assert n == 15


def test_unswept_quarter_chord_axis_kills_the_geometric_moment():
    """With no sweep and the axis at the quarter chord, CM1 should vanish.

    Every term in CM1 carries either (Xaxis - 0.25) or tan(sweep), so setting
    both to zero must leave nothing but the airfoil contribution in CM0.
    """
    r = surfcm(**{**WING, "sweep": 0.0, "Xaxis": 0.25})
    assert abs(r.CM1) < 1e-15
    assert r.CM0 != 0.0


def test_tail_planform_area_closes():
    """Span and root chord must reproduce the trapezoidal area S."""
    for kw in TCASES.values():
        p = dict(TAIL, **kw)
        r = tailpo(**p)
        S_check = 0.5 * r.b * r.co * (1.0 + p["lambda_"])
        assert abs(S_check - p["S"]) / p["S"] < 1e-14
