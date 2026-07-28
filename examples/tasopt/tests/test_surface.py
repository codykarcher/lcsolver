"""Wing/tail box port (surfw.f) against the compiled Fortran.

Three cases, chosen to cover every branch of the planform switch:

  case 1  iwplan=1  cantilever with an engine at the break station
  case 2  iwplan=0  clean cantilever (We = 0)
  case 3  iwplan=2  strut-braced (the else branch: strut tension, cosLs < 1)
"""
from __future__ import annotations

import csv
from pathlib import Path

from tasopt_py.structures.surface import surfw

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

BASE = dict(
    gee=9.81, po=120000.0, b=35.0, bs=12.0, bo=3.6, co=6.0, zs=5.0,
    lambdat=0.25, lambdas=0.65, gammat=0.22, gammas=0.70,
    Nload=3.0, iwplan=1, We=30000.0,
    Winn=12000.0, Wout=8000.0, dyWinn=40000.0, dyWout=60000.0,
    sweep=26.0, wbox=0.50, hboxo=0.1268, hboxs=0.1266, rh=0.75, fLt=-0.05,
    tauweb=1.38e8, sigcap=2.06e8, sigstrut=2.06e8,
    Ecap=6.9e10, Eweb=6.9e10, Gcap=2.6e10, Gweb=2.6e10,
    rhoweb=2700.0, rhocap=2700.0, rhostrut=2700.0, rhofuel=817.0,
)

CASES = {1: {}, 2: dict(iwplan=0, We=0.0), 3: dict(iwplan=2)}


def test_surface_matches_fortran():
    res = {ic: surfw(**{**BASE, **kw}) for ic, kw in CASES.items()}
    n = 0
    with (DATA / "surfw_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            got = getattr(res[ic], name)
            rel = abs(got - ref) / max(abs(ref), 1e-300)
            assert rel <= RTOL, f"case {ic} {name}: {got!r} vs {ref!r} rel {rel:.3e}"
            n += 1
    assert n == 111


def test_cantilever_has_no_strut():
    """iwplan 0 and 1 must produce no strut at all."""
    for iw in (0, 1):
        r = surfw(**{**BASE, "iwplan": iw})
        assert r.Astrut == 0.0
        assert r.Wstrut == 0.0
        assert r.lsp == 0.0
        assert r.cosLs == 1.0


def test_strut_braced_relieves_the_root():
    """A strut carries inboard load, so the root box sees less than a cantilever.

    With a strut the inboard box is sized to the strut-attach shear/moment
    only (So = Ss, Mo = Ms), so its gauges cannot exceed the cantilever's.
    """
    cant = surfw(**{**BASE, "iwplan": 1})
    strut = surfw(**{**BASE, "iwplan": 2})
    assert strut.Mo <= cant.Mo
    assert strut.tbcapo <= cant.tbcapo
    assert strut.Wstrut > 0.0
    assert 0.0 < strut.cosLs <= 1.0


def test_root_loads_floored_at_strut_attach():
    """So/Mo are floored at Ss/Ms so a heavy outboard engine cannot invert taper."""
    heavy = surfw(**{**BASE, "iwplan": 1, "We": 5.0e5})
    assert heavy.So >= heavy.Ss
    assert heavy.Mo >= heavy.Ms
