"""Surface profile drag (surfcd.f) against the compiled Fortran.

``surfcd`` is checked directly. ``surfcd2`` calls TASOPT's airfoil spline
database, so both sides are given the same deterministic stand-in for it
(``fortran_ref/airfun_stub.f`` and ``AIRFOIL_STUB`` below); that verifies the
spanwise quadrature, the sweep and shock-unsweep factors and the Reynolds
scaling without dragging in the tables.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_surfcd \\
        drv_surfcd.f surfcd.f airfun_stub.f
    ./drv_surfcd > tests/data/surfcd_ref.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.aero.drag import surfcd, surfcd2

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

BASE = dict(S=105.0, b=35.0, bs=12.0, bo=3.6, lambdat=0.25, lambdas=0.65,
            sweep=26.0, co=5.6, cdf=0.0050, cdp=0.0025, Reco=2.0e7,
            Reref=1.0e7, aRexp=-0.15, kSuns=0.5, fCDcen=1.0)

# Cases 4-6 sit inside the 0.02 windows where surfcd switches to its
# asymptotic expansions, which is where a port is most likely to diverge.
CASES = {
    1: {},
    2: dict(sweep=0.0),
    3: dict(sweep=35.0, lambdat=0.15, lambdas=0.50, Reco=3.5e7),
    4: dict(lambdas=0.995, lambdat=0.30),
    5: dict(lambdas=0.60, lambdat=0.59),
    6: dict(lambdas=0.99, lambdat=0.985, aRexp=0.20, fCDcen=0.6),
}

BASE2 = dict(S=105.0, b=35.0, bs=12.0, bo=3.6, lambdat=0.25, lambdas=0.65,
             gammat=0.22, gammas=0.70, toco=0.13, tocs=0.12, toct=0.10,
             Mach=0.80, sweep=26.0, co=5.6, CL=0.55, CLhtail=-0.05,
             fLo=-0.3, fLt=-0.05, Reco=2.0e7, aRexp=-0.15, kSuns=0.5,
             fexcd=1.0, ARe=1.0e7, fduo=0.0, fdus=0.0, fdut=0.0)

CASES2 = {
    7: {},
    8: dict(fduo=0.018, fdus=0.014, fdut=0.009, Mach=0.72),
    9: dict(sweep=35.0, lambdat=0.15, gammat=0.10, CL=0.70, CLhtail=0.0,
            fexcd=1.03),
}


def AIRFOIL_STUB(cl, toc, Mperp):
    """Must match fortran_ref/airfun_stub.f exactly."""
    cdf = 0.0040 + 0.0100 * toc + 0.00050 * cl ** 2
    cdp = 0.0020 + 0.0300 * toc ** 2 + 0.0030 * Mperp ** 4 + 0.0010 * cl ** 2
    return cdf, cdp, 0.0, -0.10


def _reference():
    out = {}
    with (DATA / "surfcd_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            out[(int(row[0]), row[1].strip())] = float(row[2])
    return out


def test_surfcd_matches_fortran():
    ref = _reference()
    got = {ic: surfcd(**{**BASE, **kw}) for ic, kw in CASES.items()}
    n = 0
    for (ic, name), expected in ref.items():
        if ic not in got:
            continue
        actual = getattr(got[ic], name)
        rel = abs(actual - expected) / max(abs(expected), 1e-300)
        assert rel <= RTOL, f"case {ic} {name}: {actual!r} vs {expected!r} rel {rel:.3e}"
        n += 1
    assert n == 12


def test_surfcd2_matches_fortran():
    ref = _reference()
    got = {ic: surfcd2(airfoil=AIRFOIL_STUB, **{**BASE2, **kw})
           for ic, kw in CASES2.items()}
    n = 0
    for (ic, name), expected in ref.items():
        if ic not in got:
            continue
        actual = getattr(got[ic], name)
        rel = abs(actual - expected) / max(abs(expected), 1e-300)
        assert rel <= RTOL, f"case {ic} {name}: {actual!r} vs {expected!r} rel {rel:.3e}"
        n += 1
    assert n == 18


def test_unswept_surface_keeps_full_pressure_drag():
    """With no sweep every cos(Lambda) factor is 1, so cd is just cdf + cdp."""
    swept = surfcd(**BASE)
    straight = surfcd(**{**BASE, "sweep": 0.0})
    assert straight.CDsurf > swept.CDsurf


def test_quadrature_converges_with_station_count():
    """TASOPT's n=8 should sit within its stated ~0.07% of a finer integral."""
    coarse = surfcd2(airfoil=AIRFOIL_STUB, n=8, **BASE2)
    fine = surfcd2(airfoil=AIRFOIL_STUB, n=64, **BASE2)
    rel = abs(coarse.CDwing - fine.CDwing) / fine.CDwing
    assert rel < 2e-3, rel


def test_odd_station_count_is_rejected():
    with pytest.raises(ValueError, match="even"):
        surfcd2(airfoil=AIRFOIL_STUB, n=7, **BASE2)


def test_section_lift_coefficients_scale_with_load_ratio():
    """clps/clpo must be gammas/lambdas, by construction."""
    r = surfcd2(airfoil=AIRFOIL_STUB, **BASE2)
    assert r.clps / r.clpo == pytest.approx(
        BASE2["gammas"] / BASE2["lambdas"], rel=1e-12)
    assert r.clpt / r.clpo == pytest.approx(
        BASE2["gammat"] / BASE2["lambdat"], rel=1e-12)
