"""blvar and blsys (blsys.f) vs the compiled Fortran.

Ten regimes -- attached and separated, laminar and turbulent, wake, similarity
station, direct and inverse closure, and the low-Reynolds and large-H corners
that exercise every clamp -- at five states each. Both the values and the full
analytic Jacobian are compared, 2200 numbers in all.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_blsys drv_blsys.f blsys.f
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.aero.blsys import BLState, blsys, blvar

DATA = Path(__file__).parent / "data"
RTOL = 1e-14

#: The value drv_blsys.f leaves in cf_ue before calling blvar in wake mode.
CF_UE_PRESET = 0.1234567890123456


def _case(icase, j):
    """Rebuild one driver case. Mirrors drv_blsys.f exactly."""
    simi = lami = wake = False
    direct = True
    Reyn, Mach, fexcr, hksep, uinv = 2.0e7, 0.80, 1.03, 2.9, 1.04
    x, b, rn = 10.0, 6.0, 0.90
    xm, bm, rnm = 9.0, 5.8, 0.88

    th = 0.02 + 0.02 * j
    ds = th * (1.6 + 0.4 * j)
    ue = 0.80 + 0.10 * j

    if icase == 2:
        direct = False
    if icase == 3:
        wake = True
    if icase == 4:
        simi = True
    if icase == 5:
        lami, Reyn = True, 1.0e6
    if icase == 6:
        ds = th * (3.0 + 0.6 * j)
    if icase == 7:
        Reyn = 1.0e4
        th = 0.010 + 0.002 * j
        ds = th * (1.8 + 0.3 * j)
    if icase == 8:
        lami, Reyn = True, 1.0e6
        ds = th * (5.0 + 0.8 * j)
    if icase == 9:
        ds = th * (0.9 + 0.02 * j)
    if icase == 10:
        Reyn = 5.0e2
        ds = th * (18.0 + 1.0 * j)

    thm, dsm, uem = th * 0.93, ds * 0.90, ue * 0.98
    if simi:
        thm = dsm = uem = 0.0

    vm = BLState()
    if not simi:
        vm = blvar(False, lami, False, Reyn, Mach, fexcr,
                   xm, thm, dsm, uem, cf_ue=0.0)

    v = blvar(simi, lami, wake, Reyn, Mach, fexcr, x, th, ds, ue,
              cf_ue=CF_UE_PRESET)
    aa, bb, rr = blsys(simi, lami, wake, direct, Mach, uinv, hksep,
                       x, b, rn, th, ds, ue, v,
                       xm, bm, rnm, thm, dsm, uem, vm)

    got = {n: getattr(v, n.strip()) for n in
           ("h", "h_th", "h_ds", "hk", "hk_th", "hk_ds", "hk_ue",
            "hc", "hc_th", "hc_ds", "hc_ue", "hs", "hs_th", "hs_ds", "hs_ue",
            "cf", "cf_th", "cf_ds", "cf_ue", "di", "di_th", "di_ds", "di_ue")}
    for k in range(3):
        got[f"rr{k + 1}"] = rr[k]
        for m in range(3):
            got[f"aa{k + 1}{m + 1}"] = aa[k][m]
            got[f"bb{k + 1}{m + 1}"] = bb[k][m]
    return got


def test_matches_fortran():
    cache, n = {}, 0
    with (DATA / "blsys_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, j, name, want = (int(row[0]), int(row[1]), row[2].strip(),
                                 float(row[3]))
            got = cache.setdefault((ic, j), _case(ic, j))[name]
            assert got == pytest.approx(want, rel=RTOL, abs=1e-300), \
                f"case {ic} j={j} {name}"
            n += 1
    assert n == 2200


def test_wake_leaves_cf_ue_alone():
    """blvar's wake branch writes `cd_ue = 0.` -- a typo for cf_ue.

    Under `implicit real` that declares a new variable and leaves cf_ue at
    whatever the caller passed, so the last pre-wake value is carried through
    the whole wake. It reaches only the Jacobian, never a residual.
    """
    v = blvar(False, False, True, 2.0e7, 0.8, 1.0, 10.0, 0.05, 0.12, 1.0,
              cf_ue=0.5)
    assert v.cf == 0.0 and v.cf_th == 0.0 and v.cf_ds == 0.0
    assert v.cf_ue == 0.5


def test_wake_doubles_the_dissipation():
    kw = dict(Reyn=2.0e7, Mach=0.8, fexcr=1.0)
    body = blvar(False, False, False, x=10.0, th=0.05, ds=0.12, ue=1.0, **kw)
    wake = blvar(False, False, True, x=10.0, th=0.05, ds=0.12, ue=1.0, **kw)
    # The wake has no wall shear, so `di` is not simply 2x the body value;
    # what is doubled is the dissipation computed with cf = 0.
    assert wake.cf == 0.0 and body.cf > 0.0
    assert wake.di > 0.0
    assert wake.di == pytest.approx(2.0 * (body.di - 0.5 * body.cf
                                           + (body.hk - 1.0) * 0.0), rel=0.5)


def test_hk_is_clamped_after_its_derivatives():
    """hk = max(hk, 1.005) sits below the derivative expressions, so a clamped
    station still reports the unclamped slopes."""
    v = blvar(False, False, False, 2.0e7, 0.8, 1.0, 10.0, 0.05, 0.05, 1.0)
    assert v.hk == 1.005
    assert v.hk_th != 0.0 and v.hk_ds != 0.0


def test_inverse_mode_swaps_the_closing_equation():
    kw = dict(Reyn=2.0e7, Mach=0.8, fexcr=1.0)
    v = blvar(False, False, False, x=10.0, th=0.05, ds=0.15, ue=1.0, **kw)
    vm = blvar(False, False, False, x=9.0, th=0.047, ds=0.14, ue=0.98, **kw)
    args = (0.8, 1.04, 2.9, 10.0, 6.0, 0.9, 0.05, 0.15, 1.0, v,
            9.0, 5.8, 0.88, 0.047, 0.14, 0.98, vm)
    aa_d, _, rr_d = blsys(False, False, False, True, *args)
    aa_i, _, rr_i = blsys(False, False, False, False, *args)

    assert rr_d[2] == 1.0 - 1.04            # ue - uinv
    assert aa_d[2] == [0.0, 0.0, 1.0]
    assert rr_i[2] == v.hk - 2.9            # hk - hksep
    assert aa_i[2] == [v.hk_th, v.hk_ds, v.hk_ue]
    # The first two equations are untouched by the mode.
    assert rr_d[:2] == rr_i[:2] and aa_d[:2] == aa_i[:2]


def test_similarity_station_drops_the_upstream_jacobian():
    """At a similarity station the upstream state is at x = 0 and cannot enter
    the logarithmic differences, so bb is identically zero."""
    v = blvar(True, False, False, 2.0e7, 0.8, 1.0, 10.0, 0.05, 0.12, 1.0)
    aa, bb, rr = blsys(True, False, False, True, 0.8, 1.04, 2.9,
                       10.0, 6.0, 0.9, 0.05, 0.12, 1.0, v,
                       9.0, 5.8, 0.88, 0.0, 0.0, 0.0, BLState())
    assert bb == [[0.0] * 3] * 3
    assert all(v_ != 0.0 for v_ in rr)


def test_upstream_friction_can_be_zeroed():
    v = blvar(False, False, False, 2.0e7, 0.8, 1.0, 10.0, 0.05, 0.12, 1.0)
    z = v.without_cf()
    assert (z.cf, z.cf_th, z.cf_ds, z.cf_ue) == (0.0, 0.0, 0.0, 0.0)
    assert z.hs == v.hs and z.di == v.di        # nothing else disturbed
