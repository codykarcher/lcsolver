"""Axisymmetric BL and wake (blax.f) vs the compiled Fortran.

Five 737-like fuselages -- smooth and with an excrescence factor, at M = 0.8
and M = 0.2, with a point tail and an edge tail, thin and thick -- each solved
over the 47-station body-plus-wake grid that ``axisol`` produces. 2350 numbers.

The driver dumps its *inputs* (arc length, perimeter, dr/dn, inviscid
velocity) alongside its outputs, and this test feeds those back in, so what is
measured is ``blax`` alone rather than ``blax`` composed with ``axisol``. That
matters. Driving it from the Python ``axisol`` instead, whose velocities agree
to 4.4e-16, moves the answer by up to 1.2e-12 -- an amplification of about
three thousand, worth knowing before reading anything into a small difference
downstream. It is not uniform: the state variables ``ue`` and ``th`` still
agree to 3e-15, and it is the derived ``ct`` (a nonlinear function of ``H*``
and ``Hk``, and not read by anything) that moves most.
``tests/test_fusebl.py`` exercises the composed chain.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_blax \\
        drv_blax.f blax.f blsys.f axisol.f gaussn.f
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.aero.blax import HKSEP, blax

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

#: Reynolds number per unit length, Mach, excrescence factor -- see drv_blax.f.
CASES = {1: (1.0e7, 0.80, 1.00),
         2: (1.0e7, 0.80, 1.03),
         3: (3.0e6, 0.20, 1.00),
         4: (1.0e7, 0.80, 1.00),
         5: (4.0e7, 0.80, 1.00)}

FIELDS = ["ue", "ds", "th", "ts", "dc", "cf", "cd", "ct", "hk", "ph"]


def _reference():
    """``{case: (nl, ilte, inputs, outputs)}`` straight out of the CSV."""
    meta, rows = {}, {}
    with (DATA / "blax_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, i = int(row[0]), int(row[1])
            if i == 0:
                meta[ic] = (int(float(row[2])), int(float(row[3])))
            else:
                rows.setdefault(ic, []).append([float(v) for v in row[2:16]])
    return {ic: (meta[ic][0], meta[ic][1],
                 [r[:4] for r in rows[ic]], [r[4:] for r in rows[ic]])
            for ic in rows}


REF = _reference()


def _run(ic):
    nl, ilte, inp, _ = REF[ic]
    Reyn, Mach, fexcr = CASES[ic]
    cols = list(zip(*inp))                      # s, b, rn, uinv
    return blax(nl, ilte, cols[0], cols[1], cols[2], cols[3],
                Reyn, Mach, fexcr)


RESULTS = {ic: _run(ic) for ic in CASES}


def test_matches_fortran():
    n = 0
    for ic, (nl, _ilte, _inp, out) in REF.items():
        r = RESULTS[ic]
        assert len(r.ue) == nl
        for i, want in enumerate(out):
            for f, w in zip(FIELDS, want):
                assert getattr(r, f)[i] == pytest.approx(w, rel=RTOL,
                                                         abs=1e-300), \
                    f"case {ic} i={i + 1} {f}"
                n += 1
    assert n == 2350


def test_first_station_is_left_alone():
    """Station 1 sits at x = 0, is frozen through both Newtons, and several of
    its outputs are never assigned at all -- blax reads cd(1) without ever
    writing it, and relies on the caller's array being zeroed."""
    r = RESULTS[1]
    assert r.th[0] == 0.0 and r.ds[0] == 0.0 and r.ph[0] == 0.0
    assert r.cd[0] == 0.0 and r.cf[0] == 0.0 and r.hk[0] == 0.0


def test_the_wake_carries_no_wall_friction():
    nl, ilte, _inp, _out = REF[1]
    r = RESULTS[1]
    for i in range(ilte, nl):          # 0-based: strictly past the TE point
        assert r.cf[i] == 0.0
    assert r.cf[ilte - 2] > 0.0        # ...but the last body station does


def test_boundary_layer_grows_downstream_and_the_wake_relaxes():
    r = RESULTS[1]
    nl, ilte, inp, _out = REF[1]
    # Not monotone: theta dips once, at the tail blend, where the flow
    # accelerates hard enough (ue 1.04 -> 1.12) to thin the layer. Everywhere
    # else it grows, and it grows overall.
    drops = [i for i in range(2, ilte) if r.th[i] < r.th[i - 1]]
    assert len(drops) == 1
    assert inp[drops[0]][0] > 30.0                # aft of the tail blend
    assert r.th[ilte - 1] > 100.0 * r.th[1]
    # In the wake, H falls back towards the freestream value.
    assert r.hk[nl - 1] < r.hk[ilte - 1]


def test_dissipation_integral_is_monotone():
    for ic in CASES:
        ph = RESULTS[ic].ph
        assert all(b >= a for a, b in zip(ph[1:], ph[2:]))
        assert ph[-1] > 0.0


def test_excrescence_factor_raises_friction_and_dissipation():
    """Cases 1 and 2 differ only in fexcr."""
    smooth, rough = RESULTS[1], RESULTS[2]
    _nl, ilte, _inp, _out = REF[1]
    assert rough.cf[ilte - 2] > smooth.cf[ilte - 2]
    assert rough.ph[-1] > smooth.ph[-1]


def test_higher_reynolds_number_thins_the_layer():
    """Case 3 is M = 0.2 at Re/l = 3e6; case 1 is M = 0.8 at 1e7."""
    assert RESULTS[1].th[-1] < RESULTS[3].th[-1]


def test_shape_parameter_stays_below_the_separation_fudge_on_these_bodies():
    """The direct march switches to inverse mode above Hk = 2.9. On a
    fuselage that never happens on the body, so the fudge is not what these
    answers rest on."""
    for ic in CASES:
        _nl, ilte, _inp, _out = REF[ic]
        assert max(RESULTS[ic].hk[1:ilte]) < HKSEP
