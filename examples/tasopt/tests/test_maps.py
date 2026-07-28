"""Compressor/fan efficiency map (ecmap, tfmap.f) against the compiled Fortran.

Five cases use the *historical* fan constants, which have nonzero CK/DK and so
exercise the full map expression; the sixth uses the set TASOPT actually
ships, where CK = DK = 0 and the map degenerates to a straight line.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_ecmap drv_ecmap.f tfmap.f
    ./drv_ecmap > tests/data/ecmap_ref.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.engine.maps import CMAPF, CMAPF_PENALISED, ecmap

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

BASE = dict(piD=1.60, mbD=200.0, effo=0.90, piK=1.60, effK=0.0)
ACTIVE_C6 = (3.50, 0.80, 0.03, 0.95, -0.50, 3.0, 6.0, 0.0, 0.0)
CASES = {
    1: (CMAPF_PENALISED, dict(pi=1.60, mb=200.0)),
    2: (CMAPF_PENALISED, dict(pi=1.60, mb=170.0)),
    3: (CMAPF_PENALISED, dict(pi=1.60, mb=230.0)),
    4: (CMAPF_PENALISED, dict(pi=1.45, mb=200.0)),
    5: (CMAPF_PENALISED, dict(pi=1.75, mb=185.0, effK=-0.02)),
    6: (ACTIVE_C6, dict(pi=1.72, mb=210.0, effK=-0.015)),
}
FIELDS = {"eff": "eff", "dpi": "eff_pi", "dmb": "eff_mb"}


def _solved():
    out = {}
    for ic, (cmap, kw) in CASES.items():
        args = {**BASE, **kw}
        out[ic] = ecmap(args.pop("pi"), args.pop("mb"), args["piD"],
                        args["mbD"], cmap, args["effo"], args["piK"],
                        args["effK"])
    return out


def test_matches_fortran():
    got = _solved()
    n = 0
    with (DATA / "ecmap_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            actual = getattr(got[ic], FIELDS[name])
            rel = abs(actual - ref) / max(abs(ref), 1e-300)
            assert rel <= RTOL, f"case {ic} {name}: {actual!r} vs {ref!r} rel {rel:.3e}"
            n += 1
    assert n == 18


def test_shipped_constants_make_the_map_a_straight_line():
    """With CK = DK = 0, efficiency is effo + effK*(pi - piK) and nothing else.

    This is the configuration TASOPT 2.16 actually ships. Mass flow drops out
    entirely, which is easy to miss without opening tfmap.inc.
    """
    for mb in (120.0, 200.0, 320.0):
        r = ecmap(1.72, mb, 1.60, 200.0, CMAPF, 0.90, 1.60, -0.015)
        assert r.eff == pytest.approx(0.90 - 0.015 * (1.72 - 1.60), rel=1e-12)
        assert r.eff_mb == pytest.approx(0.0, abs=1e-12)


def test_penalised_map_peaks_near_the_design_point():
    """With the historical constants, moving off design costs efficiency."""
    at_design = ecmap(1.60, 200.0, 1.60, 200.0, CMAPF_PENALISED,
                      0.90, 1.60, 0.0).eff
    for mb in (170.0, 230.0):
        off = ecmap(1.60, mb, 1.60, 200.0, CMAPF_PENALISED,
                    0.90, 1.60, 0.0).eff
        assert off < at_design


def test_derivatives_match_finite_differences():
    cmap, h = CMAPF_PENALISED, 1e-6
    pi, mb = 1.58, 195.0
    r = ecmap(pi, mb, 1.60, 200.0, cmap, 0.90, 1.60, -0.01)
    d_pi = (ecmap(pi + h, mb, 1.60, 200.0, cmap, 0.90, 1.60, -0.01).eff
            - ecmap(pi - h, mb, 1.60, 200.0, cmap, 0.90, 1.60, -0.01).eff) / (2 * h)
    d_mb = (ecmap(pi, mb + h, 1.60, 200.0, cmap, 0.90, 1.60, -0.01).eff
            - ecmap(pi, mb - h, 1.60, 200.0, cmap, 0.90, 1.60, -0.01).eff) / (2 * h)
    assert r.eff_pi == pytest.approx(d_pi, rel=1e-5)
    assert r.eff_mb == pytest.approx(d_mb, rel=1e-5)
