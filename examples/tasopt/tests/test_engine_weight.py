"""Engine and nacelle weight (tfweight.f) vs the compiled Fortran.

Models 0 (Drela), 1 and 2 (Fitzgerald basic/advanced), each geared and
ungeared, over three engine sizes.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_tfweight \\
        drv_tfweight.f tfweight.f gppre.f
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.engine.weight import (UnsupportedWeightModel, tfweight)

DATA = Path(__file__).parent / "data"
RTOL = 1e-13


def _inputs(i, geared):
    return dict(Gearf=3.0 if geared else 1.0, OPR=30.0 + 6.0 * i,
                BPR=5.0 + 2.0 * i, mdotc=40.0 + 6.0 * i, dfan=1.6 + 0.2 * i,
                rSnace=14.0, dlcomp=0.9 + 0.1 * i, neng=2.0, feadd=0.10,
                fpylon=0.10)


def test_matches_fortran():
    n = 0
    with (DATA / "tfweight_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0] == "name":
                continue
            e, k, i = (int(row[0][1:3]), int(row[0][3:5]), int(row[0][5:7]))
            g = tfweight(e, **_inputs(i, geared=(k == 2)))
            for a, b in zip((g.Weng, g.Wnac, g.Webare, g.Snace1),
                            map(float, row[1:5])):
                assert a == pytest.approx(b, rel=RTOL), row[0]
                n += 1
    assert n == 72


@pytest.mark.parametrize("iengwgt", [3, 4])
def test_surrogate_models_are_refused_not_faked(iengwgt):
    with pytest.raises(UnsupportedWeightModel, match="crddc"):
        tfweight(iengwgt, **_inputs(1, geared=False))


def test_gear_ratio_of_one_counts_as_direct_drive():
    """The switch is |Gearf - 1| < 0.001, not Gearf != 1."""
    a = tfweight(1, **dict(_inputs(1, geared=False), Gearf=1.0))
    b = tfweight(1, **dict(_inputs(1, geared=False), Gearf=1.0005))
    assert b.Webare == pytest.approx(a.Webare, rel=1e-15)
    c = tfweight(1, **dict(_inputs(1, geared=False), Gearf=1.01))
    assert c.Webare != pytest.approx(a.Webare, rel=1e-6)


def test_geared_fan_is_a_different_model_not_a_correction():
    """The geared coefficients are a separate fit, so the ratio with the
    ungeared model is not constant across bypass ratios."""
    ratios = []
    for i in (1, 3):
        u = tfweight(1, **_inputs(i, geared=False))
        g = tfweight(1, **_inputs(i, geared=True))
        ratios.append(g.Webare / u.Webare)
    assert ratios[0] != pytest.approx(ratios[1], rel=1e-3)


def test_weight_breakdown_sums():
    kw = _inputs(2, geared=False)
    g = tfweight(1, **kw)
    Weadd = g.Webare * kw["feadd"]
    Wpylon = (g.Webare + Weadd + g.Wnac) * kw["fpylon"]
    assert g.Weng == pytest.approx(g.Webare + Weadd + g.Wnac + Wpylon,
                                   rel=1e-14)


def test_pylon_fraction_compounds_with_accessories():
    """fpylon is taken on the accessories too, so raising feadd raises the
    pylon weight as well -- the two are not independent."""
    a = tfweight(1, **dict(_inputs(2, geared=False), feadd=0.10))
    b = tfweight(1, **dict(_inputs(2, geared=False), feadd=0.20))
    dWebare = 0.0
    assert a.Webare == pytest.approx(b.Webare, rel=1e-15)   # bare is unchanged
    # The total rises by more than the accessory increment alone.
    assert b.Weng - a.Weng > 0.10 * a.Webare


def test_bare_weight_scales_with_core_flow():
    """Drela's model is exactly linear in mdotc; Fitzgerald's is a power law."""
    a = tfweight(0, **dict(_inputs(2, geared=False), mdotc=40.0))
    b = tfweight(0, **dict(_inputs(2, geared=False), mdotc=80.0))
    assert b.Webare / a.Webare == pytest.approx(2.0, rel=1e-14)

    c = tfweight(1, **dict(_inputs(2, geared=False), mdotc=40.0))
    d = tfweight(1, **dict(_inputs(2, geared=False), mdotc=80.0))
    assert d.Webare / c.Webare != pytest.approx(2.0, rel=1e-3)


def test_advanced_technology_is_lighter():
    for geared in (False, True):
        basic = tfweight(1, **_inputs(2, geared=geared))
        adv = tfweight(2, **_inputs(2, geared=geared))
        assert adv.Webare < basic.Webare


def test_nacelle_area_follows_the_fan_disc():
    kw = _inputs(2, geared=False)
    g = tfweight(1, **kw)
    import math
    assert g.Snace1 == pytest.approx(
        kw["rSnace"] * 0.25 * math.pi * kw["dfan"] ** 2, rel=1e-15)
