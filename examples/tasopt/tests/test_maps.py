"""Tabulated compressor maps -- against TASOPT.jl."""
import csv
import math
import os

import pytest

from tasopt_py.engine_v3.maps import (MAP_BY_NAME, FAN_MAP, HPC_MAP,
                                      compressor_speed_and_efficiency,
                                      find_NR_inverse, bilinear)

REF = os.path.join(os.path.dirname(__file__), "..", "julia_ref")
NAMES = ["Nb", "epol", "dNb_dpi", "dNb_dmb", "depol_dpi", "depol_dmb",
         "N", "R"]


def _on_knot(m, N, R):
    return (any(abs(N - v) < 1e-12 for v in m.mN)
            or any(abs(R - v) < 1e-12 for v in m.mR))


def test_the_interpolated_values_are_exact():
    with open(os.path.join(REF, "maps_interp.csv")) as f:
        for r in csv.DictReader(f):
            m = MAP_BY_NAME[r["map"]]
            N, R = float(r["N"]), float(r["R"])
            assert m.Wc_at(N, R) == pytest.approx(float(r["Wc"]), rel=1e-15,
                                                  abs=1e-15)
            assert m.PR_at(N, R) == pytest.approx(float(r["PR"]), rel=1e-15,
                                                  abs=1e-15)
            assert m.polyeff_at(N, R) == pytest.approx(float(r["polyeff"]),
                                                       rel=1e-15, abs=1e-15)


def test_the_gradients_match_including_the_left_cell_convention():
    """A linear interpolant has no derivative on a grid line.

    Interpolations.jl resolves it by taking the cell to the *left*. Get this
    wrong and only the knots disagree -- but the design point of every map
    is a knot, so it is exactly the point that matters most.
    """
    worst = 0.0
    with open(os.path.join(REF, "maps_interp.csv")) as f:
        for r in csv.DictReader(f):
            m = MAP_BY_NAME[r["map"]]
            N, R = float(r["N"]), float(r["R"])
            for grad, keys in ((m.grad_Wc(N, R), ("dWc_dN", "dWc_dR")),
                               (m.grad_PR(N, R), ("dPR_dN", "dPR_dR")),
                               (m.grad_polyeff(N, R), ("dpe_dN", "dpe_dR"))):
                for got, k in zip(grad, keys):
                    ref = float(r[k])
                    worst = max(worst, abs(got - ref) / max(abs(ref), 1e-12))
    assert worst < 1e-11


def test_speed_and_efficiency_matches_away_from_grid_lines():
    with open(os.path.join(REF, "maps_speed_eff.csv")) as f:
        for r in csv.DictReader(f):
            m = MAP_BY_NAME[r["map"]]
            got = compressor_speed_and_efficiency(
                m, float(r["pratio"]), float(r["mb"]), float(r["piD"]),
                float(r["mbD"]), float(r["NbD"]), float(r["epol0"]))
            if _on_knot(m, got[6], got[7]):
                continue
            for k, v in zip(NAMES, got):
                assert v == pytest.approx(float(r[k]), rel=1e-8, abs=1e-13)


def test_every_reference_operating_point_inverts():
    """17 of 48 diverge without clamping the iterates onto the padded grid."""
    with open(os.path.join(REF, "maps_speed_eff.csv")) as f:
        for r in csv.DictReader(f):
            compressor_speed_and_efficiency(
                MAP_BY_NAME[r["map"]], float(r["pratio"]), float(r["mb"]),
                float(r["piD"]), float(r["mbD"]), float(r["NbD"]),
                float(r["epol0"]))


def test_the_inverse_is_solved_tighter_than_the_reference():
    """The remaining disagreement is the reference's tolerance, not ours.

    NLsolve's default ftol is 1e-8, and the reference accepts it. At the HPC
    point below its residual is 8.7e-9; this port's is at machine precision,
    which is why N and R differ in the ninth digit.
    """
    m = HPC_MAP
    got = compressor_speed_and_efficiency(m, 6.0, 0.8, 10.0, 1.0, 1.0, 0.9)
    N, R = got[6], got[7]
    Wc = 0.8 * m.Wc
    PR = 1.0 + (6.0 - 1.0) * (m.PR - 1.0) / (10.0 - 1.0)
    residual = math.hypot(m.Wc_at(N, R) - Wc, m.PR_at(N, R) - PR)
    assert residual < 1e-14


def test_the_design_point_derivative_is_two_valued():
    """The sharpest consequence of a C0 map.

    Every map's design point sits exactly on a grid line, so the reported
    sensitivity depends on which side of a 1e-16 rounding error the root
    find happens to stop. This port lands just below the Fan's design speed
    and reads the cell beneath; TASOPT.jl lands just above and reads the one
    over. The value of dNb_dpi differs by a factor of 50.

    Nothing here is a porting error -- both answers are correct readings of
    a function that has two derivatives at that point. It matters because
    the engine is always sized at its design point, and these derivatives
    feed a Newton solve.
    """
    m = FAN_MAP
    got = compressor_speed_and_efficiency(m, 1.68, 1.0, 1.68, 1.0, 1.0, 0.9)
    N, R = got[6], got[7]
    assert N == pytest.approx(m.Nc, abs=1e-14)
    assert R == pytest.approx(m.Rline, abs=1e-14)

    # Read the same quantity from either side of the knot and watch it jump.
    # It is dWc/dR that moves: below the design R-line the corrected flow
    # still rises with R-line; above it the map has flattened into choke,
    # so the derivative collapses by a factor of about 58.
    eps = 1e-9
    below = m.grad_Wc(m.Nc - eps, m.Rline - eps)[1]
    above = m.grad_Wc(m.Nc + eps, m.Rline + eps)[1]
    assert below / above > 50.0

    # That propagates straight into the inverse Jacobian the cycle solver
    # uses: dN/dPR differs by a factor of 50 across a 1e-16 step.
    def dN_dpr(N, R):
        a, b = m.grad_Wc(N, R)
        c, d = m.grad_PR(N, R)
        return -b / (a * d - b * c)
    assert (dN_dpr(m.Nc - eps, m.Rline - eps)
            / dN_dpr(m.Nc + eps, m.Rline + eps)) > 50.0


def test_efficiency_is_fenced_to_zero_off_the_map():
    """The polyeff padding is zero on all four edges, unlike Wc and PR.

    So efficiency falls linearly to nothing the moment an iterate leaves the
    tabulated region -- in every direction. It is a fence for the optimiser,
    not a model of anything.
    """
    m = FAN_MAP
    assert m.polyeff_at(0.0, 2.2) == 0.0
    assert m.polyeff_at(2.0, 2.2) == 0.0
    assert m.polyeff_at(0.99, 0.0) == 0.0
    assert m.polyeff_at(0.99, 4.0) == 0.0
    # ... while pressure ratio is padded to something non-zero instead.
    assert m.PR_at(0.99, 4.0) == pytest.approx(1.0)


def test_the_pressure_ratio_scaling_fixes_the_no_work_point():
    """PR is scaled affinely, not multiplicatively.

    A compressor doing no work has a pressure ratio of one on any map, and
    the affine form preserves that as well as the design point. A plain
    ratio would not.
    """
    m = FAN_MAP
    piD = 1.68
    for pratio, expect in ((1.0, 1.0), (piD, m.PR)):
        PR = 1.0 + (pratio - 1.0) * (m.PR - 1.0) / (piD - 1.0)
        assert PR == pytest.approx(expect, rel=1e-14)


def test_the_maps_are_the_generated_tables():
    """Guards the generator: 3 maps on a 14 x 11 grid."""
    from tasopt_py.engine_v3.map_data import MAPS
    assert set(MAPS) == {"Fan", "LPC", "HPC"}
    for d in MAPS.values():
        assert len(d["NcMap"]) == 14 and len(d["RlineMap"]) == 11
        for k in ("WcMap", "PRMap", "polyeff_Map"):
            assert len(d[k]) == 14 and all(len(r) == 11 for r in d[k])


def test_bilinear_reduces_to_the_table_at_the_nodes():
    xs, ys = [0.0, 1.0, 2.0], [0.0, 2.0]
    Z = [[1.0, 2.0], [3.0, 5.0], [8.0, 13.0]]
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            assert bilinear(xs, ys, Z, x, y) == pytest.approx(Z[i][j])
    assert bilinear(xs, ys, Z, 0.5, 1.0) == pytest.approx(
        (1.0 + 2.0 + 3.0 + 5.0) / 4.0)
