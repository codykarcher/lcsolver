"""Cross-flow heat exchanger sizing -- against TASOPT.jl."""
import csv
import math
import os

import pytest

from tasopt_py.engine_v3.hx_size import (HXGas, HXTubular, hx_size,
                                         hx_operate, hx_optimize,
                                         tube_geometry, gas_tset_single)
from tasopt_py.engine_v3.heat_exchanger import (NTU_from_effectiveness,
                                                effectiveness_from_NTU,
                                                max_effectiveness)
from tasopt_py.gas.transport import gas_Pr

REF = os.path.join(os.path.dirname(__file__), "..", "julia_ref")


def _size_case():
    g = HXGas(mdot_p=1144 / 60, mdot_c=9.95 / 60, eps=0.8, Tp_in=778,
              Tc_in=264, Mp_in=0.19, Mc_in=0.0285, pp_in=40e3, pc_in=1515e3)
    x = HXTubular(is_concentric=True, D_i=0.564, l=0.6084530646014857,
                  n_stages=4, xt_D=6, xl_D=1.25, Rfp=0.01 * 0.1761,
                  Rfc=8.815e-05, material="SS-304", Dpdes=3e6)
    return g, x


def test_hxsize_matches_tasopt_jl():
    g, x = _size_case()
    hx_size(g, x)
    ref = [731.5888605437423, 665.8848846504773, 1414.022586064259,
           62.03322510460286, 8.040614281031441, 0.004760508726403918,
           1.0189779296746375]
    got = [g.Tp_out, g.Tc_out, g.Dp_p, x.N_t, x.n_passes, x.tD_o, x.A_cs]
    for a, b in zip(got, ref):
        assert a == pytest.approx(b, rel=1e-14)


def test_hxoper_matches_tasopt_jl():
    g = HXGas(mdot_p=2 * 1144 / 60, mdot_c=2 * 9.95 / 60, eps=0.8, Tp_in=778,
              Tc_in=264, Mp_in=0.19, Mc_in=0.0285, pp_in=3 * 40e3,
              pc_in=3 * 1515e3)
    x = HXTubular(is_concentric=True, D_i=0.564, t=0.03e-2,
                  tD_o=0.004760326082769499, A_cs=1.0189779296746375,
                  l=0.6084530646014857, n_stages=4, n_passes=8, N_t=62,
                  xt_D=6, xl_D=1.25, Rfp=0.01 * 0.1761, Rfc=8.815e-05,
                  material="SS-304")
    hx_operate(g, x)
    for a, b in zip([g.Tp_out, g.Tc_out, g.Dp_p, g.eps],
                    [740.3018611354306, 591.2092288055287,
                     1707.3839192112114, 0.6504167938267722]):
        assert a == pytest.approx(b, rel=1e-14)


def test_gas_Pr_matches_tasopt_jl():
    worst = 0.0
    with open(os.path.join(REF, "hx_gaspr.csv")) as f:
        for r in csv.DictReader(f):
            got = gas_Pr(r["gas"], float(r["T"]))
            for i, k in enumerate(["R", "Pr", "gamma", "cp", "mu", "k"]):
                ref = float(r[k])
                worst = max(worst, abs(got[i] - ref) / abs(ref))
    assert worst < 1e-14


def test_gas_tset_single_matches_tasopt_jl():
    with open(os.path.join(REF, "hx_tset_single.csv")) as f:
        for r in csv.DictReader(f):
            got = gas_tset_single(int(float(r["igas"])), float(r["hspec"]),
                                  float(r["tguess"]))
            assert got == pytest.approx(float(r["T"]), rel=1e-13)


def test_the_objective_is_exact_at_the_references_own_optimum():
    """The optimiser may stop elsewhere; the model underneath must not."""
    g, x = _size_case()
    g.Mc_in, x.n_stages, x.xt_D = 0.017556551771520313, 16.023533665889893, 6.0
    x.l, x.xl_D, x.maxL = 0.353511407307948, 1, 0.5
    hx_size(g, x)
    for got, ref in [(g.Pl_p, 70015.02558557362), (g.Pl_c, 2776.7165269377065),
                     (x.n_passes, 3.2818928209900675),
                     (x.N_t, 116.66475426220994),
                     (g.Dp_p, 676.9188690545938),
                     (g.Dp_c, 13226.950572540522)]:
        assert got == pytest.approx(ref, rel=1e-14)


def test_the_optimum_is_nearly_flat_in_the_stage_count():
    """Why the two optimisers disagree on n_stages and barely on cost.

    Sweeping the stage count from 14 to 20 with everything else at the
    reference optimum moves the objective by under 0.2%. The reference's
    COBYLA stops at 16.0 and this port's at 19.3 -- both on the same flat
    floor, which is why the objective agrees to 0.1% while the design
    vector does not.
    """
    vals = []
    for ns in (14.0, 16.023533665889893, 18.0, 20.0):
        g, x = _size_case()
        g.Mc_in, x.n_stages, x.xt_D = 0.017556551771520313, ns, 6.0
        x.l, x.xl_D, x.maxL = 0.353511407307948, 1, 0.5
        hx_size(g, x)
        vals.append(g.Pl_p + g.Pl_c)
    assert (max(vals) - min(vals)) / min(vals) < 2e-3
    # ... and it is monotone down over that range, so 16.0 is not the floor.
    assert vals == sorted(vals, reverse=True)


def test_hxoptim_reaches_the_references_objective():
    g, x = _size_case()
    g.Mc_in = 0.0
    x.xl_D, x.maxL = 1, 0.5
    from tasopt_py.gas.mixture import gassum
    sp = gassum(g.alpha_p, 5, g.Tp_in)
    gam = sp.cp / (sp.cp - sp.r)
    rho = g.pp_in / (sp.r * g.Tp_in)
    V = g.Mp_in * math.sqrt(gam * sp.r * g.Tp_in)
    A = g.mdot_p / (rho * V)
    D_o = math.sqrt(4 * (A + math.pi * 0.564 ** 2 / 4) / math.pi)
    hx_optimize(g, x, [3.0, 4.0, 4.0, 1.1 * (D_o - 0.564) / 2])
    hx_size(g, x)
    I = g.Pl_p + g.Pl_c
    # Not below by accident: the port's stopping point is on the flat floor
    # the reference stops short of, so it must be <= within tolerance.
    assert I == pytest.approx(72791.74211251132, rel=2e-3)
    assert I <= 72791.74211251132


def test_the_constraints_are_respected_at_the_optimum():
    g, x = _size_case()
    g.Mc_in = 0.0
    x.xl_D, x.maxL = 1, 0.5
    hx_optimize(g, x, [3.0, 4.0, 4.0, 0.35])
    hx_size(g, x)
    assert 1.0 <= x.n_passes <= 20.0
    assert 1.0 <= x.N_t <= 200.0
    assert g.Dp_p <= 0.5 * g.pp_in
    assert g.Dp_c <= 0.5 * g.pc_in
    assert x.n_passes * x.n_stages * x.xl_D * x.tD_o <= x.maxL * 1.001


def test_effectiveness_round_trips_through_NTU():
    for C_c, C_p in ((500.0, 1000.0), (1000.0, 500.0), (800.0, 800.0)):
        for eps in (0.1, 0.3, 0.6):
            NTU, used = NTU_from_effectiveness(eps, C_c, C_p)
            assert effectiveness_from_NTU(NTU, C_c, C_p) == pytest.approx(
                used, rel=1e-12)


def test_asking_for_too_much_effectiveness_is_silently_clipped():
    """A cross-flow exchanger with one stream mixed cannot reach eps = 1.

    The reference substitutes 99% of the achievable maximum without saying
    so, which is why hx_size writes the value it used back onto the gas.
    """
    C_c, C_p = 500.0, 1000.0
    eps_max = max_effectiveness(0.5, True)
    assert eps_max == pytest.approx(0.7869386805747332, rel=1e-12)
    _, used = NTU_from_effectiveness(0.95, C_c, C_p)
    assert used == pytest.approx(0.99 * eps_max)
    assert used < 0.95           # you did not get what you asked for


def test_which_stream_is_mixed_changes_the_ceiling():
    """The two ceilings coincide only at C_r = 1."""
    assert max_effectiveness(1.0, True) == pytest.approx(
        max_effectiveness(1.0, False), rel=1e-12)
    assert max_effectiveness(0.2, True) == pytest.approx(0.9063462346100907,
                                                         rel=1e-12)
    assert max_effectiveness(0.2, False) == pytest.approx(0.9932620530009145,
                                                          rel=1e-12)


def test_the_tube_wall_has_a_floor_and_a_pole():
    K = 1.0e4
    # Low pressure: the 30 BWG minimum binds.
    t, tD_o = tube_geometry(K, 1.0e4, 215e6)
    assert t == 3.0e-4
    # High pressure: hoop stress binds and the wall thickens.
    t2, _ = tube_geometry(K, 1.0e8, 215e6)
    assert t2 > t
    # At the pole the reference divides by a vanishing denominator.
    with pytest.raises(ValueError, match="pole at C = 0.5"):
        tube_geometry(K, 215e6, 215e6)


def test_a_bigger_exchanger_transfers_more_heat():
    """Doubling the passes must raise the effectiveness -- and not past
    the ceiling."""
    def run(n_passes):
        g = HXGas(mdot_p=2 * 1144 / 60, mdot_c=2 * 9.95 / 60, Tp_in=778,
                  Tc_in=264, Mp_in=0.19, Mc_in=0.0285, pp_in=3 * 40e3,
                  pc_in=3 * 1515e3)
        x = HXTubular(is_concentric=True, D_i=0.564, t=0.03e-2,
                      tD_o=0.004760326082769499, A_cs=1.0189779296746375,
                      l=0.6084530646014857, n_stages=4, n_passes=n_passes,
                      N_t=62, xt_D=6, xl_D=1.25, Rfp=0.01 * 0.1761,
                      Rfc=8.815e-05, material="SS-304")
        hx_operate(g, x)
        return g.eps
    e8, e16 = run(8), run(16)
    assert e16 > e8
    assert e16 < 1.0
