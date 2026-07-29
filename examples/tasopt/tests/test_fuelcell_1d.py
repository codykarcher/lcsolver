"""One-dimensional PEM fuel cell models -- against TASOPT.jl."""
import math

import pytest

import tasopt_py.engine_v3.fuelcell_1d as fc
from tasopt_py.engine_v3.fuelcell_1d import (PEMInputs, ht_pemfc_voltage,
                                             lt_pemfc_voltage, pem_size,
                                             pem_operate, pem_stack_weight,
                                             binary_diffusion,
                                             porous_diffusion,
                                             solve_diffusion_ode,
                                             nafion_diffusion, eig2x2,
                                             power_density_1d)

COMMON = dict(j=1e4, p_A=3e5, p_C=3e5, x_H2O_A=0.1, x_H2O_C=0.1,
              lam_H2=3.0, lam_O2=3.0, t_M=100e-6, t_A=250e-6, t_C=250e-6)


def _ht():
    return PEMInputs(T=453.15, kind="HT-PEMFC", **COMMON)


def _lt():
    return PEMInputs(T=353.15, kind="LT-PEMFC", **COMMON)


def test_ht_voltage_matches_tasopt_jl():
    assert ht_pemfc_voltage(_ht()) == pytest.approx(0.7532410548758377,
                                                    rel=1e-15)


def test_lt_voltage_matches_tasopt_jl_to_the_references_ode_tolerance():
    """The reference integrates the membrane at reltol 1e-6.

    See ``DISCREPANCIES.md`` §81: its absolute tolerance is 6% of the area
    specific resistance it is integrating, so 1e-7 is as close as the two
    can be expected to agree. This port's answer is converged.
    """
    V, alpha = lt_pemfc_voltage(_lt())
    assert V == pytest.approx(0.7103339015901087, rel=2e-7)
    assert alpha == pytest.approx(0.13026807266756726, rel=2e-7)


def test_the_lt_answer_is_converged_in_the_ode_tolerance():
    """Tighten the integration by six orders and the answer does not move."""
    u = _lt()
    rtol0, atol0 = fc.ODE_RTOL, fc.ODE_ATOL
    try:
        vals = []
        for rt in (1e-8, 1e-11, 1e-13):
            fc.ODE_RTOL, fc.ODE_ATOL = rt, rt * 1e-2
            vals.append(lt_pemfc_voltage(u)[0])
    finally:
        fc.ODE_RTOL, fc.ODE_ATOL = rtol0, atol0
    assert max(vals) - min(vals) < 1e-9


def test_pem_size_matches_tasopt_jl():
    n, A, Q = pem_size(1e6, 200.0, _ht())
    assert n == pytest.approx(265.5192500533146, rel=1e-14)
    assert A == pytest.approx(0.5, rel=1e-14)
    assert Q == pytest.approx(664805.6978342824, rel=1e-14)


def test_pem_operate_matches_tasopt_jl():
    u = _ht()
    n, A, _ = pem_size(1e6, 200.0, u)
    V, Q = pem_operate(5e5, n, A, u)
    assert V == pytest.approx(237.6830551160271, rel=1e-13)
    assert Q == pytest.approx(200430.95710865577, rel=1e-13)


def test_stack_weight_matches_tasopt_jl():
    u = _ht()
    n, A, _ = pem_size(1e6, 200.0, u)
    assert pem_stack_weight(9.81, u, n, A, 4.0) == pytest.approx(
        7032.808376162144, rel=1e-14)


def test_the_cell_area_does_not_depend_on_the_voltage_model():
    """A_cell = P_des / (V_des * j), with V_cell cancelling exactly.

    So the two 1-D models, which disagree on voltage by 6%, size an
    identical cell area. Only the cell count and the heat load differ.
    """
    for u in (_ht(), _lt()):
        _, A, _ = pem_size(1e6, 200.0, u)
        assert A == pytest.approx(1e6 / (200.0 * u.j), rel=1e-13)


def test_only_the_low_temperature_stack_rejects_more_heat_than_power():
    """Which cell you pick decides whether cooling or power dominates.

    HT-PEMFC runs at 0.753 V against a 1.254 V thermoneutral: 60% efficient,
    rejecting 0.66 W of heat per watt delivered. LT-PEMFC runs at 0.710 V
    against 1.482 V: 48% efficient, rejecting 1.09 W per watt -- more heat
    than electricity. The heat exchanger that follows is sized by this
    number, so the choice of membrane sets the cooling problem.
    """
    _, _, Q_ht = pem_size(1e6, 200.0, _ht())
    _, _, Q_lt = pem_size(1e6, 200.0, _lt())
    assert Q_ht / 1e6 == pytest.approx(0.6648, rel=1e-3)
    assert Q_lt / 1e6 == pytest.approx(1.0863, rel=1e-3)
    assert Q_ht < 1e6 < Q_lt


def test_power_density_does_not_corrupt_the_operating_point():
    """The reference's P2Acalc writes j onto its input struct."""
    u = _ht()
    power_density_1d(u, 5.0e3)
    assert u.j == 1e4


def test_binary_diffusion_switches_correlation_for_water():
    """Water is polar and gets different leading constants.

    A pair containing water is on a different curve entirely, not a
    perturbation of the same one -- so the ratio of the two correlations is
    not close to one.
    """
    T, p = 353.15, 3e5
    d_dry = binary_diffusion(T, p, ["O2", "N2"])
    d_wet = binary_diffusion(T, p, ["O2", "H2O"])
    assert d_wet > d_dry
    with pytest.raises(ValueError, match="no critical properties"):
        binary_diffusion(T, p, ["O2", "Xe"])


def test_porous_diffusion_here_is_not_the_one_in_the_simple_model():
    """``eps**tau`` here; ``eps/tau`` in the simple polarisation module.

    Same name, same argument list, different formula -- and at the shipped
    defaults they land within 5% of each other (0.2530 against 0.2667),
    which is exactly what makes the inconsistency easy to miss. Move away
    from eps = 0.4, tau = 1.5 and they diverge fast.
    """
    from tasopt_py.engine_v3.fuelcell import porous_diffusion as simple
    D = 1.0e-5
    assert porous_diffusion(D, 0.4, 1.5) == pytest.approx(
        0.4 ** 1.5 * D, rel=1e-14)
    assert simple(D, 0.4, 1.5) == pytest.approx(0.4 / 1.5 * D, rel=1e-14)
    # Near-agreement at the defaults ...
    assert simple(D, 0.4, 1.5) / porous_diffusion(D, 0.4, 1.5) == (
        pytest.approx(1.054, rel=1e-3))
    # ... and nothing like it away from them: at eps = 0.8, tau = 4 the
    # power law gives 0.410 and the ratio gives 0.200, a factor of two the
    # other way round.
    assert simple(D, 0.8, 4.0) / porous_diffusion(D, 0.8, 4.0) == (
        pytest.approx(0.488, rel=1e-2))


def test_the_electrode_ode_is_solved_exactly():
    """Eigendecomposition, not quadrature -- so it matches its own
    closed-form solution to machine precision on a diagonal system."""
    # dx/dz = diag(-2, -5) x + (1, 1): each component relaxes independently.
    M = [[-2.0, 0.0], [0.0, -5.0]]
    B = [1.0, 1.0]
    x0 = [0.3, 0.7]
    d = 0.4
    got = solve_diffusion_ode(M, B, x0, d)
    want = [(x0[i] + B[i] / lam) * math.exp(lam * d) - B[i] / lam
            for i, lam in enumerate((-2.0, -5.0))]
    for a, b in zip(got, want):
        assert a == pytest.approx(b, rel=1e-15)


def test_a_singular_electrode_system_is_reported():
    with pytest.raises(ValueError, match="an eigenvalue is zero"):
        solve_diffusion_ode([[0.0, 0.0], [0.0, -3.0]], [1.0, 1.0],
                            [0.1, 0.2], 0.1)


def test_eigen_decomposition_is_scaling_invariant():
    """Any valid eigenbasis gives the same answer, so this port's closed
    form need not match LAPACK's normalisation."""
    l1, l2, V = eig2x2([[-3.0, 1.0], [0.5, -4.0]])
    assert l1 > l2
    for lam, k in ((l1, 0), (l2, 1)):
        vx, vy = V[0][k], V[1][k]
        assert -3.0 * vx + 1.0 * vy == pytest.approx(lam * vx, rel=1e-13)
        assert 0.5 * vx - 4.0 * vy == pytest.approx(lam * vy, rel=1e-13)


def test_the_nafion_extrapolation_slope_is_1e10_too_small():
    """A unit bug in the reference, recorded as DISCREPANCIES.md §82.

    Above lam = 16.8 the cubic is meant to be replaced by its tangent --
    "extrapolate with constant slope", says the comment. The cubic's
    derivative there is -0.01110912, and the source's constant is
    -1.1109120000000084e-12, which is that number already carrying the
    1e-10 the cubic branch applies. It is then multiplied by 1e-10 a second
    time.

    So the slope is ten orders of magnitude too small and the branch
    extrapolates with *zero* slope, not constant slope: over lam from 16.8
    to 1000 the diffusivity moves by 8e-8 percent. The intended tangent
    would have reached zero at lam = 132.8.

    The port reproduces the reference exactly. This matters little in
    practice -- saturated Nafion sits near lam = 14 to 22 -- but the branch
    does not do what its comment says.
    """
    T = 353.15
    D0 = nafion_diffusion(T, 16.8)
    assert nafion_diffusion(T, 16.8 - 1e-9) == pytest.approx(D0, rel=1e-6)
    # Frozen, not falling.
    assert nafion_diffusion(T, 1000.0) == pytest.approx(D0, rel=1e-8)
    # The intended slope would have made it fall by 63% over that range.
    intended = D0 - 0.01110912e-10 * math.exp(
        2416.0 * (1.0 / 303.0 - 1.0 / T)) * (100.0 - 16.8)
    assert intended / D0 < 0.4


def test_a_starved_cathode_reports_zero_volts():
    """Push the current far past what diffusion can supply."""
    u = PEMInputs(**{**COMMON, "j": 5.0e4, "lam_O2": 1.02},
                  T=453.15, kind="HT-PEMFC")
    assert ht_pemfc_voltage(u) == 0.0


def test_the_two_cell_types_disagree_on_voltage(): 
    """Same current, same pressures -- different membranes, different
    temperatures, and about 6% between them."""
    v_ht = ht_pemfc_voltage(_ht())
    v_lt = lt_pemfc_voltage(_lt())[0]
    assert v_ht > v_lt
    assert 0.03 < (v_ht - v_lt) / v_lt < 0.10
