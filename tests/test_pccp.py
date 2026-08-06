#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""The penalty convex-concave loop, the other way to solve a signomial program.

SIA is the default, so nothing in the rest of the suite reaches PCCP: a plain
``solve`` on a signomial routes to ``sequential/bridge.solve_sia`` and the whole
of ``sequential/pccp.py`` never runs. It is not dead code -- two documented
routes reach it, ``solve(f, solver='cvxopt')`` and ``solve(f,
sp_method='pccp')`` -- it simply had no test.

What makes it testable without pinning numbers to whatever it happens to
produce: PCCP and SIA are independent methods, so on a model with a known
closed-form optimum all three paths must land on the same number. A
transformation that quietly changes the problem shows up as a disagreement.
"""
import numpy as np
import pytest

pyo = pytest.importorskip('pyomo.environ')
pytest.importorskip('cvxopt')

from lcsolver import Formulation                            # noqa: E402
from lcsolver.solvers.solver import solve                   # noqa: E402
from lcsolver.solvers.sequential.pccp import (              # noqa: E402
    evaluate_posynomial,
    monomial_approximation,
    solve_SP,
)


def _sp():
    """min x  s.t.  x >= y/(1+y),  y >= 1/2.

    ``y/(1+y)`` is a monomial over a posynomial, which has no GP form, so the
    detector calls this signomial. It rises with y, so y sits at its lower
    bound of 1/2 and the optimum is (1/2)/(3/2) = 1/3 exactly -- a number to
    check against that no solver here computed.
    """
    f = Formulation()
    x = f.Variable('x', 1.0, '-', 'x', bounds=[1e-3, 10.0])
    y = f.Variable('y', 1.0, '-', 'y', bounds=[1e-3, 10.0])
    f.Objective(x)
    f.ConstraintList([x >= y / (1 + y), y >= 0.5])
    return f


EXACT = 1.0 / 3.0


# ---------------------------------------------------------------------------
# the two routes that reach it
# ---------------------------------------------------------------------------
def test_cvxopt_route_reaches_pccp():
    """`solver='cvxopt'` on a signomial is one of the two ways in."""
    res = solve(_sp(), solver='cvxopt')
    assert res['problem_structure'] == 'signomial_program_pccp'
    assert res['status'] == 'optimal'
    assert float(res['primal objective']) == pytest.approx(EXACT, rel=1e-5)


def test_sp_method_pccp_reaches_pccp():
    """`sp_method='pccp'` is the other, and runs the inner GP through ipopt."""
    res = solve(_sp(), sp_method='pccp')
    assert res['problem_structure'] == 'signomial_program_pccp'
    assert float(res['primal objective']) == pytest.approx(EXACT, rel=1e-5)


def test_pccp_and_sia_agree():
    """The property worth asserting: two independent methods, one answer.

    Neither number is remembered from a previous run -- 1/3 is the analytic
    optimum -- so this fails if either method changes the problem rather than
    if either changes its arithmetic.
    """
    default = solve(_sp())
    pccp = solve(_sp(), sp_method='pccp')

    assert default['problem_structure'] == 'signomial_program_sia'
    assert pccp['problem_structure'] == 'signomial_program_pccp'
    assert float(default['primal objective']) == pytest.approx(EXACT, rel=1e-5)
    assert float(pccp['primal objective']) == pytest.approx(
        float(default['primal objective']), rel=1e-4)


def test_the_solution_is_written_back_to_the_model():
    """A route that solves but does not write back leaves the model on guesses."""
    f = _sp()
    solve(f, sp_method='pccp')
    assert pyo.value(f.x) == pytest.approx(EXACT, rel=1e-5)
    assert pyo.value(f.y) == pytest.approx(0.5, rel=1e-5)


# ---------------------------------------------------------------------------
# the loop's own options
# ---------------------------------------------------------------------------
def _detected(f):
    from lcsolver.presolve.structureDetector import structure_detector
    from lcsolver.presolve.unitCorrector import unit_corrector

    corrected = unit_corrector(f)
    return structure_detector(corrected), corrected


def test_the_penalty_loop_can_be_turned_off():
    """`use_pccp=False` is the plain convex-concave loop, without the slacks.

    The penalty variables exist to keep an infeasible sub-problem solvable;
    this model's sub-problems are feasible from the start, so both settings
    must reach the same optimum. That is what makes it a fair check of the
    branch rather than of the model.
    """
    st, m = _detected(_sp())
    on = solve_SP(st, m, use_pccp=True)
    st2, m2 = _detected(_sp())
    off = solve_SP(st2, m2, use_pccp=False)

    assert float(on['primal objective']) == pytest.approx(EXACT, rel=1e-5)
    assert float(off['primal objective']) == pytest.approx(EXACT, rel=1e-5)


def test_the_penalty_exponent_does_not_move_the_optimum():
    """It weights the slacks, so it changes the path taken, not the answer."""
    for exponent in (2.0, 5.0, 8.0):
        st, m = _detected(_sp())
        res = solve_SP(st, m, penalty_exponent=exponent)
        assert float(res['primal objective']) == pytest.approx(
            EXACT, rel=1e-4), exponent


def test_running_out_of_iterations_is_an_error_not_a_wrong_answer():
    """Stopping early must raise rather than return the last iterate.

    PCCP terminates when the objective stops moving, which is a statement
    about the loop rather than about optimality. Returning that iterate as
    though it were a solution is the failure this guards.
    """
    st, m = _detected(_sp())
    with pytest.raises(RuntimeError, match='maximum iteration count'):
        solve_SP(st, m, max_iter=1, reltol=1e-16, var_reltol=1e-16)


def test_a_signomial_equality_is_handled():
    """An equality whose sides are both posynomials takes a separate branch."""
    f = Formulation()
    x = f.Variable('x', 1.0, '-', 'x', bounds=[1e-3, 10.0])
    y = f.Variable('y', 1.0, '-', 'y', bounds=[1e-3, 10.0])
    f.Objective(x + y)
    # y + y**2 == 2 has the positive root y = 1, so the optimum is x + 1 with
    # x at its lower bound
    f.ConstraintList([y + y ** 2 == 2.0, x >= 1e-3])
    res = solve(f, sp_method='pccp')
    assert res['status'] == 'optimal'
    assert pyo.value(f.y) == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# the two pieces of arithmetic underneath
# ---------------------------------------------------------------------------
#: One posynomial, ``2*x0*x1 + 3*x0**2``, in the row encoding the loop uses:
#: ``[constraint index, coefficient, exponent per variable...]``.
_POSY = [[0, 2.0, 1.0, 1.0],
         [0, 3.0, 2.0, 0.0]]


def test_evaluate_posynomial_returns_the_value_and_its_gradient():
    """At x = (2, 5): 2*2*5 + 3*4 = 32, d/dx0 = 2*5 + 6*2 = 22, d/dx1 = 2*2."""
    value, grad = evaluate_posynomial(_POSY, [2.0, 5.0])
    assert value == pytest.approx(32.0)
    assert list(grad) == pytest.approx([22.0, 4.0])


def test_monomial_approximation_touches_the_posynomial_it_approximates():
    """The condensation is exact at the point it is taken about, and below elsewhere.

    That is the defining property of the arithmetic-geometric-mean step, and
    it is what makes the convex-concave loop's sub-problem a valid inner
    approximation: a monomial that ever exceeded the posynomial would let the
    loop accept a point the original problem forbids. Checking the two
    properties pins the approximation without hard-coding its exponents.
    """
    x_star = [2.0, 5.0]
    mono = monomial_approximation(_POSY, x_star)
    assert len(mono) == 1, 'a monomial is one row'

    assert evaluate_posynomial(mono, x_star)[0] == pytest.approx(
        evaluate_posynomial(_POSY, x_star)[0], rel=1e-9)

    for probe in ([1.0, 1.0], [3.0, 7.0], [0.5, 9.0], [10.0, 0.1], [0.01, 0.01]):
        assert evaluate_posynomial(mono, probe)[0] <= (
            evaluate_posynomial(_POSY, probe)[0] + 1e-9), probe


@pytest.mark.parametrize('fn', [evaluate_posynomial, monomial_approximation])
def test_a_multi_row_object_is_not_a_posynomial(fn):
    """Rows from two different constraints are not one posynomial."""
    two_constraints = [[0, 2.0, 1.0], [1, 3.0, 1.0]]
    with pytest.raises(ValueError, match='non-posynomial'):
        fn(two_constraints, [1.0])
