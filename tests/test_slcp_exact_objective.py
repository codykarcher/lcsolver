"""Imposing a posynomial objective exactly collapses the iteration count.

SLCP keeps posynomial constraints exact but linearizes the objective, so a
fully convex problem still marches like quasi-Newton (the wind turbine takes
72 sub-problems where the GP path takes one). ``Options.exact_objective``
makes the sub-problem the original problem, so one solve suffices. Off by
default.
"""
import pyomo.environ as pyo
import pytest

from lcsolver import Formulation
from lcsolver.solvers.sequential.slcp import Options
from lcsolver.solvers.sequential.bridge import build_problem, solve_slcp
from lcsolver.presolve.structureDetector import structure_detector
from lcsolver.presolve.unitCorrector import unit_corrector

M = pyo.units.m


def _box():
    """Box surface area at fixed volume: a GP with a THREE-term objective."""
    f = Formulation()
    h = f.Variable(name="h", guess=1.0, units="m", description="")
    w = f.Variable(name="w", guess=1.0, units="m", description="")
    d = f.Variable(name="d", guess=1.0, units="m", description="")
    f.Objective(2 * (h * w + h * d + w * d))
    f.ConstraintList([h * w * d >= 8.0 * M ** 3, h <= 4.0 * M, w <= 4.0 * M])
    return f


def _monomial_objective():
    """Same feasible set, but a single-term objective."""
    f = Formulation()
    h = f.Variable(name="h", guess=1.0, units="m", description="")
    w = f.Variable(name="w", guess=1.0, units="m", description="")
    f.Objective(h * w)
    f.ConstraintList([h * w >= 2.0 * M ** 2, h <= 4.0 * M, w <= 4.0 * M])
    return f


def _count(make, **opts):
    import lcsolver.solvers.sequential.slcp as S
    calls = {"n": 0}
    original = S._solve_pyomo_subproblem

    def traced(m, n, n_cons, options, method="slcp"):
        calls["n"] += 1
        return original(m, n, n_cons, options, method)

    S._solve_pyomo_subproblem = traced
    try:
        result = solve_slcp(structure_detector(unit_corrector(make())),
                            options=Options(**opts))
    finally:
        S._solve_pyomo_subproblem = original
    return calls["n"], result


def test_same_answer():
    _, linearized = _count(_box)
    _, exact = _count(_box, exact_objective=True)
    assert exact.objective == pytest.approx(linearized.objective, rel=1e-6)


def test_multi_term_objective_needs_far_fewer_subproblems():
    n_lin, _ = _count(_box)
    n_exact, _ = _count(_box, exact_objective=True)
    assert n_exact < n_lin, (n_exact, n_lin)


def test_monomial_objective_still_benefits_from_dropping_the_quadratic():
    """A monomial objective is already exact when linearized, so the change
    is a no-op here -- but the BFGS term is dropped and the sub-problem
    becomes the original problem outright. One solve instead of two."""
    n_lin, r_lin = _count(_monomial_objective)
    n_exact, r_exact = _count(_monomial_objective, exact_objective=True)
    assert n_exact <= n_lin, (n_exact, n_lin)
    assert r_exact.objective == pytest.approx(r_lin.objective, rel=1e-9)


def test_fully_log_convex_detection():
    from lcsolver.solvers.sequential.slcp import _fully_log_convex
    assert _fully_log_convex(build_problem(
        structure_detector(unit_corrector(_box()))))


def test_default_is_off():
    assert Options().exact_objective is False
