"""The cached sub-problem must give the same answers as rebuilding it.

``Options.cache_subproblem`` builds the sub-problem's Pyomo model once and
re-points it at each new iterate through mutable Params, instead of rebuilding
every constraint symbolically every iteration. On SPaircraft the rebuild path
constructs 6077 log-sum-exp expressions over 1173 variables ~50 times, and
that -- not the Hessian -- is what dominates the run.

The caching is only possible because everything that varies turns out to be a
weighted sum of *fixed* projections: each exact term is
``exp([log c + a.log x_k] + [a.d])`` with ``a.d`` constant, and the AGM
condensation's exponent vector is ``sum_i w_i a_i``, so ``aq.d`` reuses the
same projections with scalar weights.

Off by default; anything the bridge cannot express (a Signomial body, or the
lsqp/sqp methods) silently keeps the rebuild path.
"""
import pyomo.environ as pyo
import pytest

from lcsolver import Formulation
from lcsolver.solvers.ipopt.slcp import Options, SubproblemCache
from lcsolver.solvers.ipopt.slcp_bridge import build_problem, solve_slcp
from lcsolver.presolve.structureDetector import structure_detector
from lcsolver.presolve.unitCorrector import unit_corrector

M = pyo.units.m


def _box():
    f = Formulation()
    h = f.Variable(name="h", guess=1.0, units="m", description="")
    w = f.Variable(name="w", guess=1.0, units="m", description="")
    d = f.Variable(name="d", guess=1.0, units="m", description="")
    f.Objective(2 * (h * w + h * d + w * d))
    f.ConstraintList([h * w * d >= 8.0 * M ** 3, h <= 4.0 * M, w <= 4.0 * M])
    return f


def _sp():
    """Carries a posynomial ratio, so the AGM path is exercised."""
    f = Formulation()
    x = f.Variable(name="x", guess=2.0, units="-", description="")
    y = f.Variable(name="y", guess=2.0, units="-", description="")
    f.Objective(x + y)
    f.ConstraintList([x * y >= 4.0, x + y >= 3.0, x <= 10.0, y <= 10.0])
    return f


@pytest.mark.parametrize("make", [_box, _sp])
@pytest.mark.parametrize("memory", [None, 5])
def test_cached_matches_rebuilt(make, memory):
    plain = solve_slcp(structure_detector(unit_corrector(make())))
    cached = solve_slcp(
        structure_detector(unit_corrector(make())),
        options=Options(cache_subproblem=True, hessian_memory=memory))
    assert cached.objective is not None, cached.status
    assert cached.objective == pytest.approx(plain.objective, rel=1e-6), (
        f"cached {cached.objective} vs rebuilt {plain.objective}")


def test_cache_is_off_by_default():
    assert Options().cache_subproblem is False


def test_cacheability_is_detected():
    """A problem the bridge produces is cacheable; a Signomial body is not."""
    import numpy as np
    from lcsolver.solvers.ipopt.slcp import (Constraint, Posynomial, Problem,
                                        Signomial)
    problem = build_problem(structure_detector(unit_corrector(_box())))
    assert SubproblemCache(problem, Options()).usable

    blackbox = Problem(
        1, Posynomial([(1.0, [1.0])], 1),
        [Constraint(Signomial(lambda x: (float(x[0]), np.array([1.0])), 1), "<=")])
    assert not SubproblemCache(blackbox, Options()).usable


def test_unusable_problem_falls_back_silently():
    """An uncacheable problem still solves with cache_subproblem=True."""
    result = solve_slcp(structure_detector(unit_corrector(_box())),
                        method="lsqp",
                        options=Options(cache_subproblem=True))
    assert result.objective is not None, result.status
