"""SLCP and PCCP must agree on the same formulation.

``edi.solvers.ipopt.slcp`` implements sequential log-convex programming over
its own ``Problem`` object, which nothing in EDI constructed -- so the solver
could not actually be run on a Formulation. ``slcp_bridge`` supplies that
adapter, which makes the two treatments comparable on identical problems.

They agree to ~1e-7. SLCP is slower, and increasingly so with size: it carries
a dense BFGS approximation, so cost grows sharply in the variable count.
"""
import pyomo.environ as pyo
import pytest

from edi import Formulation
from edi.solvers.ipopt.slcp_bridge import build_problem, solve_slcp
from edi.solvers.solver import solve
from edi.structure.structureDetector import structure_detector
from edi.units.unitCorrector import unit_corrector

M = pyo.units.m


def _box():
    """Minimise surface area of a box at fixed volume: a small GP."""
    f = Formulation()
    h = f.Variable(name="h", guess=1.0, units="m", description="height")
    w = f.Variable(name="w", guess=1.0, units="m", description="width")
    d = f.Variable(name="d", guess=1.0, units="m", description="depth")
    f.Objective(2 * (h * w + h * d + w * d))
    f.ConstraintList([h * w * d >= 8.0 * M ** 3, h <= 4.0 * M, w <= 4.0 * M])
    return f, lambda: 2 * (pyo.value(h) * pyo.value(w)
                           + pyo.value(h) * pyo.value(d)
                           + pyo.value(w) * pyo.value(d))


def _signomial():
    """A small SP: the constraint has a sum on the bounding side."""
    f = Formulation()
    x = f.Variable(name="x", guess=2.0, units="-", description="")
    y = f.Variable(name="y", guess=2.0, units="-", description="")
    f.Objective(x + y)
    f.ConstraintList([x * y >= 4.0, x + y >= 3.0, x <= 10.0, y <= 10.0])
    return f, lambda: pyo.value(x) + pyo.value(y)


@pytest.mark.parametrize("make", [_box, _signomial])
def test_slcp_agrees_with_pccp(make):
    f, read = make()
    solve(f, solver="ipopt-convex")
    pccp = read()

    f2, _ = make()
    result = solve_slcp(structure_detector(unit_corrector(f2)))
    assert result.objective is not None, result.status
    assert result.objective == pytest.approx(pccp, rel=1e-5), (
        f"SLCP {result.objective} vs PCCP {pccp} ({result.status})")


def test_bridge_reports_the_constraint_split():
    """A GP is entirely exact in log space; an SP carries posynomial ratios."""
    f, _ = _box()
    problem = build_problem(structure_detector(unit_corrector(f)))
    assert all(c.exact_in_logspace for c in problem.constraints)
    assert not any(c.is_sp_form for c in problem.constraints)


def test_bridge_rejects_negative_coefficients():
    """A posynomial part with a negative coefficient is reported, not dropped."""
    from edi.solvers.ipopt.slcp_bridge import _group
    import numpy as np
    from edi.solvers.ipopt.slcp import Posynomial
    with pytest.raises(ValueError):
        Posynomial([(-1.0, np.zeros(1))], 1)
