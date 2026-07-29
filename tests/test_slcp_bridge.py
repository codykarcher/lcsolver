"""The adapter from EDI's detected structure to the SLCP/SIA Problem object.

``edi.solvers.ipopt.slcp`` works on its own ``Problem`` type, which nothing
else in EDI builds. This bridge is the only path from a ``Formulation`` to
SLCP or SIA, so a defect here is invisible in the solver tests -- the paper's
problems in ``examples/slcp_cases.py`` construct ``Problem`` objects directly
and never touch it.

The test that matters here is the one that was missing: **an equality must not
become a one-sided inequality.** Writing only ``p <= 1`` for ``p == 1`` leaves
the solver free to drive ``p`` below 1, which relaxes the problem -- and a
relaxed problem yields an objective BETTER than the true optimum, which reads
as success rather than as a bug.
"""
import numpy as np
import pytest

from edi import Formulation
from edi.solvers.ipopt.sia import SIAOptions, solve_sia
from edi.solvers.ipopt.slcp import Posynomial, PosynomialRatio
from edi.solvers.ipopt.slcp_bridge import build_problem
from edi.structure.structureDetector import structure_detector
from edi.units.unitCorrector import unit_corrector


def _equality_model():
    """min z  s.t.  z >= x + y,  x*y >= 1,  x + y == 4.

    The equality forces x + y = 4, so z = 4. Drop its lower direction and
    x + y is free to fall to 2 (from x*y >= 1), giving z = 2 -- a 'better'
    objective that is not feasible for the real problem.
    """
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x')
    y = f.Variable('y', 2.0, '', 'y')
    z = f.Variable('z', 4.0, '', 'z')
    f.Objective(z)
    f.Constraint(z >= x + y)
    f.Constraint(x * y >= 1.0)
    f.Constraint(x + y == 4.0)          # multi-term posynomial equality
    for v in (x, y, z):
        f.Constraint(v <= 100.0)
        f.Constraint(0.01 <= v)
    return f


TRUE_OPTIMUM = 4.0
RELAXED_OPTIMUM = 2.0


def test_multi_term_equality_is_not_relaxed():
    """The headline regression: the bridge must not beat the true optimum."""
    st = structure_detector(unit_corrector(_equality_model()))
    prob = build_problem(st)
    r = solve_sia(prob, np.array([2.0, 2.0, 4.0]),
                  SIAOptions(max_iterations=100))
    assert r.converged, r.status
    assert r.objective == pytest.approx(TRUE_OPTIMUM, rel=1e-4), (
        f"got {r.objective}; {RELAXED_OPTIMUM} means the equality's lower "
        "direction was dropped")
    assert r.objective > RELAXED_OPTIMUM * 1.5


def test_every_equality_gets_both_directions():
    """Structural check: no '==' may leave the bridge as a lone '<='.

    A monomial equality is affine in log space and stays an '=='. Anything
    else must appear twice -- forward, and reverse as a ratio.
    """
    st = structure_detector(unit_corrector(_equality_model()))
    key = ('Signomial_Program' if st['Signomial_Program'][0]
           else 'Geometric_Program')
    operators = st[key][2]
    n_eq = sum(1 for op in operators if op == '==')
    assert n_eq >= 1, "the fixture must contain an equality to be a test"

    prob = build_problem(st)
    n_mono_eq = sum(1 for c in prob.constraints if c.operator == '==')
    n_ratio = sum(1 for c in prob.constraints
                  if isinstance(c.body, PosynomialRatio))
    # Each non-monomial equality contributes one extra constraint, and that
    # extra is always a ratio (1/p or q/p).
    assert len(prob.constraints) >= len(operators) + (n_eq - n_mono_eq)
    assert n_ratio >= n_eq - n_mono_eq


def test_the_known_optimum_stays_feasible():
    """Adding the reverse direction must not exclude the true solution."""
    st = structure_detector(unit_corrector(_equality_model()))
    prob = build_problem(st)
    x_star = np.array([2.0, 2.0, 4.0])          # x=y=2, z=4
    worst = max(c.body(x_star) for c in prob.constraints)
    assert worst <= 1.0 + 1e-9, f"true optimum reported infeasible by {worst-1:.2e}"


def test_monomial_equality_stays_exact():
    """A single-term equality is affine in log space; it must not be split."""
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x')
    y = f.Variable('y', 2.0, '', 'y')
    f.Objective(x + y)
    f.Constraint(x * y == 4.0)                  # monomial equality
    for v in (x, y):
        f.Constraint(v <= 100.0)
        f.Constraint(0.01 <= v)
    st = structure_detector(unit_corrector(f))
    prob = build_problem(st)
    eqs = [c for c in prob.constraints if c.operator == '==']
    assert len(eqs) == 1
    assert isinstance(eqs[0].body, Posynomial) and eqs[0].body.is_monomial


def test_sp_form_toggle_still_opaques_the_reverse_direction():
    """With sp_form off, BOTH directions of a split equality must be opaque."""
    st = structure_detector(unit_corrector(_equality_model()))
    a = build_problem(st, sp_form=True)
    b = build_problem(st, sp_form=False)
    assert len(a.constraints) == len(b.constraints)
    assert not any(isinstance(c.body, PosynomialRatio) for c in b.constraints)
