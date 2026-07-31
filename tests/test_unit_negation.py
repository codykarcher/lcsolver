"""Unit conversion must survive a negated subexpression.

Regression test for a bug in ``handle_negation_node``: it returned the
original Pyomo node instead of the negation of the rebuilt child, so every
conversion performed inside a negated subtree was silently discarded. No
exception was raised -- the constraint simply evaluated with the wrong
numbers, and swapping the two operands of the subtraction made it go away.

Found via SPaircraft's fuselage bending model, where the zero-bending station
is declared in feet while the neighbouring stations are in metres.
"""
import pyomo.environ as pyo

from edi import Formulation
from edi.presolve.unitCorrector import unit_corrector

FT_IN_M = 0.3048


def _corrected(build_constraint):
    f = Formulation()
    a = f.Variable(name="a", guess=10.0, units="m", description="metres")
    b = f.Variable(name="b", guess=1.0, units="m", description="metres")
    c = f.Variable(name="c", guess=32.8, units="ft", description="feet")
    y = f.Variable(name="y", guess=1.0, units="m^3", description="result")
    f.Objective(y)
    f.ConstraintList([build_constraint(a, b, c, y)])
    fc = unit_corrector(f)
    out = []
    for con in fc.component_objects(pyo.Constraint, descend_into=True,
                                    active=True):
        out.extend(c_.expr for c_ in con.values())
    assert len(out) == 1
    return out[0]


def _converts(expr):
    """True if the feet variable picked up its metres conversion factor."""
    return f"{FT_IN_M}" in str(expr) or "0.304" in str(expr)


def test_conversion_survives_negation():
    # The feet variable sits in the *subtracted* term.
    expr = _corrected(lambda a, b, c, y: y >= a * ((a - b) ** 2 - (a - c) ** 2))
    assert _converts(expr), expr


def test_conversion_order_independent():
    """Operand order must not change whether the conversion happens."""
    first = _corrected(lambda a, b, c, y: y >= a * ((a - c) ** 2 - (a - b) ** 2))
    second = _corrected(lambda a, b, c, y: y >= a * ((a - b) ** 2 - (a - c) ** 2))
    assert _converts(first) and _converts(second)


def test_bare_negation_converts():
    expr = _corrected(lambda a, b, c, y: y ** (1 / 3) >= -(a - c))
    assert _converts(expr), expr


def test_negated_expression_evaluates_correctly():
    """The numbers, not just the printed form, must be right."""
    f = Formulation()
    a = f.Variable(name="a", guess=10.0, units="m", description="metres")
    c = f.Variable(name="c", guess=32.8084, units="ft", description="feet")
    y = f.Variable(name="y", guess=1.0, units="m", description="result")
    f.Objective(y)
    f.ConstraintList([y >= a - (a - c)])
    fc = unit_corrector(f)
    con = next(c_ for cl in fc.component_objects(pyo.Constraint, active=True)
               for c_ in cl.values())
    lhs, rhs = (float(pyo.value(arg)) for arg in con.expr.args)
    # a - (a - c) == c == 32.8084 ft == 10 m, not 32.8084.
    assert abs(lhs - 10.0) < 1e-3, (lhs, rhs)
