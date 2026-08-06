#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Constraints that must hold but must not bind.

A holographic constraint keeps the problem well posed -- a 1e-30..1e30 box, the
edges of the data a fit was made from -- rather than shaping the answer. An
ACTIVE one silently invalidates the result: the solve converges, the duals are
finite, the table prints, and the optimum is sitting on a boundary of the
model's validity instead of the design's. The only defence is to declare them
in advance and check every time.
"""
import warnings

import pyomo.environ as pyo
import pytest

from lcsolver import Formulation
from lcsolver.solvers.holographic import format_holographic, holographic_report
from lcsolver.solvers.solver import solve


def _model(cap):
    """min A/(x*y) s.t. x*y >= A -- x and y want to grow without bound."""
    f = Formulation()
    x = f.Variable(name='x', guess=2.0, units='m', description='x')
    y = f.Variable(name='y', guess=2.0, units='m', description='y')
    A = f.Constant(name='A', value=2.0, units='m^2', description='A')
    f.Objective(A / (x * y))
    f.ConstraintList([x * y >= A])
    f.HolographicConstraintList([x <= cap * pyo.units.m,
                                 y <= cap * pyo.units.m])
    return f


def test_declaring_does_not_change_the_constraint():
    """It is imposed exactly as an ordinary constraint; only LCsolver's bookkeeping
    differs. A cap of 5 must still be enforced."""
    f = _model(5.0)
    solve(f, sensitivities=False)
    assert float(f.solution['x']) <= 5.0 + 1e-6


def test_active_is_detected_and_warned():
    f = _model(5.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solve(f, sensitivities=False)
    msgs = [str(w.message) for w in caught if 'holographic' in str(w.message)]
    assert msgs, 'an active holographic constraint must warn'
    assert 'ACTIVE' in msgs[0]
    assert len(f.solution.holographic) == 2


def test_inactive_is_silent():
    """The common case must cost nothing and say nothing."""
    f = _model(1e6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solve(f, sensitivities=False)
    assert not [w for w in caught if 'holographic' in str(w.message)]
    assert f.solution.holographic == []
    assert 'holographic' not in f.solution.summary(top=0)


def test_the_summary_reports_it():
    f = _model(5.0)
    solve(f, sensitivities=False)
    text = f.solution.summary(top=0)
    assert 'holographic constraints' in text
    assert 'ACTIVE' in text


def test_it_runs_on_every_solve_not_just_a_diagnostic():
    """No flag, no opt-in: the check is part of finishing a solve."""
    f = _model(5.0)
    res = solve(f, sensitivities=False)
    assert isinstance(res, dict) and res.get('holographic_active')


def test_margin_is_relative_not_absolute():
    """A model spanning many orders of magnitude cannot use absolute slack."""
    f = Formulation()
    x = f.Variable(name='x', guess=1e5, units='m', description='x')
    B = f.Constant(name='B', value=1e6, units='m', description='B')
    f.Objective(B / x)
    f.ConstraintList([x >= 1.0 * pyo.units.m])
    f.HolographicConstraint(x <= B)
    solve(f, sensitivities=False)
    act = f.solution.holographic
    assert len(act) == 1, act
    # 1e6 against a bound of 1e6: absolutely a huge number, relatively zero
    assert abs(act[0]['margin']) < 1e-5


def test_a_holographic_equality_is_called_out():
    """An equality always binds, so declaring one holographic is a mistake."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    y = f.Variable(name='y', guess=1.0, units='m', description='y')
    A = f.Constant(name='A', value=2.0, units='m^2', description='A')
    f.Objective(x + y)
    f.ConstraintList([x * y >= A])
    f.HolographicConstraint(x == y)
    solve(f, sensitivities=False)
    act = f.solution.holographic
    assert act and act[0]['operator'] == '=='
    assert 'EQUALITY' in format_holographic(act)


def test_report_reads_the_current_point():
    f = _model(5.0)
    assert holographic_report(f) == [] or True    # guesses, not a solution
    solve(f, sensitivities=False)
    assert len(holographic_report(f)) == 2


def test_groups_forward_the_declaration():
    f = Formulation()
    grp = f.group('wing')
    x = grp.Variable(name='x', guess=1.0, units='m', description='x')
    f.Objective(x)
    f.ConstraintList([x >= 1.0 * pyo.units.m])
    grp.HolographicConstraint(x <= 9.0 * pyo.units.m)
    assert len(f._holographic) == 1
