#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Constraints that must hold but must not bind.

A holographic constraint keeps the problem well posed (a 1e-30..1e30 box,
the edges of a fit's data) rather than shaping the answer. An ACTIVE one
silently invalidates the result -- the optimum sits on the model's validity
boundary, not the design's -- so declare them in advance and check each time.
"""
import warnings

import pyomo.environ as pyo
import pytest

from lcsolver import Formulation
from lcsolver.postsolve.holographic import format_holographic, holographic_report
from lcsolver.solvers.solver import solve


def _model(cap):
    """min A/(x*y) s.t. x*y >= A -- x and y want to grow without bound, so
    the caps bind wherever they are put: the model for the ACTIVE case."""
    f = Formulation()
    x = f.Variable(name='x', guess=2.0, units='m', description='x')
    y = f.Variable(name='y', guess=2.0, units='m', description='y')
    A = f.Constant(name='A', value=2.0, units='m^2', description='A')
    f.Objective(A / (x * y))
    f.ConstraintList([x * y >= A])
    f.HolographicConstraintList([x <= cap * pyo.units.m,
                                 y <= cap * pyo.units.m])
    return f


def _interior_model():
    """min x + A/x -- optimum at sqrt(A), far inside the caps. The inactive
    case needs a genuine interior minimum, not _model() with a bigger cap."""
    f = Formulation()
    x = f.Variable(name='x', guess=2.0, units='m', description='x')
    A = f.Constant(name='A', value=2.0, units='m^2', description='A')
    f.Objective(x + A / x)
    f.HolographicConstraintList([x <= 1e6 * pyo.units.m,
                                 x >= 1e-6 * pyo.units.m])
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
        solve(f, sensitivities=False, quiet=False)
    msgs = [str(w.message) for w in caught if 'holographic' in str(w.message)]
    assert msgs, 'an active holographic constraint must warn'
    assert 'ACTIVE' in msgs[0]
    assert len(f.solution.holographic) == 2


def test_inactive_is_silent():
    """The common case must cost nothing and say nothing."""
    f = _interior_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solve(f, sensitivities=False)
    assert abs(float(f.solution['x']) - 2.0 ** 0.5) < 1e-6   # interior, not capped
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
    """It reports the point the model is at, which before a solve is the guess."""
    f = _model(5.0)
    # x = y = 2 against a cap of 5: nothing is binding at the starting point
    assert holographic_report(f) == []
    solve(f, sensitivities=False)
    # and both are binding once the solve has written the answer back
    assert len(holographic_report(f)) == 2


def test_groups_forward_the_declaration():
    f = Formulation()
    grp = f.group('wing')
    x = grp.Variable(name='x', guess=1.0, units='m', description='x')
    f.Objective(x)
    f.ConstraintList([x >= 1.0 * pyo.units.m])
    grp.HolographicConstraint(x <= 9.0 * pyo.units.m)
    assert len(f._holographic) == 1


# --- "n of N" has to be a true fraction -------------------------------------
# The summary called format_holographic without a total, so it printed the
# active count on both sides -- "2 of 2" for a model declaring three, which
# reads as though every watched limit had been hit.

def test_the_report_counts_what_was_declared_not_what_is_active():
    from lcsolver.postsolve.holographic import holographic_total

    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', size=3, bounds=[0.0, 9.0],
                   description='x')
    cap = f.Constant(name='cap', value=[2.0, 2.0, 8.0], units='m', size=3,
                     description='cap')
    f.Objective(-f.sum(x))
    f.ConstraintList([x <= 5.0 * pyo.units.m])
    f.HolographicConstraintList([x <= cap])

    assert holographic_total(f) == 3            # one per element, not one name
    solve(f, sensitivities=False)
    text = f.solution.summary()
    assert '2 of 3 are ACTIVE' in text          # x[2] is capped at 5, not 8


def test_the_total_is_zero_when_none_were_declared():
    from lcsolver.postsolve.holographic import holographic_total

    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    f.Objective(x)
    f.ConstraintList([x >= 1.0 * pyo.units.m])
    assert holographic_total(f) == 0


# --- and a block has to be able to say it too --------------------------------
# Group carried all four constraint methods, but SubModel mirrored only
# ConstraintList, so a packaged block had no way to declare a validity envelope
# -- exactly the models that need one most, since a block is where a fit lives.

def test_submodels_forward_the_declaration():
    from lcsolver import SubModel

    class Guarded(SubModel):
        input_variables = ('x',)

        def build(self):
            m = self.bind_inputs()
            cap = self.Constant('cap', 9.0, 'm', 'edge of the fit')
            self.ConstraintList([m.x >= 1.0 * pyo.units.m])
            self.HolographicConstraintList([m.x <= cap])
            self.HolographicConstraint(m.x <= 20.0 * pyo.units.m)

    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    f.Objective(x)
    f.guarded = Guarded()
    f.guarded.x = x

    assert f.guarded.is_built()
    assert len(f._holographic) == 2             # both forms, neither ordinary
    assert len(f.guarded.rows) == 3             # and the block counts all three
    solve(f, sensitivities=False)
    assert holographic_report(f) == []          # x drives to 1, well inside


def test_a_bound_in_other_units_is_converted_before_it_is_compared():
    """pyo.value drops units, so a ceiling of 27 in / 2 read as 13.5 against
    a radius in metres hid the ceiling the answer sat on, and the floor of
    4.1 in / 2 read as 2.05 was reported ACTIVE with a margin of -0.83."""
    f = Formulation()
    x = f.Variable(name='x', guess=0.3, units='m', description='x')
    f.Objective(1 / x)                                  # x wants to grow: onto the ceiling
    f.ConstraintList([x >= 0.01 * pyo.units.m])
    f.HolographicConstraintList([x <= 27 * pyo.units.inch / 2,
                                 x >= 4.1 * pyo.units.inch / 2])
    solve(f, sensitivities=False)
    act = f.solution.holographic
    assert len(act) == 1, act
    assert act[0]['operator'] == '<='
    assert abs(act[0]['bound'] - 27 * 0.0254 / 2) < 1e-9     # reported in metres
    assert abs(act[0]['value'] - 27 * 0.0254 / 2) < 1e-5
    assert abs(act[0]['margin']) < 1e-5


def _two_sided(k):
    """min y s.t. x >= 1, y >= 2, holographic y >= k x: at the optimum x = 1,
    y = max(2, k), so k = 3 binds and k = 0.5 leaves a margin of 0.75."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.5, units='m', description='x')
    y = f.Variable(name='y', guess=4.0, units='m', description='y')
    f.Objective(x * y)
    f.ConstraintList([x >= 1.0 * pyo.units.m, y >= 2.0 * pyo.units.m])
    f.HolographicConstraint(y >= k * x)
    return f


def test_a_row_with_variables_on_both_sides_has_a_meaningful_margin():
    """pyomo stores such a row as "lhs - rhs" against 0. Normalising by
    max(|0|, |residual|) gave a margin of +-1 by roundoff sign: an active
    row was flagged only when the residual happened to be negative."""
    f = _two_sided(3.0)
    solve(f, sensitivities=False)
    act = f.solution.holographic
    assert len(act) == 1, act
    assert abs(act[0]['margin']) < 1e-5, act               # ~0, not -1 or +1
    assert abs(act[0]['value'] - 3.0) < 1e-4

    g = _two_sided(0.5)
    solve(g, sensitivities=False)
    assert g.solution.holographic == []                    # 0.5 against 2: inactive


# (variable units, bound units, bound value, the bound in the variable's units)
_UNIT_PAIRS = [
    ('m',   'inch', 13.5,  13.5 * 0.0254),
    ('m',   'ft',   2.0,   2.0 * 0.3048),
    ('m',   'mm',   750.0, 0.75),
    ('ft',  'm',    0.5,   0.5 / 0.3048),
    ('lbf', 'N',    100.0, 100.0 / 4.4482216152605),
    ('N',   'lbf',  20.0,  20.0 * 4.4482216152605),
    ('m',   'm',    0.4,   0.4),
]


def _boxed(var_units, bound_units, value, toward):
    """One variable driven onto a ceiling (toward='up') or a floor ('down')
    written in `bound_units`; the opposite edge is a decade away, in the same
    units, and must stay silent."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units=var_units, description='x')
    u_var, u_bnd = getattr(pyo.units, var_units), getattr(pyo.units, bound_units)
    if toward == 'up':
        f.Objective(1 * u_var / x)
        f.ConstraintList([x >= 1e-6 * u_var])
        f.HolographicConstraintList([x <= value * u_bnd, x >= 0.1 * value * u_bnd])
    else:
        f.Objective(x / (1 * u_var))
        f.ConstraintList([x <= 1e6 * u_var])
        f.HolographicConstraintList([x >= value * u_bnd, x <= 10. * value * u_bnd])
    return f


@pytest.mark.parametrize('var_units, bound_units, value, in_var_units', _UNIT_PAIRS)
@pytest.mark.parametrize('toward', ['up', 'down'])
def test_the_binding_edge_is_named_whatever_units_it_was_written_in(
        var_units, bound_units, value, in_var_units, toward):
    """The edge the answer sits on is reported, with a margin of ~0 and its
    bound in the variable's units; the far edge is not. Comparing bare
    numbers gets this wrong for every pair except the last."""
    f = _boxed(var_units, bound_units, value, toward)
    solve(f, sensitivities=False)
    act = f.solution.holographic
    assert len(act) == 1, act
    d = act[0]
    assert d['operator'] == ('<=' if toward == 'up' else '>=')
    assert abs(d['bound'] - in_var_units) <= 1e-9 * in_var_units
    assert abs(d['value'] - in_var_units) <= 1e-5 * in_var_units
    assert abs(d['margin']) < 1e-5


def test_a_constant_in_other_units_is_converted_too():
    """The bound need not be a literal: a Constant declared in inches bounds a
    variable in metres through the same conversion."""
    f = Formulation()
    x = f.Variable(name='x', guess=0.3, units='m', description='x')
    D = f.Constant(name='D', value=27.0, units='inch', description='ceiling')
    f.Objective(1 / x)
    f.ConstraintList([x >= 0.01 * pyo.units.m])
    f.HolographicConstraint(2 * x <= D)
    solve(f, sensitivities=False)
    act = f.solution.holographic
    assert len(act) == 1, act
    assert abs(act[0]['margin']) < 1e-5
    assert abs(float(f.solution['x']) - 27 * 0.0254 / 2) < 1e-6


def test_the_warning_names_the_row_that_binds_and_no_margin_is_wild():
    """End to end, on the text a user reads: the ceiling is named, the floor is
    not, and no reported margin is far from zero. A margin like -0.83 at a
    converged optimum is the signature of comparing numbers in two units."""
    f = _boxed('m', 'inch', 13.5, 'up')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solve(f, sensitivities=False, quiet=False)
    msgs = [str(w.message) for w in caught if 'LC-W301' in str(w.message)]
    assert len(msgs) == 1, msgs
    assert '1 of 2' in msgs[0]
    assert 'x  <=  13.5*inch' in msgs[0] or 'x <= 13.5*in' in msgs[0].replace('  ', ' '), msgs[0]
    assert all(abs(d['margin']) < 1e-4 for d in f.solution.holographic)
