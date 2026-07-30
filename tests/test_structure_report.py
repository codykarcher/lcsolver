"""The structure report: what class the problem is, and what blocks a simpler one.

The detector already knew -- it clears a class flag the moment a row rules that
class out -- but it never recorded *which* row, so a model that "is an SP"
could not answer the only question that fact raises: which constraint made it
one. Usually it is one or two, and usually they are a reformulation away from
posynomial, which is the difference between a label and an action.
"""
import pyomo.environ as pyo
import pytest

from edi import Formulation
from edi.presolve import structure_report
from edi.structure.structureDetector import structure_detector
from edi.units.unitCorrector import unit_corrector


def _detected(f):
    return structure_detector(unit_corrector(f), bounds_as_rows=False)


def _lp():
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    y = f.Variable(name='y', guess=1.0, units='m', description='y')
    f.Objective(x + y)
    f.ConstraintList([x >= 1.0 * pyo.units.m, y >= 2.0 * pyo.units.m])
    return f


def _qp():
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='-', description='x')
    f.Objective(x ** 2 + x)
    f.ConstraintList([x >= 0.5])
    return f


def _gp():
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    y = f.Variable(name='y', guess=1.0, units='m', description='y')
    A = f.Constant(name='A', value=2.0, units='m^2', description='A')
    f.Objective(x + y)
    f.ConstraintList([x * y >= A])
    return f


def _sp():
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='-', description='x')
    y = f.Variable(name='y', guess=1.0, units='-', description='y')
    f.Objective(x)
    f.ConstraintList([x >= y / (1 + y), y >= 0.5])
    return f


@pytest.mark.parametrize('build,label', [(_lp, 'Linear Program (LP)'),
                                         (_qp, 'Quadratic Program (QP)'),
                                         (_gp, 'Geometric Program (GP)'),
                                         (_sp, 'Signomial Program (SP)')],
                         ids=['lp', 'qp', 'gp', 'sp'])
def test_names_the_detected_class(build, label):
    assert label in structure_report(_detected(build()))


def test_an_lp_reports_nothing_blocking():
    """Nothing is simpler than an LP, so there is nothing to explain."""
    text = structure_report(_detected(_lp()))
    assert 'blocks it' not in text and 'block it' not in text


def test_sp_names_the_constraint_that_blocks_gp():
    """The headline feature: which constraint makes it an SP."""
    text = structure_report(_detected(_sp()))
    assert 'Not a Geometric Program' in text
    assert 'ratio of posynomials' in text
    assert 'y/(1 + y)' in text          # the body, not just the name


def test_qp_blames_the_objective_not_a_constraint():
    """A QP is not an LP because of its objective; say so, with grammar."""
    text = structure_report(_detected(_qp()))
    assert 'the objective blocks it' in text
    assert 'is quadratic, not affine' in text


def test_gp_does_not_complain_about_not_being_an_sp():
    """Only classes SIMPLER than the detected one are worth reporting."""
    text = structure_report(_detected(_gp()))
    assert 'Signomial' not in text
    assert 'Not a Linear or Quadratic Program' in text


def test_top_caps_the_listing_and_counts_the_rest():
    f = Formulation()
    xs = [f.Variable(name=f'x{i}', guess=1.0, units='-', description='x')
          for i in range(6)]
    f.Objective(sum(xs))
    f.ConstraintList([x * y >= 1.0 for x, y in zip(xs, xs[1:])])
    full = structure_report(_detected(f), top=None)
    capped = structure_report(_detected(f), top=2)
    assert 'more' not in full
    assert 'and 3 more' in capped


def test_formulation_method_matches_the_function():
    f = _sp()
    assert f.structure_report(quiet=True) == structure_report(_detected(f))


def test_diagnose_leads_with_the_structure_section():
    rep = _sp().diagnose(quiet=True)
    assert rep.structure.startswith('structure')
    assert 'Not a Geometric Program' in rep.structure


def test_blame_does_not_disturb_detection():
    """The instrumentation must not change what the detector concludes."""
    for build, key in ((_lp, 'Linear_Program'), (_qp, 'Quadratic_Program'),
                       (_gp, 'Geometric_Program'), (_sp, 'Signomial_Program')):
        st = _detected(build())
        assert st[key][0] is not False, key


# --- as written vs as solved ------------------------------------------------
# A constraint whose only job is to define a quantity nothing else reads is
# removed by the presolve. It can make a model an SP on paper while the solver
# is handed a GP, and the report has to distinguish the two or it misleads.

def _sp_only_on_paper():
    """A GP, plus a posynomial equality defining an output-only variable.

    `q == 1 + lam` is not GP-representable, so the model detects as an SP. But
    nothing reads `lam`, so the presolve drops it and the constraint with it.
    This is the Hoburg UAV's taper ratio, reduced.
    """
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    y = f.Variable(name='y', guess=1.0, units='m', description='y')
    q = f.Variable(name='q', guess=1.5, units='-', description='q')
    lam = f.Variable(name='lam', guess=0.5, units='-', description='taper')
    A = f.Constant(name='A', value=2.0, units='m^2', description='A')
    f.Objective(x + y)
    f.ConstraintList([x * y >= A, q >= 1.2, q == 1 + lam])
    return f


def test_reports_both_as_written_and_as_solved():
    text = structure_report(_detected(_sp_only_on_paper()))
    assert 'Signomial Program (SP) as written' in text
    assert 'Geometric Program (GP) as solved' in text
    assert 'removed by the presolve' in text


def test_a_genuine_sp_is_not_simplified():
    """The blocker here is load-bearing, so nothing may claim it goes away."""
    text = structure_report(_detected(_sp()))
    assert 'as solved' not in text
    assert 'Signomial Program (SP)' in text
    assert 'removed by the presolve' not in text


def test_simplify_can_be_turned_off():
    text = structure_report(_detected(_sp_only_on_paper()), simplify=False)
    assert 'as solved' not in text
    assert 'Signomial Program (SP)' in text


def test_no_simplification_claim_when_nothing_is_removed():
    """A plain GP must not sprout an 'as solved' line."""
    assert 'as solved' not in structure_report(_detected(_gp()))
