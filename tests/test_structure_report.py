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
from edi.presolve.reductions import structure_report
from edi.presolve.structureDetector import structure_detector
from edi.presolve.unitCorrector import unit_corrector


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
    assert f.structure_report() == structure_report(_detected(f))


def test_diagnose_leads_with_the_structure_section():
    rep = _sp().optimization_precheck()
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


# --- optimization_precheck(f) ------------------------------------------------------------
# Detecting structure means unit-correcting a clone and walking it. That is how
# this runs, not what a caller wants to say, so both entry points take the
# formulation itself.

def test_diagnose_accepts_a_formulation():
    from edi import optimization_precheck
    rep = optimization_precheck(_sp())
    assert 'Signomial Program (SP)' in rep.structure


def test_structure_report_accepts_a_formulation():
    f = _sp()
    assert structure_report(f) == structure_report(_detected(f))


def test_bad_units_are_diagnosed_not_raised():
    """The tool for asking what is wrong must survive the commonest fault.

    A unit mismatch used to propagate out of `optimization_precheck`, so the one call you
    would make to find out why a model misbehaves failed with the very error
    you were looking for, and printed nothing else.
    """
    from edi import optimization_precheck
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='a length')
    t = f.Variable(name='t', guess=1.0, units='s', description='a time')
    f.Objective(x)
    f.ConstraintList([x >= t])                    # metres against seconds
    rep = optimization_precheck(f)                 # must not raise
    assert rep.structure.startswith('units')
    assert '[s]' in rep.structure and '[m]' in rep.structure
    assert 'Nothing further can be checked' in rep.structure


def test_bad_units_report_from_structure_report_too():
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='a length')
    t = f.Variable(name='t', guess=1.0, units='s', description='a time')
    f.Objective(x)
    f.ConstraintList([x >= t])
    assert structure_report(f).startswith('units')


def test_str_of_report_carries_the_structure_section():
    """`print(optimization_precheck(...))` must be the whole report, not half of it."""
    text = str(_sp().optimization_precheck())
    assert 'structure' in text and 'presolve:' in text


# --- posynomial equalities --------------------------------------------------
# A GP admits only MONOMIAL equalities: log-sum-exp == 0 is not a convex set.
# The detector knew, but the site that cleared the flag carried no blame, ran
# only while the flag was still alive, and stopped after the first offender.
# The report then inferred "simplifies to a GP after presolve" from a blame
# list that did not mention the three constraints preventing exactly that.

def _posynomial_equality():
    """``x*y + x == A``: a posynomial equality, and not an affine one.

    The distinction matters. ``x + y == A`` is *linear*, so the model is an LP
    and the GP question never arises -- the report only discusses classes
    simpler than the one detected. A product term forces it past LP and QP, so
    GP is the next class up and the equality is what rules it out.
    """
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='-', description='x')
    y = f.Variable(name='y', guess=1.0, units='-', description='y')
    A = f.Constant(name='A', value=3.0, units='-', description='A')
    f.Objective(x)
    f.ConstraintList([x * y + x == A, x >= 0.5, y >= 0.5, y <= 2.0])
    return f


def test_a_posynomial_equality_is_not_a_gp():
    st = _detected(_posynomial_equality())
    assert st['Geometric_Program'][0] is False
    assert st['Signomial_Program'][0] is not False


def test_the_posynomial_equality_is_named_as_the_blocker():
    text = structure_report(_detected(_posynomial_equality()))
    assert 'posynomial equality' in text
    assert 'monomial equalities' in text


def test_every_offending_equality_is_blamed_not_just_the_first():
    """It used to `break` after one."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='-', description='x')
    y = f.Variable(name='y', guess=1.0, units='-', description='y')
    z = f.Variable(name='z', guess=1.0, units='-', description='z')
    A = f.Constant(name='A', value=3.0, units='-', description='A')
    f.Objective(x)
    f.ConstraintList([x * y + x == A, x * z + x == A, y * z + y == A,
                      x >= 0.1, y >= 0.1, z >= 0.1])
    st = _detected(f)
    blamed = {r[0] for r in st['blockers'].get('Geometric_Program', ())}
    assert len(blamed) == 3, blamed


def test_blame_is_recorded_even_when_gp_was_already_ruled_out():
    """The decisive one.

    The scan was guarded by `if Geometric_Program[0] != False`, so on a model
    something else had already made non-GP it never ran -- and the missing
    blame is what let the report claim a simplification that was not true.
    """
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='-', description='x')
    y = f.Variable(name='y', guess=1.0, units='-', description='y')
    lam = f.Variable(name='lam', guess=0.5, units='-', description='lam')
    q = f.Variable(name='q', guess=1.5, units='-', description='q')
    A = f.Constant(name='A', value=3.0, units='-', description='A')
    f.Objective(x)
    f.ConstraintList([q == 1 + lam,        # rules out GP, and is removable
                      x * y + x == A,      # rules out GP, and is NOT
                      x >= 0.5, y >= 0.5, y <= 2.0, q >= 1.2])
    st = _detected(f)
    reasons = ' '.join(r[1] for r in st['blockers'].get('Geometric_Program', ()))
    assert 'posynomial equality' in reasons

    text = structure_report(st)
    assert 'as solved' not in text, (
        'claimed a simplification while an unremovable signomial equality '
        'remains:\n' + text)


def test_the_claim_is_derived_from_the_reduced_rows_not_the_blame_list():
    """Where "as solved" comes from, and where it must NOT come from.

    Two earlier versions decided this by tracking which original row the
    presolve took away. Both were wrong: fold_singleton_rows,
    eliminate_monomial_equalities and reduce_columns each RENUMBER, so an
    index means something different after every pass. It is now answered by
    inspecting the reduced rows -- no fraction, no multi-term equality, no
    negative coefficient -- which needs no provenance at all.

    So blanking the blame list must not change the verdict: the two are
    independent, and that independence is the point.
    """
    from edi.presolve.reductions import _gp_after_presolve
    from edi.presolve.reductions import structure_report as report
    st = _detected(_sp_only_on_paper())
    assert _gp_after_presolve(st) is True
    with_blame = report(st)
    st['blockers'] = {}
    assert 'as solved' in with_blame
    assert 'as solved' in report(st)


def test_a_genuine_sp_is_not_gp_after_presolve():
    """The other side of it: an unremovable signomial must fail the check."""
    from edi.presolve.reductions import _gp_after_presolve
    assert _gp_after_presolve(_detected(_sp())) is False
