#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""The solve() pipeline contract: gate, split checks, result object.

Covers the workflow behaviors added 2026-08: the pre-solve error gate
(ill-posed models stop before a solver runs, batched, demotable to a
warning), the presolve_check / postsolve_check split, and the SolveResult
returned by solve().
"""

import warnings


def _ipopt_available():
    try:
        from lcsolver.environment import ipopt_available
        return bool(ipopt_available())
    except Exception:
        return False

import pytest

pyo = pytest.importorskip('pyomo.environ')

from pyomo.environ import units  # noqa: E402

from lcsolver import (Formulation, PresolveError, SolveResult,  # noqa: E402
                      postsolve_check, presolve_check, solve)


def _well_posed():
    f = Formulation()
    x = f.Variable(name='x', guess=2.0, units='', bounds=[0.1, 10.0])
    y = f.Variable(name='y', guess=2.0, units='', bounds=[0.1, 10.0])
    f.Objective(x + y)
    f.ConstraintList([x * y >= 1.0 * units.dimensionless])
    return f


def _unbounded():
    """y appears in the objective with nothing bounding it below."""
    f = Formulation()
    x = f.Variable(name='x', guess=2.0, units='', bounds=[0.1, 10.0])
    y = f.Variable(name='y', guess=2.0, units='')
    f.Objective(x + y)
    f.ConstraintList([x >= 1.0 * units.dimensionless])
    return f


def test_presolve_gate_raises_on_ill_posed():
    with pytest.raises(PresolveError) as exc:
        solve(_unbounded())
    msg = str(exc.value)
    assert 'y' in msg
    assert 'bounds' in msg          # the message says how to fix it
    assert "diagnostics='warn'" in msg  # and how to demote it


def test_presolve_gate_demotes_to_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            solve(_unbounded(), diagnostics='warn', quiet=False)
        except Exception:
            pass  # the SOLVE may fail on the unbounded model; the gate must not
    assert any('pre-solve check' in str(w.message) for w in caught)


def test_clean_model_passes_gate_silently():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = solve(_well_posed())
    assert not [w for w in caught if 'pre-solve check' in str(w.message)]
    assert res['status'] == 'optimal' or res.get('converged')


def test_solve_returns_solveresult():
    f = _well_posed()
    res = solve(f)
    assert isinstance(res, SolveResult)
    assert isinstance(res, dict)                    # backward compatible
    assert res['x'] is res.x                        # attribute == key
    # .solution is the rich printable object (built fresh per access, like
    # f.solution), not the flat name->value dict under the 'solution' key
    from lcsolver.objects.solution import Solution
    assert isinstance(res.solution, Solution)
    assert str(res.solution) == str(f.solution)
    assert isinstance(res['solution'], dict)
    assert 'Objective' in str(res.solution)
    assert res.summary() == f.solution.summary()
    assert res.objective == pytest.approx(2.0, rel=1e-5)


@pytest.mark.skipif(
    not _ipopt_available(),
    reason='the Report names the route that solved it, and the cvxopt fallback reports a different one')
def test_messages_captured_not_printed():
    """Solve warnings land in sol.messages; the console stays clean."""
    f = Formulation()
    x = f.Variable(name='x', guess=2.0, units='m', bounds=[0.5, 10.0])
    c = f.Constant(name='c', value=1.0, units='m', description='floor')
    f.Objective(x)
    f.Constraint(x >= c)
    f.Constraint(x <= 1.0 * units.m, holographic=True)  # binds at optimum
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sol = solve(f)
    assert not caught                       # nothing escaped to the console
    # messages carry codes from the master list in lcsolver.core
    from lcsolver.core import codes
    assert any(codes.HOLOGRAPHIC_ACTIVE in m for m in sol.messages)
    assert all(m.startswith('[LC-') for m in sol.messages
               if 'holographic' in m or 'sensitivities' in m)
    # sol.report is the Report text; sol.objective carries units
    assert 'auto-detected as a geometric program' in sol.report
    assert sol.objective.to('m').magnitude == pytest.approx(1.0, rel=1e-6)
    assert sol.optimality_status is True
    # the summary ends with a Post Solve Report owning status + messages
    text = sol.summary()
    assert 'Post Solve Report' in text
    post = text.split('Post Solve Report')[1]
    assert 'Status: optimal' in post
    assert codes.HOLOGRAPHIC_ACTIVE in post
    assert 'Status:' not in text.split('Objective')[0]  # not in top Report


def test_summary_report_section():
    """The Report says what was detected/prescribed and what ran."""
    f = _well_posed()
    sol = solve(f)
    text = sol.summary()
    assert 'Report' in text
    assert 'auto-detected as a geometric program' in text
    assert 'Solved with' in text
    # prescribed route reports itself as prescribed, not auto-detected
    f2 = _well_posed()
    sol2 = solve(f2, solver='ipopt', diagnostics='off')
    text2 = sol2.summary()
    assert "prescribed (solver='ipopt')" in text2
    assert 'auto-detected' not in text2.split('Objective')[0] \
        or 'bypassing auto-detection' in text2
    # a model solved outside solve() has no report and no section
    from lcsolver.objects.solution import Solution
    assert Solution().summary().count('Report') == 0


def test_named_accessors_with_units():
    """variables()/constants()/sensitivities()/dimensioned_sensitivities().

    One convention: no argument -> full flat dict keyed by display name;
    a string -> the single quantity; a list -> a dict of those names.
    Values carry units; dimensionless values come back as plain floats.
    """
    f = Formulation()
    x = f.Variable(name='x', guess=2.0, units='m', bounds=[0.1, 10.0])
    y = f.Variable(name='y', guess=2.0, units='m', bounds=[0.1, 10.0])
    c = f.Constant(name='c', value=4.0, units='m^2', description='area')
    f.Objective(x + y)
    f.ConstraintList([x * y >= c])
    sol = solve(f)

    allv = sol.variables()
    assert set(allv) == {'x', 'y'}
    # pint quantities: readable in a printed dict, convertible, strippable
    xq = sol.variables('x')
    assert xq.to('ft').magnitude == pytest.approx(2.0 / 0.3048, rel=1e-6)
    assert 'meter' in str(allv['x'].units)
    assert set(sol.variables(['x'])) == {'x'}
    assert sol.constants('c').to('m^2').magnitude == pytest.approx(4.0)
    # log-log sensitivity of 2*sqrt(c) is 1/2 ...
    assert sol.sensitivities('c') == pytest.approx(0.5, rel=1e-6)
    # ... and the dimensioned d(obj)/dc = 1/sqrt(c) = 0.5 per metre
    ds = sol.dimensioned_sensitivities('c')
    assert ds.to('1/m').magnitude == pytest.approx(0.5, rel=1e-6)
    with pytest.raises(KeyError):
        sol.variables('nope')
    # the raw dict key is untouched by the accessor of the same name
    assert isinstance(sol['sensitivities'], dict)


def test_quality_checks_attach_on_converged_solve():
    res = solve(_well_posed())
    assert 'quality' in res
    assert set(res['quality']) == {'degenerate', 'cancelling', 'at_floor'}


def test_postsolve_check_refuses_unsolved_model():
    with pytest.raises(ValueError):
        postsolve_check(_well_posed())


def test_split_checks_cover_their_halves():
    f = _well_posed()
    pre = presolve_check(f)
    assert pre.structure                 # classification always present
    solve(f)
    post = postsolve_check(f)
    assert isinstance(post.at_floor, list)
    # the combined wrapper still works and carries both halves
    from lcsolver.presolve.reductions import optimization_check
    both = optimization_check(f)
    assert both.structure
    assert isinstance(both.at_floor, list)
