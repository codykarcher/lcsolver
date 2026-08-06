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
            solve(_unbounded(), diagnostics='warn')
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
