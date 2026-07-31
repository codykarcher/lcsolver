#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""The feasibility solve, as a thing a user can call and pass on.

The elastic Phase I has been inside the SIA solver for a while, reachable only
by hand-building a low-level Problem. `feasibility(f)` takes the formulation,
and its result goes straight back into `solve(f, start=result)` -- which is the
point. A hard model started from a feasible point is a different problem from
one started at the author's guesses.
"""
import numpy as np
import pyomo.environ as pyo
import pytest

from edi import Formulation, feasibility
from edi.solvers.solver import solve


def _satisfiable():
    """min x + y s.t. x*y >= 2. The guesses (1, 1) are NOT feasible."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    y = f.Variable(name='y', guess=1.0, units='m', description='y')
    A = f.Constant(name='A', value=2.0, units='m^2', description='area')
    f.Objective(x + y)
    f.ConstraintList([x * y >= A])
    return f


def _unsatisfiable():
    """The same, but with both variables capped below what the product needs."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    y = f.Variable(name='y', guess=1.0, units='m', description='y')
    A = f.Constant(name='A', value=2.0, units='m^2', description='area')
    f.Objective(x + y)
    f.ConstraintList([x * y >= A,
                      x <= 1.0 * pyo.units.m, y <= 1.0 * pyo.units.m])
    return f


def test_finds_a_feasible_point_and_it_really_is_feasible():
    f = _satisfiable()
    r = feasibility(f)
    assert r.feasible and bool(r) is True
    assert not r.blocking
    r.apply(f)
    # the constraint the guesses violated is now satisfied
    assert pyo.value(f.x) * pyo.value(f.y) >= 2.0 - 1e-6


def test_names_the_blocking_rows_when_there_is_no_point():
    f = _unsatisfiable()
    r = feasibility(f)
    assert not r.feasible and bool(r) is False
    assert len(r.blocking) >= 1
    _row, slack, names = r.blocking[0]
    assert slack > 0
    assert set(names) == {'x', 'y'}
    assert 'INFEASIBLE' in r.summary()


def test_result_is_passable_to_solve():
    """The composition this exists for."""
    f = _satisfiable()
    r = feasibility(f)
    solve(f, start=r, sensitivities=False)
    assert float(f.solution.objective) == pytest.approx(2 * 2 ** 0.5, rel=1e-6)


def test_start_accepts_a_plain_array():
    f = _satisfiable()
    n = len(feasibility(f).x)
    solve(f, start=np.full(n, 2.0), sensitivities=False)
    assert float(f.solution.objective) == pytest.approx(2 * 2 ** 0.5, rel=1e-6)


def test_a_short_start_is_refused_rather_than_padded():
    """Silently padding would start the solve somewhere nobody asked for."""
    f = _satisfiable()
    with pytest.raises(ValueError, match='must be in'):
        solve(f, start=np.array([1.0]), sensitivities=False)


def test_x_is_indexed_like_the_model():
    from edi.preconditioner.structureDetector import structure_detector
    from edi.preconditioner.unitCorrector import unit_corrector
    f = _satisfiable()
    st = structure_detector(unit_corrector(f))
    assert len(feasibility(f).x) == len(st['variables'])
