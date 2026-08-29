#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""``options`` is overloaded: an IPOPT dict on the convex paths, an
SIAOptions object on the signomial path.  Before the type-routing fix, an
SIAOptions passed to a solve that resolved to a pure GP crashed the
structured backend into the raw-IPOPT fallback (LC-W201, the observed
failure: TypeError "'SIAOptions' object is not iterable" while the GP
backend iterated it as an options dict).  These tests pin the routing:
mismatched objects are dropped, ``sia_options`` is the unambiguous
spelling, and no combination degrades a GP solve off the structured path.
"""
import warnings

import pyomo.environ as pyo
import pytest

from lcsolver import Formulation
from lcsolver.solvers.sequential.sia import SIAOptions
from lcsolver.solvers.solver import solve


def _gp():
    """min x + y s.t. x*y >= 2 -- a pure GP."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    y = f.Variable(name='y', guess=1.0, units='m', description='y')
    A = f.Constant(name='A', value=2.0, units='m^2', description='area')
    f.Objective(x + y)
    f.ConstraintList([x * y >= A])
    return f


def _sp():
    """min x s.t. x + x^2 >= 2 + 0.5 x -- carries a signomial row."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='-', description='x')
    f.Objective(x)
    f.ConstraintList([x + x**2 >= 2. + 0.5 * x])
    return f


def _no_fallback(result):
    msgs = result.get('messages', []) or []
    return not any('LC-W201' in str(m) for m in msgs)


def test_gp_with_sia_options_stays_structured():
    """An SIAOptions on a pure GP is dropped, not a backend crash."""
    f = _gp()
    res = solve(f, options=SIAOptions(max_iterations=2000))
    assert res.optimality_status
    assert _no_fallback(res)
    assert abs(pyo.value(f.x) * pyo.value(f.y) - 2.0) < 1e-5


def test_gp_with_sia_options_alias_stays_structured():
    """The unambiguous spelling is likewise harmless on a GP."""
    f = _gp()
    res = solve(f, sia_options=SIAOptions(max_iterations=2000))
    assert res.optimality_status
    assert _no_fallback(res)


def test_sp_accepts_sia_options_both_spellings():
    """Both spellings reach SIA on a signomial program."""
    for kw in ({'options': SIAOptions(max_iterations=600)},
               {'sia_options': SIAOptions(max_iterations=600)}):
        f = _sp()
        res = solve(f, **kw)
        assert _no_fallback(res)
        # x + x^2 - 0.5x = 2  ->  x = (-0.5 + sqrt(0.25 + 8)) / 2
        assert abs(pyo.value(f.x) - 1.186141) < 1e-3


def test_sp_drops_ipopt_dict_options():
    """An IPOPT-style dict aimed at a convex backend does not reach SIA."""
    f = _sp()
    res = solve(f, options={'max_iter': 500, 'tol': 1e-9})
    assert _no_fallback(res)
    assert abs(pyo.value(f.x) - 1.186141) < 1e-3


def test_gp_ipopt_dict_still_works():
    """The historical convex-path meaning is untouched."""
    f = _gp()
    res = solve(f, options={'max_iter': 500})
    assert res.optimality_status
    assert _no_fallback(res)
