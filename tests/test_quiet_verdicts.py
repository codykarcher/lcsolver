#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""quiet=True hides chatter, not verdicts; SP sensitivities are labelled local.

Both behaviours come from one incident (lcsailboat IACC deck, 2026-09-27): an SIA solve hit its
iteration cap, lcsolver raised LC-W203 and LC-W302 (stationarity residual 0.10), quiet=True filed
both in result['messages'], nobody read them, and the solve's duals were published.  Separately,
the SIA route never recorded its structure on the model, so ``sensitivities()`` could not set its
'approximate' flag on the default signomial path.
"""
import warnings

import pytest

pyo = pytest.importorskip('pyomo.environ')

from lcsolver import Formulation                                   # noqa: E402
from lcsolver.solvers import solver as solver_module               # noqa: E402


class _Result:
    """The shape `_solve_sp` reads off a bridge result: a non-converged SIA."""

    def __init__(self, status):
        self.x = [1.0, 1.0]
        self.objective = 2.0
        self.converged = False
        self.status = status
        self.iterations = 400
        self.max_violation = 1.5e-3
        self.stationarity = 1.0e-1
        self.complementarity = 0.0
        self.phase1_feasible = None
        self.infeasibility_report = None


def _sp():
    f = Formulation()
    x = f.Variable('x', 1.0, '-', 'x')
    y = f.Variable('y', 1.0, '-', 'y')
    f.Objective(x)
    f.ConstraintList([x >= y / (1 + y), y >= 0.5, x <= 10.0, y <= 10.0])
    return f


def test_quiet_still_rewarns_a_non_converged_solve(monkeypatch):
    from lcsolver.solvers.sequential import bridge
    monkeypatch.setattr(bridge, 'solve_sia', lambda *a, **k: _Result('did not converge within 400 iterations'))
    monkeypatch.setattr(solver_module, 'any_ipopt_available', lambda: True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = solver_module.solve(_sp(), sensitivities=False)          # quiet=True by default
    shown = [str(w.message) for w in caught]
    assert any('[LC-W203]' in m for m in shown), shown
    assert any('[LC-W203]' in m for m in res['messages'])            # and still recorded


def test_quiet_stays_quiet_on_a_clean_solve():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = solver_module.solve(_sp())
    assert res.optimality_status
    assert not [w for w in caught if str(w.message).startswith(solver_module._QUIET_REWARN)]


def test_sia_sensitivities_are_labelled_local():
    f = _sp()
    res = solver_module.solve(f)
    detail = res.get('sensitivity_detail') or {}
    if 'sia' not in str(res.get('problem_structure', '')):
        pytest.skip(f"routed to {res.get('problem_structure')!r}, not SIA")
    assert detail.get('approximate') is True
