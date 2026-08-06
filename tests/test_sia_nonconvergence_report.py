#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""What `solve` says when SIA stops without a certificate.

A non-converged SIA still returns a point, and that point looks exactly like a
converged one to anything reading `res['x']`. The whole defence is the warning
built in `_solve_sp`: it has to say that the answer is the best iterate rather
than an optimum, how far from feasible it is, and -- keyed on how it failed --
what to do about it.

None of that had a test. The remedies are four separate branches selected by
substrings of the status, so a status string reworded anywhere upstream would
silently stop matching and the advice would just disappear from the message.

The failure is induced by substituting the bridge's `solve_sia` rather than by
finding a model that genuinely fails: the branches are keyed on the status
text, so the status text is the input under test, and a real diverging model
would pin this to whichever way that particular model happens to break.
"""
import warnings

import pytest

pyo = pytest.importorskip('pyomo.environ')

from lcsolver import Formulation                                   # noqa: E402
from lcsolver.solvers import solver as solver_module               # noqa: E402


class _Result:
    """The shape `_solve_sp` reads off a bridge result."""

    def __init__(self, status, infeasibility_report=None):
        self.x = [1.0, 1.0]
        self.objective = 2.0
        self.converged = False
        self.status = status
        self.iterations = 7
        self.max_violation = 1.5e-3
        self.stationarity = 4.25e-2
        self.complementarity = 0.0
        if infeasibility_report is not None:
            self.infeasibility_report = infeasibility_report


def _sp():
    """A signomial, so `solve` routes through `_solve_sp` at all."""
    f = Formulation()
    x = f.Variable('x', 1.0, '-', 'x', bounds=[1e-3, 10.0])
    y = f.Variable('y', 1.0, '-', 'y', bounds=[1e-3, 10.0])
    f.Objective(x)
    f.ConstraintList([x >= y / (1 + y), y >= 0.5])
    return f


def _solve_with_status(status, report=None, monkeypatch=None):
    from lcsolver.solvers.sequential import bridge

    monkeypatch.setattr(bridge, 'solve_sia',
                        lambda *a, **k: _Result(status, report))
    # These tests are about what the report says, not about what is installed:
    # the SIA solve itself is faked above. Without this, a machine with no
    # IPOPT routes the signomial program to cvxopt instead, the fake never
    # runs, and the test fails for a reason it is not testing.
    monkeypatch.setattr(solver_module, '_ipopt_available', lambda: True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = solver_module.solve(_sp(), sensitivities=False, quiet=False)
    messages = [str(w.message) for w in caught if 'LC-W203' in str(w.message)]
    return res, messages


def test_a_non_converged_solve_says_so(monkeypatch):
    """The number comes back, but labelled as an iterate rather than an optimum."""
    res, messages = _solve_with_status('trust region collapsed',
                                       monkeypatch=monkeypatch)
    assert messages, 'a non-converged SIA must warn'
    text = messages[0]
    assert 'did not converge' in text
    assert 'not a certified optimum' in text
    # the two numbers a reader needs to judge how bad it is
    assert '1.50e-03' in text                    # max violation
    assert '4.25e-02' in text                    # stationarity residual

    # and the result records it rather than passing the point off as optimal
    assert res['converged'] is False
    assert res['status'] == 'trust region collapsed'
    assert res['max_violation'] == pytest.approx(1.5e-3)


def test_a_converged_solve_is_silent(monkeypatch):
    """The control: the warning must not fire on a normal solve."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solver_module.solve(_sp(), sensitivities=False, quiet=False)
    assert not [w for w in caught if 'LC-W203' in str(w.message)]


@pytest.mark.parametrize('status,expected', [
    ('phase 1 failed', 'blocking constraints'),
    ('did not converge within 100 iterations', 'max_iterations'),
    ('trust region collapsed', 'condense_numerator'),
    ('sub-problem failure at iteration 3', 'variable scaling'),
])
def test_the_advice_is_keyed_on_how_it_failed(status, expected, monkeypatch):
    """Each failure mode names its own remedy, not a generic one."""
    _res, messages = _solve_with_status(status, monkeypatch=monkeypatch)
    assert messages
    assert 'What to try:' in messages[0]
    assert expected in messages[0], messages[0]


def test_an_unrecognised_status_offers_no_advice_rather_than_wrong_advice(
        monkeypatch):
    """No branch matches, so the message stops after the diagnosis."""
    _res, messages = _solve_with_status('something new', monkeypatch=monkeypatch)
    assert messages
    assert 'did not converge' in messages[0]
    assert 'What to try:' not in messages[0]


def test_a_phase_1_failure_carries_the_infeasibility_report(monkeypatch):
    """The rows that could not be closed are the actual answer to 'why'."""
    report = 'row W_fuel >= ...  unclosable by 3.2e+01'
    res, messages = _solve_with_status('phase 1 failed', report=report,
                                       monkeypatch=monkeypatch)
    assert res['infeasibility_report'] == report
    assert report in messages[0]
    assert 'Where feasibility fails' in messages[0]


def test_phase_1_without_a_report_still_advises(monkeypatch):
    """A missing report must not cost the remedy that goes with it."""
    res, messages = _solve_with_status('phase 1 failed', report=None,
                                       monkeypatch=monkeypatch)
    assert 'infeasibility_report' not in res
    assert 'Where feasibility fails' not in messages[0]
    assert 'blocking constraints' in messages[0]


def test_an_unknown_sp_method_is_rejected():
    """Neither 'sia' nor 'pccp' must fail loudly rather than pick one."""
    with pytest.raises(ValueError, match="sp_method must be"):
        solver_module.solve(_sp(), sp_method='newton')
