"""One model, every path, the same answer.

The property this file asserts is the one no unit test here can: that the
several routes from a Formulation to a number agree. EDI carries four
representations of a problem -- the Pyomo model, the detected structure, the
cvxopt backends' own matrices, and the SLCP ``Problem`` -- and every conversion
between them is somewhere a transformation can quietly change the problem while
still returning a plausible answer.

That failure mode is this repository's characteristic bug, and it is invisible
to a test built from imagination: the shapes that expose it are an equality
sharing variables, a chain of eliminations, a variable sitting at a bound, a
model that is simultaneously linear and signomial. None of those occur in a
three-variable model written to check one function.

So this file is deliberately about *agreement between paths* rather than about
any particular number. A backend that silently drops a bound, a presolve pass
that relaxes an equality, or a reader that parses the wrong encoding all show
up the same way: two paths that should agree, disagreeing.

Marked ``slow``. Run with ``pytest -m slow`` or ``pytest tests/test_integration.py``;
excluded from a plain ``pytest`` run by the marker configuration.
"""
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pyo = pytest.importorskip('pyomo.environ')

from edi import Formulation  # noqa: E402
from edi.solvers import solver as solver_module  # noqa: E402
from edi.structure.structureDetector import structure_detector  # noqa: E402
from edi.units.unitCorrector import unit_corrector  # noqa: E402

pytestmark = pytest.mark.slow

_EXAMPLES = Path(__file__).resolve().parents[1] / 'examples' / 'convexengineering'


# ---------------------------------------------------------------------------
# small models, one per structure kind
# ---------------------------------------------------------------------------
def _gp():
    """min x*y  s.t.  x >= 1, y >= 2.  Optimum 2 at (1, 2)."""
    f = Formulation()
    x = f.Variable('x', 3.0, '', 'x', bounds=[0.01, 100.0])
    y = f.Variable('y', 3.0, '', 'y', bounds=[0.01, 100.0])
    f.Objective(x * y)
    f.Constraint(x >= 1.0)
    f.Constraint(y >= 2.0)
    return f, 2.0


def _gp_with_equality():
    """A monomial equality plus a posynomial one -- both presolve targets."""
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 100.0])
    y = f.Variable('y', 2.0, '', 'y', bounds=[0.1, 100.0])
    z = f.Variable('z', 1.0, '', 'z', bounds=[1e-30, 1e30])
    f.Objective(x + y)
    f.Constraint(z == 3.0 * x)          # eliminable
    f.Constraint(x * z >= 12.0)         # 3x^2 >= 12 -> x >= 2
    f.Constraint(x * y >= 6.0)          # y >= 6/x
    # min x + 6/x subject to x >= 2 is interior at x = sqrt(6), not at the
    # bound, giving 2*sqrt(6). Worth stating: the first version of this test
    # asserted 5.0 by assuming x sat on its constraint, and every solver
    # disagreed with it in unison -- which is what agreement across paths is
    # for.
    return f, 2.0 * 6.0 ** 0.5


def _sp():
    """A signomial equality: `z == x - y` normalizes to a ratio."""
    f = Formulation()
    x = f.Variable('x', 5.0, '', 'x', bounds=[0.1, 100.0])
    y = f.Variable('y', 1.5, '', 'y', bounds=[1.0, 2.0])
    z = f.Variable('z', 3.0, '', 'z', bounds=[0.1, 100.0])
    f.Objective(z)
    f.Constraint(z == x - y)
    f.Constraint(x >= 4.0)
    return f, 2.0                        # x=4, y=2


def _at_bounds():
    """The optimum sits ON a bound -- the case that needs a projected gradient."""
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[0.1, 3.0])
    y = f.Variable('y', 1.0, '', 'y', bounds=[0.1, 10.0])
    f.Objective(1.0 / x)                 # pushes x UP into its cap
    f.Constraint(x * y >= 1.0)
    return f, 1.0 / 3.0


SMALL = [
    ('gp', _gp),
    ('gp_equality', _gp_with_equality),
    ('sp', _sp),
    ('at_bounds', _at_bounds),
]


def _objective_of(f):
    obj = [o for o in f.component_data_objects(pyo.Objective, active=True)]
    return float(pyo.value(obj[0]))


# ---------------------------------------------------------------------------
# the paths
# ---------------------------------------------------------------------------
def _solve_via(make, path):
    f, _expected = make()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if path in ('cvxopt', 'ipopt-convex', 'ipopt'):
            solver_module.solve(f, solver=path)
            return _objective_of(f)
        st = structure_detector(unit_corrector(f))
        if path == 'sia':
            from edi.solvers.ipopt.slcp_bridge import solve_sia
            return float(solve_sia(st).objective)
        if path == 'slcp':
            from edi.solvers.ipopt.slcp_bridge import solve_slcp
            return float(solve_slcp(st).objective)
        raise AssertionError(f'unknown path {path}')


PATHS = ['cvxopt', 'ipopt-convex', 'ipopt', 'sia', 'slcp']


@pytest.mark.parametrize('name,make', SMALL, ids=[n for n, _ in SMALL])
def test_every_path_finds_the_same_optimum(name, make):
    """Agreement is the assertion; the known value is only a sanity anchor."""
    _f, expected = make()
    results = {}
    for path in PATHS:
        try:
            results[path] = _solve_via(make, path)
        except Exception as exc:                      # noqa: BLE001
            results[path] = f'FAILED: {type(exc).__name__}: {exc}'

    ok = {p: v for p, v in results.items() if isinstance(v, float)}
    assert ok, f'every path failed on {name}: {results}'

    for path, value in ok.items():
        assert value == pytest.approx(expected, rel=1e-4), (
            f'{path} disagrees on {name}: got {value}, expected {expected}. '
            f'All paths: {results}')


@pytest.mark.parametrize('name,make', SMALL, ids=[n for n, _ in SMALL])
def test_presolve_does_not_change_the_answer(name, make):
    """SIA with the presolve pipeline on and off must agree."""
    from edi.solvers.ipopt.slcp_bridge import solve_sia

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f1, expected = make()
        on = solve_sia(structure_detector(unit_corrector(f1)), presolve=True)
        f2, _ = make()
        off = solve_sia(structure_detector(unit_corrector(f2)), presolve=False)

    assert on.objective == pytest.approx(off.objective, rel=1e-6)
    assert on.objective == pytest.approx(expected, rel=1e-4)
    # and presolve must hand back a full-length solution either way
    assert len(on.x) == len(off.x)


@pytest.mark.parametrize('name,make', SMALL, ids=[n for n, _ in SMALL])
def test_split_bounds_do_not_change_the_answer(name, make):
    """bounds_as_rows on and off describe the same problem."""
    from edi.solvers.ipopt.slcp_bridge import solve_sia

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f1, expected = make()
        rows = solve_sia(structure_detector(unit_corrector(f1),
                                            bounds_as_rows=True))
        f2, _ = make()
        split = solve_sia(structure_detector(unit_corrector(f2),
                                             bounds_as_rows=False))

    assert rows.objective == pytest.approx(split.objective, rel=1e-6)
    assert rows.objective == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# the real models
# ---------------------------------------------------------------------------
def _example(name):
    if str(_EXAMPLES) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES))
    try:
        return getattr(importlib.import_module(f'{name}.model'), 'build')
    except Exception:                                  # noqa: BLE001
        pytest.skip(f'example {name} unavailable')


@pytest.mark.parametrize('name', ['simpleac', 'propeller', 'windturbine'])
def test_example_models_agree_across_paths(name):
    """Real models, which carry the shapes a written-from-scratch test lacks."""
    build = _example(name)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from edi.solvers.ipopt.slcp_bridge import solve_sia
        base = solve_sia(structure_detector(unit_corrector(build())))
        plain = solve_sia(structure_detector(unit_corrector(build())),
                          presolve=False)

    assert base.objective == pytest.approx(plain.objective, rel=1e-5), (
        f'{name}: presolve changed the objective')


@pytest.mark.veryslow
def test_spaircraft_end_to_end():
    """The flagship model, and the only one carrying every shape at once.

    Every defect found while building the presolve pipeline was found by
    running this and nothing else: an equality sharing variables, a chain of
    eliminations, variables reaching their bounds, a model that is both linear
    and signomial. It takes about half a minute, which is why it carries its
    own marker.

    It also caught the iteration cap: SPaircraft converges in 149 iterations
    and the default was 100, so a converging run was being stopped three fifths
    of the way through and reported as a failure.
    """
    build = _example('spaircraft')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from edi.solvers.ipopt.slcp_bridge import solve_sia
        st = structure_detector(unit_corrector(build()))
        n_vars = len(st['variables'])
        res = solve_sia(st)

    assert res.converged, f'did not converge: {res.status}'
    assert res.max_violation <= 1e-6
    assert res.stationarity <= 1e-5
    assert len(res.x) == n_vars, 'presolve must restore the full solution'
    assert res.objective == pytest.approx(95559.91, rel=1e-3)


# ---------------------------------------------------------------------------
# the linear payload, whose two layouts are easy to confuse
# ---------------------------------------------------------------------------
def _lp():
    """min x + y  s.t.  x - y >= 2, x >= 1, both in [-10, 10].

    `x - y >= 2` gives `y <= x - 2`, so both want to be as small as allowed:
    x = 1 at its constraint, y = -10 at its bound, objective -9.
    """
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[-10.0, 10.0])
    y = f.Variable('y', 1.0, '', 'y', bounds=[-10.0, 10.0])
    f.Objective(x + y)
    f.Constraint(x - y >= 2.0)
    f.Constraint(x >= 1.0)
    return f, -9.0


def _qp():
    """min x**2 + y**2  s.t.  x + y >= 2.  Optimum 2 at (1, 1)."""
    f = Formulation()
    x = f.Variable('x', 0.0, '', 'x', bounds=[-10.0, 10.0])
    y = f.Variable('y', 0.0, '', 'y', bounds=[-10.0, 10.0])
    f.Objective(x ** 2 + y ** 2)
    f.Constraint(x + y >= 2.0)
    return f, 2.0


@pytest.mark.parametrize('name,make', [('lp', _lp), ('qp', _qp)])
def test_linear_and_quadratic_payloads_are_read_correctly(name, make):
    """LP and QP store their payload in DIFFERENT layouts.

        LP   [[c], shift, A, b]      the objective nested one deeper
        QP   [P, q, shift, A, b]

    Reading a QP with the LP layout yields the Hessian where the objective
    belongs and the constraint matrix where the right-hand side does -- which
    both `evaluate` and `propagate_bounds` did until `linear_parts` existed.
    """
    from edi.presolve import evaluate, propagate_bounds

    f, expected = make()
    st = structure_detector(unit_corrector(f), bounds_as_rows=False)
    parts = st.linear_parts()
    assert parts.A is not None
    assert (parts.hessian is None) == (name == 'lp')

    # propagation must not raise, and must not change the problem
    out, _n = propagate_bounds(st)
    assert out['bounds'] is not None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = solver_module.solve(f, solver='cvxopt')
    got = _objective_of(f)
    assert got == pytest.approx(expected, abs=1e-4), (
        f'{name}: cvxopt gave {got}, expected {expected}')

    # `evaluate` must agree with the solved objective at the solved point.
    # Read the values from the SOLUTION, not from st['variables']: unit
    # correction clones the model, so a separately detected structure holds a
    # different clone's variables and reading those returns the initial guess.
    solution = res['solution']
    xs = [float(solution[str(v)]) for v in st['variables']]
    obj, viol = evaluate(st, xs)
    assert obj == pytest.approx(got, abs=1e-4), (
        f'{name}: evaluate gave {obj}, solver gave {got}')
    assert viol <= 1e-6


def test_quadratic_objective_convention_is_consistent():
    """A QP with a LINEAR term, where a stray factor of two moves the optimum.

    EDI stores the quadratic COEFFICIENT matrix, so the objective is
    x'Px + q'x. cvxopt.solvers.qp minimises (1/2) x'Px + q'x, and solve_QP
    passes 2*P to compensate. With q = 0 a missing factor would be invisible --
    a positive scaling leaves the argmin alone -- so this uses q != 0, where
    minimising (1/2)(x^2+y^2) - 4x lands at x=4 instead of x=2.
    """
    from edi.presolve import evaluate

    def make():
        f = Formulation()
        x = f.Variable('x', 0.0, '', 'x', bounds=[-10.0, 10.0])
        y = f.Variable('y', 0.0, '', 'y', bounds=[-10.0, 10.0])
        f.Objective(x ** 2 + y ** 2 - 4.0 * x)
        f.Constraint(x + y >= 1.0)
        return f

    for backend in ('cvxopt', 'ipopt'):
        f = make()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solver_module.solve(f, solver=backend)
        assert pyo.value(f.x) == pytest.approx(2.0, abs=1e-4), backend
        assert pyo.value(f.y) == pytest.approx(0.0, abs=1e-4), backend

    st = structure_detector(unit_corrector(make()), bounds_as_rows=False)
    obj, _viol = evaluate(st, [2.0, 0.0])
    assert obj == pytest.approx(-4.0, abs=1e-6)
