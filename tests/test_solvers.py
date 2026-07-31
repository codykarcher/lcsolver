#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests for solution write-back and the IPOPT interface."""

import warnings

import pytest

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.environ import units
from pyomo.common.dependencies import attempt_import
from pyomo.core.base.units_container import pint_available

cvxopt, cvxopt_available = attempt_import("cvxopt")

try:
    from edi import Formulation

    formulation_available = True
except Exception:
    formulation_available = False


def _linear_model():
    """Trivial LP with a known answer: minimize x + y s.t. x>=1, y>=2 -> (1, 2)."""
    f = Formulation()
    f.Variable(name='x', guess=5.0, units='m', description='x')
    f.Variable(name='y', guess=5.0, units='m', description='y')
    f.Objective(f.x + f.y)
    f.ConstraintList([f.x >= 1.0 * units.m, f.y >= 2.0 * units.m])
    return f


def _rosenbrock():
    """Non-convex NLP; not LP/QP/GP/SP, so only IPOPT can solve it."""
    f = Formulation()
    f.Variable(name='x', guess=-1.0, units='', description='x')
    f.Variable(name='y', guess=2.0, units='', description='y')
    f.Objective((1 - f.x) ** 2 + 100 * (f.y - f.x ** 2) ** 2)
    f.ConstraintList([
        f.x >= -2.0 * units.dimensionless, f.x <= 2.0 * units.dimensionless,
        f.y >= -2.0 * units.dimensionless, f.y <= 3.0 * units.dimensionless,
    ])
    return f


@unittest.skipIf(not formulation_available, 'Formulation import failed')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
class TestWriteBack(unittest.TestCase):
    """A solve must leave the solution ON the model, not only in the result dict.

    This is a regression guard: previously cvxopt_solve returned a raw solver
    dictionary and never applied it, so pyo.value(m.x) still returned the initial
    guess after a successful solve.
    """

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_cvxopt_writes_solution_onto_model(self):
        from edi.solvers.solver import cvxopt_solve

        f = _linear_model()
        self.assertAlmostEqual(pyo.value(f.x), 5.0)      # the guess
        res = cvxopt_solve(f)
        self.assertEqual(res['status'], 'optimal')
        self.assertAlmostEqual(pyo.value(f.x), 1.0, places=4)
        self.assertAlmostEqual(pyo.value(f.y), 2.0, places=4)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_cvxopt_returns_solution_dict(self):
        from edi.solvers.solver import cvxopt_solve

        f = _linear_model()
        res = cvxopt_solve(f)
        self.assertIn('solution', res)
        self.assertAlmostEqual(res['solution']['x'], 1.0, places=4)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_writeback_can_be_disabled(self):
        from edi.solvers.solver import cvxopt_solve

        f = _linear_model()
        cvxopt_solve(f, write_back=False)
        self.assertAlmostEqual(pyo.value(f.x), 5.0)      # untouched

    def test_structure_detector_publishes_variable_order(self):
        from edi.presolve.structureDetector import structure_detector
        from edi.presolve.unitCorrector import unit_corrector

        f = _linear_model()
        s = structure_detector(unit_corrector(f))
        self.assertIn('variables', s)
        self.assertEqual([v.name for v in s['variables']], ['x', 'y'])


def _ipopt_route_available(route):
    try:
        from edi.solvers.ipopt.ipopt_solver_interface import _executable_available
        return _executable_available('ipopt' if route == 'pyomo' else 'cyipopt')
    except Exception:
        return False


@unittest.skipIf(not formulation_available, 'Formulation import failed')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
class TestIpopt(unittest.TestCase):
    """IPOPT interface. Each route is skipped if that backend is unavailable."""

    @unittest.skipIf(not _ipopt_route_available('pyomo'),
                     'the ipopt executable is not available')
    def test_ipopt_pyomo_route(self):
        from edi.solvers.ipopt import ipopt_solve

        f = _rosenbrock()
        r = ipopt_solve(f, method='pyomo')
        self.assertEqual(r['solver'], 'pyomo')
        self.assertAlmostEqual(pyo.value(f.x), 1.0, places=4)
        self.assertAlmostEqual(pyo.value(f.y), 1.0, places=4)

    @unittest.skipIf(not _ipopt_route_available('cyipopt'),
                     'cyipopt is not available')
    def test_ipopt_cyipopt_route(self):
        from edi.solvers.ipopt import ipopt_solve

        f = _rosenbrock()
        r = ipopt_solve(f, method='cyipopt')
        self.assertEqual(r['solver'], 'cyipopt')
        self.assertAlmostEqual(pyo.value(f.x), 1.0, places=4)

    @unittest.skipIf(not (_ipopt_route_available('pyomo')
                          or _ipopt_route_available('cyipopt')),
                     'no IPOPT backend available')
    def test_ipopt_auto_route_and_solution(self):
        from edi.solvers.ipopt import ipopt_solve

        f = _rosenbrock()
        r = ipopt_solve(f)
        self.assertIn(r['solver'], ('pyomo', 'cyipopt'))
        self.assertIn('solution', r)
        self.assertAlmostEqual(r['objective'], 0.0, places=6)

    def test_ipopt_rejects_bad_method(self):
        from edi.solvers.ipopt import ipopt_solve

        f = _rosenbrock()
        self.assertRaises(ValueError, ipopt_solve, f, **{'method': 'nonsense'})


def _unit_circle_model():
    """min x + y  s.t.  z == x**2 + y**2 (via a BLACK BOX), z <= 1.

    The analytic optimum is on the unit circle at x = y = -1/sqrt(2), z = 1.
    The black box declares its inputs/outputs in feet while the model variables
    are meters, so this also exercises unit conversion across the interface.

    Bounds on x and y are deliberate: a grey-box model is evaluated wherever the
    optimizer proposes, so an unbounded input can send the external code
    somewhere it cannot be evaluated. Without them IPOPT diverges here.
    """
    from edi import BlackBoxFunctionModel

    class UnitCircle(BlackBoxFunctionModel):
        def __init__(self):
            super().__init__()
            self.description = 'z = x**2 + y**2'
            self.inputs.append(name='x', units='ft', description='x')
            self.inputs.append(name='y', units='ft', description='y')
            self.outputs.append(name='z', units='ft**2', description='z')
            self.availableDerivative = 1

        def BlackBox(self, x, y):
            x = pyo.value(units.convert(x, self.inputs['x'].units))
            y = pyo.value(units.convert(y, self.inputs['y'].units))
            return (x ** 2 + y ** 2) * units.ft ** 2, [
                2 * x * units.ft,
                2 * y * units.ft,
            ]

    f = Formulation()
    f.Variable(name='x', guess=0.5, units='m', description='x', bounds=[-2.0, 2.0])
    f.Variable(name='y', guess=0.5, units='m', description='y', bounds=[-2.0, 2.0])
    f.Variable(name='z', guess=1.0, units='m^2', description='z', bounds=[0.0, 1.0])
    f.Constant(name='c', value=1.0, units='', description='c', size=2)
    f.Objective(f.c[0] * f.x + f.c[1] * f.y)
    f.ConstraintList([[f.z, '==', [f.x, f.y], UnitCircle()],
                      f.z <= 1 * units.m ** 2])
    return f


@unittest.skipIf(not formulation_available, 'Formulation import failed')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
class TestIpoptBlackBox(unittest.TestCase):
    """Black-box (grey-box) constraints. This is EDI's headline capability and the
    only case the AMPL-based route cannot handle, so it is covered explicitly."""

    def test_greybox_is_detected(self):
        from edi.solvers.ipopt.ipopt_solver_interface import _has_greybox

        self.assertTrue(_has_greybox(_unit_circle_model()))
        self.assertFalse(_has_greybox(_rosenbrock()))

    def test_pyomo_route_refuses_greybox(self):
        """The AMPL route cannot evaluate a Python black box; it must say so."""
        from edi.solvers.ipopt import ipopt_solve

        f = _unit_circle_model()
        self.assertRaises(RuntimeError, ipopt_solve, f, **{'method': 'pyomo'})

    @unittest.skipIf(not _ipopt_route_available('cyipopt'), 'cyipopt is not available')
    def test_greybox_constraint_is_enforced(self):
        """Regression guard: the solution must satisfy the BLACK-BOX constraint.

        If the grey-box constraint were dropped, x and y would simply run to their
        -2 bounds. Landing on the unit circle proves the external model is being
        evaluated and enforced.
        """
        from edi.solvers.ipopt import ipopt_solve

        f = _unit_circle_model()
        r = ipopt_solve(f)
        self.assertEqual(r['solver'], 'cyipopt')

        xv, yv, zv = pyo.value(f.x), pyo.value(f.y), pyo.value(f.z)
        self.assertAlmostEqual(xv, -(2 ** -0.5), places=4)
        self.assertAlmostEqual(yv, -(2 ** -0.5), places=4)
        self.assertAlmostEqual(zv, 1.0, places=4)
        # the black-box relation itself holds
        self.assertAlmostEqual(xv ** 2 + yv ** 2, zv, places=4)

    @unittest.skipIf(not _ipopt_route_available('cyipopt'), 'cyipopt is not available')
    def test_greybox_auto_routes_to_cyipopt(self):
        from edi.solvers.ipopt import ipopt_solve

        r = ipopt_solve(_unit_circle_model(), method='auto')
        self.assertEqual(r['solver'], 'cyipopt')


if __name__ == '__main__':
    unittest.main()


def _gp_known_optimum():
    """min x*y  s.t.  x >= 1, y >= 2.  Optimum: x=1, y=2, objective=2."""
    f = Formulation()
    f.Variable(name='x', guess=3.0, units='m', description='x')
    f.Variable(name='y', guess=3.0, units='m', description='y')
    f.Objective(f.x * f.y)
    f.ConstraintList([f.x >= 1.0 * units.m, f.y >= 2.0 * units.m])
    return f


@unittest.skipIf(not formulation_available, 'Formulation import failed')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
class TestConvexIpoptBackend(unittest.TestCase):
    """IPOPT as an alternative backend for structured (convex) problems.

    This is an option alongside cvxopt, not a replacement. A geometric program is
    solved in log space, where it is convex, so the global-optimality guarantee is
    preserved rather than being traded for general-NLP behavior.
    """

    @unittest.skipIf(not (_ipopt_route_available('pyomo')
                          or _ipopt_route_available('cyipopt')),
                     'no IPOPT backend available')
    def test_gp_via_ipopt_matches_analytic_optimum(self):
        from edi.presolve.structureDetector import structure_detector
        from edi.presolve.unitCorrector import unit_corrector
        from edi.solvers.ipopt.convex import solve_gp_ipopt

        f = _gp_known_optimum()
        s = structure_detector(unit_corrector(f))
        r = solve_gp_ipopt(s, model=f)
        self.assertEqual(r['status'], 'optimal')
        self.assertAlmostEqual(r['primal objective'], 2.0, places=5)
        self.assertAlmostEqual(pyo.value(f.x), 1.0, places=5)
        self.assertAlmostEqual(pyo.value(f.y), 2.0, places=5)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    @unittest.skipIf(not (_ipopt_route_available('pyomo')
                          or _ipopt_route_available('cyipopt')),
                     'no IPOPT backend available')
    def test_gp_backends_agree(self):
        """cvxopt and the IPOPT log-space path must reach the same optimum."""
        from edi.presolve.structureDetector import structure_detector
        from edi.presolve.unitCorrector import unit_corrector
        from edi.solvers.solver import cvxopt_solve
        from edi.solvers.ipopt.convex import solve_gp_ipopt

        f1 = _gp_known_optimum()
        r1 = cvxopt_solve(f1)
        f2 = _gp_known_optimum()
        r2 = solve_gp_ipopt(structure_detector(unit_corrector(f2)), model=f2)

        self.assertAlmostEqual(r1['primal objective'], r2['primal objective'], places=5)
        self.assertAlmostEqual(pyo.value(f1.x), pyo.value(f2.x), places=5)
        self.assertAlmostEqual(pyo.value(f1.y), pyo.value(f2.y), places=5)

    @unittest.skipIf(not (_ipopt_route_available('pyomo')
                          or _ipopt_route_available('cyipopt')),
                     'no IPOPT backend available')
    def test_solve_dispatcher_honours_convex_backend(self):
        from edi.solvers.solver import solve

        f = _gp_known_optimum()
        r = solve(f, convex_backend='ipopt')
        self.assertIn('ipopt', str(r.get('solver', '')).lower())
        self.assertAlmostEqual(pyo.value(f.x), 1.0, places=5)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_solve_defaults_to_cvxopt_for_structured(self):
        from edi.solvers.solver import solve

        f = _gp_known_optimum()
        r = solve(f)
        self.assertEqual(r.get('problem_structure'), 'geometric_program')
        self.assertAlmostEqual(pyo.value(f.x), 1.0, places=4)


def _indexed_gp(n=4):
    """min sum_i sK[i] + y  s.t.  sK[i]*y >= 2, y >= 0.5.

    With sK[i] = 2/y at the optimum the objective is 2n/y + y, minimized at
    y = sqrt(2n) and sK[i] = 2/sqrt(2n). For n = 4 that is y = 2*sqrt(2) and
    sK[i] = 1/sqrt(2).
    """
    f = Formulation()
    f.Variable(name='sK', guess=1.0, units='', description='sK', size=n)
    f.Variable(name='y', guess=1.0, units='', description='y')
    f.Objective(sum(f.sK[i] for i in range(n)) + f.y)
    cons = [f.sK[i] * f.y >= 2.0 * units.dimensionless for i in range(n)]
    cons.append(f.y >= 0.5 * units.dimensionless)
    f.ConstraintList(cons)
    return f


@unittest.skipIf(not formulation_available, 'Formulation import failed')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
class TestIndexedVariableWriteBack(unittest.TestCase):
    """Write-back must work for INDEXED variables, e.g. Variable(..., size=16).

    Regression guard for two coupled bugs:

    * ``structure_detector`` published only the VarData objects from the
      ``unit_corrector`` clone. Callers write ``structure_detector(unit_corrector(m))``,
      leaving the clone unreferenced; once it was collected the parent IndexedVar
      went with it and every VarData reported its name as '[Unattached VarData]'.
      Write-back then fed that string to ``find_component``, and ComponentUID
      raised ``TypeError: attribute name must be string, not 'NoneType'``.
      Scalar variables were unaffected, because there the VarData *is* the
      component that ``structures['variables']`` keeps alive -- so the failure
      only ever showed up on indexed variables.
    * ``solve()``'s auto path swallowed that exception and fell through to raw
      IPOPT, which failed later and for an unrelated-looking reason.
    """

    def test_structures_keep_the_corrected_model_alive(self):
        import gc
        from edi.presolve.structureDetector import structure_detector
        from edi.presolve.unitCorrector import unit_corrector

        # The clone is deliberately not bound to a local here: this is exactly
        # how callers invoke it, and it is what used to strand the VarData.
        s = structure_detector(unit_corrector(_indexed_gp()))
        gc.collect()
        names = [v.name for v in s['variables']]
        self.assertEqual(names, ['sK[0]', 'sK[1]', 'sK[2]', 'sK[3]', 'y'])

    def test_write_solution_resolves_indexed_variables(self):
        import gc
        from edi.presolve.structureDetector import structure_detector
        from edi.presolve.unitCorrector import unit_corrector
        from edi.solvers.writeback import write_solution

        f = _indexed_gp()
        s = structure_detector(unit_corrector(f))
        gc.collect()
        written = write_solution(s, {'x': [1.0, 2.0, 3.0, 4.0, 5.0]}, model=f)
        self.assertEqual(sorted(written), ['sK[0]', 'sK[1]', 'sK[2]', 'sK[3]', 'y'])
        # written onto the CALLER's model, not the unit-corrected clone
        self.assertAlmostEqual(pyo.value(f.sK[2]), 3.0)
        self.assertAlmostEqual(pyo.value(f.y), 5.0)

    @unittest.skipIf(not cvxopt_available, 'cvxopt is not installed')
    def test_indexed_gp_solves_and_writes_back_cvxopt(self):
        from edi.solvers.solver import solve

        f = _indexed_gp()
        r = solve(f, solver='auto')
        self.assertEqual(r.get('problem_structure'), 'geometric_program')
        self.assertIsNone(r.get('writeback_error'))
        for i in range(4):
            self.assertAlmostEqual(pyo.value(f.sK[i]), 2 ** -0.5, places=4)
        self.assertAlmostEqual(pyo.value(f.y), 8 ** 0.5, places=4)

    @unittest.skipIf(not (_ipopt_route_available('pyomo')
                          or _ipopt_route_available('cyipopt')),
                     'no IPOPT backend available')
    def test_indexed_gp_solves_and_writes_back_convex_ipopt(self):
        import warnings
        from edi.solvers.solver import solve

        f = _indexed_gp()
        with warnings.catch_warnings():
            # a fallback to raw IPOPT now warns; make that a hard failure here
            warnings.simplefilter('error', RuntimeWarning)
            r = solve(f, solver='auto', convex_backend='ipopt')
        self.assertEqual(r['status'], 'optimal')
        # the log-space convex path, NOT a silent fallback to raw IPOPT (which
        # happens to reach the same point here, so the optimum alone proves
        # nothing about which route ran)
        self.assertIn('log-transformed', str(r.get('solver', '')))
        self.assertEqual(r.get('problem_structure'), 'geometric_program')
        for i in range(4):
            self.assertAlmostEqual(pyo.value(f.sK[i]), 2 ** -0.5, places=4)
        self.assertAlmostEqual(pyo.value(f.y), 8 ** 0.5, places=4)

    @unittest.skipIf(not (_ipopt_route_available('pyomo')
                          or _ipopt_route_available('cyipopt')),
                     'no IPOPT backend available')
    def test_auto_path_survives_garbage_collection(self):
        """The clone must outlive a GC pass that lands mid-solve.

        Pyomo blocks are freed by the cyclic collector rather than by refcount,
        so before the fix this failure depended on when a collection happened to
        run -- it showed up on real (larger) models and not on small ones.
        Forcing a collection right after detection makes it deterministic.
        """
        import gc
        from edi.solvers import solver as solver_module

        original = solver_module._convex_ipopt

        def _collect_then_solve(*a, **k):
            gc.collect()                    # detection is done; the clone is on
            return original(*a, **k)        # its own from here

        solver_module._convex_ipopt = _collect_then_solve
        try:
            f = _indexed_gp()
            r = solver_module.solve(f, solver='auto', convex_backend='ipopt')
        finally:
            solver_module._convex_ipopt = original

        self.assertIn('log-transformed', str(r.get('solver', '')))
        for i in range(4):
            self.assertAlmostEqual(pyo.value(f.sK[i]), 2 ** -0.5, places=4)

    def test_auto_fallback_warns_instead_of_swallowing(self):
        """A failing structured backend must not fall through in silence."""
        import warnings
        from edi.solvers import solver as solver_module

        def _boom(*a, **k):
            raise RuntimeError('structured backend exploded')

        # Patch whichever backend `auto` actually reaches. The default convex
        # backend is ipopt; cvxopt is only used when asked for explicitly.
        original = solver_module._convex_ipopt
        solver_module._convex_ipopt = _boom
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    solver_module.solve(_gp_known_optimum(), solver='auto')
                except Exception:
                    pass                    # IPOPT may be absent; the warning is the point
            messages = [str(w.message) for w in caught]
            self.assertTrue(any('structured backend exploded' in msg for msg in messages),
                            msg=f'no explanatory warning was issued; got {messages}')
        finally:
            solver_module._convex_ipopt = original

    def test_cvxopt_is_still_reachable_on_request(self):
        """Changing the default must not remove the backend."""
        from edi.solvers import solver as solver_module

        f = _gp_known_optimum()
        solver_module.solve(f, solver='auto', convex_backend='cvxopt')
        self.assertAlmostEqual(pyo.value(f.x), 1.0, places=5)
        self.assertAlmostEqual(pyo.value(f.y), 2.0, places=5)


class TestGPObjectiveForm(unittest.TestCase):
    """The GP backend can write posynomials two ways; neither suits everything.

    'sum' hands IPOPT the posynomial itself, which is better conditioned when
    log c + a.t is O(1..30) -- the JHO sailplane solves under 'sum' and fails
    IPOPT's restoration phase under 'lse'. 'lse' takes the logarithm, which is
    required once the arguments approach the exp() overflow threshold --
    SPaircraft reaches log c = 176 with exponents to 1022.7. 'auto' picks from
    the row magnitudes and retries with 'lse' if 'sum' fails.
    """

    def _box(self):
        from edi import Formulation
        f = Formulation()
        h = f.Variable(name="h", guess=1.0, units="m", description="")
        w = f.Variable(name="w", guess=1.0, units="m", description="")
        d = f.Variable(name="d", guess=1.0, units="m", description="")
        f.Objective(2 * (h * w + h * d + w * d))
        f.ConstraintList([h * w * d >= 8.0 * pyo.units.m ** 3,
                          h <= 4.0 * pyo.units.m, w <= 4.0 * pyo.units.m])
        return f

    def test_both_forms_give_the_same_optimum(self):
        from edi.solvers.ipopt.convex import solve_gp_ipopt
        from edi.presolve.structureDetector import structure_detector
        from edi.presolve.unitCorrector import unit_corrector
        answers = {}
        for form in ("sum", "lse", "auto"):
            st = structure_detector(unit_corrector(self._box()))
            answers[form] = solve_gp_ipopt(st, form=form)["primal objective"]
        self.assertAlmostEqual(answers["sum"], answers["lse"], places=5)
        self.assertAlmostEqual(answers["sum"], answers["auto"], places=5)

    def test_auto_picks_sum_for_ordinary_magnitudes(self):
        from edi.solvers.ipopt.convex import _auto_form
        groups = {0: [(1.0, [1.0, 0.0])], 1: [(2.5, [1.0, 2.0])]}
        self.assertEqual(_auto_form(groups), "sum")

    def test_auto_picks_lse_for_large_exponents(self):
        from edi.solvers.ipopt.convex import _auto_form
        groups = {0: [(1.0, [1.0, 0.0])], 1: [(1.0, [1022.7, 0.0])]}
        self.assertEqual(_auto_form(groups), "lse")

    def test_auto_picks_lse_for_large_coefficients(self):
        from edi.solvers.ipopt.convex import _auto_form
        import math
        groups = {0: [(1.0, [1.0])], 1: [(math.exp(176.0), [1.0])]}
        self.assertEqual(_auto_form(groups), "lse")

    def test_invalid_form_is_rejected(self):
        from edi.solvers.ipopt.convex import solve_gp_ipopt
        from edi.presolve.structureDetector import structure_detector
        from edi.presolve.unitCorrector import unit_corrector
        st = structure_detector(unit_corrector(self._box()))
        with self.assertRaises(ValueError):
            solve_gp_ipopt(st, form="nonsense")


@unittest.skipIf(not formulation_available, 'Formulation import failed')
class TestWritebackFailureIsAnnounced(unittest.TestCase):
    """A failed write-back must not look like a successful solve."""

    def test_writeback_failure_warns(self):
        import warnings
        from edi.solvers import solver as solver_module

        def _boom(*a, **k):
            raise RuntimeError('writeback exploded')

        original = solver_module.write_solution
        solver_module.write_solution = _boom
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                res = solver_module.cvxopt_solve(_gp_known_optimum())
            messages = [str(w.message) for w in caught]
            self.assertTrue(
                any('writing the solution back' in m for m in messages),
                msg=f'no warning issued; got {messages}')
            self.assertIn('writeback_error', res)
        finally:
            solver_module.write_solution = original


@unittest.skipIf(not formulation_available, 'Formulation import failed')
class TestConstantOnlyConstraints(unittest.TestCase):
    """A constraint with no variables is either redundant or a proof."""

    @staticmethod
    def _model(rhs):
        f = Formulation()
        x = f.Variable('x', 2.0, '-', 'x', bounds=[0.1, 10.0])
        c = f.Constant('c', 3.0, '-', 'a constant')
        f.Objective(x)
        f.Constraint(x >= 2.0)
        f.Constraint(c >= rhs)
        return f

    def test_a_true_constant_constraint_is_filtered_out(self):
        from edi.solvers import solver as solver_module
        f = self._model(1.0)                      # c = 3 >= 1, always true
        solver_module.solve(f, diagnostics='off')
        self.assertAlmostEqual(pyo.value(f.x), 2.0, places=4)

    def test_a_false_one_is_reported_as_infeasible_before_solving(self):
        """It is a proof, and the cheapest one available -- no solve needed.

        Falling through to a general NLP solver replaces "constraint X is false
        as written" with a bare termination_condition=infeasible.
        """
        from edi.presolve.reductions import InfeasibleProblem
        from edi.solvers import solver as solver_module

        with self.assertRaises(InfeasibleProblem) as ctx:
            solver_module.solve(self._model(99.0), diagnostics='off')
        msg = str(ctx.exception)
        self.assertIn('no feasible point', msg)
        self.assertIn('c = 3', msg)               # names the value
        self.assertNotIn('dimensionless', msg)    # not the unit parameters


@unittest.skipIf(not formulation_available, 'EDI import failed')
class TestSolveDetectsOnce(unittest.TestCase):
    """`solve` walks the model once, not once per consumer.

    The checks and the structured backends used to each detect for themselves,
    because `diagnose` wants bounds separated from the rows and the backends
    read them out of the rows. `diagnose` folds single-variable rows into
    bounds itself, so it reads either form -- and the walk is the expensive
    part of a solve on a large model, seconds against seconds.
    """

    def _count_detections(self, **kwargs):
        import edi.solvers.solver as solver_mod

        calls = []
        real = solver_mod.structure_detector

        def counting(*a, **kw):
            calls.append(1)
            return real(*a, **kw)

        solver_mod.structure_detector = counting
        try:
            solver_mod.solve(_linear_model(), **kwargs)
        finally:
            solver_mod.structure_detector = real
        return sum(calls)

    def test_diagnostics_on_costs_no_extra_walk(self):
        self.assertEqual(self._count_detections(diagnostics='warn'), 1)

    def test_diagnostics_off_still_detects_for_routing(self):
        self.assertEqual(self._count_detections(diagnostics='off'), 1)

    def test_the_answer_is_unchanged_either_way(self):
        for level in ('warn', 'off'):
            f = _linear_model()
            from edi.solvers.solver import solve
            solve(f, diagnostics=level)
            self.assertAlmostEqual(pyo.value(f.x), 1.0, places=5)
            self.assertAlmostEqual(pyo.value(f.y), 2.0, places=5)


class TestIpoptUnavailableFallback:
    """What `solve()` does when there is no IPOPT to be had.

    IPOPT is the default convex backend, so a machine without it used to fail
    on models cvxopt could solve perfectly well: the structured path raised,
    and the fallback was *plain IPOPT on the raw model*, which raised for the
    same reason. A detected LP/QP/GP/SP does not need IPOPT at all.

    Availability is faked by monkeypatching `_ipopt_available` rather than by
    editing PATH, so the test is unaffected by what the machine has installed
    and cannot leak a broken PATH into later tests.
    """

    @staticmethod
    def _gp():
        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='x')
        y = f.Variable(name='y', guess=1.0, units='m', description='y')
        A = f.Constant(name='A', value=2.0, units='m^2', description='area')
        f.Objective(x + y)
        f.ConstraintList([x * y >= A])
        return f

    def test_structured_model_falls_back_to_cvxopt_and_warns(self, monkeypatch):
        from edi.solvers import solver as S
        monkeypatch.setattr(S, '_ipopt_available', lambda: False)
        f = self._gp()
        with pytest.warns(RuntimeWarning, match='cvxopt instead'):
            S.solve(f, sensitivities=False)
        # min x + y subject to x*y >= 2 is 2*sqrt(2) -- the fallback must give
        # the right answer, not merely avoid raising.
        assert float(f.solution.objective) == pytest.approx(2 * 2 ** 0.5, rel=1e-6)

    def test_unstructured_model_raises_a_named_dead_end(self, monkeypatch):
        from edi.solvers import solver as S
        monkeypatch.setattr(S, '_ipopt_available', lambda: False)
        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='-', description='x')
        f.Objective(x ** 3 - 2 * x + 5)          # not LP, QP, GP or SP
        f.ConstraintList([x >= 0.1])
        with pytest.raises(RuntimeError, match='needs IPOPT'):
            S.solve(f, sensitivities=False)

    def test_ipopt_is_still_preferred_when_present(self, monkeypatch):
        from edi.solvers import solver as S
        monkeypatch.setattr(S, '_ipopt_available', lambda: True)
        f = self._gp()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            S.solve(f, sensitivities=False)
        assert not [w for w in caught if 'cvxopt instead' in str(w.message)]


class TestSuppliedStructures:
    """`solve(f, structures=...)` -- run the chain yourself and hand it back.

    The chain a plain solve runs is unit_corrector -> structure_detector ->
    diagnose -> backend -> sensitivities. Detecting is the expensive step (four
    to six seconds on SPaircraft against an eleven-second solve), so a caller
    who has already done it for a diagnose should not pay twice.
    """

    @staticmethod
    def _gp():
        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='x')
        y = f.Variable(name='y', guess=1.0, units='m', description='y')
        A = f.Constant(name='A', value=2.0, units='m^2', description='area')
        f.Objective(x + y)
        f.ConstraintList([x * y >= A])
        return f

    def test_supplied_structures_give_the_same_answer(self):
        from edi.solvers.solver import solve
        from edi.presolve.structureDetector import structure_detector
        from edi.presolve.unitCorrector import unit_corrector

        a = self._gp()
        solve(a, sensitivities=False)

        b = self._gp()
        st = structure_detector(unit_corrector(b))
        solve(b, structures=st, sensitivities=False)

        assert float(b.solution.objective) == pytest.approx(
            float(a.solution.objective), rel=1e-9)
        assert float(b.solution.objective) == pytest.approx(2 * 2 ** 0.5,
                                                            rel=1e-6)

    def test_diagnose_does_not_consume_the_structures(self):
        """Detect once, diagnose, then solve -- the whole point of sharing."""
        from edi.presolve.reductions import diagnose
        from edi.solvers.solver import solve
        from edi.presolve.structureDetector import structure_detector
        from edi.presolve.unitCorrector import unit_corrector

        f = self._gp()
        st = structure_detector(unit_corrector(f))
        rep = diagnose(st)                       # must not mutate st
        assert 'Geometric Program' in rep.structure
        solve(f, structures=st, sensitivities=False)
        assert float(f.solution.objective) == pytest.approx(2 * 2 ** 0.5,
                                                            rel=1e-6)

    def test_the_split_bounds_form_is_refused(self):
        """The dangerous one: bounds in structures['bounds'], not in the rows.

        A backend reading only rows would solve an unbounded relaxation and
        return a perfectly reasonable-looking answer to a different question.
        """
        from edi.solvers.solver import solve
        from edi.presolve.structureDetector import structure_detector
        from edi.presolve.unitCorrector import unit_corrector

        f = self._gp()
        split = structure_detector(unit_corrector(f), bounds_as_rows=False)
        with pytest.raises(ValueError, match='bounds_as_rows'):
            solve(f, structures=split, sensitivities=False)
