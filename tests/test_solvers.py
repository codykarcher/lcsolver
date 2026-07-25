#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests for solution write-back and the IPOPT interface."""

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
        from edi.structure.structureDetector import structure_detector
        from edi.units.unitCorrector import unit_corrector

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
        from edi.structure.structureDetector import structure_detector
        from edi.units.unitCorrector import unit_corrector
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
        from edi.structure.structureDetector import structure_detector
        from edi.units.unitCorrector import unit_corrector
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
        from edi.structure.structureDetector import structure_detector
        from edi.units.unitCorrector import unit_corrector

        # The clone is deliberately not bound to a local here: this is exactly
        # how callers invoke it, and it is what used to strand the VarData.
        s = structure_detector(unit_corrector(_indexed_gp()))
        gc.collect()
        names = [v.name for v in s['variables']]
        self.assertEqual(names, ['sK[0]', 'sK[1]', 'sK[2]', 'sK[3]', 'y'])

    def test_write_solution_resolves_indexed_variables(self):
        import gc
        from edi.structure.structureDetector import structure_detector
        from edi.units.unitCorrector import unit_corrector
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

        original = solver_module.cvxopt_solve
        solver_module.cvxopt_solve = _boom
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
            solver_module.cvxopt_solve = original
