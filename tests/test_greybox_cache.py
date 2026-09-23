"""The sub-problem cache must serve black-box problems.

One grey-box row used to make the whole problem uncacheable, so every SIA
sub-problem rebuilt the full Pyomo model (20 min per iteration on a
14,000-row aircraft deck with 8 box rows). Now the algebraic rows are built
once and the linearized rows are rewritten in place each update.
"""
import numpy as np
import pytest
from pyomo.environ import units

from lcsolver import Formulation, BlackBoxFunctionModel


class UnitCircle(BlackBoxFunctionModel):
    def __init__(self):
        super().__init__()
        self.description = 'z = x**2 + y**2'
        self.inputs.append(name='x', units='', description='x')
        self.inputs.append(name='y', units='', description='y')
        self.outputs.append(name='z', units='', description='z')
        self.availableDerivative = 1
        self.post_init_setup(len(self.inputs))

    def BlackBox(self, x, y):
        x, y = self.sanitizeInputs(x, y, strip_units=True)
        return self.packOutputs(x ** 2 + y ** 2, [2 * x, 2 * y])


def _build():
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='', bounds=[0.01, 10.0])
    y = f.Variable(name='y', guess=1.0, units='', bounds=[0.01, 10.0])
    z = f.Variable(name='z', guess=2.0, units='', bounds=[0.01, 100.0])
    f.Objective(1.0 / x + 1.0 / y)
    f.ConstraintList([
        [z, '==', [x, y], UnitCircle()],
        z <= 1.0 * units.dimensionless,
    ])
    return f


def test_cache_is_usable_with_a_black_box():
    from lcsolver.solvers.sequential.bridge import build_problem
    from lcsolver.solvers.sequential.sia import SubproblemCache, SIAOptions, classify
    from lcsolver.presolve.structureDetector import structure_detector
    from lcsolver.presolve.unitCorrector import unit_corrector

    f = _build()
    st = structure_detector(unit_corrector(f))
    prob = build_problem(st, sp_form=True)
    assert classify(prob)[2] >= 1              # there is a linearized row
    cache = SubproblemCache(prob, SIAOptions())
    assert cache.usable
    assert cache.bb_idx                         # and the cache knows which


def test_cached_black_box_solve_matches_optimum():
    from lcsolver.solvers.solver import solve

    f = _build()
    res = solve(f, sensitivities=False)
    assert 'SIA' in res['solver']
    assert res['primal objective'] == pytest.approx(2.0 * 2.0 ** 0.5, rel=1e-6)


def test_cached_and_uncached_agree():
    from lcsolver.solvers.solver import solve
    from lcsolver.solvers.sequential.sia import SIAOptions

    f1, f2 = _build(), _build()
    r1 = solve(f1, sensitivities=False, options=SIAOptions())
    o2 = SIAOptions(); o2.cache_subproblem = False
    r2 = solve(f2, sensitivities=False, options=o2)
    assert r1['primal objective'] == pytest.approx(r2['primal objective'], rel=1e-6)
