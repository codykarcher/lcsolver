#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Grey-box (black-box) constraints must survive cloning and solve correctly.

Regression tests for two defects that together produced silently wrong
answers on any black-box formulation:

1. ``BBList`` (the black box's input/output container) raised inside
   ``copy.deepcopy``, so ``model.clone()`` -- which ``unit_corrector`` runs on
   every solve -- silently replaced each grey-box block's ``_ex_model`` with
   ``None``.
2. The auto-router classified the model from that gutted clone, saw a pure
   GP, and solved it *without* the black-box rows, returning a confidently
   wrong optimum with no warning.

The fix gives ``BBList`` a correct ``__deepcopy__`` and routes formulations
with grey-box constraints to SIA, which imposes each box as an opaque
signomial row.
"""

import pytest

pyo = pytest.importorskip('pyomo.environ')

from pyomo.environ import units  # noqa: E402

from lcsolver import Formulation, BlackBoxFunctionModel  # noqa: E402


class UnitCircle(BlackBoxFunctionModel):
    """z = x**2 + y**2, with exact first derivatives."""

    def __init__(self):
        super().__init__()
        self.description = 'z = x**2 + y**2'
        self.inputs.append(name='x', units='', description='x')
        self.inputs.append(name='y', units='', description='y')
        self.outputs.append(name='z', units='', description='z')
        self.availableDerivative = 1
        self.post_init_setup(len(self.inputs))

    def BlackBox(self, x, y):
        x, y = pyo.value(x), pyo.value(y)
        z = (x ** 2 + y ** 2) * units.dimensionless
        return z, [2 * x * units.dimensionless, 2 * y * units.dimensionless]


def _build():
    """min x + y  s.t.  z = bb(x, y),  z >= 1.

    Optimum x = y = 1/sqrt(2), objective sqrt(2): the circle constraint is
    active and only the black box knows its shape.
    """
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='', bounds=[0.01, 10.0])
    y = f.Variable(name='y', guess=1.0, units='', bounds=[0.01, 10.0])
    z = f.Variable(name='z', guess=2.0, units='', bounds=[0.01, 100.0])
    f.Objective(x + y)
    f.ConstraintList([
        [z, '==', [x, y], UnitCircle()],
        z >= 1.0 * units.dimensionless,
    ])
    return f


def test_clone_preserves_ex_model():
    """model.clone() must keep the grey-box external model, remapped."""
    f = _build()
    clone = f.clone()
    bb = clone.constraint_1._ex_model
    assert bb is not None, 'clone dropped the grey-box external model'
    assert [v.name for v in bb.inputs] == ['x', 'y']
    # the deepcopy must remap the optimization vars to the clone's, not
    # leave them pointing into the original model
    assert bb.inputVariables_optimization[0] is clone.x
    assert bb.inputVariables_optimization[0] is not f.x


def test_deepcopy_bblist_directly():
    import copy

    uc = UnitCircle()
    dup = copy.deepcopy(uc)
    assert [v.name for v in dup.inputs] == ['x', 'y']
    assert [v.name for v in dup.outputs] == ['z']
    assert dup.inputs['x'] is not uc.inputs['x']


def test_auto_solve_routes_blackbox_to_sia():
    """Plain solve(f) on a black-box model: SIA route, correct optimum."""
    from lcsolver.solvers.solver import solve

    f = _build()
    res = solve(f, sensitivities=False)
    assert 'SIA' in res['solver']
    assert res['primal objective'] == pytest.approx(2.0 ** 0.5, rel=1e-6)
    assert pyo.value(f.x) == pytest.approx(2.0 ** -0.5, rel=1e-5)
    assert pyo.value(f.z) == pytest.approx(1.0, rel=1e-6)
