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
        x, y = self.sanitizeInputs(x, y, strip_units=True)
        return self.packOutputs(x ** 2 + y ** 2, [2 * x, 2 * y])


def _build():
    """min 1/x + 1/y  s.t.  z = bb(x, y),  z <= 1.

    Unique optimum x = y = 1/sqrt(2), objective 2 sqrt(2): the circle
    constraint is active and only the black box knows its shape.

    (History: this used to be min x + y with z >= 1, expecting sqrt(2) ---
    but the symmetric point only satisfies FIRST-ORDER KKT there; it is a
    local max of x + y along the arc, and the true minimum sits at the
    x-bound corner (objective ~1.0099).  The old SIA stopped at the
    saddle and the test enshrined it; the 2026-08 backtracking regime
    correctly escapes to the better corner, so the model is re-posed to
    make the symmetric point the genuine, interior, unique optimum.)
    """
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


def test_sanitize_inputs_strip_units():
    """strip_units converts to declared units FIRST, then strips."""

    class FtBox(BlackBoxFunctionModel):
        def __init__(self):
            super().__init__()
            self.inputs.append(name='x', units='ft', description='x')
            self.outputs.append(name='z', units='ft**2', description='z')
            self.availableDerivative = 1

        def BlackBox(self, x):
            x = self.sanitizeInputs(x, strip_units=True)
            return self.packOutputs(x ** 2, [2 * x])

    bb = FtBox()
    x = bb.sanitizeInputs(1.0 * units.m, strip_units=True)
    assert isinstance(x, float)
    assert x == pytest.approx(3.280839895, rel=1e-9)  # 1 m in ft
    # default behavior unchanged: units kept
    xq = bb.sanitizeInputs(1.0 * units.m)
    assert pyo.value(units.convert(xq, units.ft)) == pytest.approx(x)


def test_pack_outputs_units_and_shapes():
    class FtBox(BlackBoxFunctionModel):
        def __init__(self):
            super().__init__()
            self.inputs.append(name='x', units='ft', description='x')
            self.inputs.append(name='y', units='ft', description='y')
            self.outputs.append(name='z', units='ft**2', description='z')
            self.availableDerivative = 1

        def BlackBox(self, x, y):
            x, y = self.sanitizeInputs(x, y, strip_units=True)
            return self.packOutputs(x ** 2 + y ** 2, [2 * x, 2 * y])

    bb = FtBox()
    # raw numbers get the declared units; the jacobian gets out/in units
    z, grad = bb.BlackBox(1.0 * units.m, 0.5 * units.m)
    assert pyo.value(units.convert(z, units.m ** 2)) == pytest.approx(1.25)
    assert pyo.value(units.convert(grad[0], units.m)) == pytest.approx(2.0)
    assert pyo.value(units.convert(grad[1], units.m)) == pytest.approx(1.0)
    # values-only form, and the count guard
    zonly = bb.packOutputs(4.0)
    assert pyo.value(units.convert(zonly, units.ft ** 2)) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        bb.packOutputs([1.0, 2.0])
    with pytest.raises(ValueError):
        bb.packOutputs(1.0, [1.0])  # jacobian must have one entry per input


def test_auto_solve_routes_blackbox_to_sia():
    """Plain solve(f) on a black-box model: SIA route, correct optimum."""
    from lcsolver.solvers.solver import solve

    f = _build()
    res = solve(f, sensitivities=False)
    assert 'SIA' in res['solver']
    assert res['primal objective'] == pytest.approx(2.0 * 2.0 ** 0.5,
                                                    rel=1e-6)
    assert pyo.value(f.x) == pytest.approx(2.0 ** -0.5, rel=1e-5)
    assert pyo.value(f.z) == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Variables bounded only THROUGH the box: the pre-solve gate and the KKT
# sensitivity recovery both have to account for the grey-box rows the
# algebraic scans cannot see.
# ---------------------------------------------------------------------------
def _bounded_through_the_box():
    """min w  s.t.  w >= c0 + z,  x >= xmin,  y >= ymin,  z == bb(x, y).

    Optimum x = xmin = 2, y = ymin = 1, z = xmin**2 + ymin**2 = 5, w = 6.
    Analytic log-log sensitivities: c0 -> c0/w = 1/6,
    xmin -> 2*xmin**2/w = 4/3, ymin -> 2*ymin**2/w = 1/3.

    To the algebraic rows alone, x and y are unbounded above (and look
    output-only: one monotone row each, free to escape upward) and z is
    unbounded below. All three are pinned by the box.
    """
    f = Formulation()
    x = f.Variable(name='x', guess=3.0, units='', description='x')
    y = f.Variable(name='y', guess=2.0, units='', description='y')
    z = f.Variable(name='z', guess=10.0, units='', description='bb output')
    w = f.Variable(name='w', guess=12.0, units='', description='objective')
    c0 = f.Constant(name='c0', value=1.0, units='', description='offset')
    xmin = f.Constant(name='xmin', value=2.0, units='', description='x floor')
    ymin = f.Constant(name='ymin', value=1.0, units='', description='y floor')
    f.Objective(w)
    f.ConstraintList([
        w >= c0 + z,
        x >= xmin,
        y >= ymin,
        [z, '==', [x, y], UnitCircle()],
    ])
    return f


def test_box_computed_variables_do_not_trip_the_gate():
    """The structural findings must exclude every box-referenced variable."""
    from lcsolver.presolve.reductions import presolve_check

    rep = presolve_check(_bounded_through_the_box())
    for name in ('x', 'y', 'z'):
        assert name not in rep.unbounded_above
        assert name not in rep.unbounded_below
        assert name not in rep.empty_columns
        assert name not in rep.output_columns


def test_clean_box_model_solves_without_findings():
    """Default diagnostics: no LC-E001 gate error, and no LC-W104 note --
    x and y look output-only to the rows but are not peelable."""
    import lcsolver

    sol = lcsolver.solve(_bounded_through_the_box())
    assert str(sol.status) == 'optimal'
    assert sol.variables('w') == pytest.approx(6.0, rel=1e-4)
    assert sol.variables('z') == pytest.approx(5.0, rel=1e-4)
    assert not any('LC-W104' in m or 'LC-E001' in m for m in sol.messages)


def test_greybox_duals_certify_and_sensitivities_are_exact():
    """The KKT stationarity system must carry the grey-box columns: without
    them every black-box model warned LC-W302 and the duals of the ordinary
    constraints -- hence every sensitivity -- came out wrong."""
    import lcsolver
    from lcsolver import sensitivities

    f = _bounded_through_the_box()
    sol = lcsolver.solve(f)
    assert not any('LC-W302' in m for m in sol.messages)
    s = sensitivities(f)['sensitivities']
    assert s['c0'] == pytest.approx(1.0 / 6.0, rel=1e-2)
    assert s['xmin'] == pytest.approx(4.0 / 3.0, rel=1e-2)
    assert s['ymin'] == pytest.approx(1.0 / 3.0, rel=1e-2)
