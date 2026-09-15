#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Output-only variables must not gate the solve -- and must be peeled right.

Contract: an output-only variable (one constraint computes it, nothing uses
it) does not raise; the solve notes LC-W104, peels it plus its defining row,
and recovers the value afterwards. Unbounded-but-not-output-only still
raises. A grey-box-fed variable is never peelable, and the peel runs on
grey-box models with every grey-box-referenced column protected.
"""

import numpy as np
import pytest

pytest.importorskip('lcsolver.solvers.sequential.sia')
pyo = pytest.importorskip('pyomo.environ')

import lcsolver  # noqa: E402
from lcsolver import BlackBoxFunctionModel, Formulation  # noqa: E402
from lcsolver.core import codes  # noqa: E402
from lcsolver.presolve.reductions import (  # noqa: E402
    fold_singleton_rows,
    presolve_check,
    reduce_columns,
)
from lcsolver.presolve.structureDetector import structure_detector  # noqa: E402
from lcsolver.presolve.unitCorrector import unit_corrector  # noqa: E402
from lcsolver.solvers.solver import PresolveError  # noqa: E402


class Square(BlackBoxFunctionModel):
    """z = x**2, with exact first derivative."""

    def __init__(self):
        super().__init__()
        self.description = 'z = x**2'
        self.inputs.append(name='x', units='', description='x')
        self.outputs.append(name='z', units='', description='z')
        self.availableDerivative = 1
        self.post_init_setup(len(self.inputs))

    def BlackBox(self, x):
        x = self.sanitizeInputs(x, strip_units=True)
        return self.packOutputs(x ** 2, [2 * x])


def _dangling_model():
    """min w s.t. w >= x + 1/x, x >= 2 - w, cd >= 0.1 + x**2. Optimum w = 2
    at x = 1; the signomial row keeps it on SIA; ``cd`` is output-only."""
    f = Formulation()
    w = f.Variable('w', 2.0, '', 'w')
    x = f.Variable('x', 1.5, '', 'x')
    cd = f.Variable('cd', 1.0, '', 'dangling output')
    f.Objective(w)
    f.ConstraintList([
        w >= x + 1.0 / x,
        x >= 2.0 - w,
        cd >= 0.1 + x ** 2,
    ])
    return f


def _greybox_dangling_model():
    """min w s.t. w >= 1 + z, z == bb(x) = x**2, x >= 2, cd dangling.
    ``cd`` is declared FIRST so peeling renumbers every grey-box column.
    Optimum x = 2, z = 4, w = 5; cd recovers to 4.1."""
    f = Formulation()
    cd = f.Variable('cd', 1.0, '', 'dangling output')
    x = f.Variable('x', 3.0, '', 'x')
    z = f.Variable('z', 9.0, '', 'bb output')
    w = f.Variable('w', 10.0, '', 'w')
    f.Objective(w)
    f.ConstraintList([
        w >= 1.0 + z,
        cd >= 0.1 + x ** 2,
        x >= 2.0,
        [z, '==', [x], Square()],
    ])
    return f


def test_dangling_output_variable_does_not_gate():
    """No PresolveError; LC-W104 note; value recovered, not arbitrary."""
    sol = lcsolver.solve(_dangling_model())
    assert str(sol.status) == 'optimal'
    assert sol.variables('w') == pytest.approx(2.0, rel=1e-4)
    # Recovered from its defining constraint at the solution, 0.1 + x**2
    # with x = 1 -- an IPOPT-parked free variable would sit anywhere.
    assert sol.variables('cd') == pytest.approx(1.1, rel=1e-3)
    assert any(codes.OUTPUT_ONLY in m for m in sol.messages)


def _gp_dangling_model():
    """min w s.t. w >= x + 1/x, cd >= 0.1 + x**2 -- a pure GP. Exercises the
    CENTRAL peel: the GP-IPOPT/cvxopt routes have no reduction pipeline, so
    the dangling variable used to ride through as a free column."""
    f = Formulation()
    w = f.Variable('w', 2.0, '', 'w')
    x = f.Variable('x', 1.5, '', 'x')
    cd = f.Variable('cd', 1.0, '', 'dangling output')
    f.Objective(w)
    f.ConstraintList([
        w >= x + 1.0 / x,
        cd >= 0.1 + x ** 2,
    ])
    return f


def test_dangling_variable_is_peeled_on_the_pure_gp_route():
    sol = lcsolver.solve(_gp_dangling_model())
    assert str(sol.status) == 'optimal'
    assert sol.variables('w') == pytest.approx(2.0, rel=1e-4)
    assert sol.variables('cd') == pytest.approx(1.1, rel=1e-3)
    assert any(codes.OUTPUT_ONLY in m for m in sol.messages)


def test_dangling_variable_is_peeled_on_the_cvxopt_route():
    pytest.importorskip('cvxopt')
    sol = lcsolver.solve(_gp_dangling_model(), convex_backend='cvxopt')
    assert str(sol.status) in ('optimal', 'ok')   # cvxopt says 'ok'
    assert sol.variables('w') == pytest.approx(2.0, rel=1e-4)
    assert sol.variables('cd') == pytest.approx(1.1, rel=1e-3)


def test_presolve_bypass_restores_the_error():
    """presolve=False skips the peel, so the gate must error like before."""
    with pytest.raises(PresolveError):
        lcsolver.solve(_gp_dangling_model(), presolve=False)
    with pytest.raises(PresolveError):
        lcsolver.solve(_dangling_model(), presolve=False)


def test_unbounded_but_not_output_only_still_raises():
    """Two rows touch cd, so it is not peelable -- the gate must hold."""
    f = Formulation()
    w = f.Variable('w', 2.0, '', 'w')
    x = f.Variable('x', 1.5, '', 'x')
    cd = f.Variable('cd', 1.0, '', 'unbounded, not output-only')
    f.Objective(w)
    f.ConstraintList([
        w >= x + 1.0 / x,
        x >= 2.0 - w,
        cd >= 0.1 + x ** 2,
        cd >= x,
    ])
    with pytest.raises(PresolveError):
        lcsolver.solve(f)


def test_greybox_fed_variable_is_not_peelable():
    """z looks output-only to the algebraic rows; the black box pins it."""
    rep = presolve_check(_greybox_dangling_model())
    assert 'cd' in rep.output_columns
    assert 'z' not in rep.output_columns


def test_dangling_variable_is_peeled_from_greybox_solve():
    """Grey-box models get the peel too, with referenced columns protected."""
    sol = lcsolver.solve(_greybox_dangling_model())
    assert str(sol.status) == 'optimal'
    # If the peel broke the identity-keyed grey-box column map, or wrongly
    # peeled `w >= 1 + z`, these move by whole units.
    assert sol.variables('w') == pytest.approx(5.0, rel=1e-4)
    assert sol.variables('z') == pytest.approx(4.0, rel=1e-4)
    assert sol.variables('x') == pytest.approx(2.0, rel=1e-4)
    assert sol.variables('cd') == pytest.approx(4.1, rel=1e-3)
    assert any(codes.OUTPUT_ONLY in m for m in sol.messages)


def test_protected_columns_survive_reduce():
    """protect= wins over both the disconnected and the output peel."""
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 10.0])
    f.Variable('y', 4.0, '', 'disconnected', bounds=[4.0, 20.0])
    f.Objective(x + 1.0 / x)
    f.Constraint(x >= 0.5)
    st = fold_singleton_rows(structure_detector(unit_corrector(f),
                                                bounds_as_rows=False))
    names = [str(v) for v in st['variables']]
    small, removed = reduce_columns(st, protect={names.index('y')})
    assert 'y' in [str(v) for v in small['variables']]
    assert not removed
    # ... and without protection it goes, as before.
    small, removed = reduce_columns(st)
    assert 'y' not in [str(v) for v in small['variables']]


def test_middle_position_peel_does_not_shift_writeback():
    """FALSIFICATION of the restore path: a peeled column that is NOT last.
    Before the length-aware branch, restore_columns shifted every value after
    the peeled index by one (found as wet < dry on the FireSat model)."""
    f = Formulation()
    a = f.Variable('a', 1.0, '', 'design var')
    p = f.Variable('p', 1.0, '', 'output-only, deliberately mid-list')
    b = f.Variable('b', 1.0, '', 'design var 2')
    c = f.Variable('c', 1.0, '', 'design var 3')
    f.Constraint(a >= 2.0)
    f.Constraint(p >= 10 * a)
    f.Constraint(b >= 3 * a)
    f.Constraint(c >= a + b)
    f.Objective(b + c)
    sol = lcsolver.solve(f)
    assert sol.variables('a') == pytest.approx(2.0, rel=1e-4)
    assert sol.variables('p') == pytest.approx(20.0, rel=1e-3)
    assert sol.variables('b') == pytest.approx(6.0, rel=1e-4)
    assert sol.variables('c') == pytest.approx(8.0, rel=1e-4)
