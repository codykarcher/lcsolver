#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""`from edi import units` must give Pyomo's units container.

This is a name collision worth pinning. `edi.units` is also a subpackage
(unitCorrector, unitWalker), and Python binds a submodule onto its parent
package as it imports it -- so `from edi.units.unitCorrector import ...`
anywhere would overwrite the name with the subpackage and turn `units.m` into
an AttributeError, at whatever point in a session that import first ran.

edi/__init__ loads the subpackage before rebinding, so the parent attribute is
set once and then replaced. These tests fix that behaviour in both orderings
and after the lazy imports a solve performs.
"""
import subprocess
import sys

import pyomo.environ as pyo
import pytest

from edi import Formulation, units


def test_it_is_pyomos_container():
    assert units is pyo.units
    assert str(units.m) == 'm'


def test_usable_in_a_model_and_solves():
    from edi.solvers.solver import solve
    f = Formulation()
    x = f.Variable(name='x', guess=5.0, units='m', description='x')
    y = f.Variable(name='y', guess=5.0, units='m', description='y')
    f.Objective(x + y)
    f.ConstraintList([x >= 1.0 * units.m, y >= 2.0 * units.m])
    solve(f, sensitivities=False)
    assert float(f.solution['x']) == pytest.approx(1.0, abs=1e-4)
    assert float(f.solution['y']) == pytest.approx(2.0, abs=1e-4)


def test_survives_the_submodule_imports_a_solve_performs():
    """The clobber this guards against, in the order that would trigger it."""
    import edi
    from edi.units.unitCorrector import unit_corrector    # noqa: F401
    from edi.units.unitWalker import unitsPack            # noqa: F401
    assert edi.units is pyo.units


@pytest.mark.parametrize('first', ['edi', 'submodule'],
                         ids=['edi-first', 'submodule-first'])
def test_both_import_orderings_in_a_fresh_interpreter(first):
    """Import order must not decide what `edi.units` means.

    Run out of process: within one session the modules are already in
    sys.modules, which is precisely the state that hides this bug.
    """
    lead = ('import edi' if first == 'edi'
            else 'from edi.units.unitCorrector import unit_corrector')
    code = (f'{lead}\n'
            'import edi, pyomo.environ as pyo\n'
            'from edi.units.unitCorrector import unit_corrector\n'
            'assert edi.units is pyo.units, type(edi.units)\n'
            'from edi import units\n'
            'assert units is pyo.units\n'
            'print("ok")\n')
    out = subprocess.run([sys.executable, '-c', code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert 'ok' in out.stdout


def test_the_subpackage_is_still_importable():
    """Rebinding the name must not make the real subpackage unreachable."""
    from edi.units import unitCorrector
    assert callable(unitCorrector.unit_corrector)
