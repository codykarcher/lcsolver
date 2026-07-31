#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""`from edi import units` gives Pyomo's units container.

This used to be a name collision. ``edi.units`` was the package holding
unitCorrector and unitWalker, and Python binds a submodule onto its parent
package as it imports it -- so the first ``from edi.units.unitCorrector import
...`` anywhere, including the lazy imports inside ``solve()``, replaced the
name with the package and turned ``units.m`` into an AttributeError partway
through a session.

Those modules now live in :mod:`edi.preconditioner` with the rest of the
pre-solve chain, so the name is simply free. These tests pin both halves: the
proxy is Pyomo's own object, and nothing in the package reclaims the name.
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


def test_nothing_reclaims_the_name():
    """The regression that motivated the move.

    Importing the pre-solve modules must leave ``edi.units`` alone. It did not
    when they lived under that name.
    """
    import edi
    from edi.preconditioner.unitCorrector import unit_corrector   # noqa: F401
    from edi.preconditioner.unitWalker import unitsPack           # noqa: F401
    assert edi.units is pyo.units


def test_edi_units_is_not_a_package_any_more():
    """``import edi.units.<anything>`` must fail rather than resolve."""
    with pytest.raises(ModuleNotFoundError):
        __import__('edi.units.unitCorrector')


@pytest.mark.parametrize('first', ['edi', 'preconditioner'],
                         ids=['edi-first', 'preconditioner-first'])
def test_both_import_orderings_in_a_fresh_interpreter(first):
    """Import order must not decide what ``edi.units`` means.

    Run out of process: within one session the modules are already in
    sys.modules, which is precisely the state that hid the original bug.
    """
    lead = ('import edi' if first == 'edi'
            else 'from edi.preconditioner.unitCorrector import unit_corrector')
    code = (f'{lead}\n'
            'import edi, pyomo.environ as pyo\n'
            'from edi.preconditioner.unitCorrector import unit_corrector\n'
            'from edi.preconditioner.structureDetector import structure_detector\n'
            'assert edi.units is pyo.units, type(edi.units)\n'
            'from edi import units\n'
            'assert units is pyo.units\n'
            'print("ok")\n')
    out = subprocess.run([sys.executable, '-c', code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert 'ok' in out.stdout


def test_the_preconditioner_package_exposes_the_chain():
    """One import for the whole pre-solve pipeline."""
    from edi.preconditioner import (diagnose, structure_detector,
                                    structure_report, unit_corrector)
    assert all(callable(fn) for fn in (unit_corrector, structure_detector,
                                       diagnose, structure_report))
