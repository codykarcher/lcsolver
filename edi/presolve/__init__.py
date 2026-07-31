#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Everything that happens to a model before a solver sees it.

These were three separate places -- ``edi.units``, ``edi.structure`` and a
top-level ``edi.presolve`` module -- which hid the fact that they are one
pipeline, run in one order, each step consuming the last:

1. :func:`~edi.presolve.unitCorrector.unit_check` validates that the
   constraints balance dimensionally and returns a clone converted to base
   units. Nothing downstream is meaningful until this passes.
2. :mod:`~edi.presolve.structureDetector` walks that clone and classifies
   it -- LP, QP, GP, SP -- publishing the rows, the variable ordering and the
   bounds every backend reads.
3. :mod:`~edi.presolve.reductions` folds, eliminates and reduces what the
   detector found, and reports on it -- :func:`diagnose`,
   :func:`structure_report`.

Moving them together also frees the name ``edi.units`` to mean what a modeller
expects: Pyomo's units container, re-exported as ``from edi import units``. It
used to be this subpackage, and importing a submodule of it would silently
rebind that name.

The chain, which ``solve()`` runs internally and will accept back::

    from edi.presolve import unit_check, structure_detector, diagnose, feasibility

    check      = unit_check(f)                  # print(check.summary())
    structures = structure_detector(check)      # print(structures.summary())
    report     = diagnose(structures)           # print(report.summary())
    start      = feasibility(structures)        # print(start.summary())
    solve(f, structures=structures, start=start)
"""

from edi.presolve.feasibilityCheck import (  # noqa: F401
    FeasibilityResult,
    feasibility,
)
from edi.presolve.reductions import (  # noqa: F401
    InfeasibleProblem,
    PresolveReport,
    diagnose,
    presolve,
    presolve_report,
    structure_report,
)
from edi.presolve.structureDetector import (  # noqa: F401
    structure_detector,
)
from edi.presolve.unitCorrector import (  # noqa: F401
    UnitCheck,
    UnitMismatch,
    unit_check,
    unit_corrector,
)

__all__ = [
    "unit_check",
    "UnitCheck",
    "unit_corrector",
    "UnitMismatch",
    "structure_detector",
    "diagnose",
    "structure_report",
    "presolve",
    "presolve_report",
    "PresolveReport",
    "InfeasibleProblem",
    "feasibility",
    "FeasibilityResult",
]
