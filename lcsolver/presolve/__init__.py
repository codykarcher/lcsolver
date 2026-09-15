#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Everything that happens to a model before a solver sees it.

One pipeline, run in order:
1. unit_check validates dimensions, returns a clone in base units
2. structure_detector classifies the clone (LP, QP, GP, SP) and publishes
   the rows, variable ordering, and bounds the backends read
3. reductions folds/eliminates what the detector found and reports on it

Merging these also freed lcsolver.units to mean Pyomo's units container.

solve() runs this chain internally and will accept it back:

    check      = unit_check(f)
    structures = structure_detector(check)
    report     = optimization_check(structures)
    start      = feasibility(structures)
    solve(f, structures=structures, start=start)
"""

from lcsolver.presolve.feasibilityCheck import (  # noqa: F401
    FeasibilityResult,
    feasibility,
)
from lcsolver.presolve.reductions import (  # noqa: F401
    InfeasibleProblem,
    PresolveReport,
    optimization_check,
    presolve,
    presolve_report,
    structure_report,
    rigidity_report,
    rigidity_text,
)
from lcsolver.presolve.structureDetector import (  # noqa: F401
    structure_detector,
)
from lcsolver.presolve.unitCorrector import (  # noqa: F401
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
    "optimization_check",
    "structure_report",
    "rigidity_report",
    "rigidity_text",
    "presolve",
    "presolve_report",
    "PresolveReport",
    "InfeasibleProblem",
    "feasibility",
    "FeasibilityResult",
]
