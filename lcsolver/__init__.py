#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  A Pyomo-based framework for engineering design optimization.
#
#  Originally developed as pyomo.contrib.edi (Pyomo PR #2937) at
#  National Technology and Engineering Solutions of Sandia, LLC.
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy's Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""LCsolver --- the Engineering Design Interface.

A lightweight wrapper on Pyomo for composing engineering design optimization
problems: units, black-box analysis codes, structure detection (LP/QP/GP/SP).

History note: this began as pyomo.contrib.edi, and after the split the
__init__ kept importing it inside a bare try/except -- every import failed
silently and `import lcsolver` exposed no API. The imports below are local
and deliberately NOT wrapped, so a broken install fails loudly.
"""

# Build all of the appropriate Pyomo machinery.
import pyomo.environ  # noqa: F401

from lcsolver.solvers.solver import solve, PresolveError, SolveResult
# The SIA warning messages tell users to pass options=SIAOptions(); the class
# has to be importable from the package root for that advice to be followable.
from lcsolver.solvers.sequential.sia import SIAOptions
from lcsolver.presolve.reductions import (presolve_check, postsolve_check,
                                          unbuilt_blocks, unbuilt_blocks_check)

# Which solvers this install actually has. Exposed here as well as on the
# command line (`lcsolver-check-solvers`) because "why is this model slow" and
# "which linear solver am I on" are the same question more often than not.
from lcsolver.environment import check_solvers

from lcsolver.objects.formulation import Formulation

from lcsolver.objects.submodel import SubModel

from lcsolver.objects.constraintgenerator import ConstraintGenerator

# The vector-shape vocabulary: generators that take caller-declared arrays
# use these to normalize and to refuse ambiguous shapes with the library's
# own error rather than a bare ValueError.
from lcsolver.objects.vector import ShapeMismatch, as_array

from lcsolver.objects.blackBoxFunctionModel import (
    BlackBoxFunctionModel,
    BlackBoxFunctionModel_Variable,
    BlackBoxFunctionModel_Variable as BlackBoxVariable,
    BlackBoxFunctionModel_Variable as BBVariable,
    BlackBoxFunctionModel_Variable as BBV,
    BBList,
    TypeCheckedList,
)

from lcsolver.postsolve.sensitivity import (
    sensitivities,
    constraint_duals,
    format_sensitivities,
)
from lcsolver.presolve import (FeasibilityResult, feasibility,
                          optimization_check, structure_detector,
                          structure_report, rigidity_report,
                          rigidity_text, UnitCheck, unit_check,
                          unit_corrector)

# `from lcsolver import units` gives Pyomo's units container for the places
# that need the object (`1.0 * units.m`, `units.convert(...)`). Safe now that
# nothing else is called `units`: when lcsolver.units was a package, Python
# binding the submodule onto the parent silently replaced this name
# mid-session and turned `units.m` into an AttributeError.
from pyomo.environ import units


__all__ = [
    "Formulation",
    "SubModel",
    "ConstraintGenerator",
    "ShapeMismatch",
    "as_array",
    "units",
    "optimization_check",
    "structure_report",
    "rigidity_report",
    "rigidity_text",
    "structure_detector",
    "unit_check",
    "UnitCheck",
    "unit_corrector",
    "feasibility",
    "FeasibilityResult",
    "sensitivities",
    "constraint_duals",
    "format_sensitivities",
    "BlackBoxFunctionModel",
    "BlackBoxFunctionModel_Variable",
    "BlackBoxVariable",
    "BBVariable",
    "BBV",
    "BBList",
    "TypeCheckedList",
    # solve entry points: these were reachable as lcsolver.solve but missing
    # from __all__, so `from lcsolver import *` couldn't solve anything
    "solve",
    "SIAOptions",
    "SolveResult",
    "PresolveError",
    "presolve_check",
    "postsolve_check",
    "unbuilt_blocks",
    "unbuilt_blocks_check",
    "check_solvers",
]

__version__ = "0.1.0"  # keep in sync with pyproject.toml
