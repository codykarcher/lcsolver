#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  A Pyomo-based framework for engineering design optimization.
#
#  Originally developed as pyomo.contrib.lcsolver (Pyomo PR #2937) at
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

A lightweight wrapper on Pyomo targeted at composing engineering design
optimization problems, with first-class support for units, black-box analysis
codes, and structure detection (LP/QP/GP/SP).

Note on history: this package began life as ``pyomo.contrib.lcsolver``. When it was
split into a standalone distribution the modules were reorganized into
``lcsolver.objects`` / ``lcsolver.solvers`` / ``lcsolver.presolve``, but the
package ``__init__`` continued to import from ``pyomo.contrib.lcsolver`` inside a
bare ``try/except: pass``. Because that module no longer ships with Pyomo, every
import failed silently and ``import lcsolver`` exposed none of its own API. The
imports below are local and are deliberately NOT wrapped in a bare except, so
that a broken install fails loudly instead of producing an empty namespace.
"""

# Build all of the appropriate Pyomo machinery.
import pyomo.environ  # noqa: F401

from lcsolver.solvers.solver import solve, PresolveError, SolveResult
from lcsolver.presolve.reductions import presolve_check, postsolve_check

# Which solvers this install actually has. Exposed here as well as on the
# command line (`lcsolver-check-solvers`) because "why is this model slow" and
# "which linear solver am I on" are the same question more often than not.
from lcsolver.environment import check_solvers

from lcsolver.objects.formulation import Formulation

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

# `from lcsolver import units` gives Pyomo's units container, so a model needs one
# import rather than two. Declaring a Variable already takes units as a string;
# this is for the places that need the object -- `1.0 * units.m` on the right
# of a constraint, `units.convert(...)` inside a black box.
#
# This is a plain re-export and needs no care, because nothing else is called
# `units` any more. It used to need a great deal: `lcsolver.units` was the package
# holding unitCorrector and unitWalker, and Python binds a submodule onto its
# parent package as it imports it, so the first `from lcsolver.units.unitCorrector
# import ...` anywhere -- including the lazy ones inside solve() -- silently
# replaced this name with the package and turned `units.m` into an
# AttributeError, mid-session. Those modules now live in `lcsolver.presolve`
# with the rest of the pre-solve chain, and the name is free.
from pyomo.environ import units


__all__ = [
    "Formulation",
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
]

__version__ = "0.1.0"  # keep in sync with pyproject.toml
